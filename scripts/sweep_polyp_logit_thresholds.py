#!/usr/bin/env python3
"""Sweep mask-logit thresholds for an existing Polyp checkpoint in one pass."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from export_test_predictions import collate_export, configure
from ifp_alignment import build_prompt_generator
from model import Model
from utils.eval_utils import combine_instance_masks


TEST_CATEGORIES = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
    )
    return parser.parse_args()


def score_batch(pred_logits: torch.Tensor, gt_mask: torch.Tensor, thresholds: list[float]) -> list[dict[str, float]]:
    target = torch.stack([
        combine_instance_masks(mask).squeeze(0) for mask in gt_mask
    ]).unsqueeze(1).to(pred_logits.device).bool()
    scores: list[dict[str, float]] = []
    for threshold in thresholds:
        prediction = pred_logits >= threshold
        intersection = torch.logical_and(prediction, target).flatten(1).sum(dim=1).float()
        pred_area = prediction.flatten(1).sum(dim=1).float()
        gt_area = target.flatten(1).sum(dim=1).float()
        union = pred_area + gt_area - intersection
        iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
        f1 = torch.where(pred_area + gt_area > 0, 2 * intersection / (pred_area + gt_area), torch.ones_like(intersection))
        precision = torch.where(pred_area > 0, intersection / pred_area, torch.zeros_like(intersection))
        recall = torch.where(gt_area > 0, intersection / gt_area, torch.zeros_like(intersection))
        for image_iou, image_f1, image_precision, image_recall, image_pred_area in zip(
            iou.cpu().tolist(), f1.cpu().tolist(), precision.cpu().tolist(), recall.cpu().tolist(), pred_area.cpu().tolist()
        ):
            scores.append({
                "threshold": threshold,
                "iou": float(image_iou),
                "f1": float(image_f1),
                "precision": float(image_precision),
                "recall": float(image_recall),
                "empty_prediction": int(image_pred_area == 0),
            })
    return scores


def run(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(value) for value in args.thresholds]

    first_list = source_dir / "lists" / "test_Kvasir.csv"
    model_dir, checkpoint_path, _ = configure(
        "polyp", source_dir, None, first_list, "ifp",
    )
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for threshold inference")

    model = Model(cfg)
    model.setup()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    prompt_generator = build_prompt_generator(
        cfg.prompt_generator.backend,
        device,
        checkpoint_path=cfg.prompt_generator.alignment_checkpoint,
        text_prompts=cfg.prompt_generator.text_prompts,
        dino_variant=cfg.prompt_generator.dino_variant,
        clip_variant=cfg.prompt_generator.clip_variant,
        ifp_root=cfg.prompt_generator.ifp_root,
        max_positive_points=cfg.prompt_generator.max_positive_points,
        min_positive_points=cfg.prompt_generator.min_positive_points,
        include_box=cfg.prompt_generator.include_box,
    )

    per_image: list[dict] = []
    summaries: list[dict] = []
    for category in TEST_CATEGORIES:
        test_list = source_dir / "lists" / f"test_{category}.csv"
        dataset = ISICDataset(
            cfg,
            root_dir=cfg.datasets[cfg.dataset].root_dir,
            list_file=str(test_list),
            transform=ResizeAndPad(model.image_size),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_export,
        )
        category_scores = {threshold: [] for threshold in thresholds}
        with torch.no_grad():
            for names, _, _, images, gt_masks in dataloader:
                images = images.to(device)
                prompts = (
                    prompt_generator(images, gt_masks)
                    if getattr(prompt_generator, "requires_gt_masks", False)
                    else prompt_generator(images)
                )
                _, pred_masks, _, _ = model(images, prompts)
                pred_logits = (
                    torch.stack(pred_masks, dim=0)
                    if isinstance(pred_masks, (list, tuple))
                    else pred_masks
                )
                batch_scores = score_batch(pred_logits, gt_masks, thresholds)
                position = 0
                for threshold in thresholds:
                    threshold_scores = batch_scores[position:position + len(names)]
                    position += len(names)
                    category_scores[threshold].extend(threshold_scores)
                    for name, score in zip(names, threshold_scores):
                        per_image.append({
                            "dataset": category,
                            "image": name,
                            **score,
                        })

        for threshold, scores in category_scores.items():
            count = len(scores)
            for score in scores:
                score["dataset"] = category
            summary = {
                "dataset": category,
                "threshold": threshold,
                "images": count,
                "mean_iou": sum(score["iou"] for score in scores) / count,
                "mean_f1": sum(score["f1"] for score in scores) / count,
                "mean_precision": sum(score["precision"] for score in scores) / count,
                "mean_recall": sum(score["recall"] for score in scores) / count,
                "empty_prediction_count": sum(score["empty_prediction"] for score in scores),
            }
            summary["empty_prediction_rate"] = summary["empty_prediction_count"] / count
            summaries.append(summary)

    summary_path = output_dir / "threshold_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    per_image_path = output_dir / "per_image_threshold_metrics.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("dataset", "image", "threshold", "iou", "f1", "precision", "recall", "empty_prediction")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_image)
    (output_dir / "sweep_summary.json").write_text(
        json.dumps({
            "source_experiment": str(source_dir),
            "checkpoint": str(checkpoint_path),
            "gpu": args.gpu,
            "thresholds": thresholds,
            "test_sets": list(TEST_CATEGORIES),
            "note": "Diagnostic sweep; threshold selection must use validation data before final test reporting.",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote threshold sweep for {source_dir} to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
