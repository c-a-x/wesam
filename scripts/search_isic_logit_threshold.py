#!/usr/bin/env python3
"""Calibrate an ISIC checkpoint's mask-logit threshold on validation data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from export_test_predictions import SPECS, collate_export, configure
from ifp_alignment import build_prompt_generator
from model import Model
from utils.eval_utils import combine_instance_masks
from utils.prompt_policy import apply_prompt_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def thresholds(args: argparse.Namespace) -> list[float]:
    count = round((args.threshold_max - args.threshold_min) / args.threshold_step)
    return [round(args.threshold_min + i * args.threshold_step, 6) for i in range(count + 1)]


def metric_for_thresholds(
    logits: list[torch.Tensor], targets: list[torch.Tensor], values: list[float],
) -> list[dict[str, float]]:
    result = []
    for value in values:
        iou_sum = 0.0
        f1_sum = 0.0
        for logit, target in zip(logits, targets):
            prediction = (logit >= value).int()
            stats = smp.metrics.get_stats(
                prediction, target.int(), mode="binary", threshold=0.5,
            )
            iou_sum += float(smp.metrics.iou_score(*stats, reduction="micro-imagewise"))
            f1_sum += float(smp.metrics.f1_score(*stats, reduction="micro-imagewise"))
        count = len(logits)
        result.append({
            "threshold": value,
            "mean_iou": iou_sum / count,
            "mean_f1": f1_sum / count,
        })
    return result


def collect(
    model: Model,
    prompt_generator,
    dataset: ISICDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_export,
    )
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for names, paddings, original_images, images, gt_masks in loader:
            del names, paddings, original_images
            images = images.to(device)
            prompts = prompt_generator(images)
            prompts = apply_prompt_policy(
                prompts, cfg.prompt_generator, images.shape[-2:], training=False,
            )
            _, pred_masks, _, _ = model(images, prompts)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                logits.append(pred_mask.detach().cpu())
                targets.append(combine_instance_masks(gt_mask).int().cpu())
    return logits, targets


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_list = experiment_dir / "lists/validation.csv"
    test_list = experiment_dir / "lists/test.csv"
    model_dir, checkpoint_path, _ = configure(
        "isic", experiment_dir, None, validation_list, "ifp",
    )
    device = torch.device(args.device)
    model = Model(cfg)
    model.setup()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    prompt_generator = build_prompt_generator(
        cfg.prompt_generator.backend,
        device,
        checkpoint_path=cfg.prompt_generator.alignment_checkpoint,
        text_prompts=SPECS["isic"]["prompts"],
        dino_variant=cfg.prompt_generator.dino_variant,
        clip_variant=cfg.prompt_generator.clip_variant,
        ifp_root=cfg.prompt_generator.ifp_root,
        max_positive_points=cfg.prompt_generator.max_positive_points,
        min_positive_points=cfg.prompt_generator.min_positive_points,
        include_box=cfg.prompt_generator.include_box,
    )
    transform = ResizeAndPad(model.image_size)
    val_dataset = ISICDataset(
        cfg, root_dir=SPECS["isic"]["root"], list_file=str(validation_list), transform=transform,
    )
    test_dataset = ISICDataset(
        cfg, root_dir=SPECS["isic"]["root"], list_file=str(test_list), transform=transform,
    )
    print(f"Collecting validation logits ({len(val_dataset)} images)...", flush=True)
    val_logits, val_targets = collect(model, prompt_generator, val_dataset, device, args.batch_size)
    print(f"Collecting test logits ({len(test_dataset)} images)...", flush=True)
    test_logits, test_targets = collect(model, prompt_generator, test_dataset, device, args.batch_size)

    grid = thresholds(args)
    val_rows = metric_for_thresholds(val_logits, val_targets, grid)
    selected = max(val_rows, key=lambda row: (row["mean_iou"], row["mean_f1"]))
    test_row = metric_for_thresholds(
        test_logits, test_targets, [selected["threshold"]],
    )[0]
    baseline_test = metric_for_thresholds(test_logits, test_targets, [0.0])[0]

    with (output_dir / "validation_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("threshold", "mean_iou", "mean_f1"))
        writer.writeheader()
        writer.writerows(val_rows)
    summary = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "validation_images": len(val_dataset),
        "test_images": len(test_dataset),
        "selection_metric": "validation_mean_iou_then_f1",
        "selected_threshold": selected["threshold"],
        "validation_at_selected": selected,
        "test_at_selected": test_row,
        "test_at_threshold_0": baseline_test,
        "sam_ed_reference": {"dice": 0.8806, "iou": 0.8067},
    }
    (output_dir / "threshold_search_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
