"""Evaluate frozen SAM2 directly from IFP point prompts.

The optional explicit index files make the reference set and IFP alignment
support identical to the fixed 10% split used by the other ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad, collate_fn
from ifp_alignment import build_prompt_generator
from model import Model
from run_ifp_supervised_ablation import (
    IFP_ROOT,
    ISIC_PROMPTS,
    ISIC_ROOT,
    POLYP_PROMPTS,
    POLYP_ROOT,
    TEST_CATEGORIES,
    TRAIN_CATEGORIES,
    make_support,
    read_manifest,
    read_polyp_manifest,
    rows_at_indices,
    select_indices,
    support_indices_from_args,
    train_ifp_head,
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def evaluate(dataset: str, manifest: Path, output_dir: Path, prompts: list[str], head: Path,
             args: argparse.Namespace) -> dict[str, float | int]:
    cfg.dataset = dataset
    cfg.datasets[dataset].root_dir = str(ISIC_ROOT if dataset == "ISIC" else POLYP_ROOT)
    cfg.datasets[dataset].test_list = str(manifest)
    cfg.prompt = "point"
    cfg.visual = False
    cfg.prompt_generator.backend = "ifp"
    cfg.prompt_generator.ifp_root = str(IFP_ROOT)
    cfg.prompt_generator.alignment_checkpoint = str(head)
    cfg.prompt_generator.dino_variant = "dino3h"
    cfg.prompt_generator.clip_variant = "clipl"
    cfg.prompt_generator.text_prompts = prompts
    cfg.prompt_generator.max_positive_points = 1
    cfg.prompt_generator.min_positive_points = 1
    cfg.prompt_generator.include_box = False

    device = torch.device("cuda")
    model = Model(cfg)
    model.setup()
    # LoRA is present in the wrapper but its B matrices are initialized to
    # zero and are never optimized here, leaving the pretrained SAM2 intact.
    model.to(device).eval()
    generator = build_prompt_generator("ifp", device, checkpoint_path=str(head),
                                       text_prompts=prompts, dino_variant="dino3h",
                                       clip_variant="clipl", ifp_root=str(IFP_ROOT),
                                       max_positive_points=1, min_positive_points=1,
                                       include_box=False)
    data = ISICDataset(cfg, cfg.datasets[dataset].root_dir, str(manifest),
                       transform=ResizeAndPad(model.image_size))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    ious, f1s = [], []
    with torch.no_grad():
        for images, _, gt_masks in loader:
            images = images.to(device)
            _, predictions, _, _ = model(images, generator(images))
            for prediction, target in zip(predictions, gt_masks):
                target = (target.sum(dim=0, keepdim=True) > 0).int().to(device)
                # Model.forward returns mask logits, so 0 is the probability
                # threshold 0.5 after sigmoid.
                binary = (prediction >= 0.0).int()
                stats = smp.metrics.get_stats(binary, target, mode="binary", threshold=0.5)
                ious.append(float(smp.metrics.iou_score(*stats, reduction="micro-imagewise")))
                f1s.append(float(smp.metrics.f1_score(*stats, reduction="micro-imagewise")))
    summary = {"images": len(data), "mean_iou": sum(ious) / len(ious), "mean_f1": sum(f1s) / len(f1s)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "manifest": str(manifest), **summary}, indent=2), flush=True)
    del generator, model
    torch.cuda.empty_cache()
    return summary


def run_isic(args: argparse.Namespace) -> None:
    budget = args.budget_tag or (f"{args.labeled_count}shot" if args.labeled_count is not None else "10pct")
    output_dir = args.output_dir / "ifp_only" / "ISIC" / budget
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train_rows = read_manifest(ISIC_ROOT / "manifests" / "train.csv")
    labeled_indices = select_indices(train_rows, args.seed, args.labeled_count, args.labeled_indices)
    support_indices = support_indices_from_args(labeled_indices, args, len(train_rows))
    selected = rows_at_indices(train_rows, labeled_indices)
    support_rows = rows_at_indices(train_rows, support_indices)
    train, validation, test = lists / "train.csv", lists / "validation.csv", lists / "test.csv"
    write_manifest(train, selected)
    write_manifest(validation, read_manifest(ISIC_ROOT / "manifests" / "validation.csv"))
    write_manifest(test, read_manifest(ISIC_ROOT / "manifests" / "test.csv"))
    head = train_ifp_head(output_dir, make_support(output_dir, support_rows, absolute=False), ISIC_PROMPTS, args)
    results = {"validation": evaluate("ISIC", validation, output_dir / "validation", ISIC_PROMPTS, head, args),
               "test": evaluate("ISIC", test, output_dir / "test", ISIC_PROMPTS, head, args)}
    (output_dir / "summary.json").write_text(json.dumps({
        "prompt_backend": "ifp",
        "sam2_training": False,
        "teacher_pseudo_labels": False,
        "mask_logit_threshold": 0.0,
        "references": len(support_rows),
        "labeled_count": len(selected),
        "labeled_indices": labeled_indices,
        "support_indices": support_indices,
        "seed": args.seed,
        "results": results,
    }, indent=2) + "\n")


def run_polyp(args: argparse.Namespace) -> None:
    budget = args.budget_tag or (f"{args.labeled_count}shot" if args.labeled_count is not None else "10pct")
    output_dir = args.output_dir / "ifp_only" / "Polyp" / budget
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train_rows = [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "train")]
    labeled_indices = select_indices(train_rows, args.seed, args.labeled_count, args.labeled_indices)
    support_indices = support_indices_from_args(labeled_indices, args, len(train_rows))
    selected = rows_at_indices(train_rows, labeled_indices)
    support_rows = rows_at_indices(train_rows, support_indices)
    train, validation = lists / "train.csv", lists / "validation.csv"
    write_manifest(train, selected)
    write_manifest(validation, [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "validation")])
    head = train_ifp_head(output_dir, make_support(output_dir, support_rows, absolute=True), POLYP_PROMPTS, args)
    results = {"validation": evaluate("Polyp", validation, output_dir / "validation", POLYP_PROMPTS, head, args)}
    for category in TEST_CATEGORIES:
        manifest = lists / f"test_{category}.csv"
        write_manifest(manifest, read_polyp_manifest(category, "test"))
        results[category] = evaluate("Polyp", manifest, output_dir / category, POLYP_PROMPTS, head, args)
    (output_dir / "summary.json").write_text(json.dumps({
        "prompt_backend": "ifp",
        "sam2_training": False,
        "teacher_pseudo_labels": False,
        "mask_logit_threshold": 0.0,
        "references": len(support_rows),
        "labeled_count": len(selected),
        "labeled_indices": labeled_indices,
        "support_indices": support_indices,
        "seed": args.seed,
        "results": results,
    }, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("isic", "polyp", "both"), default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output_current")
    parser.add_argument("--alignment-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--labeled-count", type=int, default=None)
    parser.add_argument("--budget-tag", default=None)
    parser.add_argument("--labeled-indices-file", type=Path)
    parser.add_argument("--support-indices-file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.labeled_indices = (
        json.loads(args.labeled_indices_file.read_text(encoding="utf-8"))
        if args.labeled_indices_file else None
    )
    args.support_indices = (
        json.loads(args.support_indices_file.read_text(encoding="utf-8"))
        if args.support_indices_file else None
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"isic", "both"}:
        run_isic(args)
    if args.dataset in {"polyp", "both"}:
        run_polyp(args)
