"""Evaluate frozen SAM2 from CLIP-only or DINO-prototype point prompts.

Explicit labeled/support index files make this prompt-only comparison use the
same fixed 10% split as the IFP-only and supervised ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
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
    write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_dino_prototypes(output_dir: Path, support: Path) -> Path:
    checkpoint = output_dir / "dino_prototypes" / "foreground_background.pt"
    subprocess.run(
        [
            sys.executable, "-m", "ifp_alignment.train_dino_prototype",
            "--reference-root", str(support), "--output-path", str(checkpoint),
            "--dino-variant", "dino3h", "--ifp-root", str(IFP_ROOT), "--device", "cuda",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return checkpoint


def evaluate(
    dataset: str,
    manifest: Path,
    result_dir: Path,
    prompts: list[str],
    prototype_checkpoint: Path | None,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    cfg.dataset = dataset
    cfg.datasets[dataset].root_dir = str(ISIC_ROOT if dataset == "ISIC" else POLYP_ROOT)
    cfg.datasets[dataset].test_list = str(manifest)
    cfg.prompt, cfg.visual = "point", False
    cfg.prompt_generator.backend = args.prompt_backend
    cfg.prompt_generator.ifp_root = str(IFP_ROOT)
    cfg.prompt_generator.alignment_checkpoint = str(prototype_checkpoint or "")
    cfg.prompt_generator.dino_variant, cfg.prompt_generator.clip_variant = "dino3h", "clipl"
    cfg.prompt_generator.text_prompts = prompts
    cfg.prompt_generator.max_positive_points = cfg.prompt_generator.min_positive_points = 1
    cfg.prompt_generator.include_box = False

    device = torch.device("cuda")
    model = Model(cfg)
    model.setup()
    model.to(device).eval()
    generator = build_prompt_generator(
        args.prompt_backend,
        device,
        checkpoint_path=str(prototype_checkpoint or ""),
        text_prompts=prompts,
        dino_variant="dino3h",
        clip_variant="clipl",
        ifp_root=str(IFP_ROOT),
        max_positive_points=1,
        min_positive_points=1,
        include_box=False,
    )
    data = ISICDataset(cfg, cfg.datasets[dataset].root_dir, str(manifest), transform=ResizeAndPad(model.image_size))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    ious, f1s = [], []
    with torch.no_grad():
        for images, _, gt_masks in loader:
            images = images.to(device)
            _, predictions, _, _ = model(images, generator(images))
            for prediction, target in zip(predictions, gt_masks):
                target = (target.sum(dim=0, keepdim=True) > 0).int().to(device)
                binary = (prediction >= 0.0).int()
                stats = smp.metrics.get_stats(binary, target, mode="binary", threshold=0.5)
                ious.append(float(smp.metrics.iou_score(*stats, reduction="micro-imagewise")))
                f1s.append(float(smp.metrics.f1_score(*stats, reduction="micro-imagewise")))
    result = {"images": len(data), "mean_iou": sum(ious) / len(ious), "mean_f1": sum(f1s) / len(f1s)}
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": dataset, "manifest": str(manifest), **result}, indent=2), flush=True)
    del generator, model
    torch.cuda.empty_cache()
    return result


def write_summary(
    output_dir: Path,
    references: int,
    labeled_indices: list[int],
    support_indices: list[int],
    results: dict[str, dict[str, float | int]],
    args: argparse.Namespace,
) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "prompt_backend": args.prompt_backend,
                "sam2_training": False,
                "teacher_pseudo_labels": False,
                "mask_logit_threshold": 0.0,
                "references": references,
                "labeled_count": args.effective_labeled_count,
                "labeled_ratio": args.effective_labeled_ratio,
                "labeled_indices": labeled_indices,
                "support_indices": support_indices,
                "seed": args.seed,
                "results": results,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def run_isic(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / args.method_dir / "ISIC" / args.budget_tag
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(ISIC_ROOT / "manifests" / "train.csv")
    count = args.labeled_count if args.labeled_count is not None else max(1, round(len(rows) * args.labeled_ratio))
    labeled_indices = select_indices(rows, args.seed, count, args.labeled_indices)
    support_indices = support_indices_from_args(labeled_indices, args, len(rows))
    selected = rows_at_indices(rows, labeled_indices)
    support_rows = rows_at_indices(rows, support_indices)
    args.effective_labeled_count, args.effective_labeled_ratio = count, count / len(rows)
    train, validation, test = lists / "train.csv", lists / "validation.csv", lists / "test.csv"
    write_manifest(train, selected)
    write_manifest(validation, read_manifest(ISIC_ROOT / "manifests" / "validation.csv"))
    write_manifest(test, read_manifest(ISIC_ROOT / "manifests" / "test.csv"))
    prototype = None
    if args.prompt_backend == "dino-prototype":
        prototype = build_dino_prototypes(output_dir, make_support(output_dir, support_rows, absolute=False))
    references = len(support_rows) if prototype else 0
    results = {
        "validation": evaluate("ISIC", validation, output_dir / "validation", ISIC_PROMPTS, prototype, args),
        "test": evaluate("ISIC", test, output_dir / "test", ISIC_PROMPTS, prototype, args),
    }
    write_summary(output_dir, references, labeled_indices, support_indices, results, args)


def run_polyp(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / args.method_dir / "Polyp" / args.budget_tag
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train_rows = [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "train")]
    count = args.labeled_count if args.labeled_count is not None else max(1, round(len(train_rows) * args.labeled_ratio))
    labeled_indices = select_indices(train_rows, args.seed, count, args.labeled_indices)
    support_indices = support_indices_from_args(labeled_indices, args, len(train_rows))
    selected = rows_at_indices(train_rows, labeled_indices)
    support_rows = rows_at_indices(train_rows, support_indices)
    args.effective_labeled_count, args.effective_labeled_ratio = count, count / len(train_rows)
    write_manifest(lists / "train.csv", selected)
    validation = lists / "validation.csv"
    write_manifest(validation, [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "validation")])
    prototype = None
    if args.prompt_backend == "dino-prototype":
        prototype = build_dino_prototypes(output_dir, make_support(output_dir, support_rows, absolute=True))
    references = len(support_rows) if prototype else 0
    results = {"validation": evaluate("Polyp", validation, output_dir / "validation", POLYP_PROMPTS, prototype, args)}
    for category in TEST_CATEGORIES:
        manifest = lists / f"test_{category}.csv"
        write_manifest(manifest, read_polyp_manifest(category, "test"))
        results[category] = evaluate("Polyp", manifest, output_dir / category, POLYP_PROMPTS, prototype, args)
    write_summary(output_dir, references, labeled_indices, support_indices, results, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-backend", choices=("clip-only", "dino-prototype"), required=True)
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--dataset", choices=("isic", "polyp", "both"), default="both")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output_current")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--labeled-indices-file", type=Path,
                        help="JSON list of reference training-row indices.")
    parser.add_argument("--support-indices-file", type=Path,
                        help="JSON list of training-row indices for DINO prototypes.")
    parser.add_argument("--budget-tag", default=None,
                        help="Explicit output budget tag; useful when the fixed index count is not 10 percent rounded.")
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--labeled-count", type=int, help="Exact number of reference images.")
    labels.add_argument("--labeled-ratio", type=float, choices=(0.01, 0.10), help="Reference-image ratio.")
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
    args.budget_tag = args.budget_tag or (
        f"{args.labeled_count}shot" if args.labeled_count is not None else f"{int(args.labeled_ratio * 100)}pct"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"isic", "both"}:
        run_isic(args)
    if args.dataset in {"polyp", "both"}:
        run_polyp(args)
