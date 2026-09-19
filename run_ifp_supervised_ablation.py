"""Run an IFP-prompted supervised-only ablation for ISIC and Polyp.

This is the supervised-only baseline used by the fair 10% comparison.  Pass
``--labeled-indices-file`` and ``--support-indices-file`` to make both the
training labels and the IFP alignment references exactly match the common
seed/split used by the semi-supervised runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from adaptation import main
from configs.config import cfg
from run_polyp_manifest import (
    POLYP_ROOT,
    PROMPTS as POLYP_PROMPTS,
    TEST_CATEGORIES,
    TRAIN_CATEGORIES,
    export_tests,
    read_manifest as read_polyp_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
ISIC_ROOT = PROJECT_ROOT / "Data" / "ISIC2018"
ISIC_PROMPTS = ["a dermoscopic photo of a skin lesion"]


def read_manifest(path: Path, *, absolute: bool = False) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not absolute:
        return rows
    return [
        {"image": str((path.parent.parent / row["image"]).resolve()),
         "mask": str((path.parent.parent / row["mask"]).resolve())}
        for row in rows
    ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "image", "mask"))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index, "image": row["image"], "mask": row["mask"]})


def make_support(output_dir: Path, rows: list[dict[str, str]], *, absolute: bool) -> Path:
    support = output_dir / "support"
    images, masks = support / "reference_images", support / "reference_masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        image = Path(row["image"])
        mask = Path(row["mask"])
        if not absolute:
            image, mask = ISIC_ROOT / image, ISIC_ROOT / mask
        image_link = images / f"{index:04d}_{image.name}"
        mask_link = masks / f"{image_link.stem}{mask.suffix}"
        for link, source in ((image_link, image), (mask_link, mask)):
            if link.is_symlink() and link.resolve() != source.resolve():
                link.unlink()
            if not link.exists():
                link.symlink_to(source)
    return support


def train_ifp_head(output_dir: Path, support: Path, prompts: list[str], args: argparse.Namespace) -> Path:
    head_dir = output_dir / "ifp_alignment"
    checkpoint = head_dir / "ifp_medical_foreground_best.pt"
    command = [
        sys.executable, "-m", "ifp_alignment.train_medical",
        "--reference-root", str(support), "--output-dir", str(head_dir),
        "--prompts", *prompts, "--target", "foreground",
        "--dino-variant", "dino3h", "--clip-variant", "clipl",
        "--ifp-root", str(IFP_ROOT), "--device", "cuda",
        "--epochs", str(args.alignment_epochs), "--batch-size", "1", "--seed", str(args.seed),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return checkpoint


def configure(dataset: str, output_dir: Path, train: Path, validation: Path, test: Path | None,
              prompts: list[str], head: Path, args: argparse.Namespace) -> None:
    cfg.gpu_ids = "0"
    cfg.dataset = dataset
    cfg.datasets[dataset].root_dir = str(ISIC_ROOT if dataset == "ISIC" else POLYP_ROOT)
    cfg.datasets[dataset].train_list = str(train)
    cfg.datasets[dataset].test_list = str(validation)
    cfg.datasets[dataset].final_test_list = str(test) if test else ""
    cfg.out_dir = str(output_dir)
    budget = args.budget_tag or (f"{args.labeled_count}shot" if args.labeled_count is not None else "1pct")
    cfg.name = f"{dataset.lower()}_ifp_supervised_{budget}_seed{args.seed}"
    cfg.batch_size, cfg.val_batchsize, cfg.num_workers, cfg.num_epochs = args.batch_size, args.val_batch_size, 0, args.epochs
    cfg.opt.learning_rate = args.learning_rate
    cfg.opt.warmup_steps = args.warmup_steps
    cfg.unsupervised_rampup_epochs = args.unsupervised_rampup_epochs
    cfg.load_type, cfg.prompt, cfg.resume = "soft", "point", False
    # Training manifests contain only the selected labeled samples.  There is
    # no unlabelled data, Teacher, pseudo labels, Anchor, or contrastive loss.
    cfg.semi.enabled, cfg.semi.labeled_ratio, cfg.semi.labeled_count = False, 1.0, None
    cfg.semi.labeled_batch_probability = args.labeled_batch_probability
    cfg.teacher_weight = cfg.anchor_weight = cfg.contrast_weight = 0.0
    cfg.supervised_weight = 1.0
    prompt = cfg.prompt_generator
    prompt.backend, prompt.ifp_root, prompt.alignment_checkpoint = "ifp", str(IFP_ROOT), str(head)
    prompt.dino_variant, prompt.clip_variant, prompt.text_prompts = "dino3h", "clipl", prompts
    prompt.max_positive_points, prompt.min_positive_points, prompt.include_box = 1, 1, False
    prompt.labeled_prompt_mode = "ifp"
    prompt.iterative_pseudo.enabled = False


def validate_indices(indices: list[int], total: int, *, name: str) -> list[int]:
    values = [int(index) for index in indices]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate indices.")
    if any(index < 0 or index >= total for index in values):
        raise ValueError(f"{name} contains an out-of-range index for {total} rows.")
    return sorted(values)


def select_indices(rows: list[dict[str, str]], seed: int, labeled_count: int | None,
                   explicit_indices: list[int] | None = None) -> list[int]:
    if explicit_indices is not None:
        return validate_indices(explicit_indices, len(rows), name="--labeled-indices-file")
    count = max(1, round(len(rows) * 0.01)) if labeled_count is None else labeled_count
    if not 1 <= count <= len(rows):
        raise ValueError(f"--labeled-count must be within [1, {len(rows)}], got {count}.")
    return sorted(random.Random(seed).sample(range(len(rows)), count))


def rows_at_indices(rows: list[dict[str, str]], indices: list[int]) -> list[dict[str, str]]:
    return [rows[index] for index in indices]


def support_indices_from_args(labeled_indices: list[int], args: argparse.Namespace, total: int) -> list[int]:
    if args.support_indices is None:
        return list(labeled_indices)
    return validate_indices(args.support_indices, total, name="--support-indices-file")


def run_isic(args: argparse.Namespace) -> None:
    budget = args.budget_tag or (f"{args.labeled_count}shot" if args.labeled_count is not None else "1pct")
    output_dir = args.output_dir / f"isic_ifp_supervised_{budget}"
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train_rows = read_manifest(ISIC_ROOT / "manifests" / "train.csv")
    labeled_indices = select_indices(train_rows, args.seed, args.labeled_count, args.labeled_indices)
    support_indices = support_indices_from_args(labeled_indices, args, len(train_rows))
    selected = rows_at_indices(train_rows, labeled_indices)
    support_rows = rows_at_indices(train_rows, support_indices)
    write_manifest(lists / "train.csv", selected)
    write_manifest(lists / "validation.csv", read_manifest(ISIC_ROOT / "manifests" / "validation.csv"))
    write_manifest(lists / "test.csv", read_manifest(ISIC_ROOT / "manifests" / "test.csv"))
    (output_dir / "dataset_summary.json").write_text(json.dumps({
        "train_total": len(train_rows),
        "selected": len(selected),
        "ratio": len(selected) / len(train_rows),
        "labeled_indices": labeled_indices,
        "support_indices": support_indices,
        "seed": args.seed,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "unsupervised_rampup_epochs": args.unsupervised_rampup_epochs,
            "labeled_batch_probability": args.labeled_batch_probability,
            "teacher_weight": 0.0,
            "anchor_weight": 0.0,
            "contrast_weight": 0.0,
        },
    }, indent=2) + "\n")
    head = train_ifp_head(output_dir, make_support(output_dir, support_rows, absolute=False), ISIC_PROMPTS, args)
    configure("ISIC", output_dir, lists / "train.csv", lists / "validation.csv", lists / "test.csv", ISIC_PROMPTS, head, args)
    main(cfg)


def run_polyp(args: argparse.Namespace) -> None:
    budget = args.budget_tag or (f"{args.labeled_count}shot" if args.labeled_count is not None else "1pct")
    output_dir = args.output_dir / f"polyp_ifp_supervised_{budget}"
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    all_train = [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "train")]
    labeled_indices = select_indices(all_train, args.seed, args.labeled_count, args.labeled_indices)
    support_indices = support_indices_from_args(labeled_indices, args, len(all_train))
    selected = rows_at_indices(all_train, labeled_indices)
    support_rows = rows_at_indices(all_train, support_indices)
    validation = [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "validation")]
    train_path, validation_path = lists / "train.csv", lists / "validation.csv"
    write_manifest(train_path, selected)
    write_manifest(validation_path, validation)
    test_paths = {}
    for category in TEST_CATEGORIES:
        path = lists / f"test_{category}.csv"
        write_manifest(path, read_polyp_manifest(category, "test"))
        test_paths[category] = path
    if args.export_only:
        args.output_dir = output_dir
        export_tests(args, test_paths)
        return
    (output_dir / "dataset_summary.json").write_text(json.dumps({
        "train_total": len(all_train),
        "selected": len(selected),
        "ratio": len(selected) / len(all_train),
        "labeled_indices": labeled_indices,
        "support_indices": support_indices,
        "seed": args.seed,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "unsupervised_rampup_epochs": args.unsupervised_rampup_epochs,
            "labeled_batch_probability": args.labeled_batch_probability,
            "teacher_weight": 0.0,
            "anchor_weight": 0.0,
            "contrast_weight": 0.0,
        },
    }, indent=2) + "\n")
    head = train_ifp_head(output_dir, make_support(output_dir, support_rows, absolute=True), POLYP_PROMPTS, args)
    configure("Polyp", output_dir, train_path, validation_path, None, POLYP_PROMPTS, head, args)
    main(cfg)
    args.output_dir = output_dir
    export_tests(args, test_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("isic", "polyp", "both"), default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ifp_supervised_1pct_ablation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alignment-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--labeled-count", type=int, default=None,
                        help="Exact number of labeled training images; defaults to 1 percent.")
    parser.add_argument("--budget-tag", default=None,
                        help="Output tag, e.g. 10pct. Defaults to the legacy count/1pct tag.")
    parser.add_argument("--labeled-indices-file", type=Path,
                        help="JSON list of training-row indices exposed to ground truth.")
    parser.add_argument("--support-indices-file", type=Path,
                        help="JSON list of training-row indices used to train IFP alignment.")
    parser.add_argument("--labeled-batch-probability", type=float, default=0.7,
                        help="Recorded for protocol consistency; semi-supervision is disabled here.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--unsupervised-rampup-epochs", type=int, default=1)
    parser.add_argument("--export-only", action="store_true",
                        help="Export Polyp test predictions from an existing best checkpoint.")
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
    # This ablation always uses the IFP point generator; export_tests shares
    # the Polyp launcher interface and needs the backend explicitly.
    args.prompt_backend = "ifp"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset in {"isic", "both"}:
        run_isic(args)
    if args.dataset in {"polyp", "both"}:
        run_polyp(args)
