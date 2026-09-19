"""Train WeSAM on combined Kvasir and CVC-ClinicDB data, then test five polyp sets."""

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


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
POLYP_ROOT = PROJECT_ROOT / "Data/kvasir-seg/organized"
TRAIN_CATEGORIES = ("Kvasir", "CVC-ClinicDB")
TEST_CATEGORIES = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)
PROMPTS = [
    "a colonoscopy image of a polyp",
    "an endoscopic view showing a colorectal polyp",
    "a gastrointestinal endoscopy image with a visible polyp",
]


def read_manifest(category: str, split: str) -> list[dict[str, str]]:
    category_root = POLYP_ROOT / category
    manifest = category_root / f"{split}.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "image": str((category_root / row["image"]).resolve()),
            "mask": str((category_root / row["mask"]).resolve()),
        }
        for row in rows
    ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "image", "mask"))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index, "image": row["image"], "mask": row["mask"]})


def prepare(
    output_dir: Path,
    ratio: float,
    seed: int,
    prompt_reference_ratio: float | None = None,
    labeled_prompt_mode: str = "gt",
    labeled_count: int | None = None,
    labeled_indices: list[int] | None = None,
    support_indices: list[int] | None = None,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    train = [row for category in TRAIN_CATEGORIES for row in read_manifest(category, "train")]
    validation = [row for category in TRAIN_CATEGORIES for row in read_manifest(category, "validation")]
    count = max(1, round(len(train) * ratio)) if labeled_count is None else labeled_count
    if not 1 <= count <= len(train):
        raise ValueError(f"labeled_count must be within [1, {len(train)}], got {count}.")
    if labeled_indices is None:
        indices = sorted(random.Random(seed).sample(range(len(train)), count))
    else:
        indices = sorted({int(index) for index in labeled_indices})
        if len(indices) != count or any(index < 0 or index >= len(train) for index in indices):
            raise ValueError("Explicit labeled indices must be unique, in range, and match labeled_count.")
    prompt_reference_ratio = ratio if prompt_reference_ratio is None else prompt_reference_ratio
    reference_count = max(1, round(len(train) * prompt_reference_ratio))
    if support_indices is None:
        reference_indices = sorted(random.Random(seed).sample(range(len(train)), reference_count))
    else:
        reference_indices = sorted({int(index) for index in support_indices})
        if not reference_indices or any(index < 0 or index >= len(train) for index in reference_indices):
            raise ValueError("Explicit support indices must be non-empty and in range.")
        reference_count = len(reference_indices)

    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train_path = lists / "train_combined.csv"
    validation_path = lists / "validation_combined.csv"
    write_manifest(train_path, train)
    write_manifest(validation_path, validation)

    test_paths: dict[str, Path] = {}
    for category in TEST_CATEGORIES:
        path = lists / f"test_{category}.csv"
        write_manifest(path, read_manifest(category, "test"))
        test_paths[category] = path

    support = output_dir / "support"
    images = support / "reference_images"
    masks = support / "reference_masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in reference_indices:
        row = train[index]
        image = Path(row["image"])
        mask = Path(row["mask"])
        image_link = images / f"{index:04d}_{image.name}"
        mask_link = masks / f"{image_link.stem}{mask.suffix}"
        for link, source in ((image_link, image), (mask_link, mask)):
            if link.is_symlink():
                if link.resolve() != source.resolve():
                    link.unlink()
                else:
                    continue
            elif link.exists():
                raise FileExistsError(f"Support path already exists and is not a symlink: {link}")
            link.symlink_to(source)

    metadata = {
        "combined_train": len(train),
        "combined_validation": len(validation),
        "labeled_count": count,
        "labeled_ratio": ratio,
        "labeled_indices": indices,
        "support_indices": reference_indices,
        "labeled_prompt_mode": labeled_prompt_mode,
        "prompt_reference_count": reference_count,
        "prompt_reference_ratio": prompt_reference_ratio,
        "seed": seed,
        "test_counts": {category: len(read_manifest(category, "test")) for category in TEST_CATEGORIES},
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return train_path, validation_path, support, test_paths


def dump_effective_config(output_dir: Path, args: argparse.Namespace) -> None:
    """Persist the resolved CLI arguments used by this run for auditability."""
    data = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output_dir / "effective_config.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train_head(args: argparse.Namespace, support: Path) -> Path:
    if args.prompt_backend in {"clip-only", "no-prompt"}:
        return Path()
    output_dir = args.output_dir / "ifp_alignment"
    if args.prompt_backend == "dino-prototype":
        checkpoint = output_dir / "dino_foreground_background_prototype.pt"
        command = [
            sys.executable, "-m", "ifp_alignment.train_dino_prototype",
            "--reference-root", str(support),
            "--output-path", str(checkpoint),
            "--dino-variant", "dino3h",
            "--ifp-root", str(IFP_ROOT),
            "--device", "cuda",
        ]
    else:
        checkpoint = output_dir / "ifp_medical_foreground_best.pt"
        command = [
            sys.executable, "-m", "ifp_alignment.train_medical",
        "--reference-root", str(support),
        "--output-dir", str(output_dir),
        "--prompts", *PROMPTS,
        "--target", "foreground",
        "--dino-variant", "dino3h",
        "--clip-variant", "clipl",
        "--ifp-root", str(IFP_ROOT),
        "--device", "cuda",
        "--epochs", str(args.alignment_epochs),
        "--batch-size", "1",
            "--seed", str(args.alignment_seed),
        ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return checkpoint


def configure(args: argparse.Namespace, train: Path, validation: Path, alignment_checkpoint: Path) -> None:
    if not 0.0 <= args.labeled_batch_probability <= 1.0:
        raise ValueError("--labeled-batch-probability must be in [0, 1].")
    fully_supervised = args.labeled_ratio == 1.0
    cfg.gpu_ids = "0"
    cfg.dataset = "Polyp"
    cfg.datasets.Polyp.root_dir = str(POLYP_ROOT)
    cfg.datasets.Polyp.train_list = str(train)
    cfg.datasets.Polyp.test_list = str(validation)
    cfg.datasets.Polyp.final_test_list = ""
    cfg.out_dir = str(args.output_dir)
    cfg.name = (
        f"polyp_{args.prompt_backend}_{args.labeled_prompt_mode}_"
        f"{args.labeled_ratio:g}gt_seed{args.seed}"
    )
    cfg.batch_size = args.batch_size
    cfg.val_batchsize = args.val_batch_size
    cfg.num_workers = 0
    cfg.num_epochs = args.epochs
    cfg.load_type = "soft"
    cfg.prompt = "none" if args.prompt_backend == "no-prompt" else "point"
    cfg.resume = False
    cfg.semi.enabled = True
    cfg.semi.labeled_ratio = args.labeled_ratio
    cfg.semi.labeled_count = args.labeled_count
    cfg.semi.labeled_indices = args.labeled_indices
    cfg.semi.seed = args.seed
    cfg.semi.labeled_batch_probability = args.labeled_batch_probability
    cfg.teacher_weight = 0.0 if fully_supervised else args.teacher_weight
    cfg.anchor_weight = args.anchor_weight
    cfg.contrast_weight = args.contrast_weight
    cfg.supervised_weight = 1.0
    cfg.unsupervised_rampup_epochs = args.unsupervised_rampup_epochs
    cfg.opt.learning_rate = args.learning_rate
    cfg.opt.warmup_steps = args.warmup_steps
    cfg.prompt_generator.backend = args.prompt_backend
    cfg.prompt_generator.ifp_root = str(IFP_ROOT)
    cfg.prompt_generator.alignment_checkpoint = str(alignment_checkpoint)
    cfg.prompt_generator.dino_variant = "dino3h"
    cfg.prompt_generator.clip_variant = "clipl"
    cfg.prompt_generator.text_prompts = PROMPTS
    cfg.prompt_generator.max_positive_points = 1
    cfg.prompt_generator.min_positive_points = 1
    cfg.prompt_generator.include_box = False
    cfg.prompt_generator.labeled_prompt_mode = args.labeled_prompt_mode
    cfg.prompt_generator.labeled_gt_prompt_probability = args.labeled_gt_prompt_probability
    cfg.prompt_generator.fallback_confidence_threshold = args.fallback_confidence_threshold
    cfg.prompt_generator.train_point_jitter_pixels = args.train_point_jitter_pixels
    cfg.prompt_generator.train_prompt_dropout = args.train_prompt_dropout
    cfg.prompt_generator.iterative_pseudo.enabled = not fully_supervised
    cfg.prompt_generator.iterative_pseudo.max_iters = 1
    cfg.prompt_generator.iterative_pseudo.teacher_iou_threshold = 0.8
    cfg.prompt_generator.iterative_pseudo.min_new_pixels = 32
    cfg.prompt_generator.iterative_pseudo.min_new_area_ratio = 0.001
    cfg.prompt_generator.iterative_pseudo.max_mask_area_ratio = 0.6
    cfg.prompt_generator.iterative_pseudo.min_overlap_ratio = 0.7
    cfg.prompt_generator.iterative_pseudo.max_new_area_ratio = 0.4


def export_tests(args: argparse.Namespace, test_paths: dict[str, Path]) -> None:
    rows: list[dict[str, str]] = []
    for category, test_path in test_paths.items():
        result_dir = args.output_dir / category
        command = [
            sys.executable, "-u", "export_test_predictions.py",
            "--dataset", "polyp",
            "--experiment-dir", str(args.output_dir),
            "--root-dir", str(POLYP_ROOT),
            "--test-list", str(test_path),
            "--result-dir", str(result_dir),
            "--device", "cuda",
            "--batch-size", str(args.val_batch_size),
            "--prompt-backend", args.prompt_backend,
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        summary = json.loads((result_dir / "prediction_summary.json").read_text(encoding="utf-8"))
        rows.append({
            "Dataset": category,
            "Images": str(summary["num_images"]),
            "Mean IoU": f"{summary['reported_mean_iou']:.6f}",
            "Mean F1": f"{summary['reported_mean_f1']:.6f}",
            "Best validation epoch": str(summary["best_validation_epoch"]),
            "Selection metric": summary["checkpoint_selection_metric"],
            "Checkpoint": summary["checkpoint"],
        })
    metrics = args.output_dir / "test_metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "Dataset", "Images", "Mean IoU", "Mean F1",
                "Best validation epoch", "Selection metric", "Checkpoint",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument(
        "--labeled-ratio", type=float, choices=(0.01, 0.10, 1.0),
        help="Fraction of train images exposed to GT; 1.0 disables pseudo supervision.",
    )
    labels.add_argument(
        "--labeled-count", type=int,
        help="Exact number of train images exposed to GT.",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--alignment-epochs", type=int, default=1,
        help="IFP alignment-head epochs. The fair-protocol default is 1.",
    )
    parser.add_argument(
        "--iterative-max-iters", type=int, default=1,
        help="Deprecated compatibility option; fair protocol uses one round.",
    )
    parser.add_argument(
        "--iterative-sim-threshold", type=float, default=None,
        help="Deprecated compatibility option; fair protocol has no similarity stop.",
    )
    parser.add_argument(
        "--alignment-seed", type=int, default=1337,
        help="Seed for the IFP alignment head; fair protocol uses 1337.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--labeled-batch-probability", type=float, default=0.5,
        help="Expected fraction of sampled training examples with GT labels.",
    )
    parser.add_argument(
        "--prompt-backend", choices=("ifp", "dino-prototype", "clip-only", "no-prompt"), default="ifp",
    )
    parser.add_argument(
        "--labeled-prompt-mode", choices=("gt", "ifp", "mixed"), default="gt",
        help="Use GT points, generated prompts, or a mixture for labeled training images.",
    )
    parser.add_argument("--labeled-gt-prompt-probability", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--unsupervised-rampup-epochs", type=int, default=3)
    parser.add_argument("--fallback-confidence-threshold", type=float, default=None)
    parser.add_argument("--train-point-jitter-pixels", type=float, default=0.0)
    parser.add_argument("--train-prompt-dropout", type=float, default=0.0)
    parser.add_argument("--labeled-indices-file", type=Path,
                        help="JSON list of explicit combined-training row indices exposed to GT.")
    parser.add_argument("--support-indices-file", type=Path,
                        help="JSON list of explicit combined-training row indices for IFP alignment support.")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.labeled_count is not None:
        total_train = sum(len(read_manifest(category, "train")) for category in TRAIN_CATEGORIES)
        if not 1 <= args.labeled_count <= total_train:
            raise ValueError(f"--labeled-count must be within [1, {total_train}].")
        args.labeled_ratio = args.labeled_count / total_train
    args.labeled_indices = (
        json.loads(args.labeled_indices_file.read_text(encoding="utf-8"))
        if args.labeled_indices_file else None
    )
    support_indices = (
        json.loads(args.support_indices_file.read_text(encoding="utf-8"))
        if args.support_indices_file else None
    )
    if args.labeled_indices is not None:
        total_train = sum(len(read_manifest(category, "train")) for category in TRAIN_CATEGORIES)
        args.labeled_count = len(args.labeled_indices)
        args.labeled_ratio = args.labeled_count / total_train
    train, validation, support, test_paths = prepare(
        args.output_dir,
        args.labeled_ratio,
        args.seed,
        args.labeled_ratio,
        args.labeled_prompt_mode,
        args.labeled_count,
        args.labeled_indices,
        support_indices,
    )
    summary_path = args.output_dir / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["training_config"] = {
        "labeled_batch_probability": args.labeled_batch_probability,
        "prompt_backend": args.prompt_backend,
        "labeled_prompt_mode": args.labeled_prompt_mode,
        "labeled_gt_prompt_probability": args.labeled_gt_prompt_probability,
        "teacher_weight": args.teacher_weight,
        "anchor_weight": args.anchor_weight,
        "contrast_weight": args.contrast_weight,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "unsupervised_rampup_epochs": args.unsupervised_rampup_epochs,
        "train_point_jitter_pixels": args.train_point_jitter_pixels,
        "train_prompt_dropout": args.train_prompt_dropout,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    dump_effective_config(args.output_dir, args)
    if args.prepare_only:
        return
    alignment_checkpoint = train_head(args, support)
    configure(args, train, validation, alignment_checkpoint)
    main(cfg)
    export_tests(args, test_paths)


if __name__ == "__main__":
    run()
