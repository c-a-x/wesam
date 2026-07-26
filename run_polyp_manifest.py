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


def prepare(output_dir: Path, ratio: float, seed: int) -> tuple[Path, Path, Path, dict[str, Path]]:
    train = [row for category in TRAIN_CATEGORIES for row in read_manifest(category, "train")]
    validation = [row for category in TRAIN_CATEGORIES for row in read_manifest(category, "validation")]
    count = max(1, round(len(train) * ratio))
    indices = sorted(random.Random(seed).sample(range(len(train)), count))

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
    for index in indices:
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
        "seed": seed,
        "test_counts": {category: len(read_manifest(category, "test")) for category in TEST_CATEGORIES},
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return train_path, validation_path, support, test_paths


def train_head(args: argparse.Namespace, support: Path) -> Path:
    output_dir = args.output_dir / "ifp_alignment"
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
        "--seed", str(args.seed),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return checkpoint


def configure(args: argparse.Namespace, train: Path, validation: Path, alignment_checkpoint: Path) -> None:
    cfg.gpu_ids = "0"
    cfg.dataset = "Polyp"
    cfg.datasets.Polyp.root_dir = str(POLYP_ROOT)
    cfg.datasets.Polyp.train_list = str(train)
    cfg.datasets.Polyp.test_list = str(validation)
    cfg.datasets.Polyp.final_test_list = ""
    cfg.out_dir = str(args.output_dir)
    cfg.name = f"polyp_combined_{args.labeled_ratio:g}gt_seed{args.seed}"
    cfg.batch_size = args.batch_size
    cfg.val_batchsize = args.val_batch_size
    cfg.num_workers = 0
    cfg.num_epochs = args.epochs
    cfg.load_type = "soft"
    cfg.prompt = "point"
    cfg.resume = False
    cfg.semi.enabled = True
    cfg.semi.labeled_ratio = args.labeled_ratio
    cfg.semi.seed = args.seed
    cfg.semi.labeled_batch_probability = 0.5
    cfg.teacher_weight = 0.1
    cfg.anchor_weight = 0.0
    cfg.contrast_weight = 0.0
    cfg.supervised_weight = 1.0
    cfg.unsupervised_rampup_epochs = 3
    cfg.prompt_generator.backend = "ifp"
    cfg.prompt_generator.ifp_root = str(IFP_ROOT)
    cfg.prompt_generator.alignment_checkpoint = str(alignment_checkpoint)
    cfg.prompt_generator.dino_variant = "dino3h"
    cfg.prompt_generator.clip_variant = "clipl"
    cfg.prompt_generator.text_prompts = PROMPTS
    cfg.prompt_generator.max_positive_points = 1
    cfg.prompt_generator.min_positive_points = 1
    cfg.prompt_generator.include_box = False
    cfg.prompt_generator.iterative_pseudo.enabled = True
    cfg.prompt_generator.iterative_pseudo.max_iters = 1
    cfg.prompt_generator.iterative_pseudo.teacher_iou_threshold = 0.8
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
    parser.add_argument("--labeled-ratio", type=float, choices=(0.01, 0.10), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alignment-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, support, test_paths = prepare(args.output_dir, args.labeled_ratio, args.seed)
    if args.prepare_only:
        return
    alignment_checkpoint = train_head(args, support)
    configure(args, train, validation, alignment_checkpoint)
    main(cfg)
    export_tests(args, test_paths)


if __name__ == "__main__":
    run()
