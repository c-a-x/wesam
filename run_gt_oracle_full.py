"""Train and evaluate full-GT SAM2 using one GT interior point per image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from adaptation import main
from configs.config import cfg
from run_polyp_manifest import (
    POLYP_ROOT,
    TEST_CATEGORIES,
    TRAIN_CATEGORIES,
    export_tests,
    read_manifest as read_polyp_manifest,
    write_manifest as write_polyp_manifest,
)
from run_semisup_manifest import SPECS, read_manifest as read_isic_manifest, write_manifest as write_isic_manifest


PROJECT_ROOT = Path(__file__).resolve().parent


def configure(dataset: str, output_dir: Path, train: Path, validation: Path, test: Path | None,
              args: argparse.Namespace) -> None:
    cfg.gpu_ids = "0"
    cfg.dataset = dataset
    cfg.datasets[dataset].root_dir = str(SPECS["isic"]["root"] if dataset == "ISIC" else POLYP_ROOT)
    cfg.datasets[dataset].train_list = str(train)
    cfg.datasets[dataset].test_list = str(validation)
    cfg.datasets[dataset].final_test_list = str(test) if test else ""
    cfg.out_dir = str(output_dir)
    cfg.name = f"{dataset.lower()}_full_gt_gt_oracle_seed{args.seed}"
    cfg.batch_size, cfg.val_batchsize, cfg.num_workers, cfg.num_epochs = args.batch_size, args.val_batch_size, 0, args.epochs
    cfg.load_type, cfg.prompt, cfg.resume = "soft", "point", False
    cfg.semi.enabled, cfg.semi.labeled_ratio, cfg.semi.labeled_count, cfg.semi.seed = False, 1.0, None, args.seed
    cfg.teacher_weight = cfg.anchor_weight = cfg.contrast_weight = 0.0
    cfg.supervised_weight = 1.0
    cfg.prompt_generator.backend = "gt-oracle"
    cfg.prompt_generator.labeled_prompt_mode = "ifp"
    cfg.prompt_generator.iterative_pseudo.enabled = False


def run_isic(args: argparse.Namespace) -> None:
    spec = SPECS["isic"]
    output_dir = args.output_dir / "gt_oracle" / "ISIC" / "full_gt"
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train = lists / "train.csv"
    validation = lists / "validation.csv"
    test = lists / "test.csv"
    write_isic_manifest(train, read_isic_manifest(spec["root"] / spec["train"]))
    write_isic_manifest(validation, read_isic_manifest(spec["root"] / spec["validation"]))
    write_isic_manifest(test, read_isic_manifest(spec["root"] / spec["test"]))
    configure("ISIC", output_dir, train, validation, test, args)
    main(cfg)


def run_polyp(args: argparse.Namespace) -> None:
    output_dir = args.output_dir / "gt_oracle" / "Polyp" / "full_gt"
    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    train = lists / "train.csv"
    validation = lists / "validation.csv"
    write_polyp_manifest(train, [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "train")])
    write_polyp_manifest(validation, [row for category in TRAIN_CATEGORIES for row in read_polyp_manifest(category, "validation")])
    test_paths = {}
    for category in TEST_CATEGORIES:
        path = lists / f"test_{category}.csv"
        write_polyp_manifest(path, read_polyp_manifest(category, "test"))
        test_paths[category] = path
    configure("Polyp", output_dir, train, validation, None, args)
    main(cfg)
    args.prompt_backend = "gt-oracle"
    export_tests(args, test_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("isic", "polyp", "both"), default="both")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output_current")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.dataset in {"isic", "both"}:
        run_isic(args)
    if args.dataset in {"polyp", "both"}:
        run_polyp(args)
