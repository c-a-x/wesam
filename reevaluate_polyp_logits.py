"""Re-evaluate existing Polyp checkpoints with an explicit logit threshold."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from export_test_predictions import export


PROJECT_ROOT = Path(__file__).resolve().parent
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--logit-threshold", type=float, default=0.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for category in TEST_CATEGORIES:
        test_list = source_dir / "lists" / f"test_{category}.csv"
        if not test_list.is_file():
            raise FileNotFoundError(test_list)
        result_dir = output_dir / category
        export_args = argparse.Namespace(
            dataset="polyp",
            experiment_dir=source_dir,
            root_dir=PROJECT_ROOT / "Data/kvasir-seg/organized",
            test_list=test_list,
            result_dir=result_dir,
            device="cuda",
            batch_size=args.batch_size,
            logit_threshold=args.logit_threshold,
        )
        export(export_args)
        summary = json.loads((result_dir / "prediction_summary.json").read_text(encoding="utf-8"))
        rows.append({
            "Dataset": category,
            "Images": str(summary["num_images"]),
            "Mean IoU": f"{summary['reported_mean_iou']:.6f}",
            "Mean F1": f"{summary['reported_mean_f1']:.6f}",
            "Mask logit threshold": str(args.logit_threshold),
            "Checkpoint": summary["checkpoint"],
        })
        torch.cuda.empty_cache()

    with (output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "source_experiment": str(source_dir),
                "mask_logit_threshold": args.logit_threshold,
                "mask_binarization": f"mask_logits >= {args.logit_threshold}",
                "test_sets": list(TEST_CATEGORIES),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    run(parse_args())
