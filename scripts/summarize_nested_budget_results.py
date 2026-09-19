#!/usr/bin/env python3
"""Summarize strict nested ISIC/Polyp budget experiments into one CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "output_current/nested_budget_control_20260823"
OUTPUT = EXPERIMENT_ROOT / "nested_budget_summary.csv"


def index_info(dataset: str, seed: int, budget: str) -> tuple[int, str, int, str]:
    root = EXPERIMENT_ROOT / "indices" / dataset / f"seed{seed}"
    labeled = root / f"labeled_{budget}.json"
    support = root / "support_10pct.json"
    return len(json.loads(labeled.read_text())), str(labeled), len(json.loads(support.read_text())), str(support)


def add_isic(rows: list[dict[str, str]]) -> None:
    for seed in (1337, 2027, 3407):
        for budget in ("1pct", "10pct"):
            experiment = EXPERIMENT_ROOT / f"isic_seed{seed}_{budget}"
            metrics = experiment / "ISIC/metrics.csv"
            entries = [row for row in csv.DictReader(metrics.open()) if "final_test_best" in row["Name"]]
            if not entries:
                raise RuntimeError(f"No final ISIC row in {metrics}")
            row = entries[-1]
            count, labeled_path, support_count, support_path = index_info("isic", seed, budget)
            rows.append({
                "dataset": "ISIC", "test_dataset": "ISIC", "budget": budget,
                "seed": str(seed), "labeled_count": str(count),
                "support_count": str(support_count), "dice_percent": f"{float(row['Mean F1']) * 100:.4f}",
                "iou_percent": f"{float(row['Mean IoU']) * 100:.4f}", "best_epoch": row["epoch"],
                "metrics_path": str(metrics), "test_metrics_path": "",
                "labeled_indices_path": labeled_path, "support_indices_path": support_path,
            })


def add_polyp(rows: list[dict[str, str]]) -> None:
    for seed in (1337, 2027, 3407):
        for budget in ("1pct", "10pct"):
            experiment = EXPERIMENT_ROOT / f"polyp_seed{seed}_{budget}"
            metrics = experiment / "Polyp/metrics.csv"
            test_metrics = experiment / "test_metrics.csv"
            validation = {row["epoch"]: row for row in csv.DictReader(metrics.open())}
            test_rows = list(csv.DictReader(test_metrics.open()))
            best_epoch = str(min(test_rows, key=lambda row: float(row["Best validation epoch"]))["Best validation epoch"])
            count, labeled_path, support_count, support_path = index_info("polyp", seed, budget)
            for row in test_rows:
                rows.append({
                    "dataset": "Polyp", "test_dataset": row["Dataset"], "budget": budget,
                    "seed": str(seed), "labeled_count": str(count),
                    "support_count": str(support_count), "dice_percent": f"{float(row['Mean F1']) * 100:.4f}",
                    "iou_percent": f"{float(row['Mean IoU']) * 100:.4f}", "best_epoch": best_epoch,
                    "metrics_path": str(metrics), "test_metrics_path": str(test_metrics),
                    "labeled_indices_path": labeled_path, "support_indices_path": support_path,
                })


def main() -> None:
    rows: list[dict[str, str]] = []
    add_isic(rows)
    add_polyp(rows)
    fields = [
        "dataset", "test_dataset", "budget", "seed", "labeled_count", "support_count",
        "dice_percent", "iou_percent", "best_epoch", "metrics_path", "test_metrics_path",
        "labeled_indices_path", "support_indices_path",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
