#!/usr/bin/env python3
"""Plot final-test Dice/IoU curves from no-prompt, IFP, and oracle experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output_current"
CHART_ROOT = PROJECT_ROOT / "comparison_examples_4" / "line_charts"
POLYP_DATASETS = ("Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB")


@dataclass(frozen=True)
class Point:
    condition: str
    display: str
    source: Path
    dice: float
    iou: float


def final_isic_metrics(path: Path) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final_rows = [row for row in rows if "final_test" in row["Name"]]
    if not final_rows:
        raise RuntimeError("No final test row in {}".format(path))
    row = final_rows[-1]
    return float(row["Mean F1"]), float(row["Mean IoU"])


def dataset_metrics(path: Path, dataset: str) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["Dataset"] == dataset]
    if not matches:
        raise RuntimeError("No {} row in {}".format(dataset, path))
    row = matches[-1]
    return float(row["Mean F1"]), float(row["Mean IoU"])


def summary_metrics(path: Path, dataset: str) -> tuple[float, float]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "test" if dataset == "ISIC" else dataset
    row = payload["results"][key]
    return float(row["mean_f1"]), float(row["mean_iou"])


def make_point(condition: str, display: str, source: Path, loader, *args: str) -> Point:
    dice, iou = loader(source, *args)
    return Point(condition, display, source, dice, iou)


def isic_points() -> List[Point]:
    return [
        make_point(
            "No semi-supervision", "No semi-supervision\n(IFP + 1% GT)",
            OUTPUT_ROOT / "no_semisupervised" / "ISIC" / "1pct" / "summary.json", summary_metrics, "ISIC",
        ),
        make_point(
            "1 GT + IFP", "1 GT + IFP\n(0.04% labels)",
            OUTPUT_ROOT / "wesam" / "ISIC" / "1" / "ISIC" / "metrics.csv", final_isic_metrics,
        ),
        make_point(
            "10 GT + IFP", "10 GT + IFP\n(0.39% labels)",
            OUTPUT_ROOT / "wesam" / "ISIC" / "10" / "ISIC" / "metrics.csv", final_isic_metrics,
        ),
        make_point(
            "1% + IFP", "1% labels + IFP",
            OUTPUT_ROOT / "wesam" / "ISIC" / "1pct" / "ISIC" / "metrics.csv", final_isic_metrics,
        ),
        make_point(
            "10% + IFP", "10% labels + IFP",
            OUTPUT_ROOT / "wesam" / "ISIC" / "10pct" / "ISIC" / "metrics.csv", final_isic_metrics,
        ),
        make_point(
            "GT-point oracle", "GT-point oracle\n(full GT)",
            OUTPUT_ROOT / "point_pvt(gt)" / "ISIC" / "full_gt" / "ISIC" / "metrics.csv", final_isic_metrics,
        ),
    ]


def polyp_points(dataset: str) -> List[Point]:
    return [
        make_point(
            "No semi-supervision", "No semi-supervision\n(IFP + 1% GT)",
            OUTPUT_ROOT / "no_semisupervised" / "Polyp" / "1pct" / "summary.json", summary_metrics, dataset,
        ),
        make_point(
            "5 GT + IFP", "5 GT + IFP\n(0.38% labels)",
            OUTPUT_ROOT / "wesam" / "Polyp" / "5" / "test_metrics.csv", dataset_metrics, dataset,
        ),
        make_point(
            "1% + IFP", "1% labels + IFP",
            OUTPUT_ROOT / "wesam" / "Polyp" / "1pct" / "test_metrics.csv", dataset_metrics, dataset,
        ),
        make_point(
            "10% + IFP", "10% labels + IFP",
            OUTPUT_ROOT / "wesam" / "Polyp" / "10pct" / "test_metrics.csv", dataset_metrics, dataset,
        ),
        make_point(
            "GT-point oracle", "GT-point oracle\n(full GT)",
            OUTPUT_ROOT / "point_pvt(gt)" / "Polyp" / "full_gt" / "test_metrics.csv", dataset_metrics, dataset,
        ),
    ]


def plot_dataset(dataset: str, points: Iterable[Point]) -> Path:
    points = list(points)
    x = np.arange(len(points))
    dice = np.asarray([point.dice * 100.0 for point in points])
    iou = np.asarray([point.iou * 100.0 for point in points])

    fig, axis = plt.subplots(figsize=(13.5, 7.2), dpi=180)
    fig.patch.set_facecolor("white")
    axis.set_facecolor("#fbfcfd")
    axis.axvspan(-0.35, 0.35, color="#e9ecef", alpha=0.8, zorder=0)
    axis.axvspan(len(points) - 1.35, len(points) - 0.65, color="#fff3cd", alpha=0.85, zorder=0)
    axis.plot(x, dice, color="#007f7b", marker="o", markersize=8, linewidth=2.6, label="Dice")
    axis.plot(x, iou, color="#c4512d", marker="s", markersize=7, linewidth=2.6, label="IoU")

    for index, value in enumerate(dice):
        axis.annotate(
            "{:.2f}".format(value), (index, value), xytext=(0, 10),
            textcoords="offset points", ha="center", color="#006965", fontsize=9, fontweight="bold",
        )
    for index, value in enumerate(iou):
        axis.annotate(
            "{:.2f}".format(value), (index, value), xytext=(0, -16),
            textcoords="offset points", ha="center", color="#9e3d21", fontsize=9, fontweight="bold",
        )

    axis.set_xlim(-0.45, len(points) - 0.55)
    axis.set_ylim(0, 100)
    axis.set_xticks(x, [point.display for point in points], fontsize=9)
    axis.set_ylabel("Final test score (%)", fontsize=11)
    axis.set_xlabel("Prompt / annotation condition", fontsize=11)
    axis.set_title("{}: Prompt and annotation progression".format(dataset), fontsize=15, pad=18, fontweight="bold")
    axis.text(
        0.5, 1.02,
        "First point removes Teacher pseudo-labels; IFP is retained. Final point is a GT-point oracle.",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=9, color="#455a64",
    )
    axis.grid(axis="y", color="#d6dde2", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower right", frameon=True, edgecolor="#cfd8dc")
    fig.tight_layout()

    output = CHART_ROOT / "{}_prompt_progression.png".format(dataset)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def write_audit(all_points: Dict[str, List[Point]]) -> Path:
    output = CHART_ROOT / "curve_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "order", "condition", "dice", "iou", "source"))
        writer.writeheader()
        for dataset, points in all_points.items():
            for order, point in enumerate(points):
                writer.writerow({
                    "dataset": dataset,
                    "order": order,
                    "condition": point.condition,
                    "dice": "{:.6f}".format(point.dice),
                    "iou": "{:.6f}".format(point.iou),
                    "source": str(point.source),
                })
    return output


def main() -> int:
    CHART_ROOT.mkdir(parents=True, exist_ok=True)
    all_points = {"ISIC": isic_points()}
    all_points.update({dataset: polyp_points(dataset) for dataset in POLYP_DATASETS})
    for dataset, points in all_points.items():
        print(plot_dataset(dataset, points))
    print(write_audit(all_points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
