#!/usr/bin/env python3
"""Diagnose existing Polyp predictions without loading a model or using a GPU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


DATASETS = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        action="append",
        required=True,
        help="Existing experiment directory; may be provided more than once.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-iou", type=float, default=0.5)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def manifest_map(path: Path) -> dict[str, Path]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image"].split("/")[-1].rsplit(".", 1)[0]: Path(row["mask"])
                for row in csv.DictReader(handle)}


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "median": 0.0, "p90": 0.0}
    result = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9])
    return {"p10": round(float(result[0]), 6), "median": round(float(result[1]), 6), "p90": round(float(result[2]), 6)}


def analyze_dataset(experiment_dir: Path, dataset: str, low_iou: float) -> tuple[list[dict], dict]:
    result_dir = experiment_dir / dataset
    metrics_path = result_dir / "prediction_metrics.csv"
    list_path = experiment_dir / "lists" / f"test_{dataset}.csv"
    if not metrics_path.is_file() or not list_path.is_file():
        raise FileNotFoundError(f"missing prediction metrics or manifest for {experiment_dir} / {dataset}")

    masks = manifest_map(list_path)
    rows: list[dict] = []
    for metric in csv.DictReader(metrics_path.open(newline="", encoding="utf-8")):
        image_id = metric["image"]
        pred_path = result_dir / metric["prediction"]
        gt_path = masks.get(image_id)
        if gt_path is None:
            raise KeyError(f"{image_id} is absent from {list_path}")
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch for {image_id}: pred={pred.shape}, gt={gt.shape}")

        intersection = int(np.logical_and(pred, gt).sum())
        pred_area = int(pred.sum())
        gt_area = int(gt.sum())
        union = pred_area + gt_area - intersection
        iou = intersection / union if union else 1.0
        f1 = 2 * intersection / (pred_area + gt_area) if pred_area + gt_area else 1.0
        precision = intersection / pred_area if pred_area else 0.0
        recall = intersection / gt_area if gt_area else 0.0
        area_ratio = pred_area / gt_area if gt_area else 0.0

        if pred_area == 0:
            error_type = "empty_prediction"
        elif iou >= low_iou:
            error_type = "acceptable"
        elif precision == 0.0 and recall == 0.0:
            # With no overlap, precision/recall cannot indicate direction.
            # Use the predicted-to-GT area ratio as the only available signal.
            if area_ratio < 0.75:
                error_type = "under_segmentation"
            elif area_ratio > 1.33:
                error_type = "over_segmentation"
            else:
                error_type = "mixed_error"
        elif area_ratio < 0.75 and recall < precision:
            error_type = "under_segmentation"
        elif area_ratio > 1.33 and precision < recall:
            error_type = "over_segmentation"
        elif recall < precision:
            error_type = "under_segmentation"
        elif precision < recall:
            error_type = "over_segmentation"
        else:
            error_type = "mixed_error"

        rows.append({
            "experiment": str(experiment_dir),
            "dataset": dataset,
            "image": image_id,
            "iou": round(iou, 6),
            "f1": round(f1, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "pred_area": pred_area,
            "gt_area": gt_area,
            "area_ratio": round(area_ratio, 6),
            "empty_prediction": int(pred_area == 0),
            "error_type": error_type,
            "prediction": str(pred_path),
            "ground_truth": str(gt_path),
        })

    ious = [row["iou"] for row in rows]
    f1s = [row["f1"] for row in rows]
    ratios = [row["area_ratio"] for row in rows]
    precisions = [row["precision"] for row in rows]
    recalls = [row["recall"] for row in rows]
    counts = {name: sum(row["error_type"] == name for row in rows) for name in (
        "acceptable", "empty_prediction", "under_segmentation", "over_segmentation", "mixed_error"
    )}
    summary = {
        "experiment": str(experiment_dir),
        "dataset": dataset,
        "images": len(rows),
        "mean_iou": round(float(np.mean(ious)), 6),
        "mean_f1": round(float(np.mean(f1s)), 6),
        "mean_precision": round(float(np.mean(precisions)), 6),
        "mean_recall": round(float(np.mean(recalls)), 6),
        "empty_prediction_count": counts["empty_prediction"],
        "empty_prediction_rate": round(counts["empty_prediction"] / len(rows), 6),
        "low_iou_count": sum(value < low_iou for value in ious),
        "low_iou_rate": round(sum(value < low_iou for value in ious) / len(rows), 6),
        "error_type_counts": counts,
        "area_ratio": quantiles(ratios),
        "precision": quantiles(precisions),
        "recall": quantiles(recalls),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for experiment_dir in args.experiment_dir:
        experiment_dir = experiment_dir.resolve()
        for dataset in DATASETS:
            rows, summary = analyze_dataset(experiment_dir, dataset, args.low_iou)
            all_rows.extend(rows)
            summaries.append(summary)

    fields = tuple(all_rows[0])
    with (output_dir / "per_image_diagnosis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "dataset_diagnosis.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = tuple(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "diagnosis_summary.json").write_text(
        json.dumps({"low_iou_threshold": args.low_iou, "datasets": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(summaries)} dataset summaries and {len(all_rows)} per-image rows to {output_dir}")


if __name__ == "__main__":
    main()
