#!/usr/bin/env python3
"""Generate reference-style LaTeX tables for the two 1% ablations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output_current"
TEX_ROOT = PROJECT_ROOT / "comparison_examples_4" / "tex"
DATASETS = ("ISIC", "Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB")

GROUPS = {
    "group1": (
        ("clip_only", "CLIP-only"),
        ("dino_only", "DINO-only"),
        ("no_semisupervised", "No semi-supervised"),
    ),
    "group2": (
        ("no_prompt", "No prompt"),
        ("no_semisupervised", "No semi-supervised"),
        ("wesam", "WeSAM"),
    ),
}


def read_json_metrics(method: str, dataset: str) -> Tuple[float, float]:
    path = OUTPUT_ROOT / method / dataset / "1pct" / "summary.json"
    if dataset == "ISIC":
        path = OUTPUT_ROOT / method / "ISIC" / "1pct" / "summary.json"
        key = "test"
    else:
        path = OUTPUT_ROOT / method / "Polyp" / "1pct" / "summary.json"
        key = dataset
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["results"][key]
    return float(row["mean_f1"]), float(row["mean_iou"])


def read_final_csv_metrics(method: str, dataset: str) -> Tuple[float, float]:
    if dataset == "ISIC":
        path = OUTPUT_ROOT / method / "ISIC" / "1pct" / "ISIC" / "metrics.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows = [row for row in rows if "final_test" in row["Name"]]
        if not rows:
            raise RuntimeError("No final test row in {}".format(path))
        row = rows[-1]
        return float(row["Mean F1"]), float(row["Mean IoU"])
    path = OUTPUT_ROOT / method / "Polyp" / "1pct" / "test_metrics.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    matches = [row for row in rows if row["Dataset"] == dataset]
    if not matches:
        raise RuntimeError("No {} row in {}".format(dataset, path))
    row = matches[-1]
    return float(row["Mean F1"]), float(row["Mean IoU"])


def read_accelerated_wesam_metrics(dataset: str) -> Tuple[float, float]:
    root = OUTPUT_ROOT / "accelerated_1pct_from_10pct_20260829"
    if dataset == "ISIC":
        path = root / "isic_seed3407" / "ISIC" / "metrics.csv"
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
        rows = [row for row in rows if "final_test" in row["Name"]]
        if not rows:
            raise RuntimeError("No final test row in {}".format(path))
        row = rows[-1]
        return float(row["Mean F1"]), float(row["Mean IoU"])

    path = root / "polyp_seed3407" / "test_metrics.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    matches = [row for row in rows if row["Dataset"] == dataset]
    if not matches:
        raise RuntimeError("No {} row in {}".format(dataset, path))
    row = matches[-1]
    return float(row["Mean F1"]), float(row["Mean IoU"])


def load_metrics(method: str, dataset: str) -> Tuple[float, float]:
    if method == "wesam":
        return read_accelerated_wesam_metrics(dataset)
    # The lightweight prompt ablations export summary.json; trained variants
    # export final test metrics CSVs.
    summary = OUTPUT_ROOT / method / ("ISIC" if dataset == "ISIC" else "Polyp") / "1pct" / "summary.json"
    if summary.exists():
        return read_json_metrics(method, dataset)
    return read_final_csv_metrics(method, dataset)


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def shown(value: float) -> str:
    return "{:.2f}".format(value * 100.0)


def write_table(group_name: str, methods: Iterable[Tuple[str, str]]) -> Path:
    methods = list(methods)
    metrics: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for method, _label in methods:
        for dataset in DATASETS:
            metrics[(method, dataset)] = load_metrics(method, dataset)

    best = {}
    for dataset in DATASETS:
        for metric_index, metric in enumerate(("Dice", "IoU")):
            best[(dataset, metric)] = max(
                metrics[(method, dataset)][metric_index] for method, _label in methods
            )

    lines = [
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}",
        "% Mean F1 is reported as Dice; values are final 1% test metrics.",
        "% WeSAM uses the nested accelerated 1% run with seed=3407; other rows use their corresponding output_current 1% controls.",
        "% Boldface marks per-dataset winners; the final block reports the unweighted mean over all six datasets.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{{Ablation results for {} under the 1\\% labeled setting.}}".format(
            tex_escape(group_name)
        ),
        "\\label{tab:ablation-" + group_name + "}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "\\multirow{2}{*}{Methods} & \\multirow{2}{*}{Dataset} & \\multicolumn{2}{c}{1\\% labeled} \\\\",
        " &  & Dice(\\%) & IoU(\\%) \\\\",
        "\\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for method_index, (method, label) in enumerate(methods):
            dataset_cell = (
                "\\multirow{{{}}}{{*}}{{\\rotatebox{{90}}{{{}}}}}".format(
                    len(methods), tex_escape(dataset)
                )
                if method_index == 0
                else ""
            )
            values = metrics[(method, dataset)]
            cells = [tex_escape(label), dataset_cell]
            for metric_index, metric in enumerate(("Dice", "IoU")):
                value = values[metric_index]
                value_text = shown(value)
                if math.isclose(value, best[(dataset, metric)], rel_tol=0.0, abs_tol=1e-12):
                    value_text = "\\textbf{" + value_text + "}"
                cells.append(value_text)
            lines.append(" & ".join(cells) + " \\\\")
        if dataset_index != len(DATASETS) - 1:
            lines.append("\\midrule")
    lines.append("\\midrule")
    mean_metrics = {}
    for method, label in methods:
        mean_metrics[method] = tuple(
            sum(metrics[(method, dataset)][metric_index] for dataset in DATASETS) / len(DATASETS)
            for metric_index in (0, 1)
        )
    best_means = {
        metric_index: max(mean_metrics[method][metric_index] for method, _label in methods)
        for metric_index in (0, 1)
    }
    for method, label in methods:
        cells = [tex_escape(label), "Mean (6 datasets)"]
        for metric_index in (0, 1):
            value = mean_metrics[method][metric_index]
            value_text = shown(value)
            if math.isclose(value, best_means[metric_index], rel_tol=0.0, abs_tol=1e-12):
                value_text = "\\textbf{" + value_text + "}"
            cells.append(value_text)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""))
    TEX_ROOT.mkdir(parents=True, exist_ok=True)
    output = TEX_ROOT / "ablation_{}.tex".format(group_name)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    for group_name, methods in GROUPS.items():
        output = write_table(group_name, methods)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
