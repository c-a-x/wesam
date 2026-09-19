#!/usr/bin/env python3
"""Export grouped recent results and historical comparison as LaTeX."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comparison_examples_4/tex/metrics_grouped.tex"


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def pct(v):
    return f"{float(v):.2f}"


def table_begin(columns, caption, label):
    return [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
    ]


def main():
    lines = [
        r"% Requires: \usepackage{booktabs,graphicx,multirow}",
        r"% This file keeps historical comparisons separate from recent ablations.",
        "",
    ]

    historical = rows(ROOT / "comparison_examples_4/metrics.csv")
    lines += table_begin("llrrrr", "Historical comparison results from metrics_combined.tex.", "tab:historical-grouped")
    lines += [r"Dataset & Method & \multicolumn{2}{c}{1\% labeled} & \multicolumn{2}{c}{10\% labeled} \\", r" &  & Dice(\%) & IoU(\%) & Dice(\%) & IoU(\%) \\", r"\midrule"]
    datasets = ["ISIC", "Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
    methods = []
    for dataset in datasets:
        ds = [r for r in historical if r["dataset"] == dataset]
        if not methods:
            methods = [r["method"] for r in ds if r["ratio"] == "1pct"]
        for method in methods:
            one = next((r for r in ds if r["method"] == method and r["ratio"] == "1pct"), None)
            ten = next((r for r in ds if r["method"] == method and r["ratio"] == "10pct"), None)
            if one and ten:
                lines.append(f"{esc(dataset)} & {esc(method)} & {pct(float(one['dice']) * 100)} & {pct(float(one['iou']) * 100)} & {pct(float(ten['dice']) * 100)} & {pct(float(ten['iou']) * 100)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}", ""]

    recent = rows(ROOT / "output_current/recent_experiments_1pct_10pct_summary.xlsx" ) if False else rows(ROOT / "output_current/nested_budget_control_20260823/nested_budget_summary.csv")
    # Controlled nested budget table: every seed is shown, with 1% and 10% side by side.
    lines += table_begin("llrrrrrr", "Strict nested-budget results; 1\% is a subset of 10\% for the same seed.", "tab:nested-budget-grouped")
    lines += [r"Dataset & Test set & Seed & \multicolumn{2}{c}{1\% labeled} & \multicolumn{2}{c}{10\% labeled} \\", r" &  &  & Dice(\%) & IoU(\%) & Dice(\%) & IoU(\%) \\", r"\midrule"]
    for dataset in ("ISIC", "Polyp"):
        testsets = ["ISIC"] if dataset == "ISIC" else ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
        for testset in testsets:
            for seed in (1337, 2027, 3407):
                subset = [r for r in recent if r["dataset"] == dataset and r["test_dataset"] == testset and int(r["seed"]) == seed]
                one = next(r for r in subset if r["budget"] == "1pct")
                ten = next(r for r in subset if r["budget"] == "10pct")
                lines.append(f"{esc(dataset)} & {esc(testset)} & {seed} & {pct(one['dice_percent'])} & {pct(one['iou_percent'])} & {pct(ten['dice_percent'])} & {pct(ten['iou_percent'])} \\\\")
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}", ""]

    # Best-result table is generated from the existing path-free Excel source CSVs.
    best = []
    for r in historical:
        best.append(("Historical", r["dataset"], r["ratio"], r["method"], "--", float(r["dice"]) * 100, float(r["iou"]) * 100))
    for dataset in ("ISIC", "Polyp"):
        for testset in (["ISIC"] if dataset == "ISIC" else ["Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]):
            for budget in ("1pct", "10pct"):
                subset = [r for r in recent if r["dataset"] == dataset and r["test_dataset"] == testset and r["budget"] == budget]
                if subset:
                    row = max(subset, key=lambda r: float(r["dice_percent"]))
                    best.append(("Nested recent", testset, budget, "Original WeSAM", row["seed"], float(row["dice_percent"]), float(row["iou_percent"])))
    lines += table_begin("llrlrr", "Best result by dataset and labeled budget; historical and nested protocols are shown separately.", "tab:best-grouped")
    lines += [r"Protocol & Dataset & Budget & Configuration & Seed & Dice(\%) & IoU(\%) \\", r"\midrule"]
    for group, dataset, budget, config, seed, dice, iou in best:
        lines.append(f"{esc(group)} & {esc(dataset)} & {esc(budget)} & {esc(config)} & {esc(seed)} & {dice:.2f} & {iou:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
