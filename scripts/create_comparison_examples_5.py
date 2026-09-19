from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "comparison_examples_5"
DATASETS = ["ISIC", "Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
METHODS = ["UA-MT", "DTC", "URPC", "CPC-SAM", "H-SAM", "SAMed", "BCP", "KnowSAM", "MC-Net", "WeSAM"]


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    old = {(r["method"], r["dataset"], r["ratio"]): (float(r["dice"]), float(r["iou"])) for r in read(ROOT / "comparison_examples_4/metrics.csv")}
    new = {}
    for r in read(ROOT / "ablation/2/ThreeMethods_SIMPLE_RESULTS_20260813_230439/summary_metrics.csv"):
        dataset = "ISIC" if r["dataset"] == "ISIC2018" else r["dataset"]
        ratio = "1pct" if r["ratio"] == "1percent" else "10pct"
        method = r["method"]
        new[(method, dataset, ratio)] = (float(r["dice"]), float(r["iou"]))
    nested = read(ROOT / "output_current/nested_budget_control_20260823/nested_budget_summary.csv")
    for dataset in DATASETS:
        for ratio in ("1pct", "10pct"):
            rows = [r for r in nested if r["test_dataset"] == dataset and r["budget"] == ratio]
            # Select the highest Dice among the three seeds, as requested.
            best = max(rows, key=lambda r: float(r["dice_percent"]))
            new[("WeSAM", dataset, ratio)] = (float(best["dice_percent"]) / 100, float(best["iou_percent"]) / 100)
    values = {}
    for method in METHODS:
        for dataset in DATASETS:
            for ratio in ("1pct", "10pct"):
                values[(method, dataset, ratio)] = new.get((method, dataset, ratio), old.get((method, dataset, ratio), (math.nan, math.nan)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.csv").write_text("method,dataset,ratio,dice,iou\n" + "\n".join(
        f"{m},{d},{r},{values[(m,d,r)][0]:.12f},{values[(m,d,r)][1]:.12f}" for d in DATASETS for m in METHODS for r in ("1pct", "10pct")
    ) + "\n", encoding="utf-8")
    tex = [
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}",
        "% SAMed/H-SAM/CPC-SAM use ThreeMethods_SIMPLE_RESULTS_20260813_230439; WeSAM uses best Dice among nested seeds.",
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        "\\caption{Quantitative results on ISIC-2018 and polyp datasets under 1\\% and 10\\% labeled settings. WeSAM reports the best result among seeds 1337, 2027, and 3407 for each dataset and budget.}",
        "\\label{tab:comparison-combined-v5}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrr}", "\\toprule",
        "\\multirow{2}{*}{Methods} & \\multirow{2}{*}{Dataset} & \\multicolumn{2}{c}{1\\% labeled} & \\multicolumn{2}{c}{10\\% labeled} \\\\",
        " &  & Dice(\\%) & IoU(\\%) & Dice(\\%) & IoU(\\%) \\\\", "\\midrule",
    ]
    for di, dataset in enumerate(DATASETS):
        for mi, method in enumerate(METHODS):
            cells = []
            if mi == 0:
                cells.append(f"\\multirow{{{len(METHODS)}}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}")
            else:
                cells.append("")
            cells.append(method)
            for ratio in ("1pct", "10pct"):
                for metric_index in (0, 1):
                    value = values[(method, dataset, ratio)][metric_index]
                    text = "--" if math.isnan(value) else f"{value * 100:.2f}"
                    # Bold the actual best method in this dataset/budget/metric.
                    available = [values[(m, dataset, ratio)][metric_index] for m in METHODS if not math.isnan(values[(m, dataset, ratio)][metric_index])]
                    if available and value >= max(available) - 1e-12:
                        text = "\\textbf{" + text + "}"
                    cells.append(text)
            tex.append(" & ".join(cells) + " \\\\")
        if di != len(DATASETS) - 1:
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    (OUT / "tex").mkdir(exist_ok=True)
    (OUT / "tex/metrics_combined.tex").write_text("\n".join(tex), encoding="utf-8")
    (OUT / "README.txt").write_text(
        "comparison_examples_5 uses the refreshed ThreeMethods_SIMPLE_RESULTS_20260813_230439 metrics for CPC-SAM, H-SAM, and SAMed. "
        "WeSAM values are selected from the best Dice among strict nested seeds 1337/2027/3407 for each dataset and budget. "
        "The selection is a best-seed comparison, not a mean±std estimate.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
