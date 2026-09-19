#!/usr/bin/env python3
"""Generate corrected 1% ablation tables (seed 1337).

Data routing:
  Group 1 CLIP-only: original directory (CLIP does not consume support indices)
  Group 1 DINO-only / IFP-only: fixed directory (support_1pct == labeled_1pct)
  Group 2 No prompt: original directory (no prompt head consumes support indices)
  Group 2 WeSAM: fixed directory
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "output_current/fair_ablation_1pct_1shot_seed1337"
FIXED = ROOT / "output_current/fair_ablation_1pct_fixed_seed1337"
OUT = ROOT / "comparison_examples_6/tex"
DATASETS = ["ISIC", "Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]
POLYP_DATASETS = DATASETS[1:]


def load_direct(group1, method):
    isic = json.loads((group1 / method / "ISIC" / "1pct" / "test" / "metrics.json").read_text())
    result = {"ISIC": (isic["mean_f1"], isic["mean_iou"])}
    for dataset in POLYP_DATASETS:
        metrics = json.loads((group1 / method / "Polyp" / "1pct" / dataset / "metrics.json").read_text())
        result[dataset] = (metrics["mean_f1"], metrics["mean_iou"])
    return result


def load_group1():
    return {
        "clip_only": load_direct(OLD / "group1", "clip_only"),
        "dino_only": load_direct(FIXED / "group1", "dino_only"),
        "ifp_only": load_direct(FIXED / "group1", "ifp_only"),
    }


def final_metrics_csv(path):
    rows = list(csv.DictReader(path.open(newline="")))
    row = next(row for row in reversed(rows) if "final_test" in row["Name"])
    return float(row["Mean F1"]), float(row["Mean IoU"])


def test_metrics_csv(path):
    rows = {row["Dataset"]: row for row in csv.DictReader(path.open(newline=""))}
    return {name: (float(row["Mean F1"]), float(row["Mean IoU"])) for name, row in rows.items()}


def load_group2(group1):
    old_group2 = OLD / "group2"
    fixed_group2 = FIXED / "group2"
    no_prompt = {
        "ISIC": final_metrics_csv(old_group2 / "1pct_isic_no_prompt" / "ISIC" / "metrics.csv"),
        **test_metrics_csv(old_group2 / "1pct_polyp_no_prompt" / "test_metrics.csv"),
    }
    wesam = {
        "ISIC": final_metrics_csv(fixed_group2 / "1pct_isic_wesam" / "ISIC" / "metrics.csv"),
        **test_metrics_csv(fixed_group2 / "1pct_polyp_wesam" / "test_metrics.csv"),
    }
    return {"ifp_only": group1["ifp_only"], "no_prompt": no_prompt, "wesam": wesam}


def fmt(value, bold=False):
    text = f"{value * 100:.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def table_header(caption, label):
    return [
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}",
        "% Dice is mean F1. Values are final test metrics multiplied by 100.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        caption,
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "\\multirow{2}{*}{Methods} & \\multirow{2}{*}{Dataset} & \\multicolumn{2}{c}{1\\% labeled} \\\\",
        " &  & Dice(\\%) & IoU(\\%) \\\\",
        "\\midrule",
    ]


def render_table(results, methods, keys, caption, label):
    lines = table_header(caption, label)
    for dataset in DATASETS:
        values = [results[keys[method]][dataset] for method in methods]
        best_f1 = max(value[0] for value in values)
        best_iou = max(value[1] for value in values)
        for index, method in enumerate(methods):
            f1, iou = results[keys[method]][dataset]
            dataset_cell = f"\\multirow{{{len(methods)}}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}" if index == 0 else ""
            lines.append(f"{method} & {dataset_cell} & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} \\\\")
        lines.append("\\midrule")

    means = {
        method: tuple(
            sum(results[keys[method]][dataset][index] for dataset in DATASETS) / len(DATASETS)
            for index in (0, 1)
        )
        for method in methods
    }
    best_f1 = max(value[0] for value in means.values())
    best_iou = max(value[1] for value in means.values())
    for method in methods:
        f1, iou = means[method]
        lines.append(f"{method} & Mean (6 datasets) & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    return "\n".join(lines)


def validate_sources():
    for dataset, expected in (("ISIC", 26), ("Polyp", 13)):
        for method in ("dino_only", "ifp_only"):
            summary = json.loads((FIXED / "group1" / method / dataset / "1pct" / "summary.json").read_text())
            assert summary["references"] == expected, (dataset, method, summary["references"])
    isic_summary = json.loads((FIXED / "group2/1pct_isic_wesam/dataset_summary.json").read_text())
    assert len(isic_summary["support_indices"]) == 26
    polyp_summary = json.loads((FIXED / "group2/1pct_polyp_wesam/dataset_summary.json").read_text())
    assert polyp_summary["prompt_reference_count"] == 13


def main():
    validate_sources()
    group1 = load_group1()
    group2 = load_group2(group1)
    OUT.mkdir(parents=True, exist_ok=True)

    group1_methods = ["CLIP-only", "DINO-only", "IFP-only"]
    group1_keys = {"CLIP-only": "clip_only", "DINO-only": "dino_only", "IFP-only": "ifp_only"}
    group1_text = render_table(
        group1,
        group1_methods,
        group1_keys,
        "\\caption{Ablation results for group 1 under the 1\\% labeled setting.}",
        "tab:ablation-group1-1pct",
    )
    (OUT / "ablation_group1_1pct.tex").write_text(group1_text)

    group2_methods = ["IFP-only", "No prompt", "WeSAM"]
    group2_keys = {"IFP-only": "ifp_only", "No prompt": "no_prompt", "WeSAM": "wesam"}
    group2_text = render_table(
        group2,
        group2_methods,
        group2_keys,
        "\\caption{Ablation of prompt-only, semi-supervised-only, and their combination under the 1\\% labeled setting. IFP-only uses frozen SAM2; No prompt uses the semi-supervised framework without IFP prompts.}",
        "tab:ablation-group2-1pct",
    )
    (OUT / "ablation_group2_1pct.tex").write_text(group2_text)
    print("validated references: ISIC=26, Polyp=13")
    print("wrote", OUT / "ablation_group1_1pct.tex")
    print("wrote", OUT / "ablation_group2_1pct.tex")


if __name__ == "__main__":
    main()
