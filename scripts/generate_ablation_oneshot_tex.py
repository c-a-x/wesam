#!/usr/bin/env python3
"""Generate ablation_group{1,2}_oneshot.tex for the fair 1-shot runs (seed 1337),
formatted identically to comparison_examples_6/tex/*10pct.tex and *1pct.tex."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output_current/fair_ablation_1pct_1shot_seed1337"
GROUP1 = RUN / "group1"
GROUP2 = RUN / "group2"
OUT = ROOT / "comparison_examples_6/tex"
DATASETS = ["ISIC", "Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]

def load_group1_oneshot():
    result = {}
    for method in ("clip_only", "dino_only", "ifp_only"):
        isic_path = GROUP1 / method / "ISIC" / "1shot" / "test" / "metrics.json"
        isic = json.loads(isic_path.read_text())
        isic_metrics = {"ISIC": (isic["mean_f1"], isic["mean_iou"])}
        polyp = {}
        for ds in DATASETS[1:]:
            p = GROUP1 / method / "Polyp" / "1shot" / ds / "metrics.json"
            m = json.loads(p.read_text())
            polyp[ds] = (m["mean_f1"], m["mean_iou"])
        result[method] = {**isic_metrics, **polyp}
    return result

def load_group2_oneshot():
    """Group2: IFP-only = group1 IFP-only; No prompt & WeSAM from trained checkpoints."""
    group1 = load_group1_oneshot()

    def final_metrics_csv(path):
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        row = next(row for row in reversed(rows) if "final_test" in row["Name"])
        return float(row["Mean F1"]), float(row["Mean IoU"])

    def test_metrics_csv(path):
        with open(path, newline="") as fh:
            rows = {r["Dataset"]: r for r in csv.DictReader(fh)}
        return {name: (float(r["Mean F1"]), float(r["Mean IoU"])) for name, r in rows.items()}

    no_prompt = {
        "ISIC": final_metrics_csv(GROUP2 / "1shot_isic_no_prompt" / "ISIC" / "metrics.csv"),
        **test_metrics_csv(GROUP2 / "1shot_polyp_no_prompt" / "test_metrics.csv"),
    }
    wesam = {
        "ISIC": final_metrics_csv(GROUP2 / "1shot_isic_wesam" / "ISIC" / "metrics.csv"),
        **test_metrics_csv(GROUP2 / "1shot_polyp_wesam" / "test_metrics.csv"),
    }
    return {"ifp_only": group1["ifp_only"], "no_prompt": no_prompt, "wesam": wesam}

def fmt(value, bold=False):
    text = f"{value * 100:.2f}"
    return f"\\textbf{{{text}}}" if bold else text

def render_group1(group1):
    methods = ["CLIP-only", "DINO-only", "IFP-only"]
    keys = {"CLIP-only": "clip_only", "DINO-only": "dino_only", "IFP-only": "ifp_only"}
    setting = "1-shot"
    lines = ["% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}",
             "% Dice is mean F1. Values are final test metrics multiplied by 100.",
             "\\begin{table*}[t]", "\\centering", "\\scriptsize",
             "\\caption{Ablation results for group 1 under the 1-shot setting.}",
             "\\label{tab:ablation-group1-oneshot}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrr}", "\\toprule",
             f"\\multirow{{2}}{{*}}{{Methods}} & \\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{2}}{{c}}{{{setting}}} " + r"\\",
             " &  & Dice(\\%) & IoU(\\%) " + r"\\", "\\midrule"]
    for dataset in DATASETS:
        values = [group1[keys[m]][dataset] for m in methods]
        best_f1, best_iou = max(x[0] for x in values), max(x[1] for x in values)
        for index, method in enumerate(methods):
            f1, iou = group1[keys[method]][dataset]
            ds = f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}" if index == 0 else ""
            lines.append(f"{method} & {ds} & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} " + r"\\")
        lines.append("\\midrule")
    means = {m: tuple(sum(group1[keys[m]][d][i] for d in DATASETS) / len(DATASETS) for i in (0, 1)) for m in methods}
    best_f1, best_iou = max(x[0] for x in means.values()), max(x[1] for x in means.values())
    for method in methods:
        f1, iou = means[method]
        lines.append(f"{method} & Mean (6 datasets) & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} " + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    return "\n".join(lines)

def render_group2(group2):
    methods = ["IFP-only", "No prompt", "WeSAM"]
    keys = {"IFP-only": "ifp_only", "No prompt": "no_prompt", "WeSAM": "wesam"}
    setting = "1-shot"
    caption = (r"\caption{Ablation of prompt-only, semi-supervised-only, and their "
               r"combination under the 1-shot setting. IFP-only uses frozen SAM2; "
               r"No prompt uses the semi-supervised framework without IFP prompts.}")
    lines = ["% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}",
             "% Dice is mean F1. Values are final test metrics multiplied by 100.",
             "\\begin{table*}[t]", "\\centering", "\\scriptsize", caption,
             "\\label{tab:ablation-group2-oneshot}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrr}", "\\toprule",
             f"\\multirow{{2}}{{*}}{{Methods}} & \\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{2}}{{c}}{{{setting}}} " + r"\\",
             " &  & Dice(\\%) & IoU(\\%) " + r"\\", "\\midrule"]
    for dataset in DATASETS:
        values = [group2[keys[m]][dataset] for m in methods]
        best_f1, best_iou = max(x[0] for x in values), max(x[1] for x in values)
        for index, method in enumerate(methods):
            f1, iou = group2[keys[method]][dataset]
            ds = f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}" if index == 0 else ""
            lines.append(f"{method} & {ds} & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} " + r"\\")
        lines.append("\\midrule")
    means = {m: tuple(sum(group2[keys[m]][d][i] for d in DATASETS) / len(DATASETS) for i in (0, 1)) for m in methods}
    best_f1, best_iou = max(x[0] for x in means.values()), max(x[1] for x in means.values())
    for method in methods:
        f1, iou = means[method]
        lines.append(f"{method} & Mean (6 datasets) & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} " + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    return "\n".join(lines)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    group1 = load_group1_oneshot()
    group2 = load_group2_oneshot()
    (OUT / "ablation_group1_oneshot.tex").write_text(render_group1(group1))
    (OUT / "ablation_group2_oneshot.tex").write_text(render_group2(group2))
    print("Wrote:", OUT / "ablation_group1_oneshot.tex")
    print("Wrote:", OUT / "ablation_group2_oneshot.tex")

if __name__ == "__main__":
    main()
