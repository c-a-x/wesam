#!/usr/bin/env python3
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "output_current/ablation_1shot_10pct_20260908"
ACCELERATED_CURRENT = ROOT / "output_current/ablation_accelerated_current_split_20260909"
FAIR_RUN = ROOT / "output_current/fair_ablation_10pct_seed1337"
FAIR_GROUP1 = FAIR_RUN / "group1"
FAIR_GROUP2 = FAIR_RUN / "group2"
FAIR_RERUN = ROOT / "output_current/fair_ablation_10pct_seed1337_rerun"
OUT = ROOT / "comparison_examples_6/tex"
DATASETS = ["ISIC", "Kvasir", "CVC-ClinicDB", "CVC-ColonDB", "CVC-300", "ETIS-LaribPolypDB"]

def final_metrics(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    row = next(row for row in reversed(rows) if "final_test" in row["Name"])
    return float(row["Mean F1"]), float(row["Mean IoU"])

def test_metrics(path):
    with open(path, newline="") as fh:
        rows = {row["Dataset"]: row for row in csv.DictReader(fh)}
    return {name: (float(row["Mean F1"]), float(row["Mean IoU"])) for name, row in rows.items()}

def direct_metrics(path):
    data = json.loads(Path(path).read_text())["results"]
    return {name: (item["mean_f1"], item["mean_iou"]) for name, item in data.items() if name != "validation"}

def combine(isic, polyp):
    return {"ISIC": isic, **polyp}

def load_legacy_1shot_results():
    paths = (ROOT / "output_current/shot1_ablation/no_prompt/ISIC/metrics.csv", ROOT / "output_current/shot1_ablation/no_semisupervised/isic_ifp_supervised_1shot/ISIC/metrics.csv", ROOT / "output_current/wesam/ISIC/1/ISIC/metrics.csv", RUN / "no_prompt/Polyp/1shot/test_metrics.csv", RUN / "no_semisupervised/Polyp/1shot/polyp_ifp_supervised_1shot/test_metrics.csv", ROOT / "output_current/wesam/Polyp/1/test_metrics.csv")
    no_prompt_isic, no_semi_isic, wesam_isic, no_prompt_polyp, no_semi_polyp, wesam_polyp = paths
    result = {}
    for method in ("clip_only", "dino_only"):
        result[method] = combine(
            direct_metrics(RUN / method / "ISIC" / "1shot" / "summary.json")["test"],
            direct_metrics(RUN / method / "Polyp" / "1shot" / "summary.json"),
        )
    result["no_prompt"] = combine(final_metrics(no_prompt_isic), test_metrics(no_prompt_polyp))
    result["no_semi"] = combine(final_metrics(no_semi_isic), test_metrics(no_semi_polyp))
    result["wesam"] = combine(final_metrics(wesam_isic), test_metrics(wesam_polyp))
    return result


def load_fair_10pct_results():
    group1 = {}
    for method in ("clip_only", "dino_only", "ifp_only"):
        group1[method] = combine(
            direct_metrics(FAIR_GROUP1 / method / "ISIC" / "10pct" / "summary.json")["test"],
            direct_metrics(FAIR_GROUP1 / method / "Polyp" / "10pct" / "summary.json"),
        )

    group2 = {
        # Prompt-only is the same frozen IFP-only evaluation used by group 1.
        "ifp_only": group1["ifp_only"],
        "no_prompt": combine(
            final_metrics(FAIR_GROUP2 / "isic_no_prompt" / "ISIC" / "metrics.csv"),
            test_metrics(FAIR_GROUP2 / "polyp_no_prompt" / "test_metrics.csv"),
        ),
        "wesam": combine(
            final_metrics(FAIR_RERUN / "isic_ifp_prompt" / "ISIC" / "metrics.csv"),
            test_metrics(FAIR_RERUN / "polyp_ifp_prompt" / "test_metrics.csv"),
        ),
    }
    return group1, group2

def fmt(value, bold=False):
    text = f"{value * 100:.2f}"
    return f"\\textbf{{{text}}}" if bold else text

def render(group, budget, results):
    if group == 1 and budget == "10pct":
        methods = ["CLIP-only", "DINO-only", "IFP-only"]
    elif group == 1:
        methods = ["CLIP-only", "DINO-only", "No semi-supervised"]
    elif budget == "10pct":
        methods = ["IFP-only", "No prompt", "WeSAM"]
    else:
        methods = ["No prompt", "No semi-supervised", "WeSAM"]
    keys = {"CLIP-only": "clip_only", "DINO-only": "dino_only", "IFP-only": "ifp_only", "No prompt": "no_prompt", "No semi-supervised": "no_semi", "WeSAM": "wesam"}
    setting = "1-shot" if budget == "1shot" else "10\\% labeled"
    suffix = "oneshot" if budget == "1shot" else "10pct"
    if group == 2 and budget == "10pct":
        caption = (r"\caption{Ablation of prompt-only, semi-supervised-only, and their "
                   r"combination under the 10\% labeled setting. IFP-only uses frozen SAM2; "
                   r"No prompt uses the semi-supervised framework without IFP prompts.}")
    else:
        caption = f"\\caption{{Ablation results for group {group} under the {setting} setting.}}"
    lines = ["% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}", "% Dice is mean F1. Values are final test metrics multiplied by 100.", "\\begin{table*}[t]", "\\centering", "\\scriptsize", caption, f"\\label{{tab:ablation-group{group}-{suffix}}}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrr}", "\\toprule", f"\\multirow{{2}}{{*}}{{Methods}} & \\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{2}}{{c}}{{{setting}}} " + r"\\", " &  & Dice(\\%) & IoU(\\%) " + r"\\", "\\midrule"]
    for dataset in DATASETS:
        values = [results[keys[m]][dataset] for m in methods]
        best_f1, best_iou = max(x[0] for x in values), max(x[1] for x in values)
        for index, method in enumerate(methods):
            f1, iou = results[keys[method]][dataset]
            ds = f"\\multirow{{3}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}" if index == 0 else ""
            lines.append(f"{method} & {ds} & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} " + r"\\")
        lines.append("\\midrule")
    means = {m: tuple(sum(results[keys[m]][d][i] for d in DATASETS) / len(DATASETS) for i in (0, 1)) for m in methods}
    best_f1, best_iou = max(x[0] for x in means.values()), max(x[1] for x in means.values())
    for method in methods:
        f1, iou = means[method]
        lines.append(f"{method} & Mean (6 datasets) & {fmt(f1, f1 == best_f1)} & {fmt(iou, iou == best_iou)} " + r"\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""]
    return "\n".join(lines)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legacy = load_legacy_1shot_results()
    for group in (1, 2):
        (OUT / f"ablation_group{group}_oneshot.tex").write_text(render(group, "1shot", legacy))

    fair_group1, fair_group2 = load_fair_10pct_results()
    (OUT / "ablation_group1_10pct.tex").write_text(render(1, "10pct", fair_group1))
    (OUT / "ablation_group2_10pct.tex").write_text(render(2, "10pct", fair_group2))

if __name__ == "__main__":
    main()
