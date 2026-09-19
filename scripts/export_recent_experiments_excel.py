#!/usr/bin/env python3
"""Create a grouped, path-free workbook for recent WeSAM experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_current/recent_experiments_1pct_10pct_summary.xlsx"
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEADER = PatternFill("solid", fgColor="1F4E78")


def csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style(sheet):
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = HEADER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for col in range(1, sheet.max_column + 1):
        width = max(len(str(sheet.cell(row, col).value or "")) for row in range(1, sheet.max_row + 1))
        sheet.column_dimensions[get_column_letter(col)].width = min(max(width + 2, 12), 34)


def yellow_best(sheet, group_cols, dice_col):
    groups = defaultdict(list)
    for row in range(2, sheet.max_row + 1):
        groups[tuple(sheet.cell(row, c).value for c in group_cols)].append(row)
    for rows in groups.values():
        vals = [sheet.cell(r, dice_col).value for r in rows if isinstance(sheet.cell(r, dice_col).value, (float, int))]
        if vals:
            best = max(vals)
            for r in rows:
                if sheet.cell(r, dice_col).value == best:
                    for c in range(1, sheet.max_column + 1):
                        sheet.cell(r, c).fill = YELLOW


def add_notes(book):
    s = book.active
    s.title = "说明"
    s.append(("内容", "说明"))
    for row in [
        ("组织方式", "每个数据集的配置和随机种子放在一起；先看 Detailed_ISIC / Detailed_Polyp，再看 Best_Results。"),
        ("黄底", "同一数据集、同一标注比例、同一实验协议内 Dice 最高的行。Best_Results 中黄底表示当前汇总实验的最佳配置。"),
        ("历史结果", "Historical_metrics_combined 是 comparison_examples_4/metrics_combined.tex 对应的数据；其最佳方法也按数据集和标注比例标黄。"),
        ("注意", "polyp_prompt_ablation 使用独立的 Polyp 10% 多 seed 消融协议；Nested_Budget 使用严格嵌套 1%/10% 协议，两者分开列出，避免混比。"),
    ]:
        s.append(row)
    style(s)


def add_historical(book):
    s = book.create_sheet("Historical_metrics_combined")
    s.append(("Dataset", "Budget", "Method", "Images", "Dice (%)", "IoU (%)"))
    for r in csv_rows(ROOT / "comparison_examples_4/metrics.csv"):
        s.append((r["dataset"], r["ratio"], r["method"], int(r["n"]), float(r["dice"]) * 100, float(r["iou"]) * 100))
    style(s)
    yellow_best(s, (1, 2), 5)


def add_isic(book):
    s = book.create_sheet("Detailed_ISIC")
    s.append(("Dataset", "Budget", "Configuration", "Seed", "Prompt", "GT/IFP Mode", "Dropout", "Anchor", "Contrastive", "Dice (%)", "IoU (%)", "Best Epoch"))
    source = ROOT / "output_current/ablation_experiments_summary.csv"
    for r in csv_rows(source):
        if r["dataset"] == "ISIC":
            s.append(("ISIC", "1pct", r["experiment"], int(r["seed"]), r["prompt_backend"], r["labeled_prompt_mode"], float(r["dropout"]), float(r["anchor_weight"]), float(r["contrast_weight"]), float(r["dice_percent"]), float(r["iou_percent"]), int(r["best_epoch"])))
    for r in csv_rows(ROOT / "output_current/nested_budget_control_20260823/nested_budget_summary.csv"):
        if r["dataset"] == "ISIC":
            s.append(("ISIC", r["budget"], "Original WeSAM (strict nested)", int(r["seed"]), "IFP", "GT", 0.0, 0.0, 0.0, float(r["dice_percent"]), float(r["iou_percent"]), int(r["best_epoch"])))
    style(s)
    yellow_best(s, (1, 2), 10)


def add_polyp(book):
    s = book.create_sheet("Detailed_Polyp")
    s.append(("Test Dataset", "Budget", "Configuration", "Seed", "Prompt", "GT/IFP Mode", "Dropout", "Anchor", "Contrastive", "Dice (%)", "IoU (%)", "Best Epoch"))
    root = ROOT / "output_current/polyp_prompt_ablation"
    configs = {
        "no_prompt": ("No prompt", "none", "none", 0.0, 0.0, 0.0),
        "wesam": ("Original WeSAM", "IFP", "GT", 0.0, 1.0, 0.1),
        "mixed_dropout_0": ("Mixed GT/IFP", "IFP", "mixed", 0.0, 1.0, 0.1),
        "mixed_dropout_005": ("Mixed GT/IFP", "IFP", "mixed", 0.05, 1.0, 0.1),
        "mixed_dropout_01": ("Mixed GT/IFP", "IFP", "mixed", 0.1, 1.0, 0.1),
    }
    for folder, config in configs.items():
        for seed in (2027, 3407):
            p = root / folder / f"seed{seed}" / "test_metrics.csv"
            if p.exists():
                for r in csv_rows(p):
                    s.append((r["Dataset"], "10pct", config[0], seed, *config[1:], float(r["Mean F1"]) * 100, float(r["Mean IoU"]) * 100, int(r["Best validation epoch"])))
    for folder, name, dropout in [("polyp_mixed_dropout0_seed1337", "Mixed GT/IFP", 0.0), ("polyp_mixed_dropout005_seed1337", "Mixed GT/IFP", 0.05), ("polyp_mixed_dropout01_seed1337", "Mixed GT/IFP", 0.1)]:
        p = ROOT / "output_current/remaining_experiments" / folder / "test_metrics.csv"
        if p.exists():
            for r in csv_rows(p):
                s.append((r["Dataset"], "1pct", name, 1337, "IFP", "mixed", dropout, 0.0, 0.0, float(r["Mean F1"]) * 100, float(r["Mean IoU"]) * 100, int(r["Best validation epoch"])))
    for r in csv_rows(ROOT / "output_current/nested_budget_control_20260823/nested_budget_summary.csv"):
        if r["dataset"] == "Polyp":
            s.append((r["test_dataset"], r["budget"], "Original WeSAM (strict nested)", int(r["seed"]), "IFP", "GT", 0.0, 0.0, 0.0, float(r["dice_percent"]), float(r["iou_percent"]), int(r["best_epoch"])))
    style(s)
    yellow_best(s, (1, 2), 10)


def add_nested(book):
    s = book.create_sheet("Nested_Budget")
    s.append(("Domain", "Test Dataset", "Budget", "Seed", "Labeled Images", "IFP Support Images", "Dice (%)", "IoU (%)", "Best Epoch"))
    for r in csv_rows(ROOT / "output_current/nested_budget_control_20260823/nested_budget_summary.csv"):
        s.append((r["dataset"], r["test_dataset"], r["budget"], int(r["seed"]), int(r["labeled_count"]), int(r["support_count"]), float(r["dice_percent"]), float(r["iou_percent"]), int(r["best_epoch"])))
    style(s)
    yellow_best(s, (1, 2, 4), 7)


def add_best(book):
    s = book.create_sheet("Best_Results")
    s.append(("Result Group", "Dataset", "Budget", "Best Configuration", "Best Seed", "Dice (%)", "IoU (%)", "Protocol"))
    historical = book["Historical_metrics_combined"]
    groups = defaultdict(list)
    for row in range(2, historical.max_row + 1):
        groups[(historical.cell(row, 1).value, historical.cell(row, 2).value)].append(row)
    for (dataset, budget), rows in sorted(groups.items()):
        best = max(rows, key=lambda row: historical.cell(row, 5).value)
        s.append(("Historical metrics_combined", dataset, budget, historical.cell(best, 3).value, "-", historical.cell(best, 5).value, historical.cell(best, 6).value, "comparison_examples_4/metrics_combined.tex"))
    for ws_name, protocol in (("Detailed_ISIC", "Recent ISIC ablations"), ("Detailed_Polyp", "Recent Polyp ablations")):
        src = book[ws_name]
        groups = defaultdict(list)
        for row in range(2, src.max_row + 1):
            groups[(src.cell(row, 1).value, src.cell(row, 2).value)].append(row)
        for (dataset, budget), rows in sorted(groups.items()):
            best = max(rows, key=lambda row: src.cell(row, 10).value)
            s.append(("Recent experiments", dataset, budget, src.cell(best, 3).value, src.cell(best, 4).value, src.cell(best, 10).value, src.cell(best, 11).value, protocol))
    style(s)
    for row in range(2, s.max_row + 1):
        for col in range(1, s.max_column + 1):
            s.cell(row, col).fill = YELLOW


def main():
    book = Workbook()
    add_notes(book)
    add_historical(book)
    add_isic(book)
    add_polyp(book)
    add_nested(book)
    add_best(book)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUT)
    check = load_workbook(OUT, read_only=True)
    print(f"Wrote {OUT} with sheets: {', '.join(check.sheetnames)}")


if __name__ == "__main__":
    main()
