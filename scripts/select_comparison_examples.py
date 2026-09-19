#!/usr/bin/env python3
"""Select GT-verified qualitative comparison examples and generate LaTeX tables.

The script compares WeSAM with binary prediction masks from all three ablation
groups.  UA-MT, DTC, and URPC masks are imported by
``organize_ablation1_raw_masks.py``; the script never scores the old RGB
visualization files in ``ablation/1``.

For every requested dataset and annotation ratio, one common test sample is
selected where WeSAM has the highest Dice among all comparable methods.  Among
those samples, the selection maximizes WeSAM's Dice margin over the strongest
baseline.  The output contains the original image, GT, each prediction mask, a
montage, CSV audit files, and LaTeX result/figure files.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "comparison_examples"
DATASETS = (
    "ISIC",
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)
RATIOS = ("1pct", "10pct")
BASELINE_METHODS = (
    "UA-MT",
    "DTC",
    "URPC",
    "CPC-SAM",
    "H-SAM",
    "SAMed",
    "BCP",
    "KnowSAM",
    "MC-Net",
)
METHOD_ORDER = ("WeSAM",) + BASELINE_METHODS


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    gt_path: Path


@dataclass
class Scores:
    dice: float
    iou: float


@dataclass
class Selection:
    dataset: str
    ratio: str
    sample_id: str
    wesam_dice: float
    wesam_iou: float
    strongest_baseline: str
    strongest_baseline_dice: float
    weakest_baseline: str
    weakest_baseline_dice: float
    mean_baseline_dice: float
    selection_margin: float
    baseline_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--ratios", nargs="+", choices=RATIOS, default=RATIOS)
    parser.add_argument(
        "--min-baselines",
        type=int,
        default=len(BASELINE_METHODS),
        help="Minimum number of baseline masks required for a selected sample.",
    )
    parser.add_argument(
        "--allow-wesam-not-best",
        action="store_true",
        help="Allow a fallback selection when WeSAM is not the Dice leader.",
    )
    parser.add_argument(
        "--min-baseline-dice",
        type=float,
        default=0.0,
        help="Require every compared baseline to have Dice strictly above this value.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Require WeSAM Dice minus the strongest baseline Dice to meet this margin.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Worker processes for aggregate metric calculation (default: 16).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge CSV audits with an earlier run in the same output directory.",
    )
    return parser.parse_args()


def canonical_id(path: Path) -> str:
    """Map a prediction filename to the image stem used by the GT manifest."""
    stem = path.stem
    for suffix in ("_pred", "_prediction", "_segmentation"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def binary_mask(path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read a PNG mask and normalize either 0/1 or 0/255 encodings to bool."""
    with Image.open(path) as image:
        image = image.convert("L")
        if target_size is not None and image.size != target_size:
            image = image.resize(target_size, Image.Resampling.NEAREST)
        values = np.asarray(image)
    if values.max(initial=0) <= 1:
        return values > 0
    return values >= 128


def score_masks(pred_path: Path, gt_path: Path) -> Scores:
    with Image.open(gt_path) as gt_image:
        target_size = gt_image.size
    prediction = binary_mask(pred_path, target_size)
    ground_truth = binary_mask(gt_path)
    intersection = np.logical_and(prediction, ground_truth).sum(dtype=np.int64)
    pred_count = prediction.sum(dtype=np.int64)
    gt_count = ground_truth.sum(dtype=np.int64)
    union = np.logical_or(prediction, ground_truth).sum(dtype=np.int64)
    dice = 1.0 if pred_count + gt_count == 0 else float(2.0 * intersection / (pred_count + gt_count))
    iou = 1.0 if union == 0 else float(intersection / union)
    return Scores(dice=dice, iou=iou)


def score_job(job: Tuple[Path, Path]) -> Scores:
    """Pickle-friendly wrapper for ProcessPoolExecutor."""
    return score_masks(*job)


def samples_for_dataset(project_root: Path, dataset: str) -> Dict[str, Sample]:
    if dataset == "ISIC":
        base = project_root / "Data" / "ISIC2018" / "test"
        image_dir = base / "images"
        mask_dir = base / "masks"
        mask_for = lambda stem: mask_dir / (stem + "_segmentation.png")
    else:
        base = project_root / "Data" / "kvasir-seg" / "organized" / dataset / "test"
        image_dir = base / "images"
        mask_dir = base / "masks"
        mask_for = lambda stem: mask_dir / (stem + ".png")

    samples = {}
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        gt_path = mask_for(image_path.stem)
        if gt_path.is_file():
            samples[image_path.stem] = Sample(image_path.stem, image_path, gt_path)
    if not samples:
        raise FileNotFoundError("No image/GT pairs found for {} under {}".format(dataset, base))
    return samples


def wesam_mask_dir(project_root: Path, dataset: str, ratio: str) -> Path:
    root = project_root / "output_current" / "wesam"
    if dataset == "ISIC":
        candidates = []
        if ratio == "1pct":
            # The repeat is the only currently exported ISIC 1% prediction set.
            candidates.append(root / "ISIC" / "1pct_repeat_20260725" / "ISIC" / "pred_masks")
        candidates.extend(
            (
                root / "ISIC" / ratio / "ISIC" / "pred_masks",
                root / "ISIC" / ratio.replace("pct", "") / "ISIC" / "pred_masks",
            )
        )
    else:
        candidates = (root / "Polyp" / ratio / dataset / "pred_masks",)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def baseline_mask_dirs(project_root: Path, dataset: str, ratio: str) -> Dict[str, Path]:
    if dataset == "ISIC":
        relative = Path("ISIC") / ratio / "ISIC" / "pred_masks"
        raw_relative = relative
    else:
        relative = Path("Polyp") / ratio / dataset / "pred_masks"
        raw_relative = relative
    return {
        "UA-MT": project_root / "ablation" / "1" / "UA-MT" / raw_relative,
        "DTC": project_root / "ablation" / "1" / "DTC" / raw_relative,
        "URPC": project_root / "ablation" / "1" / "URPC" / raw_relative,
        "CPC-SAM": project_root / "ablation" / "2" / "CPC_SAM" / relative,
        "H-SAM": project_root / "ablation" / "2" / "H_SAM" / relative,
        "SAMed": project_root / "ablation" / "2" / "SAMed" / relative,
        "BCP": project_root / "ablation" / "3" / "BCP" / relative,
        "KnowSAM": project_root / "ablation" / "3" / "KnowSAM" / relative,
        "MC-Net": project_root / "ablation" / "3" / "MC-Net" / relative,
    }


def prediction_map(mask_dir: Path) -> Dict[str, Path]:
    if not mask_dir.is_dir():
        return {}
    result = {}
    for path in sorted(mask_dir.glob("*.png")):
        result[canonical_id(path)] = path
    return result


def format_number(value: float) -> str:
    return "--" if math.isnan(value) else "{:.2f}".format(value)


def make_panel(image_path: Path, label: str, size: Tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (size[0], size[1] + 24), "white")
        x = (size[0] - image.width) // 2
        y = 24 + (size[1] - image.height) // 2
        panel.paste(image, (x, y))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), label, fill="black", font=ImageFont.load_default())
    return panel


def render_montage(
    output_path: Path,
    sample: Sample,
    method_masks: OrderedDict,
    score_by_method: Dict[str, Scores],
) -> None:
    panel_size = (280, 280)
    panels = [
        make_panel(sample.image_path, "Image", panel_size),
        make_panel(sample.gt_path, "Ground truth", panel_size),
    ]
    for method, mask_path in method_masks.items():
        score = score_by_method[method]
        panels.append(make_panel(mask_path, "{} Dice {:.3f}".format(method, score.dice), panel_size))

    columns = 3
    rows = int(math.ceil(len(panels) / float(columns)))
    canvas = Image.new("RGB", (columns * panel_size[0], rows * (panel_size[1] + 24)), "white")
    for index, panel in enumerate(panels):
        x = (index % columns) * panel_size[0]
        y = (index // columns) * (panel_size[1] + 24)
        canvas.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def output_dataset_dir(output_dir: Path, dataset: str, ratio: str) -> Path:
    if dataset == "ISIC":
        return output_dir / "ISIC" / ratio / "ISIC"
    return output_dir / "Polyp" / ratio / dataset


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_rows(
    existing: List[Dict[str, object]],
    fresh: List[Dict[str, object]],
    key_fields: Sequence[str],
) -> List[Dict[str, object]]:
    merged = {}
    for row in existing + fresh:
        merged[tuple(str(row[field]) for field in key_fields)] = row
    return [merged[key] for key in sorted(merged)]


def selection_from_row(row: Dict[str, object]) -> Selection:
    integer_fields = {"baseline_count"}
    values = {}
    for field in Selection.__dataclass_fields__:
        value = row[field]
        if field in integer_fields:
            values[field] = int(value)
        elif field in ("dataset", "ratio", "sample_id", "strongest_baseline", "weakest_baseline"):
            values[field] = str(value)
        else:
            values[field] = float(value)
    return Selection(**values)


def tex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def write_metric_tex(output_dir: Path, ratio: str, metric_rows: List[Dict[str, object]]) -> None:
    by_method_dataset = {(row["method"], row["dataset"]): row for row in metric_rows if row["ratio"] == ratio}
    methods = [method for method in METHOD_ORDER if any((method, dataset) in by_method_dataset for dataset in DATASETS)]
    best = {}
    for dataset in DATASETS:
        for metric in ("dice", "iou"):
            values = []
            for method in methods:
                row = by_method_dataset.get((method, dataset))
                value = float(row[metric]) if row and row[metric] != "" else float("nan")
                if not math.isnan(value):
                    values.append(value)
            if values:
                best[(dataset, metric)] = max(values)

    alignment = "l" + "r" * (2 * len(DATASETS))
    lines = [
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{{GT-based segmentation results for the {} labeled setting.}}".format(ratio.replace("pct", "\\%")),
        "\\label{{tab:comparison-{}}}".format(ratio),
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + alignment + "}",
        "\\toprule",
        "Method & " + " & ".join("\\multicolumn{2}{c}{" + tex_escape(dataset) + "}" for dataset in DATASETS) + " \\\\",
        " & " + " & ".join("Dice(\\%) & IoU(\\%)" for _ in DATASETS) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = [tex_escape(method)]
        for dataset in DATASETS:
            row = by_method_dataset.get((method, dataset))
            for metric in ("dice", "iou"):
                value = float(row[metric]) if row and row[metric] != "" else float("nan")
                shown = format_number(value * 100.0 if not math.isnan(value) else value)
                winner = best.get((dataset, metric))
                if winner is not None and not math.isnan(value) and abs(value - winner) < 1e-9:
                    shown = "\\textbf{" + shown + "}"
                cells.append(shown)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""))
    tex_dir = output_dir / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "metrics_{}.tex".format(ratio)).write_text("\n".join(lines), encoding="utf-8")


def write_combined_metric_tex(output_dir: Path, metric_rows: List[Dict[str, object]]) -> None:
    """Write a reference-style table with one block per dataset."""
    by_key = {
        (row["method"], row["dataset"], row["ratio"]): row
        for row in metric_rows
    }
    methods = [method for method in BASELINE_METHODS + ("WeSAM",) if any(
        (method, dataset, ratio) in by_key
        for dataset in DATASETS
        for ratio in RATIOS
    )]
    best = {}
    for dataset in DATASETS:
        for ratio in RATIOS:
            for metric in ("dice", "iou"):
                values = []
                for method in methods:
                    row = by_key.get((method, dataset, ratio))
                    value = float(row[metric]) if row and row[metric] != "" else float("nan")
                    if not math.isnan(value):
                        values.append(value)
                if values:
                    best[(dataset, ratio, metric)] = max(values)

    lines = [
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{multirow}",
        "% The available aggregate metrics are Dice and IoU; no HD95 values are stored.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Quantitative results on ISIC-2018 and polyp datasets under 1\\% and 10\\% labeled settings.}",
        "\\label{tab:comparison-combined}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "\\multirow{2}{*}{Methods} & \\multirow{2}{*}{Dataset} & \\multicolumn{2}{c}{1\\% labeled} & \\multicolumn{2}{c}{10\\% labeled} \\\\",
        " &  & Dice(\\%) & IoU(\\%) & Dice(\\%) & IoU(\\%) \\\\",
        "\\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for method_index, method in enumerate(methods):
            cells = [tex_escape(method)]
            if method_index == 0:
                cells.insert(0, "\\multirow{{{}}}{{*}}{{\\rotatebox{{90}}{{{}}}}}".format(
                    len(methods), tex_escape(dataset)
                ))
            else:
                cells.insert(0, "")
            for ratio in RATIOS:
                row = by_key.get((method, dataset, ratio))
                for metric in ("dice", "iou"):
                    value = float(row[metric]) if row and row[metric] != "" else float("nan")
                    shown = format_number(value * 100.0 if not math.isnan(value) else value)
                    winner = best.get((dataset, ratio, metric))
                    if winner is not None and not math.isnan(value) and abs(value - winner) < 1e-9:
                        shown = "\\textbf{" + shown + "}"
                    cells.append(shown)
            lines.append(" & ".join(cells) + " \\\\")
        if dataset_index != len(DATASETS) - 1:
            lines.append("\\midrule")
    lines.extend(("\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""))
    tex_dir = output_dir / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "metrics_combined.tex").write_text("\n".join(lines), encoding="utf-8")


def write_figure_tex(output_dir: Path, ratio: str, selections: List[Selection]) -> None:
    ratio_selections = [selection for selection in selections if selection.ratio == ratio]
    lines = [
        "% Requires: \\usepackage{graphicx}, \\usepackage{subfig}",
        "\\begin{figure*}[t]",
        "\\centering",
    ]
    for selection in ratio_selections:
        visualization = output_dataset_dir(output_dir, selection.dataset, ratio) / "visualizations" / (selection.sample_id + ".png")
        relative = Path(os.path.relpath(visualization, output_dir / "tex")).as_posix()
        lines.append(
            "\\subfloat[{}]{{\\includegraphics[width=0.32\\textwidth]{{{}}}}}".format(
                tex_escape(selection.dataset), relative
            )
        )
    lines.extend(
        (
            "\\caption{{Selected qualitative comparisons for the {} labeled setting. Each panel uses one common test image; WeSAM is selected as the Dice leader with the largest margin over the strongest available baseline.}}".format(
                ratio.replace("pct", "\\%")
            ),
            "\\label{fig:comparison-" + ratio + "}",
            "\\end{figure*}",
            "",
        )
    )
    tex_dir = output_dir / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "figures_{}.tex".format(ratio)).write_text("\n".join(lines), encoding="utf-8")


def evaluate_dataset_ratio(
    project_root: Path,
    dataset: str,
    ratio: str,
) -> Tuple[Dict[str, Sample], Dict[str, Dict[str, Path]], List[str]]:
    samples = samples_for_dataset(project_root, dataset)
    masks = {"WeSAM": prediction_map(wesam_mask_dir(project_root, dataset, ratio))}
    for method, mask_dir in baseline_mask_dirs(project_root, dataset, ratio).items():
        masks[method] = prediction_map(mask_dir)
    available = [method for method in BASELINE_METHODS if masks[method]]
    if not masks["WeSAM"]:
        print("SKIP {} {}: WeSAM pred_masks directory is unavailable.".format(dataset, ratio), file=sys.stderr)
    if len(available) < len(BASELINE_METHODS):
        missing = [method for method in BASELINE_METHODS if method not in available]
        print("WARN {} {}: missing baseline masks for {}.".format(dataset, ratio, ", ".join(missing)), file=sys.stderr)
    return samples, masks, available


def aggregate_metrics(
    dataset: str,
    ratio: str,
    samples: Dict[str, Sample],
    masks: Dict[str, Dict[str, Path]],
    workers: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, Scores]]]:
    rows = []
    score_maps = {}
    for method in METHOD_ORDER:
        prediction_paths = masks.get(method, {})
        jobs = []
        for sample_id, sample in samples.items():
            prediction = prediction_paths.get(sample_id)
            if prediction is not None:
                jobs.append((sample_id, prediction, sample.gt_path))
        if workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                scores = list(executor.map(score_job, [(job[1], job[2]) for job in jobs]))
        else:
            scores = [score_masks(job[1], job[2]) for job in jobs]
        if not scores:
            continue
        score_maps[method] = {job[0]: score for job, score in zip(jobs, scores)}
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "ratio": ratio,
                "n": len(scores),
                "dice": float(np.mean([score.dice for score in scores])),
                "iou": float(np.mean([score.iou for score in scores])),
            }
        )
    return rows, score_maps


def select_one(
    dataset: str,
    ratio: str,
    samples: Dict[str, Sample],
    masks: Dict[str, Dict[str, Path]],
    baseline_methods: List[str],
    min_baselines: int,
    allow_wesam_not_best: bool,
    min_baseline_dice: float,
    min_margin: float,
    score_maps: Optional[Dict[str, Dict[str, Scores]]],
    workers: int,
) -> Optional[Tuple[Selection, OrderedDict, Dict[str, Scores], Sample]]:
    def score_candidate(sample_id: str) -> Optional[Tuple[Selection, OrderedDict, Dict[str, Scores], Sample]]:
        sample = samples[sample_id]
        wesam_prediction = masks["WeSAM"].get(sample_id)
        if wesam_prediction is None:
            return None
        method_masks = OrderedDict()
        method_masks["WeSAM"] = wesam_prediction
        for method in baseline_methods:
            prediction = masks[method].get(sample_id)
            if prediction is not None:
                method_masks[method] = prediction
        if len(method_masks) - 1 < min_baselines:
            return None

        # Selection is defined only by Dice.
        scores = {
            method: (
                score_maps.get(method, {}).get(sample_id)
                if score_maps is not None
                else None
            )
            or score_masks(path, sample.gt_path)
            for method, path in method_masks.items()
        }
        wesam = scores["WeSAM"]
        baseline_scores = [(method, scores[method]) for method in method_masks if method != "WeSAM"]
        strongest_method, strongest_score = max(baseline_scores, key=lambda item: item[1].dice)
        weakest_method, weakest_score = min(baseline_scores, key=lambda item: item[1].dice)
        mean_dice = float(np.mean([score.dice for _, score in baseline_scores]))
        is_best = wesam.dice >= strongest_score.dice - 1e-12
        if not is_best and not allow_wesam_not_best:
            return None
        if any(score.dice <= min_baseline_dice for _, score in baseline_scores):
            return None
        if wesam.dice - strongest_score.dice < min_margin:
            return None
        selection = Selection(
            dataset=dataset,
            ratio=ratio,
            sample_id=sample_id,
            wesam_dice=wesam.dice,
            wesam_iou=wesam.iou,
            strongest_baseline=strongest_method,
            strongest_baseline_dice=strongest_score.dice,
            weakest_baseline=weakest_method,
            weakest_baseline_dice=weakest_score.dice,
            mean_baseline_dice=mean_dice,
            selection_margin=wesam.dice - strongest_score.dice,
            baseline_count=len(baseline_scores),
        )
        return selection, method_masks, scores, sample

    sample_ids = list(samples)
    if workers > 1 and len(sample_ids) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            candidates = [item for item in executor.map(score_candidate, sample_ids) if item is not None]
    else:
        candidates = [item for item in (score_candidate(sample_id) for sample_id in sample_ids) if item is not None]

    if not candidates:
        print("SKIP {} {}: no common sample satisfies the selection rule.".format(dataset, ratio), file=sys.stderr)
        return None
    selection, method_masks, scores, sample = max(
        candidates,
        key=lambda item: (
            item[0].selection_margin,
            item[0].wesam_dice,
            -item[0].mean_baseline_dice,
        ),
    )
    return selection, method_masks, scores, sample


def materialize_selection(
    output_dir: Path,
    selection: Selection,
    method_masks: OrderedDict,
    scores: Dict[str, Scores],
    sample: Sample,
) -> None:
    dataset_dir = output_dataset_dir(output_dir, selection.dataset, selection.ratio)
    sample_dir = dataset_dir / "selected_samples" / selection.sample_id
    masks_dir = sample_dir / "pred_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, sample_dir / ("image" + sample.image_path.suffix.lower()))
    shutil.copy2(sample.gt_path, sample_dir / "gt_mask.png")
    for method, path in method_masks.items():
        shutil.copy2(path, masks_dir / (method + ".png"))
    (sample_dir / "selection.json").write_text(json.dumps(asdict(selection), indent=2), encoding="utf-8")
    render_montage(dataset_dir / "visualizations" / (selection.sample_id + ".png"), sample, method_masks, scores)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    selections = []
    for ratio in args.ratios:
        for dataset in args.datasets:
            samples, masks, available_baselines = evaluate_dataset_ratio(
                project_root, dataset, ratio
            )
            metric_rows, score_maps = aggregate_metrics(
                dataset,
                ratio,
                samples,
                masks,
                workers=args.workers,
            )
            all_metrics.extend(metric_rows)
            selected = select_one(
                dataset,
                ratio,
                samples,
                masks,
                available_baselines,
                args.min_baselines,
                args.allow_wesam_not_best,
                args.min_baseline_dice,
                args.min_margin,
                score_maps,
                args.workers,
            )
            if selected is not None:
                selection, method_masks, scores, sample = selected
                materialize_selection(output_dir, selection, method_masks, scores, sample)
                selections.append(selection)
                print(
                    "SELECT {} {} {}: WeSAM Dice={:.4f}, strongest {}={:.4f}, weakest {}={:.4f}".format(
                        dataset,
                        ratio,
                        selection.sample_id,
                        selection.wesam_dice,
                        selection.strongest_baseline,
                        selection.strongest_baseline_dice,
                        selection.weakest_baseline,
                        selection.weakest_baseline_dice,
                    )
                )

    metric_fields = ("method", "dataset", "ratio", "n", "dice", "iou")
    metric_output = output_dir / "metrics.csv"
    selection_output = output_dir / "selected_examples.csv"
    selection_fields = tuple(Selection.__dataclass_fields__.keys())
    metric_csv_rows = all_metrics
    selection_csv_rows = [asdict(item) for item in selections]
    if args.append:
        metric_csv_rows = merge_rows(read_csv(metric_output), metric_csv_rows, ("method", "dataset", "ratio"))
        selection_csv_rows = merge_rows(read_csv(selection_output), selection_csv_rows, ("dataset", "ratio"))
    write_csv(metric_output, metric_fields, metric_csv_rows)
    write_csv(selection_output, selection_fields, selection_csv_rows)
    all_selections = [selection_from_row(row) for row in selection_csv_rows]
    for ratio in args.ratios:
        write_metric_tex(output_dir, ratio, metric_csv_rows)
        write_figure_tex(output_dir, ratio, all_selections)
    write_combined_metric_tex(output_dir, metric_csv_rows)
    print("Wrote {} selections to {}".format(len(selections), output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
