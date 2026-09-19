#!/usr/bin/env python3
"""Regenerate the v5 qualitative montages from GT-scored prediction masks.

The panels retain the v4 layout and baseline order.  Every WeSAM panel is
strictly above 0.95 Dice and remains the Dice leader for its common sample.
All selections preserve the v4 rule that the nine baseline Dice scores are
positive, except CVC-300: no saved 1% or 10% WeSAM run has a >0.95 Dice leader
with all nine baseline scores above 0.01.  Its two documented fallbacks retain
all ten method masks and have eight positive baseline scores.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_comparison_examples as comparison  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "comparison_examples_5"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"


@dataclass(frozen=True)
class PlannedSelection:
    dataset: str
    ratio: str
    sample_id: str
    wesam_dir: Path
    positive_baseline_requirement: int
    exception: str = ""


PLANNED_SELECTIONS = (
    PlannedSelection(
        "ISIC",
        "1pct",
        "ISIC_0022457",
        PROJECT_ROOT / "output_current/wesam/ISIC/1pct_repeat_20260725/ISIC/pred_masks",
        9,
    ),
    PlannedSelection(
        "ISIC",
        "10pct",
        "ISIC_0017409",
        PROJECT_ROOT / "output_current/wesam/ISIC/10pct/ISIC/pred_masks",
        9,
    ),
    PlannedSelection(
        "Kvasir",
        "1pct",
        "cju2wve9v7esz0878mxsdcy04",
        PROJECT_ROOT / "output_current/accelerated_1pct_from_10pct_20260829/polyp_seed3407/Kvasir/pred_masks",
        9,
    ),
    PlannedSelection(
        "Kvasir",
        "10pct",
        "cju7ecl9i2i060987xawjp4l0",
        PROJECT_ROOT / "output_current/accelerated_10pct_20260828/polyp_seed3407/Kvasir/pred_masks",
        9,
    ),
    PlannedSelection(
        "CVC-ClinicDB",
        "1pct",
        "545",
        PROJECT_ROOT / "output_current/accelerated_1pct_from_10pct_20260829/polyp_seed3407/CVC-ClinicDB/pred_masks",
        9,
    ),
    PlannedSelection(
        "CVC-ClinicDB",
        "10pct",
        "106",
        PROJECT_ROOT / "output_current/accelerated_10pct_20260828/polyp_seed3407/CVC-ClinicDB/pred_masks",
        9,
    ),
    PlannedSelection(
        "CVC-ColonDB",
        "1pct",
        "121",
        PROJECT_ROOT / "output_current/accelerated_1pct_from_10pct_20260829/polyp_seed3407/CVC-ColonDB/pred_masks",
        9,
    ),
    PlannedSelection(
        "CVC-ColonDB",
        "10pct",
        "124",
        PROJECT_ROOT / "output_current/accelerated_10pct_20260828/polyp_seed3407/CVC-ColonDB/pred_masks",
        9,
    ),
    PlannedSelection(
        "CVC-300",
        "1pct",
        "159",
        PROJECT_ROOT / "output_current/accelerated_1pct_from_10pct_20260829/polyp_seed3407/CVC-300/pred_masks",
        8,
        "URPC Dice is 0.0; no saved 1% WeSAM run meets the original 9/9-positive rule above 0.95 Dice.",
    ),
    PlannedSelection(
        "CVC-300",
        "10pct",
        "166",
        PROJECT_ROOT / "output_current/accelerated_10pct_20260828/polyp_seed3407/CVC-300/pred_masks",
        8,
        "UA-MT Dice is 0.0; no saved 10% WeSAM run meets the original 9/9-positive rule above 0.95 Dice.",
    ),
    PlannedSelection(
        "ETIS-LaribPolypDB",
        "1pct",
        "124",
        PROJECT_ROOT / "output_current/accelerated_1pct_from_10pct_20260829/polyp_seed3407/ETIS-LaribPolypDB/pred_masks",
        9,
    ),
    PlannedSelection(
        "ETIS-LaribPolypDB",
        "10pct",
        "159",
        PROJECT_ROOT / "output_current/accelerated_10pct_20260828/polyp_seed3407/ETIS-LaribPolypDB/pred_masks",
        9,
    ),
)


def output_name(dataset: str, ratio: str) -> str:
    return "{}_{}pt.png".format(dataset, "1" if ratio == "1pct" else "10")


def baseline_masks(dataset: str, ratio: str) -> Dict[str, Dict[str, Path]]:
    return {
        method: comparison.prediction_map(path)
        for method, path in comparison.baseline_mask_dirs(PROJECT_ROOT, dataset, ratio).items()
    }


def copy_selection_assets(
    selection: PlannedSelection,
    sample: comparison.Sample,
    method_masks: OrderedDict,
) -> None:
    dataset_dir = comparison.output_dataset_dir(OUTPUT_DIR, selection.dataset, selection.ratio)
    sample_dir = dataset_dir / "selected_samples" / selection.sample_id
    masks_dir = sample_dir / "pred_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image_path, sample_dir / ("image" + sample.image_path.suffix.lower()))
    shutil.copy2(sample.gt_path, sample_dir / "gt_mask.png")
    for method, mask_path in method_masks.items():
        shutil.copy2(mask_path, masks_dir / (method + ".png"))


def render_selection(selection: PlannedSelection) -> Dict[str, object]:
    samples = comparison.samples_for_dataset(PROJECT_ROOT, selection.dataset)
    sample = samples.get(selection.sample_id)
    if sample is None:
        raise RuntimeError("Unknown test sample: {}".format(selection.sample_id))

    wesam_masks = comparison.prediction_map(selection.wesam_dir)
    baseline_by_method = baseline_masks(selection.dataset, selection.ratio)
    if selection.sample_id not in wesam_masks:
        raise RuntimeError("Missing WeSAM mask for {}".format(selection.sample_id))

    method_masks = OrderedDict((("WeSAM", wesam_masks[selection.sample_id]),))
    for method in comparison.BASELINE_METHODS:
        mask_path = baseline_by_method[method].get(selection.sample_id)
        if mask_path is None:
            raise RuntimeError("Missing {} mask for {}".format(method, selection.sample_id))
        method_masks[method] = mask_path

    scores = {
        method: comparison.score_masks(mask_path, sample.gt_path)
        for method, mask_path in method_masks.items()
    }
    wesam = scores["WeSAM"]
    strongest_method = max(comparison.BASELINE_METHODS, key=lambda method: scores[method].dice)
    strongest_score = scores[strongest_method]
    positive_methods = [
        method for method in comparison.BASELINE_METHODS if scores[method].dice > 0.01
    ]
    zero_methods = [
        method for method in comparison.BASELINE_METHODS if method not in positive_methods
    ]

    if wesam.dice <= 0.95:
        raise RuntimeError("WeSAM Dice is not above 0.95 for {}".format(selection.sample_id))
    if wesam.dice < strongest_score.dice - 1e-12:
        raise RuntimeError("WeSAM is not the Dice leader for {}".format(selection.sample_id))
    if len(positive_methods) < selection.positive_baseline_requirement:
        raise RuntimeError("Too few positive baseline scores for {}".format(selection.sample_id))
    if selection.positive_baseline_requirement == len(comparison.BASELINE_METHODS) and zero_methods:
        raise RuntimeError("A strict selection has a non-positive baseline score")

    comparison.render_montage(
        VISUALIZATION_DIR / output_name(selection.dataset, selection.ratio),
        sample,
        method_masks,
        scores,
    )
    copy_selection_assets(selection, sample, method_masks)

    planned_fields = asdict(selection)
    planned_fields["wesam_dir"] = str(planned_fields["wesam_dir"])
    record = {
        **planned_fields,
        "wesam_source": str(selection.wesam_dir),
        "wesam_dice": wesam.dice,
        "wesam_iou": wesam.iou,
        "strongest_baseline": strongest_method,
        "strongest_baseline_dice": strongest_score.dice,
        "selection_margin": wesam.dice - strongest_score.dice,
        "positive_baseline_count": len(positive_methods),
        "nonpositive_baselines": ";".join(zero_methods),
        "visualization": str(VISUALIZATION_DIR / output_name(selection.dataset, selection.ratio)),
    }
    (comparison.output_dataset_dir(OUTPUT_DIR, selection.dataset, selection.ratio)
     / "selected_samples" / selection.sample_id / "selection.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    records = [render_selection(selection) for selection in PLANNED_SELECTIONS]
    fieldnames = list(records[0])
    with (OUTPUT_DIR / "visualization_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print("Wrote {} qualitative montages to {}".format(len(records), VISUALIZATION_DIR))
    for record in records:
        print(
            "{dataset:22} {ratio:5} {sample_id:28} WeSAM={wesam_dice:.6f} "
            "margin={selection_margin:.6f} positive_baselines={positive_baseline_count}/9".format(**record)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
