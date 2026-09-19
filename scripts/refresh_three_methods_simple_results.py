#!/usr/bin/env python3
"""Install the corrected ThreeMethods masks and metrics into the standard layout.

The 20260813 archive is the authoritative source for SAMed, H-SAM, and
CPC-SAM.  It retains the original masks under ``ablation/2``; this script
normalizes those masks into the existing ``pred_masks`` directories and
refreshes the comparison metric tables.
"""

from __future__ import annotations

import csv
import gzip
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ablation/2/ThreeMethods_SIMPLE_RESULTS_20260813_230439"
METHOD_DIR = {"SAMed": "SAMed", "H-SAM": "H_SAM", "CPC-SAM": "CPC_SAM"}
RATIOS = {"1percent": "1pct", "10percent": "10pct"}


def png_from_nifti(source: Path, destination: Path) -> None:
    """Convert the archive's binary float32 NIfTI mask into an 8-bit PNG."""
    with gzip.open(source, "rb") as handle:
        payload = handle.read()
    header = payload[:352]
    if struct.unpack("<i", header[:4])[0] != 348:
        raise ValueError("Unsupported NIfTI header: {}".format(source))
    dimensions = struct.unpack("<8h", header[40:56])
    datatype, bitpix = struct.unpack("<2h", header[70:74])
    offset = int(struct.unpack("<f", header[108:112])[0])
    if datatype != 16 or bitpix != 32 or dimensions[0] != 2:
        raise ValueError("Expected a 2-D float32 NIfTI mask: {}".format(source))
    width, height = dimensions[1], dimensions[2]
    values = np.frombuffer(payload, dtype="<f4", offset=offset)
    if values.size != width * height:
        raise ValueError("Unexpected NIfTI data size: {}".format(source))
    # NIfTI stores axes in Fortran order.  The archive masks are binary.
    mask = values.reshape((width, height), order="F").T > 0.5
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(destination)


def archive_predictions(method: str, ratio: str, dataset: str) -> Path:
    root = SOURCE / "masks" / method / ratio
    return root / ("ISIC2018" if dataset == "ISIC2018" else "Polyp/{}".format(dataset)) / "predictions"


def target_predictions(method: str, ratio: str, dataset: str) -> Path:
    root = ROOT / "ablation/2" / METHOD_DIR[method]
    if dataset == "ISIC2018":
        return root / "ISIC" / RATIOS[ratio] / "ISIC" / "pred_masks"
    return root / "Polyp" / RATIOS[ratio] / dataset / "pred_masks"


def replace_masks(summary_rows: list[dict[str, str]]) -> list[str]:
    incomplete = []
    for row in summary_rows:
        method, ratio, dataset = row["method"], row["ratio"], row["dataset"]
        source = archive_predictions(method, ratio, dataset)
        target = target_predictions(method, ratio, dataset)
        files = sorted(source.glob("*.png")) + sorted(source.glob("*.nii.gz"))
        expected = int(row["n"])
        if len(files) != expected:
            message = "SKIP {} {} {}: archive has {} masks; summary requires {}".format(
                method, ratio, dataset, len(files), expected
            )
            print(message)
            incomplete.append(message)
            continue

        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for source_file in files:
            name = source_file.name.replace(".nii.gz", ".png")
            destination = target / name
            if source_file.suffix == ".png":
                shutil.copy2(source_file, destination)
            else:
                png_from_nifti(source_file, destination)
        count = len(list(target.glob("*.png")))
        if count != expected:
            raise RuntimeError("{} contains {} converted masks; expected {}".format(target, count, expected))
        print("{} {} {}: {} masks".format(method, ratio, dataset, count))
    return incomplete


def refresh_metrics(summary_rows: list[dict[str, str]]) -> None:
    metrics_path = ROOT / "comparison_examples_4/metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    if not fields:
        raise RuntimeError("Missing header in {}".format(metrics_path))
    authoritative = {}
    for row in summary_rows:
        dataset = "ISIC" if row["dataset"] == "ISIC2018" else row["dataset"]
        ratio = RATIOS[row["ratio"]]
        authoritative[(row["method"], dataset, ratio)] = row
    changed = 0
    for row in rows:
        source = authoritative.get((row["method"], row["dataset"], row["ratio"]))
        if source:
            row.update(n=source["n"], dice=source["dice"], iou=source["iou"])
            changed += 1
    if changed != len(authoritative):
        raise RuntimeError("Only updated {} of {} corrected metric rows".format(changed, len(authoritative)))
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("Updated {} corrected rows in {}".format(changed, metrics_path))


def main() -> None:
    summary_path = SOURCE / "summary_metrics.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 36:
        raise RuntimeError("Expected 36 rows in {}".format(summary_path))
    incomplete = replace_masks(summary_rows)
    refresh_metrics(summary_rows)

    # Regenerate the three standard comparison tables from the corrected CSV.
    from select_comparison_examples import read_csv, write_combined_metric_tex, write_metric_tex
    metrics = read_csv(ROOT / "comparison_examples_4/metrics.csv")
    for ratio in ("1pct", "10pct"):
        write_metric_tex(ROOT / "comparison_examples_4", ratio, metrics)
    write_combined_metric_tex(ROOT / "comparison_examples_4", metrics)
    if incomplete:
        print("WARNING: corrected CPC-SAM 1% ISIC masks were not installed because the archive is incomplete.")


if __name__ == "__main__":
    main()
