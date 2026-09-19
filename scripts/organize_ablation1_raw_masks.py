#!/usr/bin/env python3
"""Import ablation/1 raw masks into the standard prediction-mask layout.

The supplied UA-MT, DTC, and URPC masks are named by their zero-based test
index.  This script maps each index to the corresponding sorted test image and
stores the raw binary mask under that image's stem.  It never reads or derives
predictions from the existing visualization folders.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("UA-MT", "DTC", "URPC")
RATIOS = ("1pct", "10pct")
POLYP_DATASETS = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace differing imported masks.")
    return parser.parse_args()


def samples(project_root: Path, dataset: str) -> List[Path]:
    if dataset == "ISIC":
        image_dir = project_root / "Data" / "ISIC2018" / "test" / "images"
    else:
        image_dir = project_root / "Data" / "kvasir-seg" / "organized" / dataset / "test" / "images"
    result = sorted(path for path in image_dir.glob("*") if path.is_file())
    if not result:
        raise FileNotFoundError("No test images found for {} under {}".format(dataset, image_dir))
    return result


def indexed_masks(mask_dir: Path) -> Iterable[Tuple[int, Path]]:
    for path in sorted(mask_dir.glob("*.png")):
        try:
            index = int(path.stem)
        except ValueError as error:
            raise ValueError("Expected a numeric mask filename, found {}".format(path)) from error
        yield index, path


def copy_mask(source: Path, target: Path, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not filecmp.cmp(source, target, shallow=False):
        if not overwrite:
            raise FileExistsError(
                "{} already exists and differs from {}; rerun with --overwrite to replace it".format(target, source)
            )
    if not target.exists() or overwrite:
        shutil.copy2(source, target)


def import_setting(
    project_root: Path,
    source_root: Path,
    dataset: str,
    ratio: str,
    method: str,
    overwrite: bool,
) -> List[dict]:
    image_paths = samples(project_root, dataset)
    if dataset == "ISIC":
        source_dir = source_root / "masks_isic" / ratio / method
        target_dir = project_root / "ablation" / "1" / method / "ISIC" / ratio / "ISIC" / "pred_masks"
    else:
        source_dir = source_root / "masks_polyp" / ratio / dataset / method
        target_dir = project_root / "ablation" / "1" / method / "Polyp" / ratio / dataset / "pred_masks"
    if not source_dir.is_dir():
        raise FileNotFoundError("Raw mask directory is missing: {}".format(source_dir))

    rows = []
    for index, source in indexed_masks(source_dir):
        if index >= len(image_paths):
            raise IndexError("{} has index {}, but {} has only {} test images".format(source, index, dataset, len(image_paths)))
        sample_id = image_paths[index].stem
        target = target_dir / (sample_id + ".png")
        copy_mask(source, target, overwrite)
        rows.append(
            {
                "dataset": dataset,
                "ratio": ratio,
                "method": method,
                "source_index": index,
                "sample_id": sample_id,
                "source": source.relative_to(project_root).as_posix(),
                "target": target.relative_to(project_root).as_posix(),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_root = project_root / "ablation" / "1"
    rows = []
    for ratio in RATIOS:
        for method in METHODS:
            rows.extend(import_setting(project_root, source_root, "ISIC", ratio, method, args.overwrite))
        for dataset in POLYP_DATASETS:
            for method in METHODS:
                rows.extend(import_setting(project_root, source_root, dataset, ratio, method, args.overwrite))

    audit_path = source_root / "raw_mask_import_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "ratio", "method", "source_index", "sample_id", "source", "target"))
        writer.writeheader()
        writer.writerows(rows)
    print("Imported {} raw masks; wrote {}".format(len(rows), audit_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
