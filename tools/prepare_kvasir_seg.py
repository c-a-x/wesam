#!/usr/bin/env python3
"""Extract and organize the bundled polyp datasets into stable splits.

The training archive combines Kvasir (non-numeric filenames) and
CVC-ClinicDB (numeric filenames). The test archive already separates five
source datasets. Official test samples are never moved into training.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import zipfile
from pathlib import Path


TEST_CATEGORIES = (
    "Kvasir",
    "CVC-ClinicDB",
    "CVC-ColonDB",
    "CVC-300",
    "ETIS-LaribPolypDB",
)


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract an archive once, rejecting unsafe member paths."""
    marker = destination / f".{archive.stem}.complete"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    marker.touch()


def collect_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    images = {path.name: path for path in image_dir.iterdir() if path.is_file()}
    masks = {path.name: path for path in mask_dir.iterdir() if path.is_file()}
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files under {image_dir.parent}: "
            f"images_without_masks={missing_masks[:5]}, masks_without_images={missing_images[:5]}"
        )
    return [(images[name], masks[name]) for name in sorted(images)]


def link_file(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.samefile(source):
            return
        raise FileExistsError(f"Refusing to replace existing file: {destination}")
    os.link(source, destination)


def write_split(output_category: Path, split: str, pairs: list[tuple[Path, Path]]) -> None:
    image_dir = output_category / split / "images"
    mask_dir = output_category / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_category / f"{split}.csv"

    rows: list[tuple[int, str, str]] = []
    for index, (image, mask) in enumerate(pairs):
        destination_image = image_dir / image.name
        destination_mask = mask_dir / mask.name
        link_file(image, destination_image)
        link_file(mask, destination_mask)
        rows.append(
            (
                index,
                destination_image.relative_to(output_category).as_posix(),
                destination_mask.relative_to(output_category).as_posix(),
            )
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "image", "mask"))
        writer.writerows(rows)


def split_training_pairs(pairs: list[tuple[Path, Path]], validation_ratio: float, seed: int) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not pairs:
        return [], []
    validation_count = max(1, round(len(pairs) * validation_ratio))
    validation_indices = set(random.Random(seed).sample(range(len(pairs)), validation_count))
    train = [pair for index, pair in enumerate(pairs) if index not in validation_indices]
    validation = [pair for index, pair in enumerate(pairs) if index in validation_indices]
    return train, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("Data/kvasir-seg"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")

    root = args.root.resolve()
    raw_root = root / "raw"
    extract_zip(root / "TrainDataset.zip", raw_root)
    extract_zip(root / "TestDataset.zip", raw_root)

    train_pairs = collect_pairs(raw_root / "TrainDataset" / "image", raw_root / "TrainDataset" / "masks")
    training_categories = {"Kvasir": [], "CVC-ClinicDB": []}
    for pair in train_pairs:
        category = "CVC-ClinicDB" if pair[0].stem.isdecimal() else "Kvasir"
        training_categories[category].append(pair)

    test_pairs = {}
    for category in TEST_CATEGORIES:
        category_root = raw_root / "TestDataset" / category
        test_pairs[category] = collect_pairs(category_root / "images", category_root / "masks")

    organized_root = root / "organized"
    if organized_root.exists() and any(organized_root.iterdir()):
        raise FileExistsError(f"Output already exists: {organized_root}")
    organized_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[str, int, int, int, str]] = []
    for category in TEST_CATEGORIES:
        category_root = organized_root / category
        train, validation = split_training_pairs(
            training_categories.get(category, []), args.validation_ratio, args.seed
        )
        test = test_pairs[category]
        for split, pairs in (("train", train), ("validation", validation), ("test", test)):
            write_split(category_root, split, pairs)
        source = "official train + official test" if category in training_categories else "official test only"
        summary_rows.append((category, len(train), len(validation), len(test), source))
        print(f"{category}: train={len(train)}, validation={len(validation)}, test={len(test)}", flush=True)

    with (organized_root / "split_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("category", "train", "validation", "test", "source"))
        writer.writerows(summary_rows)

    with (organized_root / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# Polyp Dataset Splits\n\n"
            "Kvasir and CVC-ClinicDB keep their official test sets; validation is a "
            f"seed-{args.seed} split of their supplied training data. CVC-300, "
            "CVC-ColonDB, and ETIS-LaribPolypDB only ship official test data, so "
            "their train and validation directories are intentionally empty.\n"
        )


if __name__ == "__main__":
    main()
