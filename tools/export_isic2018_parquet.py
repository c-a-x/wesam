#!/usr/bin/env python3
"""Export the Hugging Face ISIC2018 parquet shards into image/mask files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import pyarrow.parquet as pq


SPLITS = ("train", "validation", "test")


def atomic_write(path: Path, payload: bytes) -> None:
    """Write a file atomically so an interrupted export can safely resume."""
    temporary_path = path.with_suffix(path.suffix + ".part")
    temporary_path.write_bytes(payload)
    os.replace(temporary_path, path)


def export_split(parquet_dir: Path, output_root: Path, split: str, batch_size: int) -> int:
    shards = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No {split} parquet shards under {parquet_dir}")

    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifests" / f"{split}.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str]] = []
    skipped = 0
    written = 0
    for shard_index, shard in enumerate(shards, start=1):
        parquet_file = pq.ParquetFile(shard)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                image = row["image"]
                label = row["label"]
                image_name = Path(image["path"]).name
                mask_name = Path(label["path"]).name
                image_path = image_dir / image_name
                mask_path = mask_dir / mask_name

                if image_path.is_file() and mask_path.is_file():
                    skipped += 1
                else:
                    atomic_write(image_path, image["bytes"])
                    atomic_write(mask_path, label["bytes"])
                    written += 1

                records.append(
                    (
                        image_path.relative_to(output_root).as_posix(),
                        mask_path.relative_to(output_root).as_posix(),
                    )
                )
        print(
            f"{split}: shard {shard_index}/{len(shards)}, "
            f"written={written}, skipped={skipped}",
            flush=True,
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "image", "mask"))
        writer.writerows((index, image, mask) for index, (image, mask) in enumerate(records))

    print(f"{split}: exported {len(records)} samples to {manifest_path}", flush=True)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("Data/ISIC2018/hf_mirror_parquet/data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/ISIC2018"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    counts = {
        split: export_split(args.parquet_dir, args.output_root, split, args.batch_size)
        for split in SPLITS
    }
    print("Export complete: " + ", ".join(f"{split}={count}" for split, count in counts.items()))


if __name__ == "__main__":
    main()
