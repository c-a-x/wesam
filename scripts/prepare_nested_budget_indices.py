#!/usr/bin/env python3
"""Create nested 1%/10% labeled index sets with a fixed 10% IFP support set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DATASETS = {"isic": 2594, "polyp": 1305}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1337, 2027, 3407))
    args = parser.parse_args()

    for dataset, total in DATASETS.items():
        count_1pct = round(total * 0.01)
        count_10pct = round(total * 0.10)
        for seed in args.seeds:
            # The first 1% members of this deterministic 10% draw form the
            # nested low-budget set. The entire 10% draw stays fixed as IFP support.
            sampled_10pct = random.Random(seed).sample(range(total), count_10pct)
            root = args.output_dir / dataset / f"seed{seed}"
            root.mkdir(parents=True, exist_ok=True)
            (root / "labeled_1pct.json").write_text(
                json.dumps(sorted(sampled_10pct[:count_1pct]), indent=2) + "\n",
                encoding="utf-8",
            )
            (root / "labeled_10pct.json").write_text(
                json.dumps(sorted(sampled_10pct), indent=2) + "\n",
                encoding="utf-8",
            )
            (root / "support_10pct.json").write_text(
                json.dumps(sorted(sampled_10pct), indent=2) + "\n",
                encoding="utf-8",
            )
            metadata = {
                "dataset": dataset,
                "seed": seed,
                "train_count": total,
                "labeled_1pct_count": count_1pct,
                "labeled_10pct_count": count_10pct,
                "nested": True,
                "ifp_support": "support_10pct.json shared by 1pct and 10pct",
            }
            (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
