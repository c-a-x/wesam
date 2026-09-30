"""Fast CPU sweep over saved no-prompt/IFP logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def remove_pad(array: np.ndarray, padding) -> np.ndarray:
    left, top, right, bottom = [int(v) for v in padding]
    return array[
        top: array.shape[0] - bottom if bottom else array.shape[0],
        left: array.shape[1] - right if right else array.shape[1],
    ]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-grid", action="store_true")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=False)
    names = data["name"]
    paddings = data["padding"]
    base_logits = data["base_logits"].astype(np.float32)
    ifp_logits = data["ifp_logits"].astype(np.float32)
    targets = data["target"].astype(bool)

    records = []
    for index in range(len(names)):
        padding = paddings[index]
        base = remove_pad(base_logits[index], padding)
        ifp = remove_pad(ifp_logits[index], padding)
        target = remove_pad(targets[index], padding)
        base_bin = base >= 0.0
        ifp_bin = ifp >= 0.0
        base_prob = sigmoid(base)
        ifp_prob = sigmoid(ifp)
        # Confidence summaries that do not use GT.
        base_conf = float(np.mean(np.abs(base_prob - 0.5)) * 2.0)
        ifp_conf = float(np.mean(np.abs(ifp_prob - 0.5)) * 2.0)
        records.append({
            "name": str(names[index]),
            "base": base_bin,
            "ifp": ifp_bin,
            "base_prob": base_prob,
            "ifp_prob": ifp_prob,
            "base_logit": base,
            "ifp_logit": ifp,
            "target": target,
            "base_area": float(base_bin.mean()),
            "ifp_area": float(ifp_bin.mean()),
            "base_conf": base_conf,
            "ifp_conf": ifp_conf,
        })

    candidates = []

    def add(strategy, alpha=0.0, base_conf_gate=0.0, ifp_conf_gate=0.0,
            area_low=0.0, area_high=1.0):
        ious, f1s, precisions, recalls = [], [], [], []
        chosen = 0
        for record in records:
            base = record["base"]
            ifp = record["ifp"]
            if strategy == "base":
                pred = base
            elif strategy == "ifp":
                pred = ifp
            elif strategy == "prob":
                pred = (1.0 - alpha) * record["base_prob"] + alpha * record["ifp_prob"] >= 0.5
            elif strategy == "logit":
                pred = (1.0 - alpha) * record["base_logit"] + alpha * record["ifp_logit"] >= 0.0
            elif strategy == "union_gate":
                gate = (
                    record["base_conf"] < base_conf_gate
                    or record["base_area"] < area_low
                    or record["base_area"] > area_high
                    or (record["ifp_conf"] >= ifp_conf_gate and record["ifp_area"] > record["base_area"])
                )
                pred = np.logical_or(base, ifp) if gate else base
                chosen += int(gate)
            elif strategy == "intersect_gate":
                gate = (
                    record["base_conf"] < base_conf_gate
                    or record["base_area"] < area_low
                    or record["base_area"] > area_high
                    or (record["ifp_conf"] >= ifp_conf_gate and record["ifp_area"] < record["base_area"])
                )
                pred = np.logical_and(base, ifp) if gate else base
                chosen += int(gate)
            else:
                raise ValueError(strategy)

            target = record["target"]
            tp = float(np.logical_and(pred, target).sum())
            fp = float(np.logical_and(pred, ~target).sum())
            fn = float(np.logical_and(~pred, target).sum())
            iou = tp / max(tp + fp + fn, 1e-7)
            f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-7)
            precision = tp / max(tp + fp, 1e-7)
            recall = tp / max(tp + fn, 1e-7)
            ious.append(iou)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        candidates.append({
            "strategy": strategy,
            "alpha": float(alpha),
            "base_conf_gate": float(base_conf_gate),
            "ifp_conf_gate": float(ifp_conf_gate),
            "area_low": float(area_low),
            "area_high": float(area_high),
            "IoU": float(np.mean(ious)),
            "F1": float(np.mean(f1s)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "gate_fraction": chosen / len(records),
        })

    # Mandatory baselines and simple weighted ensembles.
    add("base")
    add("ifp")
    for alpha in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        add("prob", alpha=alpha)
        add("logit", alpha=alpha)

    # Conservative gates. Keep the grid small enough to run interactively.
    if args.small_grid:
        area_pairs = ((0.02, 0.70), (0.03, 0.75), (0.05, 0.80))
        base_conf_values = (0.00, 0.20, 0.40, 0.60)
        ifp_conf_values = (0.30, 0.50, 0.70)
    else:
        area_pairs = ((0.01, 0.70), (0.02, 0.75), (0.03, 0.80),
                      (0.05, 0.85), (0.03, 0.70))
        base_conf_values = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
        ifp_conf_values = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
    for area_low, area_high in area_pairs:
        for base_conf_gate in base_conf_values:
            for ifp_conf_gate in ifp_conf_values:
                add("union_gate", base_conf_gate=base_conf_gate,
                    ifp_conf_gate=ifp_conf_gate,
                    area_low=area_low, area_high=area_high)
                add("intersect_gate", base_conf_gate=base_conf_gate,
                    ifp_conf_gate=ifp_conf_gate,
                    area_low=area_low, area_high=area_high)

    candidates.sort(key=lambda item: (item["F1"], item["IoU"]), reverse=True)
    summary = {
        "npz": str(args.npz.resolve()),
        "num_images": len(records),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    base = next(item for item in candidates if item["strategy"] == "base")
    ifp = next(item for item in candidates if item["strategy"] == "ifp")
    print(json.dumps({
        "base": base,
        "ifp": ifp,
        "top10": candidates[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
