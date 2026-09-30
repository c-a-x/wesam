"""Summarize streamed no-prompt / IFP logits efficiently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def metrics(pred, target):
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    iou = tp / max(tp + fp + fn, 1e-7)
    f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-7)
    return iou, f1, tp / max(tp + fp, 1e-7), tp / max(tp + fn, 1e-7)


def add(candidates, strategy, arrays, alpha=0.0, base_conf_gate=0.0,
        ifp_conf_gate=0.0, area_low=0.0, area_high=1.0):
    sums = np.zeros(4, dtype=np.float64)
    gate_count = 0
    for base, ifp, target, base_prob, ifp_prob, base_conf, ifp_conf, base_area, ifp_area in arrays:
        if strategy == "base":
            pred = base >= 0.0
        elif strategy == "ifp":
            pred = ifp >= 0.0
        elif strategy == "prob":
            pred = (1.0 - alpha) * base_prob + alpha * ifp_prob >= 0.5
        elif strategy == "logit":
            pred = (1.0 - alpha) * base + alpha * ifp >= 0.0
        elif strategy == "union_gate":
            gate = (
                base_conf < base_conf_gate
                or base_area < area_low or base_area > area_high
                or (ifp_conf >= ifp_conf_gate and ifp_area > base_area)
            )
            pred = np.logical_or(base >= 0.0, ifp >= 0.0) if gate else (base >= 0.0)
            gate_count += int(gate)
        elif strategy == "intersect_gate":
            gate = (
                base_conf < base_conf_gate
                or base_area < area_low or base_area > area_high
                or (ifp_conf >= ifp_conf_gate and ifp_area < base_area)
            )
            pred = np.logical_and(base >= 0.0, ifp >= 0.0) if gate else (base >= 0.0)
            gate_count += int(gate)
        else:
            raise ValueError(strategy)
        sums += metrics(pred, target)
    count = len(arrays)
    iou, f1, precision, recall = sums / count
    candidates.append({
        "strategy": strategy, "alpha": alpha,
        "base_conf_gate": base_conf_gate, "ifp_conf_gate": ifp_conf_gate,
        "area_low": area_low, "area_high": area_high,
        "IoU": float(iou), "F1": float(f1),
        "precision": float(precision), "recall": float(recall),
        "gate_fraction": gate_count / count,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small-grid", action="store_true")
    args = parser.parse_args()
    files = sorted((args.root / "base_logits").glob("*.npy"))
    if not files:
        raise SystemExit("no logits found")

    arrays = []
    for path in files:
        base = np.load(path).astype(np.float32).squeeze(0)
        ifp = np.load(args.root / "ifp_logits" / path.name).astype(np.float32).squeeze(0)
        target = np.load(args.root / "targets" / path.name).astype(bool)
        base_prob = sigmoid(base)
        ifp_prob = sigmoid(ifp)
        base_conf = float(np.mean(np.abs(base_prob - 0.5)) * 2.0)
        ifp_conf = float(np.mean(np.abs(ifp_prob - 0.5)) * 2.0)
        base_area = float((base >= 0.0).mean())
        ifp_area = float((ifp >= 0.0).mean())
        arrays.append((base, ifp, target, base_prob, ifp_prob,
                       base_conf, ifp_conf, base_area, ifp_area))

    candidates = []
    add(candidates, "base", arrays)
    add(candidates, "ifp", arrays)
    for alpha in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        add(candidates, "prob", arrays, alpha=alpha)
        add(candidates, "logit", arrays, alpha=alpha)
    if args.small_grid:
        area_pairs = ((0.02, 0.70), (0.03, 0.75), (0.05, 0.80))
        base_confs = (0.0, 0.2, 0.4, 0.6)
        ifp_confs = (0.3, 0.5, 0.7)
    else:
        area_pairs = ((0.01, 0.70), (0.02, 0.75), (0.03, 0.80), (0.05, 0.85), (0.03, 0.70))
        base_confs = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        ifp_confs = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    for area_low, area_high in area_pairs:
        for base_conf in base_confs:
            for ifp_conf in ifp_confs:
                add(candidates, "union_gate", arrays,
                    base_conf_gate=base_conf, ifp_conf_gate=ifp_conf,
                    area_low=area_low, area_high=area_high)
                add(candidates, "intersect_gate", arrays,
                    base_conf_gate=base_conf, ifp_conf_gate=ifp_conf,
                    area_low=area_low, area_high=area_high)
    candidates.sort(key=lambda x: (x["F1"], x["IoU"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "num_images": len(files), "candidates": candidates,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base": next(x for x in candidates if x["strategy"] == "base"),
        "ifp": next(x for x in candidates if x["strategy"] == "ifp"),
        "top5": candidates[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
