"""Report base and pure-IFP metrics from streamed logits only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def metrics(pred: np.ndarray, target: np.ndarray):
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    return (
        tp / max(tp + fp + fn, 1e-7),
        2.0 * tp / max(2.0 * tp + fp + fn, 1e-7),
        tp / max(tp + fp, 1e-7),
        tp / max(tp + fn, 1e-7),
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    files = sorted((a.root / "base_logits").glob("*.npy"))
    sums = {"base": np.zeros(4), "ifp": np.zeros(4)}
    for i, path in enumerate(files, 1):
        base = np.load(path).astype(np.float32).squeeze(0)
        ifp = np.load(a.root / "ifp_logits" / path.name).astype(np.float32).squeeze(0)
        target = np.load(a.root / "targets" / path.name).astype(bool)
        sums["base"] += metrics(base >= 0.0, target)
        sums["ifp"] += metrics(ifp >= 0.0, target)
        if i % 250 == 0:
            print(f"{i}/{len(files)}", flush=True)
    result = {"num_images": len(files)}
    for name, values in sums.items():
        vals = values / len(files)
        result[name] = dict(zip(("IoU", "F1", "precision", "recall"), map(float, vals)))
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
