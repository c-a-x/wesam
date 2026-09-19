#!/usr/bin/env python3
"""Print current progress of the IfpPrompt rerun experiments."""
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_current/fair_ablation_10pct_seed1337_rerun"
BASELINE = ROOT / "output_current/fair_ablation_10pct_seed1337/group2"

RUNS = {
    "isic_ifp_prompt": ("isic", "isic_wesam"),
    "isic_ifp_prompt_reg": ("isic", "isic_wesam"),
    "polyp_ifp_prompt": ("polyp", "polyp_wesam"),
    "polyp_ifp_prompt_reg": ("polyp", "polyp_wesam"),
}


def progress(log: Path):
    if not log.exists():
        return "not started"
    text = log.read_text(errors="ignore").replace("\r", "\n")
    iters = [l for l in text.split("\n") if l.startswith("Epoch: [")]
    vals = [l for l in text.split("\n") if l.startswith("Validation [")]
    if iters:
        last = iters[-1]
        m = re.match(r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\]", last)
        return f"epoch {m.group(1)} iter {m.group(2)}/{m.group(3)}" if m else iters[-1][:60]
    lines = [l for l in text.split("\n") if "epoch 1/1" in l]
    return "alignment head" if not lines else lines[-1]


def main():
    for name in RUNS:
        d = OUT / name
        log = OUT / "logs" / f"{name}.log"
        print(f"{name:26s} {progress(log)}")
        for suffix in ("ISIC/metrics.csv", "Polyp/metrics.csv"):
            p = d / suffix
            if p.exists():
                rows = list(csv.DictReader(p.open()))
                final = [r for r in rows if "final_test" in r["Name"]]
                if final:
                    r = final[-1]
                    print(f"    {suffix:18s} test IoU={r['Mean IoU']} F1={r['Mean F1']} epoch={r['epoch']}")


if __name__ == "__main__":
    main()
