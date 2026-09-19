"""Wait for the six ISIC screen runs, select by validation IoU, then run full data."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
SCREEN = ROOT / "output_current" / "isic_fast_screen"
FULL = ROOT / "output_current" / "best_full_after_screen_20260820"
METHODS = ("no_prompt", "full_wesam", "jitter", "dropout", "fallback", "gt_ifp_mix")


def metrics_path(method: str) -> Path:
    return SCREEN / method / "ISIC" / "metrics.csv"


def best_val(method: str) -> float | None:
    path = metrics_path(method)
    if not path.is_file():
        return None
    values = []
    completed = False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Name"].endswith("_student"):
                values.append(float(row["Mean IoU"]))
            if "final_test_best_student" in row["Name"]:
                completed = True
    return max(values) if values and completed else None


def args_for(dataset: str, method: str, gpu: int) -> list[str]:
    is_no_prompt = method == "no_prompt"
    gt_probability = "0.5" if method == "gt_ifp_mix" else "1.0"
    fallback = "0.2" if method == "fallback" else None
    jitter = "16" if method == "jitter" else "0"
    dropout = "0.2" if method == "dropout" else "0"
    return [
        PYTHON,
        "run_semisup_manifest.py" if dataset == "isic" else "run_polyp_manifest.py",
        *( ["--dataset", "isic"] if dataset == "isic" else [] ),
        "--labeled-ratio", "0.01", "--gpu", str(gpu),
        "--output-dir", str(FULL / dataset), "--epochs", "10",
        "--batch-size", "4", "--val-batch-size", "16", "--seed", "1337",
        "--labeled-batch-probability", "0.25",
        "--prompt-backend", "no-prompt" if is_no_prompt else "ifp",
        "--labeled-prompt-mode", "gt", "--labeled-gt-prompt-probability", gt_probability,
        "--teacher-weight", "0.1",
        "--anchor-weight", "0" if is_no_prompt else "1.0",
        "--contrast-weight", "0" if is_no_prompt else "0.1",
        "--train-point-jitter-pixels", jitter, "--train-prompt-dropout", dropout,
        *( ["--fallback-confidence-threshold", fallback] if fallback else [] ),
    ]


def main() -> None:
    FULL.mkdir(parents=True, exist_ok=True)
    log_dir = FULL / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        scores = {method: best_val(method) for method in METHODS}
        print(json.dumps({"scores": scores}), flush=True)
        if all(score is not None for score in scores.values()):
            break
        time.sleep(60)

    selected = max(scores, key=lambda method: scores[method])
    selection = {"scores": scores, "selected_method": selected, "selection_metric": "validation_mean_iou"}
    (FULL / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    jobs = []
    for dataset, gpu in (("isic", 0), ("polyp", 1)):
        handle = (log_dir / f"{dataset}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(args_for(dataset, selected, gpu), cwd=ROOT, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((dataset, process, handle))
        with (FULL / "launch.log").open("a", encoding="utf-8") as launch:
            launch.write(f"started {dataset} gpu={gpu} pid={process.pid} method={selected}\n")
    for dataset, process, handle in jobs:
        code = process.wait()
        handle.close()
        with (FULL / "launch.log").open("a", encoding="utf-8") as launch:
            launch.write(f"finished {dataset} return_code={code}\n")


if __name__ == "__main__":
    main()
