"""Run the selected full-data WeSAM configuration on ISIC and Polyp."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python"
OUTPUT = ROOT / "output_current" / "best_full_wesam_20260820"


def command(dataset: str, gpu: int) -> list[str]:
    runner = "run_semisup_manifest.py" if dataset == "isic" else "run_polyp_manifest.py"
    args = [
        PYTHON,
        runner,
        "--labeled-ratio", "0.01",
        "--gpu", str(gpu),
        "--output-dir", str(OUTPUT / dataset),
        "--epochs", "10",
        "--batch-size", "4",
        "--val-batch-size", "16",
        "--seed", "1337",
        "--labeled-batch-probability", "0.25",
        "--prompt-backend", "ifp",
        "--labeled-prompt-mode", "gt",
        "--labeled-gt-prompt-probability", "1.0",
        "--teacher-weight", "0.1",
        "--anchor-weight", "1.0",
        "--contrast-weight", "0.1",
        "--train-point-jitter-pixels", "0",
        "--train-prompt-dropout", "0",
    ]
    if dataset == "isic":
        args[2:2] = ["--dataset", "isic"]
    return args


def main() -> None:
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    jobs = [("isic", 0), ("polyp", 1)]
    processes = []
    for dataset, gpu in jobs:
        log_path = logs / f"{dataset}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command(dataset, gpu), cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        processes.append((dataset, process, handle))
        print(f"started {dataset} on GPU {gpu}: pid={process.pid}", flush=True)

    failures = []
    for dataset, process, handle in processes:
        return_code = process.wait()
        handle.close()
        print(f"finished {dataset}: return_code={return_code}", flush=True)
        if return_code:
            failures.append((dataset, return_code))
    if failures:
        raise SystemExit(failures)


if __name__ == "__main__":
    main()
