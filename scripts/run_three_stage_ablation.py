"""Run the dependency-aware ISIC/Polyp WeSAM ablation workflow.

Model choices are made from validation mIoU only. Test metrics are emitted by
the underlying runners after checkpoint selection and are never used to choose
the sampling probability or a prompt-robustness strategy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python")
DATASETS = ("isic",)
SEEDS = (1337, 2027, 3407)
PROBABILITIES = (0.1, 0.25, 0.5)


@dataclass(frozen=True)
class Task:
    stage: str
    dataset: str
    method: str
    probability: float
    seed: int


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pids(pids: list[int]) -> None:
    pending = set(pids)
    while pending:
        pending = {pid for pid in pending if alive(pid)}
        if pending:
            print(f"Waiting for existing queues: {sorted(pending)}", flush=True)
            time.sleep(60)


def task_dir(output: Path, task: Task) -> Path:
    return output / task.stage / task.dataset / task.method / f"p{task.probability:g}" / f"seed{task.seed}"


def metric_path(output: Path, task: Task) -> Path:
    dataset_dir = "ISIC" if task.dataset == "isic" else "Polyp"
    return task_dir(output, task) / dataset_dir / "metrics.csv"


def best_validation_iou(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [
        float(row["Mean IoU"])
        for row in rows
        if row["Name"].endswith("_student")
    ]
    if not values:
        raise RuntimeError(f"No student validation rows in {path}")
    return max(values)


def command(task: Task, output: Path, gpu: int) -> list[str]:
    out_dir = task_dir(output, task)
    method = task.method
    anchor = 0.0
    contrast = 0.0
    jitter = 0.0
    dropout = 0.0
    gt_probability = 1.0
    prompt_backend = "no-prompt" if method == "no_prompt" else "ifp"
    if method == "anchor_only":
        anchor = 1.0
    elif method in {"full_wesam", "jitter", "dropout", "gt_ifp_mix"}:
        anchor, contrast = 1.0, 0.1
    if method == "jitter":
        jitter = 16.0
    elif method == "dropout":
        dropout = 0.2
    elif method == "gt_ifp_mix":
        gt_probability = 0.5

    runner = "run_semisup_manifest.py" if task.dataset == "isic" else "run_polyp_manifest.py"
    args = [
        str(PYTHON), runner,
        "--labeled-ratio", "0.01",
        "--gpu", str(gpu),
        "--output-dir", str(out_dir),
        "--epochs", "10",
        "--batch-size", "4",
        "--val-batch-size", "16",
        "--seed", str(task.seed),
        "--labeled-batch-probability", str(task.probability),
        "--prompt-backend", prompt_backend,
        "--labeled-prompt-mode", "gt",
        "--labeled-gt-prompt-probability", str(gt_probability),
        "--teacher-weight", "0.1",
        "--anchor-weight", str(anchor),
        "--contrast-weight", str(contrast),
        "--train-point-jitter-pixels", str(jitter),
        "--train-prompt-dropout", str(dropout),
    ]
    if task.dataset == "isic":
        args[2:2] = ["--dataset", "isic"]
    return args


def run_tasks(tasks: list[Task], output: Path) -> None:
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def worker(gpu: int, assigned: list[Task]) -> None:
        for task in assigned:
            out_dir = task_dir(output, task)
            done = metric_path(output, task)
            if done.is_file() and any("final_test_best_student" in line for line in done.read_text().splitlines()):
                print(f"Skipping completed task: {task}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            marker = out_dir / ".running"
            try:
                marker.mkdir()
            except FileExistsError:
                print(f"Skipping externally running task: {task}", flush=True)
                continue
            log = logs / f"{task.stage}_{task.dataset}_{task.method}_p{task.probability:g}_seed{task.seed}.log"
            print(f"GPU {gpu}: {task}", flush=True)
            try:
                with log.open("w", encoding="utf-8") as handle:
                    environment = os.environ.copy()
                    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                    subprocess.run(
                        command(task, output, gpu), cwd=ROOT, env=environment,
                        stdout=handle, stderr=subprocess.STDOUT, check=True,
                    )
            finally:
                marker.rmdir()

    # Two independent task streams per physical GPU. This caps concurrency at
    # two jobs on GPU0 and two jobs on GPU1 while keeping both cards busy.
    assignments = ([], [], [], [])
    for index, task in enumerate(tasks):
        assignments[index % len(assignments)].append(task)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(worker, worker_index % 2, assigned)
            for worker_index, assigned in enumerate(assignments)
        ]
        for future in futures:
            future.result()


def mean_validation(output: Path, stage: str, dataset: str, method: str, probability: float, seeds=SEEDS) -> float:
    values = [
        best_validation_iou(metric_path(output, Task(stage, dataset, method, probability, seed)))
        for seed in seeds
    ]
    return sum(values) / len(values)


def write_summary(output: Path, summary: dict) -> None:
    (output / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    wait_for_pids(args.wait_for_pid)

    phase1 = [
        Task("phase1", dataset, method, probability, seed)
        for dataset in DATASETS
        for method in ("no_prompt", "original_ifp")
        for probability in PROBABILITIES
        for seed in SEEDS
    ]
    run_tasks(phase1, output)

    selected = {}
    for dataset in DATASETS:
        scores = {
            probability: mean_validation(output, "phase1", dataset, "original_ifp", probability)
            for probability in PROBABILITIES
        }
        selected[dataset] = max(scores, key=scores.get)

    phase2 = [
        Task("phase2", dataset, method, selected[dataset], seed)
        for dataset in DATASETS
        for method in ("anchor_only", "full_wesam")
        for seed in SEEDS
    ]
    run_tasks(phase2, output)

    phase3_candidates = []
    for dataset, probability in selected.items():
        full = mean_validation(output, "phase2", dataset, "full_wesam", probability)
        no_prompt = mean_validation(output, "phase1", dataset, "no_prompt", probability)
        if full < no_prompt:
            phase3_candidates.extend(
                Task("phase3_screen", dataset, method, probability, 1337)
                for method in ("jitter", "dropout", "gt_ifp_mix")
            )
    run_tasks(phase3_candidates, output)

    confirmed = {}
    for dataset, probability in selected.items():
        candidates = [task for task in phase3_candidates if task.dataset == dataset]
        if not candidates:
            continue
        baseline = best_validation_iou(metric_path(output, Task("phase2", dataset, "full_wesam", probability, 1337)))
        winner = max(candidates, key=lambda task: best_validation_iou(metric_path(output, task)))
        if best_validation_iou(metric_path(output, winner)) > baseline:
            confirmed[dataset] = winner.method

    phase3_confirm = [
        Task("phase3_confirm", dataset, method, selected[dataset], seed)
        for dataset, method in confirmed.items()
        for seed in SEEDS
    ]
    run_tasks(phase3_confirm, output)

    summary = {"selected_probability": selected, "phase3_confirmed_method": confirmed}
    write_summary(output, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, action="append", default=[])
    main(parser.parse_args())
