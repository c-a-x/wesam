"""Run the Polyp prompt ablation on two GPUs.

The five methods are evaluated with seeds 2027 and 3407:

* no_prompt
* wesam (the original IFP/WeSAM configuration)
* mixed_dropout_0
* mixed_dropout_005
* mixed_dropout_01

Each GPU owns one serial worker, so at most one training process can use a
physical GPU at a time.  The underlying ``run_polyp_manifest.py`` runner is
used for data preparation, training, checkpoint selection, and test export.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_polyp_manifest.py"
SEEDS = (2027, 3407)
DEFAULT_PYTHON = Path("/LAI_Data/Anaconda_envs/2024U/UAP-SAM/bin/python")


@dataclass(frozen=True)
class Task:
    name: str
    seed: int
    prompt_backend: str
    labeled_prompt_mode: str
    labeled_gt_prompt_probability: float
    dropout: float
    anchor_weight: float
    contrast_weight: float


def tasks() -> list[Task]:
    result: list[Task] = []
    methods = (
        ("no_prompt", "no-prompt", "gt", 1.0, 0.0, 0.0, 0.0),
        ("wesam", "ifp", "gt", 1.0, 0.0, 1.0, 0.1),
        ("mixed_dropout_0", "ifp", "mixed", 0.5, 0.0, 1.0, 0.1),
        ("mixed_dropout_005", "ifp", "mixed", 0.5, 0.05, 1.0, 0.1),
        ("mixed_dropout_01", "ifp", "mixed", 0.5, 0.10, 1.0, 0.1),
    )
    for seed in SEEDS:
        for name, backend, mode, gt_probability, dropout, anchor, contrast in methods:
            result.append(Task(name, seed, backend, mode, gt_probability, dropout, anchor, contrast))
    return result


def output_dir(root: Path, task: Task) -> Path:
    return root / task.name / f"seed{task.seed}"


def command(task: Task, out_dir: Path, gpu: int, args: argparse.Namespace) -> list[str]:
    python = os.environ.get("WESAM_PYTHON", str(DEFAULT_PYTHON))
    if not Path(python).is_file():
        python = sys.executable
    return [
        python, "-u", str(RUNNER),
        "--labeled-ratio", str(args.labeled_ratio),
        "--gpu", str(gpu), "--output-dir", str(out_dir),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
        "--val-batch-size", str(args.val_batch_size), "--seed", str(task.seed),
        "--labeled-batch-probability", str(args.labeled_batch_probability),
        "--prompt-backend", task.prompt_backend,
        "--labeled-prompt-mode", task.labeled_prompt_mode,
        "--labeled-gt-prompt-probability", str(task.labeled_gt_prompt_probability),
        "--teacher-weight", str(args.teacher_weight),
        "--anchor-weight", str(task.anchor_weight),
        "--contrast-weight", str(task.contrast_weight),
        "--train-prompt-dropout", str(task.dropout),
    ]


def completed(out_dir: Path) -> bool:
    metrics = out_dir / "Polyp" / "metrics.csv"
    if not metrics.is_file():
        return False
    text = metrics.read_text(encoding="utf-8")
    return "final_test_best_student" in text


def run_worker(gpu: int, assigned: list[Task], root: Path, args: argparse.Namespace) -> list[str]:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for task in assigned:
        out_dir = output_dir(root, task)
        if completed(out_dir) and not args.rerun_completed:
            print(f"GPU {gpu}: skip completed {task.name} seed={task.seed}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / ".running"
        try:
            marker.mkdir()
        except FileExistsError:
            print(f"GPU {gpu}: skip already-running {task.name} seed={task.seed}", flush=True)
            continue

        log_path = log_dir / f"{task.name}_seed{task.seed}.log"
        print(f"GPU {gpu}: start {task.name} seed={task.seed}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command(task, out_dir, gpu, args), cwd=ROOT,
                    env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
                )
        except subprocess.CalledProcessError as error:
            failures.append(f"{task.name}/seed{task.seed} (exit {error.returncode})")
            print(f"GPU {gpu}: FAILED {task.name} seed={task.seed}; see {log_path}", flush=True)
        finally:
            marker.rmdir()
        print(f"GPU {gpu}: finished {task.name} seed={task.seed}", flush=True)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "polyp_prompt_ablation")
    parser.add_argument("--labeled-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--labeled-batch-probability", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument("--rerun-completed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.labeled_ratio <= 1.0:
        raise ValueError("--labeled-ratio must be in (0, 1].")
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_tasks = tasks()
    # Round-robin assignment keeps both cards occupied while each card stays serial.
    assignments = ([task for index, task in enumerate(all_tasks) if index % 2 == gpu] for gpu in (0, 1))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_worker, gpu, list(assigned), args.output_root, args) for gpu, assigned in enumerate(assignments)]
        failures = [failure for future in futures for failure in future.result()]
    if failures:
        print("Failed tasks:", ", ".join(failures), file=sys.stderr)
        return 1
    print("All Polyp prompt-ablation tasks completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
