"""Run ISIC/Kvasir semi-supervised prompt ablations from local manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from adaptation import main
from configs.config import cfg


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
SPECS = {
    "isic": {
        "config_name": "ISIC", "root": PROJECT_ROOT / "Data/ISIC2018",
        "train": "manifests/train.csv", "validation": "manifests/validation.csv",
        "test": "manifests/test.csv", "prompts": [
            "a dermoscopic photo of a skin lesion",
            "a close-up photo of a mole on human skin",
            "a medical photo of melanoma on the skin",
        ],
    },
    "kvasir": {
        "config_name": "Kvasir", "root": PROJECT_ROOT / "Data/kvasir-seg/organized/Kvasir",
        "train": "train.csv", "validation": "validation.csv", "test": "test.csv",
        "prompts": [
            "a colonoscopy image of a polyp",
            "an endoscopic view showing a colorectal polyp",
            "a gastrointestinal endoscopy image with a visible polyp",
        ],
    },
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "image", "mask"))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"index": index, "image": row["image"], "mask": row["mask"]})


def prepare(
    output_dir: Path,
    spec: dict,
    ratio: float,
    seed: int,
    labeled_count: int | None = None,
    train_subset_ratio: float = 1.0,
    labeled_indices: list[int] | None = None,
    support_indices: list[int] | None = None,
) -> tuple[Path, Path, Path, Path]:
    root = spec["root"]
    train = read_manifest(root / spec["train"])
    validation = read_manifest(root / spec["validation"])
    test = read_manifest(root / spec["test"])
    if not 0.0 < train_subset_ratio <= 1.0:
        raise ValueError("train_subset_ratio must be in (0, 1].")
    if train_subset_ratio < 1.0:
        subset_count = max(1, round(len(train) * train_subset_ratio))
        subset_indices = sorted(random.Random(seed + 1000003).sample(range(len(train)), subset_count))
        train = [train[index] for index in subset_indices]
    count = max(1, round(len(train) * ratio)) if labeled_count is None else labeled_count
    if not 1 <= count <= len(train):
        raise ValueError(f"labeled_count must be within [1, {len(train)}], got {count}.")
    if labeled_indices is None:
        indices = sorted(random.Random(seed).sample(range(len(train)), count))
    else:
        indices = sorted({int(index) for index in labeled_indices})
        if len(indices) != count or any(index < 0 or index >= len(train) for index in indices):
            raise ValueError("Explicit labeled indices must be unique, in range, and match labeled_count.")
    if support_indices is None:
        support_indices = indices
    else:
        support_indices = sorted({int(index) for index in support_indices})
        if not support_indices or any(index < 0 or index >= len(train) for index in support_indices):
            raise ValueError("Explicit support indices must be non-empty and in range.")

    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_manifest(lists / f"{name}.csv", rows)

    support = output_dir / "support"
    images = support / "reference_images"
    masks = support / "reference_masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in support_indices:
        row = train[index]
        image = root / row["image"]
        mask = root / row["mask"]
        image_link = images / image.name
        # train_medical pairs a reference mask with an image by stem. ISIC
        # images are .jpg while masks end in _segmentation.png, so name each
        # support mask after the corresponding reference image.
        mask_link = masks / f"{image.stem}{mask.suffix}"
        for link, source in ((image_link, image), (mask_link, mask)):
            if link.is_symlink():
                if link.resolve() != source.resolve():
                    link.unlink()
                else:
                    continue
            elif link.exists():
                raise FileExistsError(f"Support path already exists and is not a symlink: {link}")
            link.symlink_to(source)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "train_count": len(train),
                "validation_count": len(validation),
                "test_count": len(test),
                "labeled_count": count,
                "labeled_ratio": ratio,
                "labeled_indices": indices,
                "support_indices": support_indices,
                "train_subset_ratio": train_subset_ratio,
                "seed": seed,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return lists / "train.csv", lists / "validation.csv", lists / "test.csv", support


def dump_effective_config(output_dir: Path, args: argparse.Namespace) -> None:
    """Persist the resolved CLI arguments used by this run for auditability."""
    data = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output_dir / "effective_config.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train_head(args, support: Path, prompts: list[str]) -> Path:
    if args.prompt_backend in {"clip-only", "no-prompt"}:
        return Path()
    out = args.output_dir / "ifp_alignment"
    if args.prompt_backend == "dino-prototype":
        checkpoint = out / "dino_foreground_background_prototype.pt"
        command = [
            sys.executable, "-m", "ifp_alignment.train_dino_prototype",
            "--reference-root", str(support), "--output-path", str(checkpoint),
            "--dino-variant", "dino3h", "--ifp-root", str(IFP_ROOT), "--device", "cuda",
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        return checkpoint

    checkpoint = out / "ifp_medical_foreground_best.pt"
    if getattr(args, "reuse_alignment", False) and checkpoint.is_file():
        print(f"[reuse-alignment] using existing head: {checkpoint}", flush=True)
        return checkpoint
    command = [sys.executable, "-m", "ifp_alignment.train_medical", "--reference-root", str(support),
               "--output-dir", str(out), "--prompts", *prompts, "--target", "foreground",
               "--dino-variant", "dino3h", "--clip-variant", "clipl", "--ifp-root", str(IFP_ROOT),
               "--device", "cuda", "--epochs", str(args.alignment_epochs), "--batch-size", "1",
               "--seed", str(args.alignment_seed)]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return checkpoint


def configure(args, spec: dict, train: Path, validation: Path, test: Path, head: Path) -> None:
    if not 0.0 <= args.labeled_batch_probability <= 1.0:
        raise ValueError("--labeled-batch-probability must be in [0, 1].")
    name = spec["config_name"]
    # The launcher masks each process to its requested physical GPU. Fabric
    # therefore sees exactly one local device, always numbered zero.
    cfg.gpu_ids = "0"
    cfg.dataset = name
    cfg.datasets[name].root_dir = str(spec["root"])
    cfg.datasets[name].train_list = str(train)
    cfg.datasets[name].test_list = str(validation)
    cfg.datasets[name].final_test_list = str(test)
    cfg.out_dir = str(args.output_dir)
    label_tag = f"{args.labeled_count}gt" if args.labeled_count is not None else f"{args.labeled_ratio:g}gt"
    cfg.name = f"{args.dataset}_{label_tag}_seed{args.seed}"
    cfg.batch_size, cfg.val_batchsize, cfg.num_workers, cfg.num_epochs = args.batch_size, args.val_batch_size, 0, args.epochs
    cfg.load_type = "soft"
    cfg.prompt = "none" if args.prompt_backend in {"no-prompt", "hybrid"} else "point"
    cfg.resume = False
    fully_supervised = args.labeled_ratio == 1.0
    cfg.semi.enabled, cfg.semi.labeled_ratio, cfg.semi.seed = True, args.labeled_ratio, args.seed
    cfg.semi.labeled_count = args.labeled_count
    cfg.semi.labeled_indices = args.labeled_indices
    cfg.semi.labeled_batch_probability = args.labeled_batch_probability
    cfg.teacher_weight = 0.0 if fully_supervised else args.teacher_weight
    cfg.anchor_weight, cfg.contrast_weight, cfg.supervised_weight = (
        args.anchor_weight, args.contrast_weight, args.supervised_weight
    )
    cfg.unsupervised_rampup_epochs = args.unsupervised_rampup_epochs
    cfg.opt.learning_rate = args.learning_rate
    cfg.opt.warmup_steps = args.warmup_steps
    prompt = cfg.prompt_generator
    prompt_backend = "ifp" if args.prompt_backend == "hybrid" else args.prompt_backend
    prompt.backend, prompt.ifp_root, prompt.alignment_checkpoint = prompt_backend, str(IFP_ROOT), str(head)
    prompt.dino_variant, prompt.clip_variant, prompt.text_prompts = "dino3h", "clipl", spec["prompts"]
    prompt.max_positive_points, prompt.min_positive_points, prompt.include_box = 1, 1, False
    prompt.labeled_prompt_mode = args.labeled_prompt_mode
    prompt.labeled_gt_prompt_probability = args.labeled_gt_prompt_probability
    prompt.fallback_confidence_threshold = args.fallback_confidence_threshold
    prompt.train_point_jitter_pixels = args.train_point_jitter_pixels
    prompt.train_prompt_dropout = args.train_prompt_dropout
    # Hybrid keeps the main no-prompt semi-supervised path. Running the
    # iterative IFP teacher here would leak prompt errors into the pseudo-label
    # target and defeat the purpose of the auxiliary-only branch.
    prompt.iterative_pseudo.enabled = (
        not fully_supervised and args.prompt_backend != "hybrid"
    )
    prompt.iterative_pseudo.max_iters = 1
    prompt.iterative_pseudo.teacher_iou_threshold = 0.8
    prompt.iterative_pseudo.min_new_pixels = 32
    prompt.iterative_pseudo.min_new_area_ratio = 0.001
    prompt.iterative_pseudo.max_mask_area_ratio = 0.6
    prompt.iterative_pseudo.min_overlap_ratio = 0.7
    prompt.iterative_pseudo.max_new_area_ratio = 0.4
    prompt.iterative_pseudo.min_student_teacher_iou = args.min_student_teacher_iou
    cfg.ifp_consistency_weight = (
        args.ifp_consistency_weight if args.prompt_backend == "hybrid" else 0.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument(
        "--labeled-ratio", type=float, choices=(0.01, 0.10, 1.0),
        help="Fraction of train images exposed to GT; 1.0 disables pseudo supervision.",
    )
    labels.add_argument(
        "--labeled-count", type=int,
        help="Exact number of train images exposed to GT.",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--alignment-epochs", type=int, default=1,
        help="IFP alignment-head epochs. The fair-protocol default is 1.",
    )
    parser.add_argument(
        "--iterative-max-iters", type=int, default=1,
        help="Deprecated compatibility option; fair protocol uses one round.",
    )
    parser.add_argument(
        "--iterative-sim-threshold", type=float, default=None,
        help="Deprecated compatibility option; fair protocol has no similarity stop.",
    )
    parser.add_argument(
        "--alignment-seed", type=int, default=1337,
        help="Seed for the IFP alignment head; fair protocol uses 1337.",
    )
    parser.add_argument(
        "--reuse-alignment", action="store_true",
        help="Reuse an existing IFP alignment checkpoint instead of retraining the head.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--labeled-batch-probability", type=float, default=0.5,
        help="Expected fraction of sampled training examples with GT labels.",
    )
    parser.add_argument(
        "--prompt-backend",
        choices=("ifp", "dino-prototype", "clip-only", "no-prompt", "hybrid"),
        default="ifp",
    )
    parser.add_argument(
        "--labeled-prompt-mode", choices=("gt", "ifp", "mixed"), default="gt",
        help="Prompt source for labeled training samples; validation and test use the selected backend.",
    )
    parser.add_argument("--labeled-gt-prompt-probability", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument(
        "--ifp-consistency-weight", type=float, default=0.1,
        help=(
            "Weight for the hybrid IFP auxiliary consistency loss. Only used "
            "with --prompt-backend hybrid (default: 0.1)."
        ),
    )
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument(
        "--supervised-weight", type=float, default=1.0,
        help="Weight for GT mask focal+dice loss on labeled samples.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--unsupervised-rampup-epochs", type=int, default=3)
    parser.add_argument("--fallback-confidence-threshold", type=float, default=None)
    parser.add_argument("--train-point-jitter-pixels", type=float, default=0.0)
    parser.add_argument("--train-prompt-dropout", type=float, default=0.0)
    parser.add_argument(
        "--min-student-teacher-iou", type=float, default=0.0,
        help=(
            "Only apply pseudo supervision on unlabeled samples whose student "
            "mask agrees with the teacher pseudo mask at this IoU. "
            "0 disables the gate (default)."
        ),
    )
    parser.add_argument("--labeled-indices-file", type=Path,
                        help="JSON list of explicit training-row indices exposed to GT.")
    parser.add_argument("--support-indices-file", type=Path,
                        help="JSON list of explicit training-row indices for IFP alignment support.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--train-subset-ratio", type=float, default=1.0,
        help="Fraction of the training manifest to use; validation and test stay complete.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spec = SPECS[args.dataset]
    # Set this before starting the projection-head subprocess so it and the
    # subsequent SAM2 process stay on the same requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.labeled_count is not None:
        total_train = len(read_manifest(spec["root"] / spec["train"]))
        if not 1 <= args.labeled_count <= total_train:
            raise ValueError(f"--labeled-count must be within [1, {total_train}].")
        args.labeled_ratio = args.labeled_count / total_train
    args.labeled_indices = (
        json.loads(args.labeled_indices_file.read_text(encoding="utf-8"))
        if args.labeled_indices_file else None
    )
    support_indices = (
        json.loads(args.support_indices_file.read_text(encoding="utf-8"))
        if args.support_indices_file else None
    )
    if args.labeled_indices is not None:
        args.labeled_count = len(args.labeled_indices)
        args.labeled_ratio = args.labeled_count / len(read_manifest(spec["root"] / spec["train"]))
    train, validation, test, support = prepare(
        args.output_dir, spec, args.labeled_ratio, args.seed, args.labeled_count,
        args.train_subset_ratio,
        args.labeled_indices,
        support_indices,
    )
    dump_effective_config(args.output_dir, args)
    if args.prepare_only:
        raise SystemExit(0)
    head = train_head(args, support, spec["prompts"])
    configure(args, spec, train, validation, test, head)
    main(cfg)
