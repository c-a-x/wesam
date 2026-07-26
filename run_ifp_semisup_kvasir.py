"""Run iterative IFP pseudo-label adaptation on a Kvasir subset."""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path

from adaptation import main
from configs.config import cfg


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
DATA_ROOT = PROJECT_ROOT / "Data" / "kvasir-seg" / "organized" / "Kvasir"
TEXT_PROMPTS = [
    "a colonoscopy image of a polyp",
    "an endoscopic view showing a colorectal polyp",
    "a gastrointestinal endoscopy image with a visible polyp",
]


def find_mask(image: Path) -> Path:
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = DATA_ROOT / "target_masks" / f"{image.stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No target mask for {image.name}")


def prepare_data(
    output_dir: Path, count: int, val_count: int, shot: int, seed: int
) -> tuple[Path, Path, Path]:
    images = sorted(
        path for path in (DATA_ROOT / "target_images").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if count > len(images):
        raise ValueError(f"Requested {count} images, but Kvasir only contains {len(images)}")
    random.Random(seed).shuffle(images)
    images = images[:count]
    if val_count:
        test_images = images[:val_count]
        train_images = images[val_count:]
    else:
        # Preserve the original quick-experiment behavior when no independent
        # validation subset is requested.
        train_images = images
        test_images = images

    # Match ISICDataset._build_has_gt_list() so alignment references are the
    # same samples that receive ground-truth supervision during adaptation.
    support_indices = sorted(random.Random(seed).sample(range(len(train_images)), shot))
    support_images = [train_images[index] for index in support_indices]

    list_dir = output_dir / "lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    for split, split_images in (("train", train_images), ("test", test_images)):
        list_path = list_dir / f"{split}.csv"
        with list_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("index", "image", "mask"))
            for index, image in enumerate(split_images):
                mask = find_mask(image)
                writer.writerow((index, f"target_images/{image.name}", f"target_masks/{mask.name}"))

    support_root = output_dir / "support"
    image_dir = support_root / "reference_images"
    mask_dir = support_root / "reference_masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for directory in (image_dir, mask_dir):
        for existing in directory.iterdir():
            if existing.is_symlink() or existing.is_file():
                existing.unlink()
    for image in support_images:
        mask = find_mask(image)
        (image_dir / image.name).symlink_to(image)
        (mask_dir / mask.name).symlink_to(mask)

    manifest = output_dir / "support.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset_index", "image", "mask"))
        for index, image in zip(support_indices, support_images):
            writer.writerow((index, image.name, find_mask(image).name))

    return list_dir / "train.csv", list_dir / "test.csv", support_root


def train_alignment(args, support_root: Path, output_dir: Path) -> Path:
    alignment_dir = output_dir / "ifp_alignment"
    checkpoint = alignment_dir / "ifp_medical_foreground_best.pt"
    if checkpoint.is_file() and not args.retrain_alignment:
        print(f"Using existing experiment checkpoint: {checkpoint}", flush=True)
        return checkpoint

    command = [
        sys.executable,
        "-m",
        "ifp_alignment.train_medical",
        "--reference-root",
        str(support_root),
        "--output-dir",
        str(alignment_dir),
        "--prompts",
        *TEXT_PROMPTS,
        "--target",
        "foreground",
        "--dino-variant",
        "dino3h",
        "--clip-variant",
        "clipl",
        "--ifp-root",
        str(IFP_ROOT),
        "--device",
        "cuda",
        "--epochs",
        str(args.alignment_epochs),
        "--batch-size",
        "1",
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).parent)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Alignment training did not create {checkpoint}")
    return checkpoint


def configure(args, output_dir: Path, train_list: Path, test_list: Path, checkpoint: Path) -> None:
    cfg.gpu_ids = "0"
    cfg.dataset = "Kvasir"
    cfg.datasets.Kvasir.root_dir = str(DATA_ROOT)
    cfg.datasets.Kvasir.train_list = str(train_list)
    cfg.datasets.Kvasir.test_list = str(test_list)
    cfg.batch_size = args.batch_size
    cfg.val_batchsize = args.val_batch_size
    cfg.num_workers = args.num_workers
    cfg.num_epochs = args.epochs
    cfg.eval_interval = 1
    cfg.load_type = "soft"
    cfg.prompt = "point"
    cfg.out_dir = str(output_dir)
    train_count = args.count - args.val_count
    cfg.name = (
        f"kvasir_{train_count}train_{args.val_count}val_"
        f"ifp_semisup_{args.shot}shot_seed{args.seed}"
    )
    cfg.resume = False

    cfg.semi.enabled = True
    cfg.semi.labeled_ratio = args.shot / train_count
    cfg.semi.seed = args.seed
    cfg.teacher_weight = args.teacher_weight
    cfg.anchor_weight = args.anchor_weight
    cfg.contrast_weight = 0.0
    cfg.supervised_weight = 1.0
    cfg.unsupervised_rampup_epochs = 3
    cfg.semi.labeled_batch_probability = 0.5
    prompt = cfg.prompt_generator
    prompt.backend = "ifp"
    prompt.ifp_root = str(IFP_ROOT)
    prompt.alignment_checkpoint = str(checkpoint)
    prompt.background_alignment_checkpoint = ""
    prompt.dino_variant = "dino3h"
    prompt.clip_variant = "clipl"
    prompt.text_prompts = TEXT_PROMPTS
    prompt.background_text_prompts = []
    prompt.resize = 0
    prompt.score_threshold = None
    prompt.max_positive_points = 1
    prompt.min_positive_points = 1
    prompt.include_box = False
    prompt.iterative_pseudo.enabled = True
    prompt.iterative_pseudo.max_iters = args.max_iters
    prompt.iterative_pseudo.teacher_iou_threshold = args.teacher_iou_threshold
    prompt.iterative_pseudo.min_new_pixels = 32
    prompt.iterative_pseudo.min_new_area_ratio = 0.001
    prompt.iterative_pseudo.max_mask_area_ratio = args.max_mask_area_ratio
    prompt.iterative_pseudo.min_overlap_ratio = args.min_overlap_ratio
    prompt.iterative_pseudo.max_new_area_ratio = 0.4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alignment-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--teacher-weight", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--teacher-iou-threshold", type=float, default=0.8)
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.7)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.6)
    parser.add_argument("--retrain-alignment", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_count = args.count - args.val_count
    if args.val_count < 0 or train_count <= 0:
        raise ValueError("val-count must be in [0, count)")
    if args.shot <= 0 or args.shot > train_count:
        raise ValueError("shot must be in [1, count - val-count]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_list, test_list, support_root = prepare_data(
        args.output_dir, args.count, args.val_count, args.shot, args.seed
    )
    alignment_checkpoint = train_alignment(args, support_root, args.output_dir)
    configure(args, args.output_dir, train_list, test_list, alignment_checkpoint)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.gpu_ids)
    main(cfg)
