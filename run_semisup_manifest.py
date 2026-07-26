"""Run the retained IFP-WeSAM semi-supervised method from local manifests."""

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
SPECS = {
    "isic": {
        "config_name": "ISIC", "root": PROJECT_ROOT / "Data/ISIC2018",
        "train": "manifests/train.csv", "validation": "manifests/validation.csv",
        "test": "manifests/test.csv", "prompts": ["a dermoscopic photo of a skin lesion"],
    },
    "kvasir": {
        "config_name": "Kvasir", "root": PROJECT_ROOT / "Data/kvasir-seg/organized/Kvasir",
        "train": "train.csv", "validation": "validation.csv", "test": "test.csv",
        "prompts": ["a colonoscopy image of a polyp", "an endoscopic view showing a colorectal polyp"],
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


def prepare(output_dir: Path, spec: dict, ratio: float, seed: int) -> tuple[Path, Path, Path, Path]:
    root = spec["root"]
    train = read_manifest(root / spec["train"])
    validation = read_manifest(root / spec["validation"])
    test = read_manifest(root / spec["test"])
    count = max(1, round(len(train) * ratio))
    indices = sorted(random.Random(seed).sample(range(len(train)), count))

    lists = output_dir / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_manifest(lists / f"{name}.csv", rows)

    support = output_dir / "support"
    images = support / "reference_images"
    masks = support / "reference_masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for index in indices:
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
    return lists / "train.csv", lists / "validation.csv", lists / "test.csv", support


def train_head(args, support: Path, prompts: list[str]) -> Path:
    out = args.output_dir / "ifp_alignment"
    checkpoint = out / "ifp_medical_foreground_best.pt"
    command = [sys.executable, "-m", "ifp_alignment.train_medical", "--reference-root", str(support),
               "--output-dir", str(out), "--prompts", *prompts, "--target", "foreground",
               "--dino-variant", "dino3h", "--clip-variant", "clipl", "--ifp-root", str(IFP_ROOT),
               "--device", "cuda", "--epochs", str(args.alignment_epochs), "--batch-size", "1",
               "--seed", str(args.seed)]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return checkpoint


def configure(args, spec: dict, train: Path, validation: Path, test: Path, head: Path) -> None:
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
    cfg.name = f"{args.dataset}_{args.labeled_ratio:g}gt_seed{args.seed}"
    cfg.batch_size, cfg.val_batchsize, cfg.num_workers, cfg.num_epochs = args.batch_size, args.val_batch_size, 0, args.epochs
    cfg.load_type, cfg.prompt, cfg.resume = "soft", "point", False
    cfg.semi.enabled, cfg.semi.labeled_ratio, cfg.semi.seed = True, args.labeled_ratio, args.seed
    cfg.semi.labeled_batch_probability = 0.5
    cfg.teacher_weight, cfg.anchor_weight, cfg.contrast_weight, cfg.supervised_weight = 0.1, 0.0, 0.0, 1.0
    cfg.unsupervised_rampup_epochs = 3
    prompt = cfg.prompt_generator
    prompt.backend, prompt.ifp_root, prompt.alignment_checkpoint = "ifp", str(IFP_ROOT), str(head)
    prompt.dino_variant, prompt.clip_variant, prompt.text_prompts = "dino3h", "clipl", spec["prompts"]
    prompt.max_positive_points, prompt.min_positive_points, prompt.include_box = 1, 1, False
    prompt.iterative_pseudo.enabled, prompt.iterative_pseudo.max_iters = True, 1
    prompt.iterative_pseudo.teacher_iou_threshold, prompt.iterative_pseudo.max_mask_area_ratio = 0.8, 0.6
    prompt.iterative_pseudo.min_overlap_ratio, prompt.iterative_pseudo.max_new_area_ratio = 0.7, 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--labeled-ratio", type=float, choices=(0.01, 0.10), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--alignment-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spec = SPECS[args.dataset]
    # Set this before starting the projection-head subprocess so it and the
    # subsequent SAM2 process stay on the same requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test, support = prepare(args.output_dir, spec, args.labeled_ratio, args.seed)
    head = train_head(args, support, spec["prompts"])
    configure(args, spec, train, validation, test, head)
    main(cfg)
