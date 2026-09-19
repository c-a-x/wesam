"""Export test-set masks from an existing WeSAM experiment without training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from ifp_alignment import build_prompt_generator
from model import Model
from utils.eval_utils import AverageMeter, combine_instance_masks, remove_padding


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
SPECS = {
    "isic": {
        "config_name": "ISIC",
        "root": PROJECT_ROOT / "Data/ISIC2018",
        "prompts": ["a dermoscopic photo of a skin lesion"],
    },
    "kvasir": {
        "config_name": "Kvasir",
        "root": PROJECT_ROOT / "Data/kvasir-seg/organized/Kvasir",
        "prompts": [
            "a colonoscopy image of a polyp",
            "an endoscopic view showing a colorectal polyp",
        ],
    },
    "polyp": {
        "config_name": "Polyp",
        "root": PROJECT_ROOT / "Data/kvasir-seg/organized",
        "prompts": [
            "a colonoscopy image of a polyp",
            "an endoscopic view showing a colorectal polyp",
        ],
    },
}


def collate_export(batch):
    names, paddings, original_images, _, _, images, _, masks = zip(*batch)
    return names, paddings, original_images, torch.stack(images), masks


def configure(
    dataset: str,
    experiment_dir: Path,
    root_dir: Path | None,
    test_list_override: Path | None,
    prompt_backend: str,
    prompt_checkpoint_override: Path | None = None,
) -> tuple[Path, Path, Path]:
    spec = SPECS[dataset]
    name = spec["config_name"]
    model_dir = experiment_dir / name
    checkpoint = model_dir / "save/best-student.pth"
    if prompt_checkpoint_override is not None:
        alignment_checkpoint = prompt_checkpoint_override
    elif prompt_backend == "dino-prototype":
        alignment_checkpoint = experiment_dir / "ifp_alignment/dino_foreground_background_prototype.pt"
    elif prompt_backend == "clip-only":
        alignment_checkpoint = Path()
    else:
        alignment_checkpoint = experiment_dir / "ifp_alignment/ifp_medical_foreground_best.pt"
    test_list = test_list_override or experiment_dir / "lists/test.csv"
    root_dir = root_dir or spec["root"]
    required_paths = (checkpoint, test_list)
    if prompt_backend not in {"clip-only", "no-prompt", "gt-oracle"}:
        required_paths = (*required_paths, alignment_checkpoint)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg.gpu_ids = "0"
    cfg.dataset = name
    # Export needs the original image name and padding so masks can be saved
    # at native resolution. This changes only dataset return values.
    cfg.visual = True
    cfg.datasets[name].root_dir = str(root_dir)
    cfg.datasets[name].test_list = str(test_list)
    cfg.out_dir = str(model_dir)
    cfg.load_type = "soft"
    cfg.prompt = "none" if prompt_backend == "no-prompt" else "point"
    cfg.prompt_generator.backend = prompt_backend
    cfg.prompt_generator.ifp_root = str(IFP_ROOT)
    cfg.prompt_generator.alignment_checkpoint = str(alignment_checkpoint)
    cfg.prompt_generator.background_alignment_checkpoint = ""
    cfg.prompt_generator.dino_variant = "dino3h"
    cfg.prompt_generator.clip_variant = "clipl"
    cfg.prompt_generator.text_prompts = spec["prompts"]
    cfg.prompt_generator.max_positive_points = 1
    cfg.prompt_generator.min_positive_points = 1
    cfg.prompt_generator.include_box = False
    return model_dir, checkpoint, test_list


def recorded_test_metric(metrics_path: Path) -> tuple[float, float] | None:
    if not metrics_path.is_file():
        return None
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in reversed(rows):
        if row["Name"].endswith("_final_test_best_student"):
            return float(row["Mean IoU"]), float(row["Mean F1"])
    return None


def export(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    root_dir = args.root_dir.resolve() if args.root_dir else None
    test_list_override = args.test_list.resolve() if args.test_list else None
    model_dir, checkpoint_path, test_list = configure(
        args.dataset,
        experiment_dir,
        root_dir,
        test_list_override,
        args.prompt_backend,
        args.prompt_checkpoint,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = Model(cfg)
    model.setup()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    prompt_generator = build_prompt_generator(
        cfg.prompt_generator.backend,
        device,
        checkpoint_path=cfg.prompt_generator.alignment_checkpoint,
        text_prompts=cfg.prompt_generator.text_prompts,
        dino_variant=cfg.prompt_generator.dino_variant,
        clip_variant=cfg.prompt_generator.clip_variant,
        ifp_root=cfg.prompt_generator.ifp_root,
        max_positive_points=cfg.prompt_generator.max_positive_points,
        min_positive_points=cfg.prompt_generator.min_positive_points,
        include_box=cfg.prompt_generator.include_box,
    )

    dataset = ISICDataset(
        cfg,
        root_dir=cfg.datasets[cfg.dataset].root_dir,
        list_file=str(test_list),
        transform=ResizeAndPad(model.image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_export,
    )

    result_dir = args.result_dir.resolve() if args.result_dir else model_dir
    output_dir = result_dir / "pred_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = result_dir / "prediction_metrics.csv"
    ious, f1_scores = AverageMeter(), AverageMeter()
    image_ious: list[float] = []
    image_f1s: list[float] = []

    with per_image_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
        writer = csv.DictWriter(
            handle,
            fieldnames=("image", "IoU", "F1", "prediction"),
        )
        writer.writeheader()
        for names, paddings, original_images, images, gt_masks in dataloader:
            images = images.to(device)
            prompts = (prompt_generator(images, gt_masks)
                       if getattr(prompt_generator, "requires_gt_masks", False)
                       else prompt_generator(images))
            _, pred_masks, _, _ = model(images, prompts)
            num_images = images.size(0)

            for name, padding, original_image, pred_mask, gt_mask in zip(
                names, paddings, original_images, pred_masks, gt_masks
            ):
                target = combine_instance_masks(gt_mask).to(device)
                # SAM2 returns raw mask logits. Convert them explicitly so
                # the saved mask and its metrics use the same decision rule.
                prediction = (pred_mask >= args.logit_threshold).int()
                batch_stats = smp.metrics.get_stats(
                    prediction, target.int(), mode="binary", threshold=0.5
                )
                image_iou = float(smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise"))
                image_f1 = float(smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise"))
                # Each prediction is scored independently, so no batch-size weighting applies.
                ious.update(image_iou, 1)
                f1_scores.update(image_f1, 1)
                image_ious.append(image_iou)
                image_f1s.append(image_f1)

                mask = remove_padding(prediction.squeeze(0).cpu(), padding).numpy().astype("uint8")
                height, width = original_image.shape[:2]
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) * 255
                prediction_path = output_dir / f"{name}.png"
                Image.fromarray(mask).save(prediction_path)
                writer.writerow({
                    "image": name,
                    "IoU": f"{image_iou:.6f}",
                    "F1": f"{image_f1:.6f}",
                    "prediction": str(prediction_path.relative_to(result_dir)),
                })

    recorded = recorded_test_metric(model_dir / "metrics.csv") if result_dir == model_dir else None
    summary = {
        "dataset": args.dataset,
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(checkpoint_path),
        "prompt_backend": args.prompt_backend,
        "best_validation_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_selection_metric": "validation_mean_iou",
        "mask_logit_threshold": args.logit_threshold,
        "mask_binarization": f"mask_logits >= {args.logit_threshold}",
        "test_manifest": str(test_list),
        "num_images": len(dataset),
        "reported_mean_iou": round(float(ious.avg), 6),
        "reported_mean_f1": round(float(f1_scores.avg), 6),
        "per_image_mean_iou": round(sum(image_ious) / len(image_ious), 6),
        "per_image_mean_f1": round(sum(image_f1s) / len(image_f1s), 6),
        "recorded_final_test": (
            {"mean_iou": recorded[0], "mean_f1": recorded[1]} if recorded else None
        ),
    }
    if recorded:
        summary["matches_recorded_final_test"] = (
            abs(summary["reported_mean_iou"] - recorded[0]) < 5e-4
            and abs(summary["reported_mean_f1"] - recorded[1]) < 5e-4
        )
    summary_path = result_dir / "prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path,
                        help="Dataset root; required for cross-dataset evaluation.")
    parser.add_argument("--test-list", type=Path,
                        help="Manifest to evaluate instead of <experiment-dir>/lists/test.csv.")
    parser.add_argument("--result-dir", type=Path,
                        help="Directory for pred_masks and per-test metrics.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--prompt-backend", choices=("ifp", "dino-prototype", "clip-only", "no-prompt", "gt-oracle"), default="ifp",
    )
    parser.add_argument(
        "--prompt-checkpoint", type=Path,
        help="Use a prompt-generator checkpoint from another experiment directory.",
    )
    parser.add_argument(
        "--logit-threshold", type=float, default=0.0,
        help="Threshold applied directly to SAM2 mask logits (0.0 equals probability 0.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
