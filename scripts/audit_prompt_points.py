"""Audit prompt localization against GT masks without running SAM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from ifp_alignment import build_prompt_generator


ROOT = Path(__file__).resolve().parents[1]


def collate(batch):
    images, _, masks = zip(*batch)
    return torch.stack(images), masks


def prompt_point(prompt):
    if prompt is None or prompt.get("in_points") is None:
        return None
    coords = prompt["in_points"][0]
    return coords.reshape(-1, 2)[0]


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    experiment = args.experiment_dir.resolve()
    dino_checkpoint = args.dino_checkpoint.resolve()
    ifp_checkpoint = (experiment / "ifp_alignment/ifp_medical_foreground_best.pt").resolve()
    list_path = (experiment / "lists/test.csv").resolve()

    cfg.dataset = "ISIC"
    cfg.visual = False
    cfg.datasets.ISIC.root_dir = str(ROOT / "Data/ISIC2018")
    cfg.semi.enabled = False
    cfg.datasets.ISIC.test_list = str(list_path)
    cfg.prompt = "point"

    transform = ResizeAndPad(1024)
    dataset = ISICDataset(cfg, cfg.datasets.ISIC.root_dir, str(list_path), transform=transform)
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    def make(backend, checkpoint):
        return build_prompt_generator(
            backend, device, checkpoint_path=str(checkpoint),
            text_prompts=["a dermoscopic photo of a skin lesion"],
            dino_variant="dino3h", clip_variant="clipl", ifp_root=str(ROOT / "assets"),
            max_positive_points=1, min_positive_points=1, include_box=False,
        )

    generators = {
        "ifp": make("ifp", ifp_checkpoint),
        "dino": make("dino-prototype", dino_checkpoint),
    }
    stats = {name: {"inside": 0, "total": 0, "distance": [], "confidence": [], "agreement": 0} for name in generators}
    rows = []
    for images, masks in loader:
        images = images.to(device)
        prompts = {name: generator(images) for name, generator in generators.items()}
        for index, mask_item in enumerate(masks):
            mask = mask_item[0].to(device) > 0
            points = {name: prompt_point(prompt[index]) for name, prompt in prompts.items()}
            entry = {}
            centers = []
            foreground = torch.nonzero(mask, as_tuple=False)
            center_yx = foreground.float().mean(dim=0)
            diagonal = float((mask.shape[-2] ** 2 + mask.shape[-1] ** 2) ** 0.5)
            for name, point in points.items():
                if point is None:
                    continue
                x, y = point
                xi = int(round(float(x)))
                yi = int(round(float(y)))
                inside = 0 <= yi < mask.shape[-2] and 0 <= xi < mask.shape[-1] and bool(mask[yi, xi])
                distance = float((((y - center_yx[0]) ** 2 + (x - center_yx[1]) ** 2) ** 0.5) / diagonal)
                stats[name]["inside"] += int(inside)
                stats[name]["total"] += 1
                stats[name]["distance"].append(distance)
                confidence = prompts[name][index].get("confidence")
                if confidence is not None:
                    confidence = float(confidence)
                    stats[name]["confidence"].append(confidence)
                entry[name] = {
                    "x": float(x), "y": float(y), "inside": inside,
                    "distance_to_center": distance, "confidence": confidence,
                }
                centers.append(point)
            if len(centers) == 2:
                point_distance = float(torch.linalg.vector_norm(centers[0] - centers[1]).item())
                stats["ifp"]["agreement"] += int(point_distance < args.agreement_pixels)
                entry["point_distance"] = point_distance
            rows.append(entry)

    summary = {}
    for name, value in stats.items():
        summary[name] = {
            "inside_rate": value["inside"] / max(1, value["total"]),
            "mean_normalized_distance_to_gt_center": sum(value["distance"]) / max(1, len(value["distance"])),
            "num_images": value["total"],
        }
        if value["confidence"]:
            confidence = torch.tensor(value["confidence"])
            summary[name]["confidence"] = {
                "mean": float(confidence.mean()),
                "p10": float(torch.quantile(confidence, 0.1)),
                "p50": float(torch.quantile(confidence, 0.5)),
                "p90": float(torch.quantile(confidence, 0.9)),
            }
    summary["ifp_dino_agreement_rate"] = stats["ifp"]["agreement"] / max(1, len(rows))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "ifp_inside", "dino_inside", "ifp_distance", "dino_distance", "ifp_confidence", "dino_confidence", "point_distance"))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                "index": index,
                "ifp_inside": row.get("ifp", {}).get("inside", ""),
                "dino_inside": row.get("dino", {}).get("inside", ""),
                "ifp_distance": row.get("ifp", {}).get("distance_to_center", ""),
                "dino_distance": row.get("dino", {}).get("distance_to_center", ""),
                "ifp_confidence": row.get("ifp", {}).get("confidence", ""),
                "dino_confidence": row.get("dino", {}).get("confidence", ""),
                "point_distance": row.get("point_distance", ""),
            })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--agreement-pixels", type=float, default=64.0)
    parser.add_argument("--max-images", type=int, default=None)
    run(parser.parse_args())
