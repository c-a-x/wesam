"""Stream no-prompt / IFP logits for Polyp and sweep inference-time fusion.

This script is deliberately Polyp-oriented because the fair Polyp runner uses
``cfg.visual=True`` and the dataset therefore returns a different tuple layout
than the ISIC fusion evaluator.  The metric path mirrors
``export_test_predictions.py``: raw SAM2 logits are thresholded at 0.0,
instance masks are merged, and image-wise metrics are averaged over images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from ifp_alignment import build_prompt_generator
from model import Model
from utils.eval_utils import combine_instance_masks


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
POLYP_ROOT = PROJECT_ROOT / "Data/kvasir-seg/organized"
TRAIN_PROMPTS = [
    "a colonoscopy image of a polyp",
    "an endoscopic view showing a colorectal polyp",
    "a gastrointestinal endoscopy image with a visible polyp",
]
# This is the prompt set used by the existing fair Polyp export script.  It is
# kept as the default so the base/pure-IFP columns are directly comparable with
# the current ablation tables.  ``--prompt-set train`` is available as a
# protocol diagnostic because the alignment heads were trained with 3 prompts.
OFFICIAL_PROMPTS = [
    "a colonoscopy image of a polyp",
    "an endoscopic view showing a colorectal polyp",
]


def collate_eval(batch):
    names, paddings, _, _, _, images, _, masks = zip(*batch)
    return names, paddings, torch.stack(images), masks


def set_config(root_dir: Path, list_file: Path, prompt_set: str) -> None:
    prompts = TRAIN_PROMPTS if prompt_set == "train" else OFFICIAL_PROMPTS
    cfg.gpu_ids = "0"
    cfg.dataset = "Polyp"
    cfg.visual = True
    cfg.load_type = "soft"
    cfg.datasets.Polyp.root_dir = str(root_dir)
    cfg.datasets.Polyp.test_list = str(list_file)
    cfg.prompt = "none"
    prompt = cfg.prompt_generator
    prompt.backend = "ifp"
    prompt.ifp_root = str(IFP_ROOT)
    prompt.background_alignment_checkpoint = ""
    prompt.dino_variant = "dino3h"
    prompt.clip_variant = "clipl"
    prompt.text_prompts = prompts
    prompt.max_positive_points = 1
    prompt.min_positive_points = 1
    prompt.include_box = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    iou = tp / max(tp + fp + fn, 1e-7)
    f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-7)
    precision = tp / max(tp + fp, 1e-7)
    recall = tp / max(tp + fn, 1e-7)
    return iou, f1, precision, recall


def candidate_names():
    names = ["base", "ifp"]
    for alpha in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        names.append(f"logit_alpha_{alpha:g}")
        names.append(f"prob_alpha_{alpha:g}")
    for threshold in (0.005, 0.01, 0.02, 0.05, 0.1):
        names.append(f"fallback_base_area_lt_{threshold:g}")
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        names.append(f"fallback_base_conf_lt_{threshold:g}")
    return names


def predict_candidate(base, ifp, base_prob, ifp_prob, name):
    if name == "base":
        return base >= 0.0
    if name == "ifp":
        return ifp >= 0.0
    if name.startswith("logit_alpha_"):
        alpha = float(name.removeprefix("logit_alpha_"))
        return (1.0 - alpha) * base + alpha * ifp >= 0.0
    if name.startswith("prob_alpha_"):
        alpha = float(name.removeprefix("prob_alpha_"))
        return (1.0 - alpha) * base_prob + alpha * ifp_prob >= 0.5
    if name.startswith("fallback_base_area_lt_"):
        threshold = float(name.removeprefix("fallback_base_area_lt_"))
        base_binary = base >= 0.0
        return (ifp >= 0.0) if float(base_binary.mean()) < threshold else base_binary
    if name.startswith("fallback_base_conf_lt_"):
        threshold = float(name.removeprefix("fallback_base_conf_lt_"))
        base_conf = float(np.mean(np.abs(base_prob - 0.5)) * 2.0)
        return (ifp >= 0.0) if base_conf < threshold else (base >= 0.0)
    raise ValueError(name)


def evaluate_records(root: Path, output: Path, selected: dict | None = None) -> dict:
    base_dir = root / "base_logits"
    ifp_dir = root / "ifp_logits"
    target_dir = root / "targets"
    files = sorted(base_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No logits found in {base_dir}")
    names = candidate_names()
    sums = {name: np.zeros(4, dtype=np.float64) for name in names}
    finite = {name: 0 for name in names}
    per_candidate_selected = None
    selected_name = None
    if selected:
        selected_name = selected.get("name") or selected.get("strategy")
        if selected_name and selected_name not in sums:
            raise ValueError(f"Unknown selected candidate {selected_name!r}")
    rows = []
    for index, path in enumerate(files, 1):
        base = np.load(path).astype(np.float32).squeeze()
        ifp = np.load(ifp_dir / path.name).astype(np.float32).squeeze()
        target = np.load(target_dir / path.name).astype(bool)
        base_prob = sigmoid(base)
        ifp_prob = sigmoid(ifp)
        row = {"name": path.stem}
        for name in names:
            pred = predict_candidate(base, ifp, base_prob, ifp_prob, name)
            values = metrics(pred, target)
            sums[name] += values
            finite[name] += int(np.isfinite(pred).all())
            if name == selected_name:
                row["selected_IoU"], row["selected_F1"], row["selected_precision"], row["selected_recall"] = values
        rows.append(row)
        if index % 100 == 0:
            print(f"summarized {index}/{len(files)}", flush=True)
    candidates = []
    for name, values in sums.items():
        mean = values / max(len(files), 1)
        candidates.append({
            "name": name,
            "IoU": float(mean[0]),
            "F1": float(mean[1]),
            "precision": float(mean[2]),
            "recall": float(mean[3]),
            "finite_predictions": finite[name],
        })
    candidates.sort(key=lambda item: (item["F1"], item["IoU"]), reverse=True)
    by_name = {item["name"]: item for item in candidates}
    selected_item = by_name.get(selected_name) if selected_name else None
    result = {
        "num_images": len(files),
        "selection_source": selected,
        "selected_candidate": selected_item,
        "best_by_F1": candidates[:10],
        "candidates": candidates,
        "base": by_name["base"],
        "ifp": by_name["ifp"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "num_images": len(files),
        "base": by_name["base"],
        "ifp": by_name["ifp"],
        "selected": selected_item,
        "best_by_F1": candidates[:8],
    }, indent=2))
    return result


def stream(args: argparse.Namespace) -> None:
    set_config(args.root_dir.resolve(), args.list_file.resolve(), args.prompt_set)
    device = torch.device(args.device)
    model = Model(cfg)
    model.setup()
    checkpoint = torch.load(args.no_prompt_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    generator = build_prompt_generator(
        "ifp",
        device,
        checkpoint_path=str(args.prompt_checkpoint.resolve()),
        text_prompts=cfg.prompt_generator.text_prompts,
        dino_variant=cfg.prompt_generator.dino_variant,
        clip_variant=cfg.prompt_generator.clip_variant,
        ifp_root=cfg.prompt_generator.ifp_root,
        max_positive_points=1,
        min_positive_points=1,
        include_box=False,
    )
    dataset = ISICDataset(
        cfg,
        root_dir=str(args.root_dir.resolve()),
        list_file=str(args.list_file.resolve()),
        transform=ResizeAndPad(model.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )
    out = args.output_dir
    base_dir = out / "base_logits"
    ifp_dir = out / "ifp_logits"
    target_dir = out / "targets"
    for directory in (base_dir, ifp_dir, target_dir):
        directory.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"
    records_written = 0
    with index_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(("index", "name", "padding", "base_pred_iou", "ifp_pred_iou"))
        for names, paddings, images, gt_masks in loader:
            if args.max_images is not None and records_written >= args.max_images:
                break
            images = images.to(device)
            prompts = generator(images)
            model.encode(images)
            base_masks, base_ious, _ = model.decode(images.shape[-2:], None, prompt_mode="none")
            ifp_masks, ifp_ious, _ = model.decode(images.shape[-2:], prompts, prompt_mode="point")
            for name, padding, base, ifp, gt, base_iou, ifp_iou in zip(
                names, paddings, base_masks, ifp_masks, gt_masks, base_ious, ifp_ious
            ):
                if args.max_images is not None and records_written >= args.max_images:
                    break
                target = combine_instance_masks(gt).float()
                np.save(base_dir / f"{records_written:06d}.npy", base.squeeze(0).cpu().numpy().astype(np.float16))
                np.save(ifp_dir / f"{records_written:06d}.npy", ifp.squeeze(0).cpu().numpy().astype(np.float16))
                np.save(target_dir / f"{records_written:06d}.npy", (target.squeeze(0).numpy() > 0).astype(np.uint8))
                writer.writerow((
                    records_written,
                    name,
                    json.dumps([int(value) for value in (padding.tolist() if hasattr(padding, "tolist") else padding)]),
                    float(base_iou.reshape(-1)[0].item()),
                    float(ifp_iou.reshape(-1)[0].item()),
                ))
                records_written += 1
            print(f"streamed {records_written}/{len(dataset)}", flush=True)
    evaluate_records(out, args.summary_output, args.selected_config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--selected-config", type=Path,
                        help="JSON containing the validation-selected candidate name.")
    parser.add_argument("--root-dir", type=Path, default=POLYP_ROOT)
    parser.add_argument("--prompt-set", choices=("official", "train"), default="official")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = json.loads(args.selected_config.read_text()) if args.selected_config else None
    stream(args)
    if selected is None and args.max_images is None:
        # Write a first-pass validation selection file for the driver.
        summary = json.loads(args.summary_output.read_text())
        best = summary["best_by_F1"][0]
        selection = {"name": best["name"], "validation_F1": best["F1"], "validation_IoU": best["IoU"]}
        (args.summary_output.parent / "selected_by_validation.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
