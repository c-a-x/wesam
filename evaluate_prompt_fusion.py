"""Evaluate conservative no-prompt + IFP inference-time fusion.

The base checkpoint is always the no-prompt model. IFP only contributes a
secondary prediction through a non-destructive gate/ensemble, so the learned
no-prompt solution is preserved exactly when the gate chooses it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from ifp_alignment import build_prompt_generator
from model import Model
from utils.eval_utils import combine_instance_masks, remove_padding


PROJECT_ROOT = Path(__file__).resolve().parent
IFP_ROOT = PROJECT_ROOT / "assets"
SPECS = {
    "isic": {
        "config_name": "ISIC",
        "root": PROJECT_ROOT / "Data/ISIC2018",
        "prompts": [
            "a dermoscopic photo of a skin lesion",
            "a close-up photo of a mole on human skin",
            "a medical photo of melanoma on the skin",
        ],
    },
    "polyp": {
        "config_name": "Polyp",
        "root": PROJECT_ROOT / "Data/kvasir-seg/organized",
        "prompts": [
            "a colonoscopy image of a polyp",
            "an endoscopic view showing a colorectal polyp",
            "a gastrointestinal endoscopy image with a visible polyp",
        ],
    },
}


def collate_eval(batch):
    names, paddings, original_images, _, _, images, _, masks = zip(*batch)
    return names, paddings, original_images, torch.stack(images), masks


def set_config(dataset: str, root_dir: Path, list_file: Path, model_dir: Path,
               prompt_checkpoint: Path) -> None:
    spec = SPECS[dataset]
    name = spec["config_name"]
    cfg.gpu_ids = "0"
    cfg.dataset = name
    cfg.visual = True
    cfg.load_type = "soft"
    cfg.datasets[name].root_dir = str(root_dir)
    cfg.datasets[name].test_list = str(list_file)
    cfg.out_dir = str(model_dir)
    cfg.prompt = "none"
    prompt = cfg.prompt_generator
    prompt.backend = "ifp"
    prompt.ifp_root = str(IFP_ROOT)
    prompt.alignment_checkpoint = str(prompt_checkpoint)
    prompt.background_alignment_checkpoint = ""
    prompt.dino_variant = "dino3h"
    prompt.clip_variant = "clipl"
    prompt.text_prompts = spec["prompts"]
    prompt.background_text_prompts = []
    prompt.max_positive_points = 1
    prompt.min_positive_points = 1
    prompt.include_box = False


def metric_from_mask(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
    pred = pred.bool()
    target = target.bool()
    inter = float((pred & target).sum().item())
    union = float((pred | target).sum().item())
    pred_area = float(pred.sum().item())
    target_area = float(target.sum().item())
    iou = inter / max(union, 1e-7)
    f1 = 2.0 * inter / max(pred_area + target_area, 1e-7)
    precision = inter / max(pred_area, 1e-7)
    recall = inter / max(target_area, 1e-7)
    return iou, f1, precision, recall


def mask_area_ratio(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())


def evaluate_split(
    model: Model,
    prompt_generator,
    dataset: ISICDataset,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_eval,
    )
    records: list[dict] = []
    with torch.no_grad():
        for names, paddings, _, images, gt_masks in loader:
            images = images.to(device)
            prompts = prompt_generator(images)
            model.encode(images)
            base_masks, base_ious, _ = model.decode(
                images.shape[-2:], None, prompt_mode="none"
            )
            ifp_masks, ifp_ious, _ = model.decode(
                images.shape[-2:], prompts, prompt_mode="point"
            )
            if isinstance(base_ious, list):
                base_pred_ious = torch.tensor(
                    [float(value.reshape(-1)[0].item()) for value in base_ious],
                    device=device,
                )
            else:
                base_pred_ious = base_ious.reshape(len(base_masks), -1)[:, 0]
            if isinstance(ifp_ious, list):
                ifp_pred_ious = torch.tensor(
                    [float(value.reshape(-1)[0].item()) for value in ifp_ious],
                    device=device,
                )
            else:
                ifp_pred_ious = ifp_ious.reshape(len(ifp_masks), -1)[:, 0]

            for index, (name, padding, base, ifp, gt) in enumerate(
                zip(names, paddings, base_masks, ifp_masks, gt_masks)
            ):
                target = combine_instance_masks(gt).to(device)
                # Keep padded coordinates so thresholding exactly matches the
                # existing export/evaluation path: binarize, then remove pad.
                records.append({
                    "name": name,
                    "padding": tuple(int(v) for v in padding),
                    "base_logits": base.squeeze(0).cpu().numpy().astype(np.float16),
                    "ifp_logits": ifp.squeeze(0).cpu().numpy().astype(np.float16),
                    "target": target.squeeze(0).cpu().numpy().astype(np.uint8),
                    "base_pred_iou": float(base_pred_ious[index].item()),
                    "ifp_pred_iou": float(ifp_pred_ious[index].item()),
                })
            print(f"evaluated {len(records)} images", flush=True)
    return records


def predict_from_logits(record: dict, strategy: str, alpha: float,
                        gate_base_q: float, gate_ifp_q: float,
                        area_low: float, area_high: float) -> np.ndarray:
    base_logits = record["base_logits"].astype(np.float32)
    ifp_logits = record["ifp_logits"].astype(np.float32)
    base_prob = 1.0 / (1.0 + np.exp(-base_logits))
    ifp_prob = 1.0 / (1.0 + np.exp(-ifp_logits))
    base_binary = base_logits >= 0.0
    ifp_binary = ifp_logits >= 0.0
    base_area = float(base_binary.mean())
    ifp_area = float(ifp_binary.mean())
    base_q = record["base_pred_iou"]
    ifp_q = record["ifp_pred_iou"]

    if strategy == "base":
        return base_binary
    if strategy == "ifp":
        return ifp_binary
    if strategy == "prob":
        return (1.0 - alpha) * base_prob + alpha * ifp_prob >= 0.5
    if strategy == "logit":
        return (1.0 - alpha) * base_logits + alpha * ifp_logits >= 0.0
    if strategy == "union_gate":
        gate = (
            base_area < area_low
            or base_area > area_high
            or base_q < gate_base_q
            or (ifp_q >= gate_ifp_q and ifp_area > base_area)
        )
        return np.logical_or(base_binary, ifp_binary) if gate else base_binary
    if strategy == "intersect_gate":
        gate = (
            base_area > area_high
            or base_q < gate_base_q
            or (ifp_q >= gate_ifp_q and ifp_area < base_area)
        )
        return np.logical_and(base_binary, ifp_binary) if gate else base_binary
    raise ValueError(strategy)


def summarize(records: list[dict], strategy: str, alpha: float,
              gate_base_q: float, gate_ifp_q: float,
              area_low: float, area_high: float) -> dict:
    rows = []
    for record in records:
        pred = predict_from_logits(
            record, strategy, alpha, gate_base_q, gate_ifp_q, area_low, area_high
        )
        pred = remove_padding(torch.from_numpy(pred), record["padding"]).numpy()
        target = remove_padding(
            torch.from_numpy(record["target"].astype(bool)), record["padding"]
        ).numpy()
        iou, f1, precision, recall = metric_from_mask(
            torch.from_numpy(pred), torch.from_numpy(target)
        )
        rows.append({"name": record["name"], "IoU": iou, "F1": f1,
                     "precision": precision, "recall": recall})
    mean = {key: float(np.mean([row[key] for row in rows]))
            for key in ("IoU", "F1", "precision", "recall")}
    return {"strategy": strategy, "alpha": alpha, "gate_base_q": gate_base_q,
            "gate_ifp_q": gate_ifp_q, "area_low": area_low,
            "area_high": area_high, **mean, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, default=SPECS["isic"]["root"])
    parser.add_argument("--no-prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--list-file", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(SPECS), default="isic")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep conservative fusion settings on this split.")
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    list_file = args.list_file or args.experiment_dir / "lists" / f"{args.split}.csv"
    set_config(args.dataset, args.root_dir.resolve(), list_file.resolve(),
               args.experiment_dir.resolve(), args.prompt_checkpoint.resolve())
    device = torch.device(args.device)

    model = Model(cfg)
    model.setup()
    checkpoint = torch.load(args.no_prompt_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    prompt_generator = build_prompt_generator(
        "ifp", device,
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
        list_file=str(list_file.resolve()),
        transform=ResizeAndPad(model.image_size),
    )
    records = evaluate_split(model, prompt_generator, dataset, device, args.batch_size)
    np.savez_compressed(
        args.result_dir / f"{args.split}_logits.npz",
        **{key: np.array([record[key] for record in records]) for key in
           ("name", "padding", "base_logits", "ifp_logits", "target")},
    )

    candidates = [summarize(records, "base", 0, 0, 0, 0, 1)]
    if args.sweep:
        for alpha in (0.01, 0.02, 0.05, 0.10, 0.20):
            candidates.append(summarize(records, "prob", alpha, 0, 0, 0, 1))
            candidates.append(summarize(records, "logit", alpha, 0, 0, 0, 1))
        for gate_base_q in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            for gate_ifp_q in (0.50, 0.60, 0.70, 0.80, 0.90):
                for area_low, area_high in (
                    (0.02, 0.80), (0.03, 0.75), (0.05, 0.70),
                    (0.02, 0.70), (0.05, 0.80),
                ):
                    candidates.append(summarize(
                        records, "union_gate", 0, gate_base_q, gate_ifp_q,
                        area_low, area_high,
                    ))
                    candidates.append(summarize(
                        records, "intersect_gate", 0, gate_base_q, gate_ifp_q,
                        area_low, area_high,
                    ))
    else:
        config_path = args.result_dir / "selected_fusion.json"
        selected = json.loads(config_path.read_text(encoding="utf-8"))
        candidates.append(summarize(records, **selected))

    candidates.sort(key=lambda item: (item["F1"], item["IoU"]), reverse=True)
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "num_images": len(records),
        "base_checkpoint": str(args.no_prompt_checkpoint.resolve()),
        "prompt_checkpoint": str(args.prompt_checkpoint.resolve()),
        "candidates": candidates,
    }
    compact = dict(summary)
    compact["candidates"] = [
        {key: value for key, value in item.items() if key != "rows"}
        for item in candidates
    ]
    (args.result_dir / f"{args.split}_fusion_summary.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "split": args.split,
        "num_images": len(records),
        "best": [{k: v for k, v in item.items() if k != "rows"} for item in candidates[:10]],
    }, indent=2))


if __name__ == "__main__":
    main()
