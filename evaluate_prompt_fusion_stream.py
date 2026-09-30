"""Stream no-prompt/IFP mask logits to disk, one image at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.config import cfg
from datasets.ISIC import ISICDataset
from datasets.tools import ResizeAndPad
from ifp_alignment import build_prompt_generator
from model import Model


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
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), default="isic")
    parser.add_argument("--root-dir", type=Path, default=SPECS["isic"]["root"])
    parser.add_argument("--list-file", type=Path, required=True)
    parser.add_argument("--no-prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    spec = SPECS[args.dataset]
    name = spec["config_name"]
    cfg.gpu_ids = "0"
    cfg.dataset = name
    cfg.visual = True
    cfg.load_type = "soft"
    cfg.datasets[name].root_dir = str(args.root_dir.resolve())
    cfg.datasets[name].test_list = str(args.list_file.resolve())
    prompt = cfg.prompt_generator
    prompt.backend = "ifp"
    prompt.ifp_root = str(IFP_ROOT)
    prompt.alignment_checkpoint = str(args.prompt_checkpoint.resolve())
    prompt.background_alignment_checkpoint = ""
    prompt.dino_variant = "dino3h"
    prompt.clip_variant = "clipl"
    prompt.text_prompts = spec["prompts"]
    prompt.max_positive_points = 1
    prompt.min_positive_points = 1
    prompt.include_box = False

    device = torch.device(args.device)
    model = Model(cfg)
    model.setup()
    checkpoint = torch.load(args.no_prompt_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    generator = build_prompt_generator(
        "ifp", device,
        checkpoint_path=str(args.prompt_checkpoint.resolve()),
        text_prompts=prompt.text_prompts,
        dino_variant=prompt.dino_variant,
        clip_variant=prompt.clip_variant,
        ifp_root=prompt.ifp_root,
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = args.output_dir / "base_logits"
    ifp_dir = args.output_dir / "ifp_logits"
    target_dir = args.output_dir / "targets"
    for directory in (base_dir, ifp_dir, target_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "index.csv"
    indices = []

    with manifest_path.open("w", encoding="utf-8") as handle, torch.no_grad():
        handle.write("name\n")
        for index in range(len(dataset)):
            name, _, _, _, _, image, _, gt_masks = dataset[index]
            images = image.unsqueeze(0).to(device)
            prompts = generator(images)
            model.encode(images)
            base, _, _ = model.decode(images.shape[-2:], None, prompt_mode="none")
            ifp, _, _ = model.decode(images.shape[-2:], prompts, prompt_mode="point")
            target = torch.stack([mask.float() for mask in gt_masks])
            if target.shape[0] > 1:
                target = target.amax(dim=0, keepdim=True)
            else:
                target = (target > 0).float()
            # Save the model-space logits. The GPU preprocess is deterministic
            # and its output is already the native padded image space.
            np.save(base_dir / f"{index:06d}.npy", base[0].cpu().numpy().astype(np.float16))
            np.save(ifp_dir / f"{index:06d}.npy", ifp[0].cpu().numpy().astype(np.float16))
            np.save(target_dir / f"{index:06d}.npy", target[0].cpu().numpy().astype(np.uint8))
            indices.append({"name": str(name), "index": index})
            handle.write(f"{name}\n")
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"saved {index + 1}/{len(dataset)}", flush=True)


if __name__ == "__main__":
    main()
