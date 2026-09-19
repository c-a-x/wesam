"""Train IFP image-text projection heads from medical few-shot references.

The reference root must contain ``reference_images/`` and ``reference_masks/``
with matching file stems. Run once for foreground and optionally again with
``--target background`` to produce the background alignment checkpoint.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ifp_alignment.common import (
    DEFAULT_IFP_ROOT,
    ProjectionHead,
    default_resize,
    extract_clip_patch_tokens,
    extract_patch_tokens,
    get_patch_size,
    load_clip_model,
    load_dino_model,
    pixel_mask_to_patch_mask,
    text_features,
)


class MedicalReferenceDataset(Dataset):
    def __init__(
        self, root: Path, extract_tokens, resize: int, patch_size: int, target: str, device,
        max_references: int | None = None, reference_seed: int = 42,
    ):
        self.image_dir = root / "reference_images"
        self.mask_dir = root / "reference_masks"
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError("reference root requires reference_images/ and reference_masks/")
        self.files = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if max_references is not None:
            if max_references > len(self.files):
                raise ValueError(
                    f"Requested {max_references} references but only {len(self.files)} are available"
                )
            # Match IFP's few-shot selection: random.sample(os.listdir(...), seed=42).
            original_order = [
                self.image_dir / name
                for name in os.listdir(self.image_dir)
                if (self.image_dir / name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            self.files = random.Random(reference_seed).sample(original_order, max_references)
        self.extract_tokens = extract_tokens
        self.resize = resize
        self.patch_size = patch_size
        self.target = target
        self.device = device

        valid_files = []
        skipped_files = []
        for image_path in self.files:
            mask_path = self._mask_path(image_path)
            foreground = np.asarray(Image.open(mask_path).convert("L")) > 127
            patch_mask = pixel_mask_to_patch_mask(
                foreground, self.resize, self.patch_size
            ).flatten()
            if self.target == "background":
                patch_mask = ~patch_mask
            if bool(patch_mask.any()) and bool((~patch_mask).any()):
                valid_files.append(image_path)
            else:
                skipped_files.append(image_path.name)
        self.files = valid_files
        if skipped_files:
            print(
                f"Skipping {len(skipped_files)} references without usable "
                f"{self.target}/other patches: {', '.join(skipped_files)}",
                flush=True,
            )
        if not self.files:
            raise RuntimeError(
                f"No usable reference image/mask pairs found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.files)

    def _mask_path(self, image_path: Path) -> Path:
        mask_path = next(
            (
                self.mask_dir / f"{image_path.stem}{extension}"
                for extension in (".png", ".jpg", ".jpeg")
                if (self.mask_dir / f"{image_path.stem}{extension}").is_file()
            ),
            None,
        )
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        return mask_path

    def __getitem__(self, index):
        image_path = self.files[index]
        mask_path = self._mask_path(image_path)

        image = Image.open(image_path).convert("RGB")
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().div(255).unsqueeze(0)
        tokens, _ = self.extract_tokens(image_tensor.to(self.device))
        foreground = np.asarray(Image.open(mask_path).convert("L")) > 127
        patch_mask = pixel_mask_to_patch_mask(foreground, self.resize, self.patch_size).flatten().to(self.device)
        if self.target == "background":
            patch_mask = ~patch_mask
        selected = tokens[0][patch_mask]
        other = tokens[0][~patch_mask]
        if selected.numel() == 0 or other.numel() == 0:
            raise RuntimeError(f"Reference mask for {image_path.name} has no usable {self.target}/other patches")
        return selected.mean(dim=0).cpu(), other.mean(dim=0).cpu()


def alignment_loss(selected, other, text, margin: float):
    positive = 1 - F.cosine_similarity(selected, text).mean()
    negative = F.relu(F.cosine_similarity(other, text).mean() + margin)
    separation = F.relu(F.cosine_similarity(selected, other).mean() + margin)
    return positive + negative + 0.5 * separation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, help="Directory containing reference_images/ and reference_masks/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", required=True, nargs="+", help="Foreground or background text templates")
    parser.add_argument("--target", choices=("foreground", "background"), default="foreground")
    parser.add_argument("--dino-variant", choices=("dino3b", "dino3h", "dino3l"), default="dino3h")
    parser.add_argument("--clip-variant", choices=("clipb", "clipl"), default="clipb")
    parser.add_argument("--vision-backbone", choices=("dino", "clip"), default="dino")
    parser.add_argument("--ifp-root", default=DEFAULT_IFP_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-references", type=int, default=None,
        help="Use N reference image/mask pairs for a few-shot run.",
    )
    parser.add_argument(
        "--reference-seed", type=int, default=42,
        help="Random seed for few-shot reference selection, matching IFP's training script.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    clip, clip_processor = load_clip_model(args.clip_variant, device, args.ifp_root)
    raw_text = text_features(clip, clip_processor, args.prompts, device).mean(dim=0, keepdim=True)

    if args.vision_backbone == "dino":
        resize = args.resize or default_resize(args.dino_variant)
        patch_size = get_patch_size(args.dino_variant)
        resize = resize // patch_size * patch_size
        dino = load_dino_model(args.dino_variant, device, args.ifp_root)
        extract_tokens = lambda images: extract_patch_tokens(dino, images, resize, patch_size)
    else:
        patch_size = int(clip.config.vision_config.patch_size)
        crop_size = clip_processor.image_processor.crop_size
        resize = int(crop_size.get("height", crop_size.get("shortest_edge", 224)))
        extract_tokens = lambda images: extract_clip_patch_tokens(clip, clip_processor, images)

    dataset = MedicalReferenceDataset(
        Path(args.reference_root), extract_tokens, resize, patch_size, args.target, device,
        max_references=args.max_references,
        reference_seed=args.reference_seed,
    )
    # The frozen DINO model runs inside Dataset.__getitem__, so worker processes must stay disabled.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    sample_selected, _ = dataset[0]
    img_head = ProjectionHead(sample_selected.numel()).to(device)
    txt_head = ProjectionHead(raw_text.shape[-1]).to(device)
    optimizer = torch.optim.AdamW([*img_head.parameters(), *txt_head.parameters()], lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("ifp_medical_background_best.pt" if args.target == "background" else "ifp_medical_foreground_best.pt")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for selected, other in loader:
            selected, other = selected.to(device), other.to(device)
            text = txt_head(raw_text.expand(selected.shape[0], -1))
            loss = alignment_loss(img_head(selected), img_head(other), text, args.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / max(1, len(loader))
        print(f"epoch {epoch}/{args.epochs}: loss={average_loss:.4f}")
        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {
                    "format": "ifp_alignment_v1",
                    "kind": "medical",
                    "target": args.target,
                    "img_head": img_head.state_dict(),
                    "txt_head": txt_head.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "dino_variant": args.dino_variant,
                    "clip_variant": args.clip_variant,
                    "vision_backbone": args.vision_backbone,
                    "prompts": args.prompts,
                    "reference_images": [path.name for path in dataset.files],
                    "resize": resize,
                },
                output_path,
            )
    print(f"saved best checkpoint: {output_path}")


if __name__ == "__main__":
    main()
