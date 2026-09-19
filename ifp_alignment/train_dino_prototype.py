"""Build frozen DINO foreground/background patch prototypes from GT references."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ifp_alignment.common import DEFAULT_IFP_ROOT, default_resize, extract_patch_tokens, get_patch_size, load_dino_model
from ifp_alignment.train_medical import MedicalReferenceDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dino-variant", choices=("dino3b", "dino3h", "dino3l"), default="dino3h")
    parser.add_argument("--ifp-root", default=DEFAULT_IFP_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    patch_size = get_patch_size(args.dino_variant)
    resize = default_resize(args.dino_variant)
    dino = load_dino_model(args.dino_variant, device, args.ifp_root)
    extract_tokens = lambda images: extract_patch_tokens(dino, images, resize, patch_size)
    dataset = MedicalReferenceDataset(
        Path(args.reference_root), extract_tokens, resize, patch_size, "foreground", device
    )

    foreground_sum = None
    background_sum = None
    for index in range(len(dataset)):
        foreground, background = dataset[index]
        foreground_sum = foreground if foreground_sum is None else foreground_sum + foreground
        background_sum = background if background_sum is None else background_sum + background

    foreground_prototype = torch.nn.functional.normalize(foreground_sum, dim=0)
    background_prototype = torch.nn.functional.normalize(background_sum, dim=0)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "dino_foreground_background_prototype_v1",
            "dino_variant": args.dino_variant,
            "resize": resize,
            "patch_size": patch_size,
            "reference_count": len(dataset),
            "foreground_prototype": foreground_prototype,
            "background_prototype": background_prototype,
        },
        output_path,
    )
    print(f"saved DINO prototypes from {len(dataset)} references: {output_path}")


if __name__ == "__main__":
    main()
