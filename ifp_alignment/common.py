"""Shared frozen-backbone utilities for the migrated IFP modules."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# IFP alignment assets are kept in this project so training and export do not
# depend on the separate IFP repository.
DEFAULT_IFP_ROOT = str(Path(__file__).resolve().parents[1] / "assets")

DINO3_MODELS = {
    "dino3b": ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "dino3h": ("dinov3_vith16plus", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
    "dino3l": ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
}

CLIP_PATHS = {
    "clipb": "CLIP/clip-vit-base-patch16",
    "clipl": "CLIP/clip-vit-large-patch14",
}


class ProjectionHead(nn.Module):
    """The two-layer normalized projection head used by IFP checkpoints."""

    def __init__(self, in_dim: int, out_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


def resolve_clip_path(variant_or_path: str, ifp_root: str = DEFAULT_IFP_ROOT) -> str:
    path = CLIP_PATHS.get(variant_or_path, variant_or_path)
    if not os.path.isabs(path):
        path = os.path.join(ifp_root, path)
    return path


def get_patch_size(variant: str) -> int:
    if variant.startswith("dino3"):
        return 16
    if variant.startswith("dino2"):
        return 14
    raise ValueError(f"Unsupported DINO variant: {variant}")


def default_resize(variant: str) -> int:
    return 512 if variant.startswith("dino3") else 518


def load_dino_model(variant: str, device: torch.device | str, ifp_root: str = DEFAULT_IFP_ROOT):
    """Load a local IFP DINO backbone. The migrated generic pipeline uses DINOv3."""

    if variant not in DINO3_MODELS:
        raise ValueError(
            f"Unsupported DINO variant {variant!r}. Available migrated variants: "
            f"{', '.join(DINO3_MODELS)}"
        )

    model_name, weight_name = DINO3_MODELS[variant]
    dino_root = Path(ifp_root) / "DINO" / "dinov3"
    weights = dino_root / weight_name
    if not weights.is_file():
        raise FileNotFoundError(f"DINO weights not found: {weights}")

    model = torch.hub.load(
        repo_or_dir=str(dino_root),
        model=model_name,
        source="local",
        pretrained=False,
    )
    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_clip_model(variant_or_path: str, device: torch.device | str, ifp_root: str = DEFAULT_IFP_ROOT):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("IFP alignment requires transformers. Install requirements.txt.") from exc

    clip_path = resolve_clip_path(variant_or_path, ifp_root)
    if not os.path.isdir(clip_path):
        raise FileNotFoundError(f"CLIP model directory not found: {clip_path}")
    model = CLIPModel.from_pretrained(clip_path).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, CLIPProcessor.from_pretrained(clip_path, use_fast=False)


@torch.no_grad()
def text_features(clip_model, clip_processor, prompts: Sequence[str], device: torch.device | str) -> torch.Tensor:
    if not prompts:
        raise ValueError("At least one text prompt is required.")
    inputs = clip_processor(text=list(prompts), return_tensors="pt", padding=True).to(device)
    features = clip_model.get_text_features(**inputs)
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    return F.normalize(features, dim=-1)


@torch.no_grad()
def extract_patch_tokens(model, images: torch.Tensor, resize: int, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Extract a batch of patch tokens while preserving IFP's fixed patch grid."""

    images = F.interpolate(images.float(), size=(resize, resize), mode="bilinear", align_corners=False)
    expected_h = resize // patch_size
    expected_w = resize // patch_size
    expected_n = expected_h * expected_w

    if hasattr(model, "get_intermediate_layers"):
        tokens = model.get_intermediate_layers(images, n=1)[0]
    else:
        output = model.forward_features(images)
        if not isinstance(output, dict) or "x_norm_patchtokens" not in output:
            raise ValueError("DINO model does not expose patch tokens")
        tokens = output["x_norm_patchtokens"]

    if tokens.shape[1] == expected_n + 1:
        tokens = tokens[:, 1:]
    if tokens.shape[1] < expected_n:
        padding = tokens[:, -1:].expand(-1, expected_n - tokens.shape[1], -1)
        tokens = torch.cat((tokens, padding), dim=1)
    return tokens[:, :expected_n], (expected_h, expected_w)


def pixel_mask_to_patch_mask(mask, resize: int, patch_size: int, threshold: float = 0.5) -> torch.Tensor:
    """Map a binary pixel mask to the DINO patch grid."""

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(resize, resize), mode="nearest").squeeze()
    grid_h = resize // patch_size
    grid_w = resize // patch_size
    blocks = mask.reshape(grid_h, patch_size, grid_w, patch_size)
    return blocks.mean(dim=(1, 3)) >= threshold


def projection_head_from_state_dict(state_dict: dict[str, torch.Tensor]) -> ProjectionHead:
    first_weight = state_dict["net.0.weight"]
    last_weight = state_dict["net.2.weight"]
    head = ProjectionHead(
        in_dim=first_weight.shape[1],
        out_dim=last_weight.shape[0],
        hidden_dim=first_weight.shape[0],
    )
    head.load_state_dict(state_dict)
    return head
