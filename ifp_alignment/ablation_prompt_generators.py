"""Single-encoder point-prompt baselines for IFP ablations."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .common import (
    DEFAULT_IFP_ROOT,
    default_resize,
    extract_clip_patch_tokens,
    extract_patch_tokens,
    get_patch_size,
    load_clip_model,
    load_dino_model,
    text_features,
)


class _TopPatchPromptGenerator:
    def __init__(self, device):
        self.device = torch.device(device)

    def _prompts_from_scores(self, scores: torch.Tensor, grid_h: int, grid_w: int, height: int, width: int):
        indices = torch.argmax(scores, dim=1)
        rows = torch.div(indices, grid_w, rounding_mode="floor")
        cols = indices % grid_w
        coords = torch.stack(
            ((cols.float() + 0.5) * width / grid_w, (rows.float() + 0.5) * height / grid_h),
            dim=-1,
        )
        return [
            {
                "in_points": (
                    point.reshape(1, 1, 2),
                    torch.ones((1, 1), dtype=torch.int, device=self.device),
                ),
                "in_box": None,
            }
            for point in coords
        ]

    @torch.no_grad()
    def iterative_teacher_pseudo_labels(
        self,
        teacher_model,
        images: torch.Tensor,
        *,
        teacher_iou_threshold: float | None = 0.5,
        min_new_pixels: int = 32,
        min_new_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.8,
        **_: object,
    ):
        """One-pass high-confidence Teacher masks for single-encoder baselines."""

        _, _, height, width = images.shape
        image_area = height * width
        min_pixels = max(min_new_pixels, int(round(image_area * min_new_area_ratio)))
        image_embeddings = teacher_model.encode(images)
        masks, iou_predictions, res_masks = teacher_model.decode((height, width), self(images))
        pseudo_masks = []
        for mask, quality in zip(masks, iou_predictions):
            binary = (mask > 0).float()
            area = int(binary.sum().item())
            accepted = (
                area >= min_pixels
                and area / image_area <= max_mask_area_ratio
                and (
                    teacher_iou_threshold is None
                    or float(quality.reshape(-1)[0].item()) >= teacher_iou_threshold
                )
            )
            pseudo_masks.append(binary if accepted else torch.zeros_like(binary))
        return image_embeddings, pseudo_masks, iou_predictions, res_masks


class DINOPrototypePointPromptGenerator(_TopPatchPromptGenerator):
    """DINO-only baseline: foreground-vs-background prototype patch scoring."""

    def __init__(
        self,
        device,
        *,
        checkpoint_path: str,
        dino_variant: str = "dino3h",
        ifp_root: str = DEFAULT_IFP_ROOT,
    ):
        super().__init__(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        try:
            foreground = checkpoint["foreground_prototype"]
            background = checkpoint["background_prototype"]
        except KeyError as exc:
            raise KeyError(f"DINO prototype checkpoint is missing {exc.args[0]}: {checkpoint_path}") from exc
        self.foreground = F.normalize(foreground.to(self.device), dim=-1)
        self.background = F.normalize(background.to(self.device), dim=-1)
        self.patch_size = get_patch_size(dino_variant)
        self.resize = int(checkpoint.get("resize", default_resize(dino_variant)))
        self.dino = load_dino_model(dino_variant, self.device, ifp_root)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        tokens, (grid_h, grid_w) = extract_patch_tokens(
            self.dino, images.to(self.device), self.resize, self.patch_size
        )
        foreground_score = F.cosine_similarity(tokens, self.foreground.view(1, 1, -1), dim=-1)
        background_score = F.cosine_similarity(tokens, self.background.view(1, 1, -1), dim=-1)
        scores = foreground_score - background_score
        _, _, height, width = images.shape
        return self._prompts_from_scores(scores, grid_h, grid_w, height, width)


class CLIPOnlyPointPromptGenerator(_TopPatchPromptGenerator):
    """CLIP-only baseline: CLIP vision patch tokens scored by CLIP text features."""

    def __init__(
        self,
        device,
        *,
        text_prompts: Sequence[str],
        clip_variant: str = "clipl",
        ifp_root: str = DEFAULT_IFP_ROOT,
    ):
        super().__init__(device)
        self.clip, self.clip_processor = load_clip_model(clip_variant, self.device, ifp_root)
        embedding = text_features(self.clip, self.clip_processor, text_prompts, self.device).mean(dim=0)
        self.text_embedding = F.normalize(embedding, dim=-1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        tokens, (grid_h, grid_w) = extract_clip_patch_tokens(
            self.clip, self.clip_processor, images.to(self.device)
        )
        scores = F.cosine_similarity(tokens, self.text_embedding.view(1, 1, -1), dim=-1)
        _, _, height, width = images.shape
        return self._prompts_from_scores(scores, grid_h, grid_w, height, width)


class NoPromptGenerator(_TopPatchPromptGenerator):
    """Teacher-Student baseline that supplies no points or boxes to SAM2."""

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        # Deliberately use None rather than an empty prompt dictionary. This
        # reaches SAM2's prompt encoder as points=None, boxes=None, masks=None.
        return None


class GTOraclePointPromptGenerator:
    """One positive interior point from each sample's ground-truth mask."""

    requires_gt_masks = True

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, gt_masks) -> list[dict]:
        prompts = []
        for gt_mask in gt_masks:
            mask = gt_mask.to(images.device, dtype=torch.float32)
            while mask.ndim > 2:
                mask = mask.amax(dim=0)
            foreground = mask > 0
            if not bool(foreground.any()):
                raise ValueError("GT-oracle prompting requires a non-empty foreground mask.")
            eroded = -F.max_pool2d(-foreground[None, None].float(), 7, 1, 3)[0, 0] > 0.5
            candidates = torch.nonzero(foreground & eroded, as_tuple=False)
            if candidates.numel() == 0:
                candidates = torch.nonzero(foreground, as_tuple=False)
            center = candidates.float().mean(dim=0, keepdim=True)
            point_yx = candidates[((candidates.float() - center) ** 2).sum(dim=1).argmin()]
            point_xy = point_yx[[1, 0]].float().reshape(1, 1, 2)
            labels = torch.ones((1, 1), dtype=torch.int, device=images.device)
            prompts.append({"in_points": (point_xy, labels), "in_box": None})
        return prompts
