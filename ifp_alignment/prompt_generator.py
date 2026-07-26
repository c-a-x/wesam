"""Runtime IFP image-text prompt generator for WeSAM."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .common import (
    DEFAULT_IFP_ROOT,
    default_resize,
    extract_patch_tokens,
    get_patch_size,
    load_clip_model,
    load_dino_model,
    projection_head_from_state_dict,
    text_features,
)


class IFPPointPromptGenerator:
    """Frozen IFP alignment model that emits WeSAM-compatible point prompts."""

    def __init__(
        self,
        device,
        *,
        checkpoint_path: str,
        background_checkpoint_path: str | None = None,
        text_prompts: Sequence[str],
        background_text_prompts: Sequence[str] | None = None,
        dino_variant: str = "dino3h",
        clip_variant: str = "clipb",
        ifp_root: str = DEFAULT_IFP_ROOT,
        resize: int | None = None,
        score_threshold: float | None = None,
        max_positive_points: int = 1,
        min_positive_points: int = 1,
        include_box: bool = False,
    ):
        self.device = torch.device(device)
        self.dino_variant = dino_variant
        self.patch_size = get_patch_size(dino_variant)
        self.resize = resize or default_resize(dino_variant)
        self.score_threshold = score_threshold
        self.max_positive_points = max_positive_points
        self.min_positive_points = min_positive_points
        self.include_box = include_box

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "img_head" not in checkpoint or "txt_head" not in checkpoint:
            raise KeyError(f"Alignment checkpoint must contain img_head and txt_head: {checkpoint_path}")

        self.dino = load_dino_model(dino_variant, self.device, ifp_root)
        self.clip, self.clip_processor = load_clip_model(clip_variant, self.device, ifp_root)
        self.img_head = projection_head_from_state_dict(checkpoint["img_head"]).to(self.device).eval()
        self.txt_head = projection_head_from_state_dict(checkpoint["txt_head"]).to(self.device).eval()
        for module in (self.img_head, self.txt_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        raw_text = text_features(self.clip, self.clip_processor, text_prompts, self.device)
        self.text_embedding = self.txt_head(raw_text).mean(dim=0)
        self.text_embedding = F.normalize(self.text_embedding, dim=0)

        self.background_img_head = None
        self.background_text_embedding = None
        if background_checkpoint_path:
            background_checkpoint = torch.load(
                background_checkpoint_path, map_location="cpu", weights_only=False
            )
            if "img_head" not in background_checkpoint or "txt_head" not in background_checkpoint:
                raise KeyError(
                    "Background alignment checkpoint must contain img_head and txt_head: "
                    f"{background_checkpoint_path}"
                )
            if not background_text_prompts:
                raise ValueError("background_text_prompts is required with a background checkpoint")
            self.background_img_head = projection_head_from_state_dict(
                background_checkpoint["img_head"]
            ).to(self.device).eval()
            background_txt_head = projection_head_from_state_dict(
                background_checkpoint["txt_head"]
            ).to(self.device).eval()
            for module in (self.background_img_head, background_txt_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            background_raw_text = text_features(
                self.clip, self.clip_processor, background_text_prompts, self.device
            )
            self.background_text_embedding = background_txt_head(background_raw_text).mean(dim=0)
            self.background_text_embedding = F.normalize(self.background_text_embedding, dim=0)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor):
        scores, background_scores, (grid_h, grid_w) = self.score_maps(images)
        _, _, height, width = images.shape

        prompts = []
        for index, image_scores in enumerate(scores):
            selected = self._select_indices(image_scores)
            coords = self._indices_to_coords(selected, grid_h, grid_w, height, width)
            all_coords = [coords]
            all_labels = [torch.ones(coords.shape[0], dtype=torch.int, device=self.device)]
            if background_scores is not None:
                negative = self._select_indices(background_scores[index], exclude=selected)
                negative_coords = self._indices_to_coords(
                    negative, grid_h, grid_w, height, width
                )
                all_coords.append(negative_coords)
                all_labels.append(
                    torch.zeros(negative_coords.shape[0], dtype=torch.int, device=self.device)
                )
            point_coords = torch.cat(all_coords, dim=0)
            point_labels = torch.cat(all_labels, dim=0)
            prompts.append(
                {
                    "in_points": (point_coords.unsqueeze(0), point_labels.unsqueeze(0)),
                    "in_box": self._box_from_points(coords, width, height),
                }
            )
        return prompts

    @torch.no_grad()
    def score_maps(self, images: torch.Tensor):
        """Return foreground and optional background scores on the DINO patch grid."""
        tokens, grid_shape = extract_patch_tokens(
            self.dino, images.to(self.device), self.resize, self.patch_size
        )
        projected = self.img_head(tokens)
        scores = F.cosine_similarity(projected, self.text_embedding.view(1, 1, -1), dim=-1)
        background_scores = None
        if self.background_img_head is not None:
            background_projected = self.background_img_head(tokens)
            background_scores = F.cosine_similarity(
                background_projected, self.background_text_embedding.view(1, 1, -1), dim=-1
            )
        return scores, background_scores, grid_shape

    @torch.no_grad()
    def iterative_teacher_pseudo_labels(
        self,
        teacher_model,
        images: torch.Tensor,
        *,
        max_iters: int = 3,
        teacher_iou_threshold: float | None = 0.5,
        min_new_pixels: int = 32,
        min_new_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.8,
        min_overlap_ratio: float = 0.05,
        max_new_area_ratio: float = 0.4,
    ):
        """Generate multi-round pseudo masks with one frozen teacher image encode.

        The IFP score map chooses a positive point for each round. The teacher
        SAM decoder is reused on the cached image embedding, and patch scores
        covered by an accepted mask are suppressed before the next round.
        ``teacher_iou_threshold`` applies to SAM2's raw IoU prediction, which
        is trained by WeSAM against the observed mask IoU.
        """
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")

        scores, _, (grid_h, grid_w) = self.score_maps(images)
        working_scores = scores.clone()
        _, _, height, width = images.shape
        image_area = height * width
        required_new_pixels = max(
            min_new_pixels, int(round(image_area * min_new_area_ratio))
        )

        image_embeddings = teacher_model.encode(images)
        pseudo_masks = [
            torch.zeros((1, height, width), device=images.device, dtype=torch.float32)
            for _ in range(images.shape[0])
        ]
        active = torch.ones(images.shape[0], dtype=torch.bool, device=images.device)

        for _ in range(max_iters):
            if not bool(active.any()):
                break

            selected = torch.argmax(working_scores, dim=1)
            batch_indices = torch.arange(images.shape[0], device=images.device)
            selected_scores = working_scores[batch_indices, selected].clone()
            # Always consume the selected patch. A rejected candidate should
            # lead to the next-best patch, not terminate the whole image.
            working_scores[batch_indices, selected] = -torch.inf
            prompts = []
            for batch_index in range(images.shape[0]):
                index = selected[batch_index]
                coords = self._indices_to_coords(
                    index.reshape(1), grid_h, grid_w, height, width
                )
                prompts.append({
                    "in_points": (
                        coords.unsqueeze(0),
                        torch.ones((1, 1), dtype=torch.int, device=images.device),
                    ),
                    "in_box": None,
                })

            masks, iou_predictions, res_masks = teacher_model.decode(
                (height, width), prompts
            )

            for batch_index in range(images.shape[0]):
                if not bool(active[batch_index]):
                    continue

                selected_score = selected_scores[batch_index]
                if self.score_threshold is not None and selected_score < self.score_threshold:
                    active[batch_index] = False
                    continue

                quality = float(iou_predictions[batch_index].reshape(-1)[0].item())
                if (
                    teacher_iou_threshold is not None
                    and quality < teacher_iou_threshold
                ):
                    continue

                current = (masks[batch_index] > 0).float()
                current_area = int(current.sum().item())
                if current_area == 0 or current_area / image_area > max_mask_area_ratio:
                    continue

                union_area = int(pseudo_masks[batch_index].sum().item())
                if union_area:
                    intersection = int(
                        (current * pseudo_masks[batch_index]).sum().item()
                    )
                    # A later mask must mostly belong to the existing object.
                    # The old denominator used min(area), allowing a tiny mask
                    # inside a large unrelated region to pass.
                    overlap_ratio = intersection / max(1, current_area)
                    if overlap_ratio < min_overlap_ratio:
                        continue

                    new_area = current_area - intersection
                    if new_area / max(1, union_area) > max_new_area_ratio:
                        continue

                new_pixels = current * (1.0 - pseudo_masks[batch_index])
                if int(new_pixels.sum().item()) < required_new_pixels:
                    continue

                candidate_union = torch.maximum(
                    pseudo_masks[batch_index], current
                )
                if candidate_union.sum().item() / image_area > max_mask_area_ratio:
                    continue
                pseudo_masks[batch_index] = candidate_union

                patch_mask = F.interpolate(
                    current.unsqueeze(0),
                    size=(grid_h, grid_w),
                    mode="area",
                )[0, 0] > 0.1
                working_scores[batch_index, patch_mask.flatten()] = -torch.inf

            # Avoid selecting an already exhausted image on the next round.
            active &= torch.isfinite(working_scores).any(dim=1)
            working_scores[~active] = -torch.inf

        return image_embeddings, pseudo_masks, iou_predictions, res_masks

    def _indices_to_coords(self, indices, grid_h, grid_w, height, width):
        rows = torch.div(indices, grid_w, rounding_mode="floor")
        cols = indices % grid_w
        return torch.stack(
            ((cols.float() + 0.5) * width / grid_w, (rows.float() + 0.5) * height / grid_h),
            dim=-1,
        )

    def _select_indices(self, scores: torch.Tensor, exclude: torch.Tensor | None = None) -> torch.Tensor:
        max_points = min(self.max_positive_points, scores.numel())
        ranked = torch.argsort(scores, descending=True)
        if exclude is not None and exclude.numel():
            keep = ~torch.isin(ranked, exclude)
            ranked = ranked[keep]
        if self.score_threshold is None:
            return ranked[:max(max_points, self.min_positive_points)]

        selected = ranked[scores[ranked] >= self.score_threshold][:max_points]
        if selected.numel() < self.min_positive_points:
            selected = ranked[: min(max_points, self.min_positive_points)]
        return selected

    def _box_from_points(self, points: torch.Tensor, width: int, height: int):
        if not self.include_box or points.shape[0] < 2:
            return None
        x0, y0 = points.amin(dim=0)
        x1, y1 = points.amax(dim=0)
        if x0 == x1:
            x1 = torch.clamp(x1 + 1, max=width - 1)
        if y0 == y1:
            y1 = torch.clamp(y1 + 1, max=height - 1)
        return torch.stack((x0, y0, x1, y1)).unsqueeze(0).float()
