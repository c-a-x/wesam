"""Prompt robustness policy shared by training and evaluation."""

from __future__ import annotations

from typing import Any

import torch


def apply_prompt_policy(
    prompts: list[dict[str, Any] | None] | None, policy: Any,
    image_shape: tuple[int, int], *, training: bool,
) -> list[dict[str, Any] | None] | None:
    """Apply confidence fallback and optional train-time point corruption."""
    if prompts is None:
        return None
    threshold = getattr(policy, "fallback_confidence_threshold", None)
    jitter = float(getattr(policy, "train_point_jitter_pixels", 0.0)) if training else 0.0
    dropout = float(getattr(policy, "train_prompt_dropout", 0.0)) if training else 0.0
    if not 0.0 <= dropout <= 1.0:
        raise ValueError("train_prompt_dropout must be in [0, 1]")

    height, width = image_shape
    result: list[dict[str, Any] | None] = []
    for prompt in prompts:
        if prompt is None:
            result.append(None)
            continue
        confidence = prompt.get("confidence")
        if threshold is not None and confidence is not None and float(confidence) < float(threshold):
            result.append(None)
            continue
        points = prompt.get("in_points")
        if dropout and points is not None and bool(torch.rand((), device=points[0].device) < dropout):
            result.append(None)
            continue
        updated = dict(prompt)
        if points is not None and jitter:
            coords, labels = points
            coords = coords.clone()
            coords.add_(torch.empty_like(coords).uniform_(-jitter, jitter))
            coords[..., 0].clamp_(0, width - 1)
            coords[..., 1].clamp_(0, height - 1)
            updated["in_points"] = (coords, labels)
        result.append(updated)
    return result
