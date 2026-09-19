"""IFP-style image-text alignment training and prompt generation."""

from .common import ProjectionHead
from .ablation_prompt_generators import (
    CLIPOnlyPointPromptGenerator,
    DINOPrototypePointPromptGenerator,
    GTOraclePointPromptGenerator,
    NoPromptGenerator,
)
from .prompt_generator import IFPPointPromptGenerator


def build_prompt_generator(backend: str, device, **kwargs):
    if backend == "ifp":
        return IFPPointPromptGenerator(device, **kwargs)
    if backend == "dino-prototype":
        return DINOPrototypePointPromptGenerator(
            device,
            checkpoint_path=kwargs["checkpoint_path"],
            dino_variant=kwargs["dino_variant"],
            ifp_root=kwargs["ifp_root"],
        )
    if backend == "clip-only":
        return CLIPOnlyPointPromptGenerator(
            device,
            text_prompts=kwargs["text_prompts"],
            clip_variant=kwargs["clip_variant"],
            ifp_root=kwargs["ifp_root"],
        )
    if backend == "no-prompt":
        return NoPromptGenerator(device)
    if backend == "gt-oracle":
        return GTOraclePointPromptGenerator()
    raise ValueError(f"Unsupported prompt backend: {backend}")


__all__ = [
    "build_prompt_generator",
    "CLIPOnlyPointPromptGenerator",
    "DINOPrototypePointPromptGenerator",
    "GTOraclePointPromptGenerator",
    "IFPPointPromptGenerator",
    "NoPromptGenerator",
    "ProjectionHead",
]
