"""IFP-style image-text alignment training and prompt generation."""

from .common import ProjectionHead
from .prompt_generator import IFPPointPromptGenerator

__all__ = ["IFPPointPromptGenerator", "ProjectionHead"]
