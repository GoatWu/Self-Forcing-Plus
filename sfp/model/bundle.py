from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelBundle:
    """Container for model-facing components used by trainers/pipelines."""

    generator: torch.nn.Module
    text_encoder: torch.nn.Module
    vae: torch.nn.Module
    scheduler: object
    image_encoder: Optional[torch.nn.Module] = None
    high_noise_model: Optional[torch.nn.Module] = None
