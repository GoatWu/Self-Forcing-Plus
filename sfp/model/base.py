from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """Base model configuration marker type."""


@dataclass
class BuildContext:
    """Runtime context passed to model builders."""

    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
