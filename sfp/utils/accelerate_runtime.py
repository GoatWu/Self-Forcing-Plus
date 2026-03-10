from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from accelerate import Accelerator


@dataclass
class AccelerateConfig:
    mixed_precision: Optional[str] = None


def create_accelerator(cfg: Optional[AccelerateConfig] = None) -> Accelerator:
    """Create a process-aware Accelerator runtime used by v2 scripts."""
    cfg = cfg or AccelerateConfig()
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    return accelerator
