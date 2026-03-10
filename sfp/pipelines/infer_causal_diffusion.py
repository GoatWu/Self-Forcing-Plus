from __future__ import annotations

import torch
from easydict import EasyDict

from sfp.legacy.pipeline import CausalDiffusionInferencePipeline
from sfp.pipelines.infer_common import load_config


def build_pipeline(
    config_path: str,
    default_config_path: str,
    checkpoint_path: str,
    use_ema: bool,
    device: torch.device,
):
    cfg = EasyDict(load_config(config_path, default_config_path))
    pipeline = CausalDiffusionInferencePipeline(cfg, device=device)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        key = "generator_ema" if use_ema else "generator"
        pipeline.generator.load_state_dict(state_dict[key])
    return pipeline.to(device=device, dtype=torch.bfloat16)
