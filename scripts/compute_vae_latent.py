from __future__ import annotations

import _bootstrap  # noqa: F401
import sys

import draccus

import compute_vae_latent_legacy as legacy
from sfp.utils.accelerate_runtime import AccelerateConfig, create_accelerator
from sfp.utils.config import ComputeVaeLatentConfig
from sfp.utils.seeding import set_seed


@draccus.wrap()
def main(cfg: ComputeVaeLatentConfig) -> None:
    accelerator = create_accelerator(
        AccelerateConfig(mixed_precision=cfg.runtime.mixed_precision)
    )
    set_seed(cfg.runtime.seed + accelerator.process_index)

    argv = [
        "compute_vae_latent_legacy.py",
        "--input_video_folder",
        cfg.input_video_folder,
        "--output_latent_folder",
        cfg.output_latent_folder,
        "--model_name",
        cfg.model_name,
        "--prompt_folder",
        cfg.prompt_folder,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        legacy.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
