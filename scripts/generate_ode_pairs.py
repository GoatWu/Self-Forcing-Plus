from __future__ import annotations

import _bootstrap  # noqa: F401
import sys

import draccus

import generate_ode_pairs_legacy as legacy
from sfp.utils.accelerate_runtime import AccelerateConfig, create_accelerator
from sfp.utils.config import GenerateOdePairsConfig
from sfp.utils.seeding import set_seed


@draccus.wrap()
def main(cfg: GenerateOdePairsConfig) -> None:
    accelerator = create_accelerator(
        AccelerateConfig(mixed_precision=cfg.runtime.mixed_precision)
    )
    set_seed(cfg.runtime.seed + accelerator.process_index)

    argv = [
        "generate_ode_pairs_legacy.py",
        "--output_folder",
        cfg.output_folder,
        "--caption_path",
        cfg.caption_path,
        "--guidance_scale",
        str(cfg.guidance_scale),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        legacy.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
