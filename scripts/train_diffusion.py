from __future__ import annotations

import _bootstrap  # noqa: F401

import draccus

from sfp.trainers import diffusion
from sfp.utils.accelerate_runtime import AccelerateConfig, create_accelerator
from sfp.utils.config import TrainCommandConfig
from sfp.utils.seeding import set_seed


@draccus.wrap()
def main(cfg: TrainCommandConfig) -> None:
    accelerator = create_accelerator(
        AccelerateConfig(mixed_precision=cfg.runtime.mixed_precision)
    )
    set_seed(cfg.runtime.seed + accelerator.process_index)
    diffusion.run(cfg)


if __name__ == "__main__":
    main()
