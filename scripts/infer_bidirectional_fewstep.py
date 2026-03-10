from __future__ import annotations

import _bootstrap  # noqa: F401

import draccus

from sfp.pipelines.infer_bidirectional_fewstep import build_pipeline
from sfp.pipelines.infer_common import (
    build_dataset,
    get_dataloader,
    run_generation_loop,
)
from sfp.utils.accelerate_runtime import AccelerateConfig, create_accelerator
from sfp.utils.config import InferenceCommandConfig
from sfp.utils.seeding import set_seed


@draccus.wrap()
def main(cfg: InferenceCommandConfig) -> None:
    accelerator = create_accelerator(
        AccelerateConfig(mixed_precision=cfg.runtime.mixed_precision)
    )
    set_seed(cfg.runtime.seed + accelerator.process_index)

    if cfg.i2v and accelerator.num_processes > 1:
        raise ValueError(
            "I2V currently supports single-process inference in v2 scripts."
        )

    pipeline = build_pipeline(
        cfg.legacy_config_path,
        cfg.default_config_path,
        cfg.checkpoint_path,
        cfg.use_ema,
        accelerator.device,
    )
    dataset = build_dataset(cfg.data_path, cfg.i2v, cfg.extended_prompt_path)
    dataloader = get_dataloader(dataset, accelerator)
    run_generation_loop(
        pipeline,
        dataloader,
        accelerator,
        cfg.output_folder,
        cfg.i2v,
        cfg.num_output_frames,
        cfg.num_samples,
        cfg.save_with_index,
    )


if __name__ == "__main__":
    main()
