from __future__ import annotations

from sfp.legacy.trainer.gan import Trainer as LegacyTrainer

from sfp.utils.config import TrainCommandConfig
from sfp.trainers.common import load_train_config


def run(cfg: TrainCommandConfig) -> None:
    runtime_cfg = load_train_config(cfg.legacy_config_path, cfg.default_config_path)
    runtime_cfg.no_save = cfg.no_save
    runtime_cfg.no_visualize = cfg.no_visualize
    runtime_cfg.logdir = cfg.logdir
    runtime_cfg.wandb_save_dir = cfg.wandb_save_dir
    runtime_cfg.disable_wandb = cfg.disable_wandb
    trainer = LegacyTrainer(runtime_cfg)
    trainer.train()
