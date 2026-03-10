from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sfp.model.wan_wrapper import WanModelConfig


@dataclass
class RuntimeConfig:
    seed: int = 0
    mixed_precision: Optional[str] = None


@dataclass
class TrainCommandConfig:
    legacy_config_path: str
    default_config_path: str = "configs/default_config.yaml"
    logdir: str = ""
    wandb_save_dir: str = ""
    no_save: bool = False
    no_visualize: bool = False
    disable_wandb: bool = False
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: WanModelConfig = field(default_factory=WanModelConfig)


@dataclass
class InferenceCommandConfig:
    legacy_config_path: str
    default_config_path: str = "configs/default_config.yaml"
    checkpoint_path: str = ""
    data_path: str = ""
    extended_prompt_path: str = ""
    output_folder: str = "outputs"
    num_output_frames: int = 21
    i2v: bool = False
    use_ema: bool = False
    num_samples: int = 1
    save_with_index: bool = False
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: WanModelConfig = field(default_factory=WanModelConfig)


@dataclass
class DemoCommandConfig:
    legacy_config_path: str = "configs/self_forcing_dmd.yaml"
    default_config_path: str = "configs/default_config.yaml"
    checkpoint_path: str = "./checkpoints/self_forcing_dmd.pt"
    host: str = "0.0.0.0"
    port: int = 5001
    trt: bool = False
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: WanModelConfig = field(default_factory=WanModelConfig)


@dataclass
class ComputeVaeLatentConfig:
    input_video_folder: str
    output_latent_folder: str
    prompt_folder: str
    model_name: str = "Wan2.1-T2V-14B"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class GenerateOdePairsConfig:
    output_folder: str
    caption_path: str
    guidance_scale: float = 6.0
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class CreateLmdbShardsConfig:
    data_path: str
    prompt_path: str = ""
    video_path: str = ""
    lmdb_path: str = ""
    num_shards: int = 16


@dataclass
class CreateLmdbIterativeConfig:
    data_path: str
    lmdb_path: str


@dataclass
class DMDTrainConfig:
    """Typed config for DMD distillation training (high-noise target)."""

    # Model names (paths under wan_models/)
    generator_name: str = "Wan2.2-T2V-A14B/high_noise_model"
    real_score_name: str = "Wan2.2-T2V-A14B/high_noise_model"
    fake_score_name: str = "Wan2.2-T2V-A14B/high_noise_model"

    # Architecture
    generator_is_causal: bool = False
    timestep_shift: float = 5.0
    target: str = "high_noise"
    boundary_step: int = 500
    gradient_checkpointing: bool = True
    num_train_timestep: int = 1000

    # DMD backward simulation
    denoising_step_list: list[int] = field(default_factory=lambda: [1000, 750])

    # Training schedule
    dfake_gen_update_ratio: int = 5
    guidance_scale: float = 4.0

    # Generator optimizer
    lr: float = 2e-6
    beta1: float = 0.0
    beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm_generator: float = 10.0

    # Critic optimizer
    lr_critic: float = 4e-7
    beta1_critic: float = 0.0
    beta2_critic: float = 0.999
    max_grad_norm_critic: float = 10.0

    # Data
    data_type: str = "text_folder"
    data_path: str = "prompts/good_prompts/"
    data_max_count: int = 200000
    batch_size: int = 1
    negative_prompt: str = ""

    # EMA
    ema_weight: float = 0.99
    ema_start_step: int = 200

    # Shape: [B, F, C, H, W]
    image_or_video_shape: list[int] = field(
        default_factory=lambda: [1, 21, 16, 60, 104]
    )

    # Logging / checkpointing
    logdir: str = "logs/dmd_distillation"
    log_iters: int = 200
    gc_interval: int = 100
    no_save: bool = False
    disable_wandb: bool = True
    wandb_host: str = ""
    wandb_key: str = ""
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_save_dir: str = ""
    config_name: str = "dmd_distillation"

    # Resume
    resume_ckpt: str = ""
    resume_step: int = 0

    # Runtime
    seed: int = 0
    mixed_precision: Optional[str] = None


@dataclass
class OdeTrainConfig:
    """Typed config for ODE regression training (CausVid Sec 4.3)."""

    # Model
    generator_name: str = "Wan2.1-T2V-1.3B"
    timestep_shift: float = 5.0
    gradient_checkpointing: bool = True
    num_frame_per_block: int = 3
    independent_first_frame: bool = False
    i2v: bool = False

    # ODE trajectory timesteps
    denoising_step_list: list[int] = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )

    # Optimizer
    lr: float = 2e-6
    beta1: float = 0.0
    beta2: float = 0.999
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0

    # Data (LMDB)
    data_path: str = ""
    max_pair: int = 100_000_000
    batch_size: int = 1

    # Logging / checkpointing
    logdir: str = "logs/ode_regression"
    log_iters: int = 50
    gc_interval: int = 100
    no_save: bool = False
    no_visualize: bool = True
    disable_wandb: bool = True
    wandb_host: str = ""
    wandb_key: str = ""
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_save_dir: str = ""
    config_name: str = "ode_regression"

    # Resume
    generator_ckpt: str = ""
    resume_step: int = 0

    # Runtime
    seed: int = 0
    mixed_precision: Optional[str] = None
