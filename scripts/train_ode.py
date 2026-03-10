"""ODE regression training script — self-contained, functional, accelerate-based.

Replaces the legacy class hierarchy (BaseModel → ODERegression + Trainer)
with flat functions and a single training loop.  Causal generator, LMDB data.

See CausVid Sec 4.3 (https://arxiv.org/abs/2412.07772) for algorithm details.

Usage:
    accelerate launch scripts/train_ode.py \
        --config_path configs/v2/train_ode.yaml
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Tuple

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import draccus
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from sfp.model.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from sfp.pipelines.train_ode import prepare_ode_input, sample_timesteps_per_block
from sfp.utils.accelerate_runtime import AccelerateConfig, create_accelerator
from sfp.utils.config import OdeTrainConfig
from sfp.utils.data import ODERegressionLMDBDataset, cycle
from sfp.utils.seeding import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def compute_ode_loss(
    generator: WanDiffusionWrapper,
    ode_latent: torch.Tensor,
    conditional_dict: dict,
    denoising_step_list: torch.Tensor,
    num_frame_per_block: int,
    independent_first_frame: bool,
    i2v: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """Compute ODE regression loss (CausVid Sec 4.3).

    1. target = ode_latent[:, -1] (clean latent)
    2. Sample per-block timestep indices
    3. Gather noisy input from ODE trajectories
    4. Generator forward pass
    5. MSE loss with mask (exclude timestep=0 frames)
    """
    target = ode_latent[:, -1]  # [B, F, C, H, W]
    batch_size, _, num_frames = ode_latent.shape[:3]
    num_steps = len(denoising_step_list)

    # Sample per-block timestep indices
    indices = sample_timesteps_per_block(
        min_index=0,
        max_index=num_steps,
        batch_size=batch_size,
        num_frames=num_frames,
        num_frame_per_block=num_frame_per_block,
        independent_first_frame=independent_first_frame,
        device=device,
    )

    # Gather noisy latents at sampled indices
    noisy_input, raw_timesteps = prepare_ode_input(
        ode_latent=ode_latent,
        denoising_step_list=denoising_step_list,
        timestep_indices=indices,
        i2v=i2v,
    )

    # Convert raw timestep values to timestep_id for WanDiffusionWrapper
    # raw=1000 → id=0, raw=750 → id=250, raw=0 → id=1000
    timestep_id = 1000 - raw_timesteps

    _, pred = generator(
        noisy_image_or_video=noisy_input,
        conditional_dict=conditional_dict,
        timestep_id=timestep_id,
    )

    # Mask: exclude frames where timestep=0 (already clean)
    mask = raw_timesteps != 0  # [B, F]

    if mask.any():
        loss = F.mse_loss(pred[mask], target[mask], reduction="mean")
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    log_dict = {
        "timestep": raw_timesteps.float().mean().detach(),
        "mask_ratio": mask.float().mean().detach(),
    }
    return loss, log_dict


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_generator(cfg: OdeTrainConfig):
    """Instantiate causal generator and text encoder for ODE training."""
    generator = WanDiffusionWrapper(
        model_name=cfg.generator_name,
        timestep_shift=cfg.timestep_shift,
        is_causal=True,
        target="high_noise",
        timestep_bound=0,
    )
    generator.model.requires_grad_(True)

    # Set per-block causal config on the underlying CausalWanModel
    if cfg.num_frame_per_block > 1:
        generator.model.num_frame_per_block = cfg.num_frame_per_block
    if cfg.independent_first_frame:
        generator.model.independent_first_frame = True

    if cfg.gradient_checkpointing:
        generator.enable_gradient_checkpointing()

    text_encoder = WanTextEncoder(model_name=cfg.generator_name)
    text_encoder.requires_grad_(False)

    return generator, text_encoder


def create_dataset(cfg: OdeTrainConfig) -> ODERegressionLMDBDataset:
    """Create ODE regression LMDB dataset."""
    return ODERegressionLMDBDataset(cfg.data_path, max_pair=cfg.max_pair)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(accelerator, generator, cfg: OdeTrainConfig, step: int):
    """Gather state dict and save on main process."""
    gen_state = accelerator.get_state_dict(generator)

    if accelerator.is_main_process:
        ckpt_dir = os.path.join(cfg.logdir, f"checkpoint_model_{step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"generator": gen_state}, os.path.join(ckpt_dir, "model.pt"))
        logger.info("Saved checkpoint to %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: OdeTrainConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    accelerator = create_accelerator(
        AccelerateConfig(mixed_precision=cfg.mixed_precision)
    )
    device = accelerator.device
    dtype = torch.bfloat16 if cfg.mixed_precision else torch.float32

    # Seed — broadcast a random seed if seed==0
    if cfg.seed == 0:
        random_seed = torch.randint(0, 10_000_000, (1,), device=device)
        if dist.is_initialized():
            dist.broadcast(random_seed, src=0)
        cfg.seed = random_seed.item()
    set_seed(cfg.seed + accelerator.process_index)

    # W&B init
    if accelerator.is_main_process and not cfg.disable_wandb:
        wandb.login(host=cfg.wandb_host, key=cfg.wandb_key)
        wandb.init(
            config=draccus.encode(cfg),
            name=cfg.config_name,
            mode="online",
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            dir=cfg.wandb_save_dir or None,
        )

    # Models
    generator, text_encoder = create_generator(cfg)

    # Prepare with accelerate
    generator, text_encoder = accelerator.prepare(generator, text_encoder)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    # Dataset
    dataset = create_dataset(cfg)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, drop_last=True
    )
    dataloader = cycle(
        torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=8
        )
    )

    if accelerator.is_main_process:
        logger.info("Dataset size: %d", len(dataset))

    # Pre-compute denoising step tensor
    denoising_step_list = torch.tensor(
        cfg.denoising_step_list, dtype=torch.long
    )

    # Resume
    step = cfg.resume_step
    if cfg.generator_ckpt:
        if accelerator.is_main_process:
            logger.info("Resuming from %s", cfg.generator_ckpt)
        ckpt = torch.load(cfg.generator_ckpt, map_location="cpu")
        generator.load_state_dict(ckpt["generator"], strict=True)

    # Training loop
    start_step = step
    previous_time = None

    while True:
        if accelerator.is_main_process:
            logger.info("Step %d", step)

        batch = next(dataloader)
        ode_latent = batch["ode_latent"].to(device=device, dtype=dtype)
        prompts = batch["prompts"]

        with torch.no_grad():
            conditional_dict = text_encoder(text_prompts=prompts)

        optimizer.zero_grad(set_to_none=True)
        loss, log_dict = compute_ode_loss(
            generator=generator,
            ode_latent=ode_latent,
            conditional_dict=conditional_dict,
            denoising_step_list=denoising_step_list,
            num_frame_per_block=cfg.num_frame_per_block,
            independent_first_frame=cfg.independent_first_frame,
            i2v=cfg.i2v,
            device=device,
        )

        accelerator.backward(loss)
        if hasattr(generator, "clip_grad_norm_"):
            grad_norm = generator.clip_grad_norm_(cfg.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in generator.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
        optimizer.step()

        step += 1

        # Checkpointing
        if (
            not cfg.no_save
            and (step - start_step) > 0
            and step % cfg.log_iters == 0
        ):
            torch.cuda.empty_cache()
            _save_checkpoint(accelerator, generator, cfg, step)
            torch.cuda.empty_cache()

        # Logging
        if accelerator.is_main_process:
            wandb_dict = {
                "generator_loss": loss.item(),
                "generator_grad_norm": grad_norm.item()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm,
                "avg_timestep": log_dict["timestep"].item(),
            }
            logger.info("loss=%.6f", loss.item())

            if not cfg.disable_wandb:
                wandb.log(wandb_dict, step=step)

            current_time = time.time()
            if previous_time is not None and not cfg.disable_wandb:
                wandb.log({"per_iteration_time": current_time - previous_time}, step=step)
            previous_time = current_time

        # Periodic GC
        if step % cfg.gc_interval == 0:
            if accelerator.is_main_process:
                logger.info("Running GC")
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
