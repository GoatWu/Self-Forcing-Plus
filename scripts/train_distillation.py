"""DMD distillation training script — self-contained, functional, accelerate-based.

Replaces the legacy class hierarchy (BaseModel → SelfForcingModel → DMD + Trainer)
with flat functions and a single training loop.  High-noise target only.

Usage:
    accelerate launch scripts/train_distillation.py \
        --config_path configs/v2/train_distillation.yaml
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
from sfp.pipelines.train_dmd import (
    backward_simulate,
    compute_timestep_bound,
    encode_prompts,
    prepare_latents,
)
from sfp.utils.accelerate_runtime import AccelerateConfig, create_accelerator
from sfp.utils.config import DMDTrainConfig
from sfp.utils.data import TextDataset, TextFolderDataset, cycle
from sfp.utils.ema import EMATracker
from sfp.utils.seeding import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional helpers (extracted from legacy DMD + BaseModel)
# ---------------------------------------------------------------------------


def sample_timesteps(
    min_t: int,
    max_t: int,
    batch_size: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample uniform timesteps shared across all frames.

    From BaseModel._get_timestep with uniform_timestep=True.
    Returns shape [B, F].
    """
    return torch.randint(
        min_t, max_t, [batch_size, 1], device=device, dtype=torch.long
    ).repeat(1, num_frames)


def compute_kl_gradient(
    fake_score: WanDiffusionWrapper,
    real_score: WanDiffusionWrapper,
    noisy_latent: torch.Tensor,
    clean_latent: torch.Tensor,
    timestep_id: torch.Tensor,
    conditional_dict: dict,
    unconditional_dict: dict,
    guidance_scale: float,
    y: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute KL gradient (DMD paper eq. 7).

    Three forward passes: fake_score conditional, real_score conditional,
    real_score unconditional.  CFG on the real score.  Gradient normalized
    by |clean - pred_real|.
    """
    _, pred_fake = fake_score(
        noisy_image_or_video=noisy_latent,
        conditional_dict=conditional_dict,
        timestep_id=timestep_id,
        y=y,
    )
    _, pred_real_cond = real_score(
        noisy_image_or_video=noisy_latent,
        conditional_dict=conditional_dict,
        timestep_id=timestep_id,
        y=y,
    )
    _, pred_real_uncond = real_score(
        noisy_image_or_video=noisy_latent,
        conditional_dict=unconditional_dict,
        timestep_id=timestep_id,
        y=y,
    )

    pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * guidance_scale

    grad = pred_fake - pred_real

    # Gradient normalization (DMD paper eq. 8)
    normalizer = torch.abs(clean_latent - pred_real).mean(dim=[1, 2, 3, 4], keepdim=True)
    grad = grad / normalizer
    grad = torch.nan_to_num(grad)

    log_dict = {
        "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
        "timestep": timestep_id.detach(),
    }
    return grad, log_dict


def compute_dmd_loss(
    generator: WanDiffusionWrapper,
    fake_score: WanDiffusionWrapper,
    real_score: WanDiffusionWrapper,
    image_or_video_shape: list[int],
    conditional_dict: dict,
    unconditional_dict: dict,
    scheduler,
    cfg: DMDTrainConfig,
    timestep_bound: torch.Tensor,
    min_timestep: int,
    max_timestep: int,
    device: torch.device,
    dtype: torch.dtype,
    y: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute generator loss via DMD.

    1. backward_simulate -> pred_image (with gradients at exit step)
    2. sample_timesteps, add_noise_high
    3. compute_kl_gradient (all no_grad)
    4. loss = 0.5 * MSE(pred, (pred - grad).detach())
    """
    noise = prepare_latents(image_or_video_shape, device, dtype)

    # Backward simulation — gradient flows through the exit step
    flow_pred, pred_image = backward_simulate(
        generator=generator,
        noise=noise,
        conditional_dict=conditional_dict,
        denoising_step_list=cfg.denoising_step_list,
        scheduler=scheduler,
        timestep_bound=timestep_bound,
        training_target=cfg.target,
        y=y,
    )
    del flow_pred

    batch_size, num_frames = pred_image.shape[:2]

    with torch.no_grad():
        timestep = sample_timesteps(
            min_timestep, max_timestep, batch_size, num_frames, device
        )
        min_step = int(0.02 * cfg.num_train_timestep)
        max_step = int(0.98 * cfg.num_train_timestep)
        timestep = timestep.clamp(min_step, max_step)
        timestep_id = 1000 - timestep

        noise_for_kl = torch.randn_like(pred_image)
        noisy_latent = scheduler.add_noise_high(
            pred_image.flatten(0, 1),
            noise_for_kl.flatten(0, 1),
            timestep_id.flatten(0, 1),
            timestep_bound,
        ).detach().unflatten(0, (batch_size, num_frames))

        grad, log_dict = compute_kl_gradient(
            fake_score=fake_score,
            real_score=real_score,
            noisy_latent=noisy_latent,
            clean_latent=pred_image,
            timestep_id=timestep_id,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            guidance_scale=cfg.guidance_scale,
            y=y,
        )

    dmd_loss = 0.5 * F.mse_loss(
        pred_image.double(),
        (pred_image.double() - grad.double()).detach(),
        reduction="mean",
    )
    return dmd_loss, log_dict


def compute_critic_loss(
    generator: WanDiffusionWrapper,
    fake_score: WanDiffusionWrapper,
    image_or_video_shape: list[int],
    conditional_dict: dict,
    scheduler,
    cfg: DMDTrainConfig,
    timestep_bound: torch.Tensor,
    sigma_bound: torch.Tensor,
    min_timestep: int,
    max_timestep: int,
    device: torch.device,
    dtype: torch.dtype,
    y: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute critic (fake_score) denoising loss on generated samples.

    1. backward_simulate (no_grad) -> generated_image
    2. sample_timesteps, add_noise_high
    3. fake_score forward -> flow_pred_fake
    4. Convert flow to x_bound via calculate_alpha_beta_high
    5. loss = MSE(x_bound, generated_image)
    """
    noise = prepare_latents(image_or_video_shape, device, dtype)

    with torch.no_grad():
        _, generated_image = backward_simulate(
            generator=generator,
            noise=noise,
            conditional_dict=conditional_dict,
            denoising_step_list=cfg.denoising_step_list,
            scheduler=scheduler,
            timestep_bound=timestep_bound,
            training_target=cfg.target,
            y=y,
        )

    batch_size, num_frames = image_or_video_shape[:2]

    critic_timestep = sample_timesteps(
        min_timestep, max_timestep, batch_size, num_frames, device
    )
    min_step = int(0.02 * cfg.num_train_timestep)
    max_step = int(0.98 * cfg.num_train_timestep)
    critic_timestep = critic_timestep.clamp(min_step, max_step)
    critic_timestep_id = 1000 - critic_timestep

    critic_noise = torch.randn_like(generated_image)
    noisy_generated = scheduler.add_noise_high(
        generated_image.flatten(0, 1),
        critic_noise.flatten(0, 1),
        critic_timestep_id.flatten(0, 1),
        timestep_bound,
    ).unflatten(0, (batch_size, num_frames))

    flow_pred_fake, _ = fake_score(
        noisy_image_or_video=noisy_generated,
        conditional_dict=conditional_dict,
        timestep_id=critic_timestep_id,
        y=y,
    )

    # Convert flow prediction to x_bound estimate
    scheduler.sigmas = scheduler.sigmas.to(noisy_generated.device)
    t = scheduler.sigmas[critic_timestep_id].reshape(-1, 1, 1, 1)
    s = sigma_bound.to(noisy_generated.device)
    alpha, beta = scheduler.calculate_alpha_beta_high(t, s)
    fake_image = (
        (1 - s) * (t - beta * beta) * noisy_generated
        - (1 - s) * (1 - t) * beta * beta * flow_pred_fake
    ) / ((1 - t) * beta * beta + (1 - s) * (t - beta * beta) * alpha)

    denoising_loss = torch.mean((fake_image - generated_image) ** 2)

    log_dict = {"critic_timestep": critic_timestep_id.detach()}
    return denoising_loss, log_dict


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def create_models(cfg: DMDTrainConfig):
    """Instantiate generator, real_score, fake_score, text_encoder."""
    model_kwargs = dict(timestep_shift=cfg.timestep_shift)

    generator = WanDiffusionWrapper(
        **model_kwargs,
        model_name=cfg.generator_name,
        is_causal=cfg.generator_is_causal,
        timestep_bound=cfg.boundary_step,
        target=cfg.target,
    )
    generator.model.requires_grad_(True)

    real_score = WanDiffusionWrapper(
        **model_kwargs,
        model_name=cfg.real_score_name,
        is_causal=False,
        timestep_bound=cfg.boundary_step,
        target=cfg.target,
    )
    real_score.model.requires_grad_(False)

    fake_score = WanDiffusionWrapper(
        **model_kwargs,
        model_name=cfg.fake_score_name,
        is_causal=False,
        timestep_bound=cfg.boundary_step,
        target=cfg.target,
    )
    fake_score.model.requires_grad_(True)

    if cfg.gradient_checkpointing:
        generator.enable_gradient_checkpointing()
        fake_score.enable_gradient_checkpointing()

    text_encoder = WanTextEncoder(model_name=cfg.generator_name)
    text_encoder.requires_grad_(False)

    return generator, real_score, fake_score, text_encoder


def create_dataset(cfg: DMDTrainConfig):
    if cfg.data_type == "text_folder":
        return TextFolderDataset(cfg.data_path, cfg.data_max_count)
    elif cfg.data_type == "text_file":
        return TextDataset(cfg.data_path)
    else:
        raise ValueError(f"Unsupported data_type: {cfg.data_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: DMDTrainConfig) -> None:
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
    generator, real_score, fake_score, text_encoder = create_models(cfg)
    scheduler = generator.get_scheduler()

    # Prepare with accelerate (handles FSDP wrapping based on accelerate config)
    generator, real_score, fake_score, text_encoder = accelerator.prepare(
        generator, real_score, fake_score, text_encoder
    )

    # Optimizers
    generator_optimizer = torch.optim.AdamW(
        [p for p in generator.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        [p for p in fake_score.parameters() if p.requires_grad],
        lr=cfg.lr_critic,
        betas=(cfg.beta1_critic, cfg.beta2_critic),
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

    # Pre-compute constants
    timestep_bound = compute_timestep_bound(cfg.boundary_step, cfg.timestep_shift)
    sigma_bound = timestep_bound / 1000
    moe_train_step = cfg.num_train_timestep - cfg.boundary_step
    min_timestep = int(cfg.boundary_step + moe_train_step * 0.04)
    max_timestep = int(cfg.boundary_step + moe_train_step * 0.96)

    # EMA
    step = 0
    generator_ema = None
    if cfg.ema_weight > 0 and step >= cfg.ema_start_step:
        generator_ema = EMATracker(generator, decay=cfg.ema_weight)

    # Cache unconditional dict (computed once)
    unconditional_dict = None

    # Resume
    if cfg.resume_ckpt:
        if accelerator.is_main_process:
            logger.info("Resuming from %s", cfg.resume_ckpt)
        step = cfg.resume_step

        ema_path = os.path.join(cfg.resume_ckpt, "generator_ema.pt")
        if os.path.exists(ema_path) and cfg.ema_weight > 0:
            ema_state = torch.load(ema_path, map_location="cpu")
            generator.load_state_dict(ema_state, strict=True)
            generator_ema = EMATracker(generator, decay=cfg.ema_weight)

        gen_path = os.path.join(cfg.resume_ckpt, "generator.pt")
        if os.path.exists(gen_path):
            generator.load_state_dict(
                torch.load(gen_path, map_location="cpu"), strict=True
            )

        critic_path = os.path.join(cfg.resume_ckpt, "critic.pt")
        if os.path.exists(critic_path):
            fake_score.load_state_dict(
                torch.load(critic_path, map_location="cpu"), strict=True
            )

    # Training loop
    start_step = step
    previous_time = None

    while True:
        if accelerator.is_main_process:
            logger.info("Step %d", step)

        train_generator = step % cfg.dfake_gen_update_ratio == 0

        if step % 20 == 0:
            torch.cuda.empty_cache()

        # Generator step (uses its own batch, matching legacy behavior)
        if train_generator:
            gen_batch = next(dataloader)
            gen_prompts = gen_batch["prompts"]
            gen_bs = len(gen_prompts)
            gen_shape = list(cfg.image_or_video_shape)
            gen_shape[0] = gen_bs

            with torch.no_grad():
                gen_cond = text_encoder(text_prompts=gen_prompts)
                if unconditional_dict is None:
                    _, unconditional_dict = encode_prompts(
                        text_encoder, gen_prompts, cfg.negative_prompt, device
                    )

            generator_optimizer.zero_grad(set_to_none=True)
            gen_loss, gen_log = compute_dmd_loss(
                generator=generator,
                fake_score=fake_score,
                real_score=real_score,
                image_or_video_shape=gen_shape,
                conditional_dict=gen_cond,
                unconditional_dict=unconditional_dict,
                scheduler=scheduler,
                cfg=cfg,
                timestep_bound=timestep_bound,
                min_timestep=min_timestep,
                max_timestep=max_timestep,
                device=device,
                dtype=dtype,
            )
            torch.cuda.empty_cache()
            accelerator.backward(gen_loss)
            if hasattr(generator, "clip_grad_norm_"):
                gen_grad_norm = generator.clip_grad_norm_(cfg.max_grad_norm_generator)
            else:
                gen_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in generator.parameters() if p.requires_grad],
                    cfg.max_grad_norm_generator,
                )
            generator_optimizer.step()

            if generator_ema is not None:
                generator_ema.update(generator)

        # Critic step (separate batch)
        critic_batch = next(dataloader)
        critic_prompts = critic_batch["prompts"]
        critic_bs = len(critic_prompts)
        critic_shape = list(cfg.image_or_video_shape)
        critic_shape[0] = critic_bs

        with torch.no_grad():
            critic_cond = text_encoder(text_prompts=critic_prompts)
            if unconditional_dict is None:
                _, unconditional_dict = encode_prompts(
                    text_encoder, critic_prompts, cfg.negative_prompt, device
                )

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss, critic_log = compute_critic_loss(
            generator=generator,
            fake_score=fake_score,
            image_or_video_shape=critic_shape,
            conditional_dict=critic_cond,
            scheduler=scheduler,
            cfg=cfg,
            timestep_bound=timestep_bound,
            sigma_bound=sigma_bound,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            device=device,
            dtype=dtype,
        )
        accelerator.backward(critic_loss)
        if hasattr(fake_score, "clip_grad_norm_"):
            critic_grad_norm = fake_score.clip_grad_norm_(cfg.max_grad_norm_critic)
        else:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in fake_score.parameters() if p.requires_grad],
                cfg.max_grad_norm_critic,
            )
        critic_optimizer.step()

        step += 1

        # Lazy EMA init
        if cfg.ema_weight > 0 and step >= cfg.ema_start_step and generator_ema is None:
            generator_ema = EMATracker(generator, decay=cfg.ema_weight)

        # Checkpointing
        if (
            not cfg.no_save
            and (step - start_step) > 0
            and step % cfg.log_iters == 0
        ):
            torch.cuda.empty_cache()
            _save_checkpoint(accelerator, generator, fake_score, generator_ema,
                             cfg, step)
            torch.cuda.empty_cache()

        # Logging
        if accelerator.is_main_process:
            log_dict = {
                "critic_loss": critic_loss.item(),
                "critic_grad_norm": critic_grad_norm.item()
                if isinstance(critic_grad_norm, torch.Tensor)
                else critic_grad_norm,
            }
            if train_generator:
                log_dict.update({
                    "generator_loss": gen_loss.item(),
                    "generator_grad_norm": gen_grad_norm.item()
                    if isinstance(gen_grad_norm, torch.Tensor)
                    else gen_grad_norm,
                    "dmdtrain_gradient_norm": gen_log["dmdtrain_gradient_norm"].item(),
                })
                logger.info("gen_loss=%.6f  critic_loss=%.6f", gen_loss.item(), critic_loss.item())
            else:
                logger.info("critic_loss=%.6f", critic_loss.item())

            if not cfg.disable_wandb:
                wandb.log(log_dict, step=step)

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


def _save_checkpoint(accelerator, generator, fake_score, generator_ema, cfg, step):
    """Gather FSDP state dicts and save on main process."""
    gen_state = accelerator.get_state_dict(generator)
    critic_state = accelerator.get_state_dict(fake_score)

    payload = {
        "generator": gen_state,
        "critic": critic_state,
    }
    if generator_ema is not None:
        payload["generator_ema"] = generator_ema.state_dict()

    if accelerator.is_main_process:
        ckpt_dir = os.path.join(cfg.logdir, f"checkpoint_model_{step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(payload, os.path.join(ckpt_dir, "model.pt"))
        logger.info("Saved checkpoint to %s", ckpt_dir)


if __name__ == "__main__":
    main()
