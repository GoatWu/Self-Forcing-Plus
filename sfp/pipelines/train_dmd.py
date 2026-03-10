"""Pipeline helpers for DMD distillation training.

Provides reusable functions for backward simulation, prompt encoding, and
latent preparation — shared across training scripts.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist

from sfp.model.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from sfp.utils.schedulers import FlowMatchScheduler


def encode_prompts(
    text_encoder: WanTextEncoder,
    prompts: List[str],
    negative_prompt: str,
    device: torch.device,
) -> Tuple[dict, dict]:
    """Encode text prompts into conditional and unconditional dicts."""
    with torch.no_grad():
        conditional_dict = text_encoder(text_prompts=prompts)
        unconditional_dict = text_encoder(
            text_prompts=[negative_prompt] * len(prompts)
        )
        unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
    return conditional_dict, unconditional_dict


def prepare_latents(
    shape: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample initial noise latents of shape [B, F, C, H, W]."""
    return torch.randn(shape, device=device, dtype=dtype)


def compute_timestep_bound(
    boundary_step: int, timestep_shift: float
) -> torch.Tensor:
    """Shift the raw boundary step through the flow-matching schedule.

    Mirrors BidirectionalTrainingPipeline.__init__:28-35 and DMD.__init__:60-68.
    """
    t_bound = torch.tensor([boundary_step], dtype=torch.float32)
    if timestep_shift > 1:
        t_bound = (
            timestep_shift
            * (t_bound / 1000)
            / (1 + (timestep_shift - 1) * (t_bound / 1000))
            * 1000
        )
    return t_bound


def _sync_exit_index(num_steps: int, device: torch.device) -> int:
    """Pick a random exit step on rank 0 and broadcast to all ranks."""
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        idx = torch.randint(0, num_steps, (1,), device=device)
    else:
        idx = torch.empty(1, dtype=torch.long, device=device)

    if dist.is_initialized():
        dist.broadcast(idx, src=0)
    return idx.item()


def backward_simulate(
    generator: WanDiffusionWrapper,
    noise: torch.Tensor,
    conditional_dict: dict,
    denoising_step_list: List[int],
    scheduler: FlowMatchScheduler,
    timestep_bound: torch.Tensor,
    training_target: str = "high_noise",
    y: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run backward simulation to produce generator input.

    Extracted from BidirectionalTrainingPipeline.inference_with_trajectory().
    Iterates through denoising steps, picks a random exit step (synced across
    ranks), runs all steps before exit with no_grad, and the exit step with
    gradients enabled for backpropagation.

    Returns:
        (flow_pred, denoised_pred) — both [B, F, C, H, W].
    """
    noisy_image_or_video = noise
    num_steps = len(denoising_step_list)
    exit_idx = _sync_exit_index(num_steps, device=noise.device)

    for index, current_timestep in enumerate(denoising_step_list):
        is_exit = index == exit_idx
        timestep_id = (
            1000
            - torch.ones(noise.shape[:2], device=noise.device, dtype=torch.int64)
            * current_timestep
        )

        if not is_exit:
            with torch.no_grad():
                flow_pred, denoised_pred = generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep_id=timestep_id,
                    y=y,
                )
                next_timestep_id = 1000 - denoising_step_list[
                    index + 1
                ] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device
                )

                if training_target == "high_noise":
                    noisy_image_or_video = scheduler.add_noise_high(
                        denoised_pred.flatten(0, 1),
                        noise.flatten(0, 1),
                        next_timestep_id.flatten(0, 1),
                        timestep_bound,
                    ).unflatten(0, denoised_pred.shape[:2])
                elif training_target == "low_noise":
                    noisy_image_or_video = scheduler.add_noise_low(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep_id.flatten(0, 1),
                        timestep_bound,
                    ).unflatten(0, denoised_pred.shape[:2])
        else:
            flow_pred, denoised_pred = generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep_id=timestep_id,
                y=y,
            )
            break

    return flow_pred, denoised_pred
