"""Pipeline helpers for ODE regression training.

Provides reusable functions for per-block timestep sampling and ODE input
preparation — extracted from legacy BaseModel._get_timestep and
ODERegression._prepare_generator_input.
"""

from __future__ import annotations

from typing import Tuple

import torch


def sample_timesteps_per_block(
    min_index: int,
    max_index: int,
    batch_size: int,
    num_frames: int,
    num_frame_per_block: int,
    independent_first_frame: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample per-block timestep indices into denoising_step_list.

    Within each block of frames, all frames share the same timestep index.
    Returns shape [B, F] of indices in [min_index, max_index).
    """
    indices = torch.randint(
        min_index, max_index, [batch_size, num_frames], device=device, dtype=torch.long
    )

    if independent_first_frame:
        # First frame keeps its own index; remaining frames are blocked
        from_second = indices[:, 1:]
        from_second = from_second.reshape(batch_size, -1, num_frame_per_block)
        from_second[:, :, 1:] = from_second[:, :, 0:1]
        from_second = from_second.reshape(batch_size, -1)
        indices = torch.cat([indices[:, 0:1], from_second], dim=1)
    else:
        indices = indices.reshape(batch_size, -1, num_frame_per_block)
        indices[:, :, 1:] = indices[:, :, 0:1]
        indices = indices.reshape(batch_size, -1)

    return indices


def prepare_ode_input(
    ode_latent: torch.Tensor,
    denoising_step_list: torch.Tensor,
    timestep_indices: torch.Tensor,
    i2v: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather noisy latents from ODE trajectories at sampled indices.

    Args:
        ode_latent: [B, T, F, C, H, W] — full ODE trajectory (noise→clean).
        denoising_step_list: [T] — raw timestep values at each trajectory step.
        timestep_indices: [B, F] — indices from sample_timesteps_per_block.
        i2v: if True, force first frame to the last (clean) index.

    Returns:
        (noisy_input [B, F, C, H, W], raw_timesteps [B, F])
    """
    batch_size, num_steps, num_frames, num_channels, height, width = ode_latent.shape

    if i2v:
        timestep_indices = timestep_indices.clone()
        timestep_indices[:, 0] = num_steps - 1

    # Gather: index into dim=1 (trajectory axis)
    gather_idx = (
        timestep_indices.reshape(batch_size, 1, num_frames, 1, 1, 1)
        .expand(-1, -1, -1, num_channels, height, width)
        .to(ode_latent.device)
    )
    noisy_input = torch.gather(ode_latent, dim=1, index=gather_idx).squeeze(1)

    # Map indices → raw timestep values
    raw_timesteps = denoising_step_list[timestep_indices].to(ode_latent.device)

    return noisy_input, raw_timesteps
