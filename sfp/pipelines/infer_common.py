from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
from tqdm import tqdm
import yaml

from sfp.utils.data import TextDataset, TextImagePairDataset


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str, default_config_path: str) -> dict:
    with open(default_config_path, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f) or {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return deep_merge(default_cfg, cfg)


def build_dataset(data_path: str, i2v: bool, extended_prompt_path: str = ""):
    if i2v:
        transform = transforms.Compose(
            [
                transforms.Resize((480, 832)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return TextImagePairDataset(data_path, transform=transform)
    return TextDataset(
        prompt_path=data_path, extended_prompt_path=extended_prompt_path or None
    )


def get_dataloader(dataset, accelerator):
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False
    )


def run_generation_loop(
    pipeline,
    dataloader,
    accelerator,
    output_folder: str,
    i2v: bool,
    num_output_frames: int,
    num_samples: int,
    save_with_index: bool,
):
    if accelerator.is_main_process:
        os.makedirs(output_folder, exist_ok=True)
    accelerator.wait_for_everyone()

    num_prompts = len(dataloader.dataset)

    for _, batch in tqdm(
        enumerate(dataloader), disable=not accelerator.is_main_process
    ):
        idx = batch["idx"].item()
        if i2v:
            prompt = batch["prompts"][0]
            prompts = [prompt] * num_samples
            image = (
                batch["image"]
                .squeeze(0)
                .unsqueeze(0)
                .unsqueeze(2)
                .to(device=accelerator.device, dtype=torch.bfloat16)
            )
            initial_latent = pipeline.vae.encode_to_latent(image).to(
                device=accelerator.device, dtype=torch.bfloat16
            )
            initial_latent = initial_latent.repeat(num_samples, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [
                    num_samples,
                    num_output_frames - 1,
                    16,
                    60,
                    104,
                ],
                device=accelerator.device,
                dtype=torch.bfloat16,
            )
        else:
            prompt = batch["prompts"][0]
            extended_prompt = batch.get("extended_prompts", [None])[0]
            prompts = [
                extended_prompt if extended_prompt is not None else prompt
            ] * num_samples
            initial_latent = None
            sampled_noise = torch.randn(
                [
                    num_samples,
                    num_output_frames,
                    16,
                    60,
                    104,
                ],
                device=accelerator.device,
                dtype=torch.bfloat16,
            )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
        )
        _ = latents
        video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()

        pipeline.vae.model.clear_cache()

        if idx < num_prompts:
            for seed_idx in range(num_samples):
                if save_with_index:
                    out = os.path.join(output_folder, f"{idx}-{seed_idx}.mp4")
                else:
                    out = os.path.join(output_folder, f"{prompt[:100]}-{seed_idx}.mp4")
                write_video(out, video[seed_idx], fps=16)
