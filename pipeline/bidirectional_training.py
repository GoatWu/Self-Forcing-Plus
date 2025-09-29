from typing import List
import torch

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint


class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanDiffusionWrapper,
        boundary_step: int,
        training_target: str
    ):
        super().__init__()
        self.model_name = model_name
        self.training_target = training_target
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.boundary_step = boundary_step
        self.timestep_bound = torch.tensor([self.boundary_step])
        if self.scheduler.shift > 1:
            self.timestep_bound = self.scheduler.shift * \
                (self.timestep_bound / 1000) / (1 + (self.scheduler.shift - 1) * (self.timestep_bound / 1000)) * 1000

    def generate_and_sync_list(self, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
        else:
            indices = torch.empty(1, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(self, noise: torch.Tensor, y, **conditional_dict) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """

        # initial point
        noisy_image_or_video = noise
        x_bound = noise
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(num_denoising_steps, device=noise.device)

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = (index == exit_flags[0])
            timestep_id = 1000 - torch.ones(
                noise.shape[:2],
                device=noise.device,
                dtype=torch.int64) * current_timestep
            if not exit_flag:
                with torch.no_grad():
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_image_or_video,
                        conditional_dict=conditional_dict,
                        timestep_id=timestep_id,
                        y=y
                    )  # [B, F, C, H, W]

                    next_timestep_id = 1000 - self.denoising_step_list[index + 1] * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device)

                    if self.training_target == "high_noise":
                        noisy_image_or_video = self.scheduler.add_noise_high(
                            denoised_pred.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep_id.flatten(0, 1),
                            self.timestep_bound
                        ).unflatten(0, denoised_pred.shape[:2])
                    elif self.training_target == "low_noise":
                        noisy_image_or_video = self.scheduler.add_noise_low(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep_id.flatten(0, 1),
                            self.timestep_bound
                        ).unflatten(0, denoised_pred.shape[:2])           
            else:
                flow_pred, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep_id=timestep_id,
                    y=y
                )  # [B, F, C, H, W]
                break

        return flow_pred, denoised_pred
