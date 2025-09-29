from typing import Tuple
from einops import rearrange
from torch import nn
from safetensors.torch import load_file
import torch.distributed as dist
import torch
import os

from pipeline import SelfForcingTrainingPipeline, BidirectionalTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanCLIPEncoder


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.is_causal = args.generator_type == "causal"
        self.i2v = args.i2v
        self.training_target = args.training_target
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.boundary_step = args.boundary_step

        self._initialize_models(args, device)
        
        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)

        self.x_bound = None

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.2-T2V-A14B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.2-T2V-A14B")
        self.generator_name = getattr(args, "generator_name", "Wan2.2-T2V-A14B")

        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
            model_name=self.generator_name,
            is_causal=self.is_causal,
            timestep_bound=self.boundary_step,
            target=self.training_target
        )
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
            model_name=self.real_model_name,
            is_causal=False,
            timestep_bound=self.boundary_step,
            target=self.training_target
        )
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
            model_name=self.fake_model_name,
            is_causal=False,
            timestep_bound=self.boundary_step,
            target=self.training_target
        )
        self.fake_score.model.requires_grad_(True)

        if self.training_target == "low_noise":
            self.high_noise_name = getattr(args, "high_noise_name", "Wan2.2-T2V-A14B")
            self.high_noise_model = WanDiffusionWrapper(
                **getattr(args, "model_kwargs", {}),
                model_name=self.high_noise_name,
                is_causal=False,
                timestep_bound=self.boundary_step,
                target="high_noise"
            )
            distill_path = os.path.join("./wan_models", getattr(args, "high_noise_distill_ckpt", ""))
            weights = load_file(distill_path)
            weights = {f"model.{k}": v for k, v in weights.items()}
            self.high_noise_model.load_state_dict(weights, strict=True)
            self.high_noise_model.model.requires_grad_(False)

        self.text_encoder = WanTextEncoder(model_name=self.generator_name)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=self.generator_name)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        if self.training_target == "low_noise":
            noise = self.x_bound.to(self.device).to(self.dtype)
        elif self.training_target == "high_noise":
            noise = torch.randn(noise_shape,
                                device=self.device, dtype=self.dtype)

        flow_pred, pred_image_or_video = self._consistency_backward_simulation(
            noise=noise,
            y=y,
            **conditional_dict
        )
        return flow_pred, pred_image_or_video, None, noise

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        y: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, y=y, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        if self.is_causal:
            self.inference_pipeline = SelfForcingTrainingPipeline(
                model_name=self.generator_name,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.args.independent_first_frame,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.num_training_frames,
                context_noise=self.args.context_noise
            )
        else:
            self.inference_pipeline = BidirectionalTrainingPipeline(
                model_name=self.generator_name,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                boundary_step=self.boundary_step,
                training_target=self.training_target
            )
