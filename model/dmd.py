from builtins import NotImplementedError
from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
from utils.wan_wrapper import WanDiffusionWrapper
from model.base import SelfForcingModel
import torch.distributed as dist


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if self.training_target == "high_noise":
            moe_train_step = self.num_train_timestep - self.boundary_step
            self.min_timestep = int(self.boundary_step + moe_train_step * 0.04)
            self.max_timestep = int(self.boundary_step + moe_train_step * 0.96)
        elif self.training_target == "low_noise":
            moe_train_step = self.boundary_step
            self.min_timestep = int(moe_train_step * 0.04)
            self.max_timestep = int(moe_train_step * 0.96)

        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        self.timestep_bound = torch.tensor([self.boundary_step])
        if self.timestep_shift > 1:
            self.timestep_bound = self.timestep_shift * \
                (self.timestep_bound / 1000) / (1 + (self.timestep_shift - 1) * (self.timestep_bound / 1000)) * 1000
        self.sigma_bound = self.timestep_bound / 1000

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep_id: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        y = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep_id=timestep_id,
            y=y
        )

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep_id=timestep_id,
            y=y
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep_id=timestep_id,
            y=y
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep_id.detach()
        }

    def compute_distribution_matching_loss(
        self,
        flow_pred: torch.Tensor,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            timestep = self._get_timestep(
                self.min_timestep,
                self.max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            timestep = timestep.clamp(self.min_step, self.max_step)

            timestep_id = 1000 - timestep

            noise = torch.randn_like(image_or_video)
            if self.training_target == "high_noise":
                noisy_latent = self.scheduler.add_noise_high(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep_id.flatten(0, 1),
                    self.timestep_bound
                ).detach().unflatten(0, (batch_size, num_frame))
            elif self.training_target == "low_noise":
                noisy_latent = self.scheduler.add_noise_low(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep_id.flatten(0, 1),
                    self.timestep_bound
                ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep_id=timestep_id,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                y=y
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        flow_pred, pred_image, gradient_mask, noise = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            y=y
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            flow_pred=flow_pred,
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            y=y
        )

        if dist.get_rank() == 0:
            print(f"dmd_loss: {dmd_loss.item()}")

        del flow_pred, pred_image, gradient_mask

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        y: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            _, generated_image, _, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                y=y
            )

        # Step 2: Compute the fake prediction
        critic_timestep = self._get_timestep(
            self.min_timestep,
            self.max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_timestep_id = 1000 - critic_timestep

        critic_noise = torch.randn_like(generated_image)
        if self.training_target == "high_noise":
            noisy_generated_image = self.scheduler.add_noise_high(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep_id.flatten(0, 1),
                self.timestep_bound
            ).unflatten(0, image_or_video_shape[:2])
        elif self.training_target == "low_noise":
            noisy_generated_image = self.scheduler.add_noise_low(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep_id.flatten(0, 1),
                self.timestep_bound
            ).unflatten(0, image_or_video_shape[:2])

        flow_pred_fake, _ = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep_id=critic_timestep_id,
            y=y
        )
        if self.training_target == "high_noise":
            self.scheduler.sigmas = self.scheduler.sigmas.to(noisy_generated_image.device)
            t = self.scheduler.sigmas[critic_timestep_id].reshape(-1, 1, 1, 1)
            s = self.sigma_bound.to(noisy_generated_image.device)
            alpha, beta = self.scheduler.calculate_alpha_beta_high(t, s)
            fake_image = ((1 - s) * (t - beta * beta) * noisy_generated_image - (1 - s) * (1 - t) * beta * beta * flow_pred_fake) / ((1 - t) * beta * beta + (1 - s) * (t - beta * beta) * alpha)
        elif self.training_target == "low_noise":
            self.scheduler.sigmas = self.scheduler.sigmas.to(noisy_generated_image.device)
            t = self.scheduler.sigmas[critic_timestep_id].reshape(-1, 1, 1, 1)
            s = self.sigma_bound.to(noisy_generated_image.device)
            fake_image = noisy_generated_image - flow_pred_fake * t
        denoising_loss = torch.mean((fake_image - generated_image) ** 2)
        if dist.get_rank() == 0:
            print(f"denoising_loss: {denoising_loss.item()}")

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep_id.detach()
        }

        return denoising_loss, critic_log_dict
