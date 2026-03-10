import os
import types
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import nn

from sfp.model.base import BuildContext, ModelConfig
from sfp.model.bundle import ModelBundle
from sfp.utils.schedulers import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.clip import CLIPModel
from wan.modules.causal_model import CausalWanModel


@dataclass
class WanModelConfig(ModelConfig):
    model_name: str = "Wan2.1-T2V-14B"
    timestep_shift: float = 8.0
    local_attn_size: int = -1
    sink_size: int = 0
    timestep_bound: int = 0
    target: str = "high_noise"
    is_causal: bool = True

    def validate(self) -> None:
        if self.target not in {"high_noise", "low_noise"}:
            raise ValueError(f"Unsupported target '{self.target}'.")
        if self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be > 0.")
        if self.local_attn_size < -1:
            raise ValueError("local_attn_size must be -1 or a non-negative integer.")


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-14B") -> None:
        super().__init__()
        self.model_name = model_name

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder.load_state_dict(
            torch.load(
                f"wan_models/{self.model_name}/models_t5_umt5-xxl-enc-bf16.pth",
                map_location="cpu",
                weights_only=False,
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=f"wan_models/{self.model_name}/google/umt5-xxl/",
            seq_len=512,
            clean="whitespace",
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanCLIPEncoder(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-14B"):
        super().__init__()
        self.model_name = model_name
        self.image_encoder = CLIPModel(
            dtype=torch.float16,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(
                f"wan_models/{self.model_name}/",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, img):
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        img = img[:, None, :, :].to(self.device)
        clip_encoder_out = self.image_encoder.visual([img]).squeeze(0)
        return clip_encoder_out


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name="Wan2.1-T2V-14B"):
        super().__init__()
        self.model_name = model_name
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(
                pretrained_path=f"wan_models/{self.model_name}/Wan2.1_VAE.pth",
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

        self.dtype = torch.bfloat16

        self.vae_stride = (4, 8, 8)
        self.target_video_length = 81

    def encode(self, pixel):
        device, dtype = pixel[0].device, self.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]
        output = [
            self.model.encode(u.to(self.dtype).unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        return output

    def run_vae_encoder(self, img):
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5).cuda()
        img = img.to(torch.bfloat16).cuda()
        h, w = img.shape[1:]
        lat_h = h // self.vae_stride[1]
        lat_w = w // self.vae_stride[2]

        msk = torch.ones(
            1,
            self.target_video_length,
            lat_h,
            lat_w,
            device=torch.device("cuda"),
        )
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        vae_encode_out = self.encode(
            [
                torch.concat(
                    [
                        torch.nn.functional.interpolate(
                            img[None].cpu(), size=(h, w), mode="bicubic"
                        ).transpose(0, 1),
                        torch.zeros(3, self.target_video_length - 1, h, w),
                    ],
                    dim=1,
                ).cuda()
            ],
        )[0]
        vae_encode_out = torch.concat([msk, vae_encode_out]).to(torch.bfloat16)
        return [vae_encode_out]

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(
                decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Wan2.1-T2V-14B",
        timestep_shift=8.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0,
        timestep_bound=0,
        target="high_noise",
    ):
        super().__init__()
        self.model_name = model_name
        self.target = target
        self.dim = 5120 if "14B" in model_name else 1536
        self.timestep_shift = timestep_shift
        self.timestep_bound = torch.tensor([timestep_bound])
        if self.timestep_shift > 1:
            self.timestep_bound = (
                self.timestep_shift
                * (self.timestep_bound / 1000)
                / (1 + (self.timestep_shift - 1) * (self.timestep_bound / 1000))
                * 1000
            )

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/",
                local_attn_size=local_attn_size,
                sink_size=sink_size,
            )
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.gradient_checkpointing = True

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            # nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.Linear(atten_dim * 3 + time_embed_dim, self.dim),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class),
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(
            num_registers=num_registers, dim=atten_dim
        )
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _convert_flow_pred_to_x_bound(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps, timestep_bound = map(
            lambda x: x.double().to(flow_pred.device),
            [
                flow_pred,
                xt,
                self.scheduler.sigmas,
                self.scheduler.timesteps,
                self.timestep_bound,
            ],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        timestep_id_bound = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_bound.unsqueeze(1)).abs(), dim=1
        )
        sigma_t_bound = sigmas[timestep_id_bound].reshape(-1, 1, 1, 1)
        x_bound_pred = xt - (sigma_t - sigma_t_bound) * flow_pred
        return x_bound_pred.to(original_dtype)

    @staticmethod
    def _convert_x_bound_to_flow_pred(
        scheduler,
        x_bound_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        timestep_bound: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = x_bound_pred.dtype
        x_bound_pred, xt, sigmas, timesteps, timestep_bound = map(
            lambda x: x.double().to(x_bound_pred.device),
            [x_bound_pred, xt, scheduler.sigmas, scheduler.timesteps, timestep_bound],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        timestep_id_bound = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_bound.unsqueeze(1)).abs(), dim=1
        )
        sigma_t_bound = sigmas[timestep_id_bound].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x_bound_pred) / (sigma_t - sigma_t_bound)
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep_id: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        self.scheduler.timesteps = self.scheduler.timesteps.to(timestep_id.device)
        timestep = self.scheduler.timesteps[timestep_id]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                y=y,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    y=y,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        y=y,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        y=y,
                    ).permute(0, 2, 1, 3, 4)

        if self.target == "high_noise":
            pred_x = self._convert_flow_pred_to_x_bound(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])
        elif self.target == "low_noise":
            pred_x = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x, logits

        return flow_pred, pred_x

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()


def build_model_bundle(
    cfg: WanModelConfig, ctx: Optional[BuildContext] = None
) -> ModelBundle:
    """Build a WAN-backed model bundle from a typed config."""
    cfg.validate()
    generator = WanDiffusionWrapper(
        model_name=cfg.model_name,
        timestep_shift=cfg.timestep_shift,
        is_causal=cfg.is_causal,
        local_attn_size=cfg.local_attn_size,
        sink_size=cfg.sink_size,
        timestep_bound=cfg.timestep_bound,
        target=cfg.target,
    )
    text_encoder = WanTextEncoder(model_name=cfg.model_name)
    vae = WanVAEWrapper(model_name=cfg.model_name)
    scheduler = generator.get_scheduler()
    image_encoder = WanCLIPEncoder(model_name=cfg.model_name)

    if ctx is not None and ctx.device is not None:
        generator = generator.to(ctx.device)
        text_encoder = text_encoder.to(ctx.device)
        vae = vae.to(ctx.device)
        image_encoder = image_encoder.to(ctx.device)
    if ctx is not None and ctx.dtype is not None:
        generator = generator.to(dtype=ctx.dtype)

    return ModelBundle(
        generator=generator,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        image_encoder=image_encoder,
    )
