"""Integration tests for the DMD distillation training script.

Tests the full training loop with mock models (no real Wan weights needed).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


B, F, C, H, W = 1, 4, 16, 8, 8


def _make_mock_diffusion_wrapper():
    """Create a mock WanDiffusionWrapper that behaves like the real one."""
    from sfp.utils.schedulers import FlowMatchScheduler

    mock = MagicMock(spec=["__call__", "model", "parameters", "named_parameters",
                          "enable_gradient_checkpointing", "get_scheduler",
                          "requires_grad_", "train", "eval", "load_state_dict",
                          "state_dict", "to"])

    # Real scheduler so math works correctly
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    mock.get_scheduler.return_value = scheduler

    # Parameters: a single trainable tensor
    param = torch.nn.Parameter(torch.randn(4, 4))
    mock.parameters.return_value = [param]
    mock.named_parameters.return_value = [("weight", param)]
    mock.model = MagicMock()
    mock.model.requires_grad_ = MagicMock()

    def forward_fn(noisy_image_or_video, conditional_dict, timestep_id, y=None, **kw):
        shape = noisy_image_or_video.shape
        flow = torch.randn(shape, requires_grad=True)
        denoised = torch.randn(shape, requires_grad=True)
        return flow, denoised

    mock.__call__ = forward_fn
    mock.side_effect = forward_fn
    return mock


def _make_mock_text_encoder():
    mock = MagicMock()
    mock.requires_grad_ = MagicMock()

    def encode_fn(text_prompts):
        return {"prompt_embeds": torch.randn(len(text_prompts), 77, 512)}

    mock.__call__ = encode_fn
    mock.side_effect = encode_fn
    return mock


@pytest.fixture
def dummy_dataset(tmp_path):
    """Create a dummy text folder dataset."""
    for i in range(5):
        (tmp_path / f"prompt_{i}.txt").write_text(f"a dog playing in the park {i}")
    return str(tmp_path)


class TestIntegrationTrainLoop:
    """Test the full training loop with mocked models."""

    def test_compute_dmd_loss_end_to_end(self, dummy_dataset):
        """Test that compute_dmd_loss runs without errors with mock models."""
        from scripts.train_distillation import compute_dmd_loss
        from sfp.pipelines.train_dmd import compute_timestep_bound
        from sfp.utils.config import DMDTrainConfig

        generator = _make_mock_diffusion_wrapper()
        fake_score = _make_mock_diffusion_wrapper()
        real_score = _make_mock_diffusion_wrapper()
        scheduler = generator.get_scheduler()

        cfg = DMDTrainConfig(
            denoising_step_list=[1000, 750],
            boundary_step=500,
            timestep_shift=5.0,
            guidance_scale=4.0,
        )

        timestep_bound = compute_timestep_bound(cfg.boundary_step, cfg.timestep_shift)
        moe_train_step = cfg.num_train_timestep - cfg.boundary_step
        min_timestep = int(cfg.boundary_step + moe_train_step * 0.04)
        max_timestep = int(cfg.boundary_step + moe_train_step * 0.96)

        cond = {"prompt_embeds": torch.randn(B, 77, 512)}
        uncond = {"prompt_embeds": torch.randn(B, 77, 512)}

        with patch("sfp.pipelines.train_dmd.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False

            loss, log_dict = compute_dmd_loss(
                generator=generator,
                fake_score=fake_score,
                real_score=real_score,
                image_or_video_shape=[B, F, C, H, W],
                conditional_dict=cond,
                unconditional_dict=uncond,
                scheduler=scheduler,
                cfg=cfg,
                timestep_bound=timestep_bound,
                min_timestep=min_timestep,
                max_timestep=max_timestep,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)
        assert "dmdtrain_gradient_norm" in log_dict

    def test_compute_critic_loss_end_to_end(self):
        """Test that compute_critic_loss runs without errors."""
        from scripts.train_distillation import compute_critic_loss
        from sfp.pipelines.train_dmd import compute_timestep_bound
        from sfp.utils.config import DMDTrainConfig

        generator = _make_mock_diffusion_wrapper()
        fake_score = _make_mock_diffusion_wrapper()
        scheduler = generator.get_scheduler()

        cfg = DMDTrainConfig(
            denoising_step_list=[1000, 750],
            boundary_step=500,
            timestep_shift=5.0,
        )

        timestep_bound = compute_timestep_bound(cfg.boundary_step, cfg.timestep_shift)
        sigma_bound = timestep_bound / 1000
        moe_train_step = cfg.num_train_timestep - cfg.boundary_step
        min_timestep = int(cfg.boundary_step + moe_train_step * 0.04)
        max_timestep = int(cfg.boundary_step + moe_train_step * 0.96)

        cond = {"prompt_embeds": torch.randn(B, 77, 512)}

        with patch("sfp.pipelines.train_dmd.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False

            loss, log_dict = compute_critic_loss(
                generator=generator,
                fake_score=fake_score,
                image_or_video_shape=[B, F, C, H, W],
                conditional_dict=cond,
                scheduler=scheduler,
                cfg=cfg,
                timestep_bound=timestep_bound,
                sigma_bound=sigma_bound,
                min_timestep=min_timestep,
                max_timestep=max_timestep,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert "critic_timestep" in log_dict

    def test_ema_tracker(self):
        """Test EMATracker works with a simple model."""
        from sfp.utils.ema import EMATracker

        model = torch.nn.Linear(4, 4)
        ema = EMATracker(model, decay=0.99)

        # Shadow should be initialized
        assert len(ema.shadow) == 2  # weight + bias

        # Update should not error
        model.weight.data.fill_(1.0)
        ema.update(model)

        # Shadow should have moved toward 1.0
        assert ema.shadow["weight"].mean().item() > 0

        # State dict round-trip
        sd = ema.state_dict()
        ema2 = EMATracker(model, decay=0.99)
        ema2.load_state_dict(sd)
        assert torch.allclose(ema.shadow["weight"], ema2.shadow["weight"])
