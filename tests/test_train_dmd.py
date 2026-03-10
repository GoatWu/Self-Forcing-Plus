"""Tests for DMD distillation training helpers.

These tests mock the WanDiffusionWrapper to avoid loading actual model weights,
while verifying that the mathematical operations and control flow are correct.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, F, C, H, W = 2, 4, 16, 8, 8  # small shapes for testing


@pytest.fixture
def scheduler():
    """Create a real FlowMatchScheduler for testing."""
    from sfp.utils.schedulers import FlowMatchScheduler

    sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(1000, training=True)
    return sched


@pytest.fixture
def mock_generator():
    """Mock WanDiffusionWrapper that returns random flow_pred and denoised_pred."""
    gen = MagicMock()
    gen.device = torch.device("cpu")

    def forward_fn(noisy_image_or_video, conditional_dict, timestep_id, y=None, **kw):
        shape = noisy_image_or_video.shape
        flow = torch.randn(shape)
        denoised = torch.randn(shape)
        return flow, denoised

    gen.side_effect = forward_fn
    gen.__call__ = forward_fn
    return gen


@pytest.fixture
def mock_score():
    """Mock score network returning random predictions."""
    score = MagicMock()

    def forward_fn(noisy_image_or_video, conditional_dict, timestep_id, y=None, **kw):
        shape = noisy_image_or_video.shape
        flow = torch.randn(shape)
        pred = torch.randn(shape)
        return flow, pred

    score.side_effect = forward_fn
    score.__call__ = forward_fn
    return score


@pytest.fixture
def cond_dict():
    return {"prompt_embeds": torch.randn(B, 77, 512)}


@pytest.fixture
def uncond_dict():
    return {"prompt_embeds": torch.randn(B, 77, 512)}


# ---------------------------------------------------------------------------
# Tests for sfp/pipelines/train_dmd.py
# ---------------------------------------------------------------------------


class TestComputeTimestepBound:
    def test_no_shift(self):
        from sfp.pipelines.train_dmd import compute_timestep_bound

        t = compute_timestep_bound(500, 1.0)
        assert t.item() == pytest.approx(500.0)

    def test_with_shift(self):
        from sfp.pipelines.train_dmd import compute_timestep_bound

        t = compute_timestep_bound(500, 5.0)
        # shift * (b/1000) / (1 + (shift-1)*(b/1000)) * 1000
        # = 5 * 0.5 / (1 + 4*0.5) * 1000 = 2.5/3.0 * 1000 = 833.33
        expected = 5.0 * 0.5 / (1 + 4.0 * 0.5) * 1000
        assert t.item() == pytest.approx(expected, rel=1e-4)

    def test_returns_tensor(self):
        from sfp.pipelines.train_dmd import compute_timestep_bound

        t = compute_timestep_bound(500, 5.0)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1,)


class TestPrepareLatents:
    def test_shape_and_dtype(self):
        from sfp.pipelines.train_dmd import prepare_latents

        latents = prepare_latents([B, F, C, H, W], torch.device("cpu"), torch.float32)
        assert latents.shape == (B, F, C, H, W)
        assert latents.dtype == torch.float32

    def test_bfloat16(self):
        from sfp.pipelines.train_dmd import prepare_latents

        latents = prepare_latents([1, 2, 3, 4, 4], torch.device("cpu"), torch.bfloat16)
        assert latents.dtype == torch.bfloat16


class TestBackwardSimulate:
    @patch("sfp.pipelines.train_dmd.dist")
    def test_returns_correct_shapes(self, mock_dist, mock_generator, scheduler):
        from sfp.pipelines.train_dmd import backward_simulate, compute_timestep_bound

        mock_dist.is_initialized.return_value = False

        noise = torch.randn(B, F, C, H, W)
        cond = {"prompt_embeds": torch.randn(B, 77, 512)}
        t_bound = compute_timestep_bound(500, 5.0)

        flow, denoised = backward_simulate(
            generator=mock_generator,
            noise=noise,
            conditional_dict=cond,
            denoising_step_list=[1000, 750],
            scheduler=scheduler,
            timestep_bound=t_bound,
            training_target="high_noise",
        )

        assert flow.shape == (B, F, C, H, W)
        assert denoised.shape == (B, F, C, H, W)

    @patch("sfp.pipelines.train_dmd.dist")
    def test_single_step_list(self, mock_dist, mock_generator, scheduler):
        """With a single denoising step, exit is always index 0 (with gradients)."""
        from sfp.pipelines.train_dmd import backward_simulate, compute_timestep_bound

        mock_dist.is_initialized.return_value = False

        noise = torch.randn(B, F, C, H, W)
        cond = {"prompt_embeds": torch.randn(B, 77, 512)}
        t_bound = compute_timestep_bound(500, 5.0)

        flow, denoised = backward_simulate(
            generator=mock_generator,
            noise=noise,
            conditional_dict=cond,
            denoising_step_list=[1000],
            scheduler=scheduler,
            timestep_bound=t_bound,
        )

        assert flow.shape == noise.shape


# ---------------------------------------------------------------------------
# Tests for scripts/train_distillation.py helpers
# ---------------------------------------------------------------------------


class TestSampleTimesteps:
    def test_shape(self):
        from scripts.train_distillation import sample_timesteps

        t = sample_timesteps(100, 900, B, F, torch.device("cpu"))
        assert t.shape == (B, F)

    def test_uniform_across_frames(self):
        from scripts.train_distillation import sample_timesteps

        t = sample_timesteps(100, 900, B, F, torch.device("cpu"))
        # All frames should have the same timestep
        for b in range(B):
            assert (t[b] == t[b, 0]).all()

    def test_range(self):
        from scripts.train_distillation import sample_timesteps

        t = sample_timesteps(100, 200, 64, F, torch.device("cpu"))
        assert (t >= 100).all()
        assert (t < 200).all()


class TestComputeKlGradient:
    def test_output_shape(self, mock_score, cond_dict, uncond_dict):
        from scripts.train_distillation import compute_kl_gradient

        noisy = torch.randn(B, F, C, H, W)
        clean = torch.randn(B, F, C, H, W)
        timestep_id = torch.randint(0, 1000, (B, F))

        grad, log_dict = compute_kl_gradient(
            fake_score=mock_score,
            real_score=mock_score,
            noisy_latent=noisy,
            clean_latent=clean,
            timestep_id=timestep_id,
            conditional_dict=cond_dict,
            unconditional_dict=uncond_dict,
            guidance_scale=4.0,
        )

        assert grad.shape == (B, F, C, H, W)
        assert "dmdtrain_gradient_norm" in log_dict
        assert "timestep" in log_dict

    def test_no_nans(self, mock_score, cond_dict, uncond_dict):
        from scripts.train_distillation import compute_kl_gradient

        noisy = torch.randn(B, F, C, H, W)
        clean = torch.randn(B, F, C, H, W)
        timestep_id = torch.randint(0, 1000, (B, F))

        grad, _ = compute_kl_gradient(
            fake_score=mock_score,
            real_score=mock_score,
            noisy_latent=noisy,
            clean_latent=clean,
            timestep_id=timestep_id,
            conditional_dict=cond_dict,
            unconditional_dict=uncond_dict,
            guidance_scale=4.0,
        )

        assert not torch.isnan(grad).any()


class TestDMDTrainConfig:
    def test_defaults(self):
        from sfp.utils.config import DMDTrainConfig

        cfg = DMDTrainConfig()
        assert cfg.target == "high_noise"
        assert cfg.boundary_step == 500
        assert cfg.timestep_shift == 5.0
        assert cfg.denoising_step_list == [1000, 750]
        assert cfg.lr == pytest.approx(2e-6)
        assert cfg.lr_critic == pytest.approx(4e-7)

    def test_yaml_load(self, tmp_path):
        """Test that draccus can load the config from YAML."""
        import draccus
        from sfp.utils.config import DMDTrainConfig

        yaml_content = """
generator_name: Wan2.2-T2V-A14B/high_noise_model
target: high_noise
boundary_step: 500
timestep_shift: 5.0
denoising_step_list: [1000, 750]
lr: 2.0e-06
"""
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = draccus.load(DMDTrainConfig, yaml_path)
        assert cfg.generator_name == "Wan2.2-T2V-A14B/high_noise_model"
        assert cfg.denoising_step_list == [1000, 750]
        assert cfg.lr == pytest.approx(2e-6)


class TestCreateDataset:
    def test_text_folder(self, tmp_path):
        from scripts.train_distillation import create_dataset
        from sfp.utils.config import DMDTrainConfig

        # Create dummy prompt files
        for i in range(3):
            (tmp_path / f"prompt_{i}.txt").write_text(f"a cat doing thing {i}")

        cfg = DMDTrainConfig(data_type="text_folder", data_path=str(tmp_path))
        dataset = create_dataset(cfg)
        assert len(dataset) == 3

    def test_invalid_type(self):
        from scripts.train_distillation import create_dataset
        from sfp.utils.config import DMDTrainConfig

        cfg = DMDTrainConfig(data_type="invalid")
        with pytest.raises(ValueError, match="Unsupported data_type"):
            create_dataset(cfg)
