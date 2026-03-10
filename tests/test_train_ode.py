"""Tests for ODE regression training helpers.

These tests verify the per-block timestep sampling, ODE input preparation,
and loss computation without loading actual model weights.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B, F, C, H, W = 2, 6, 16, 8, 8  # F=6 divisible by num_frame_per_block=3
NUM_FRAME_PER_BLOCK = 3
NUM_STEPS = 4  # e.g., [1000, 750, 500, 250]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def denoising_step_list():
    return torch.tensor([1000, 750, 500, 250], dtype=torch.long)


@pytest.fixture
def ode_latent():
    """Fake ODE trajectory: [B, T, F, C, H, W]."""
    return torch.randn(B, NUM_STEPS, F, C, H, W)


@pytest.fixture
def mock_generator():
    gen = MagicMock()

    def forward_fn(noisy_image_or_video, conditional_dict, timestep_id, **kw):
        shape = noisy_image_or_video.shape
        return torch.randn(shape), torch.randn(shape)

    gen.side_effect = forward_fn
    gen.__call__ = forward_fn
    return gen


@pytest.fixture
def cond_dict():
    return {"prompt_embeds": torch.randn(B, 77, 512)}


# ---------------------------------------------------------------------------
# Tests for sfp/pipelines/train_ode.py
# ---------------------------------------------------------------------------


class TestSampleTimestepsPerBlock:
    def test_shape(self):
        from sfp.pipelines.train_ode import sample_timesteps_per_block

        indices = sample_timesteps_per_block(
            0, NUM_STEPS, B, F, NUM_FRAME_PER_BLOCK
        )
        assert indices.shape == (B, F)

    def test_range(self):
        from sfp.pipelines.train_ode import sample_timesteps_per_block

        indices = sample_timesteps_per_block(
            1, NUM_STEPS, 64, F, NUM_FRAME_PER_BLOCK
        )
        assert (indices >= 1).all()
        assert (indices < NUM_STEPS).all()

    def test_within_block_uniformity(self):
        from sfp.pipelines.train_ode import sample_timesteps_per_block

        torch.manual_seed(42)
        indices = sample_timesteps_per_block(
            0, NUM_STEPS, B, F, NUM_FRAME_PER_BLOCK
        )
        # F=6 with block_size=3 → 2 blocks: [0:3] and [3:6]
        for b in range(B):
            assert indices[b, 0] == indices[b, 1] == indices[b, 2]
            assert indices[b, 3] == indices[b, 4] == indices[b, 5]

    def test_independent_first_frame(self):
        from sfp.pipelines.train_ode import sample_timesteps_per_block

        # F=6 with independent_first_frame: frame 0 independent,
        # frames 1-5 blocked in groups of 3 → but 5 isn't divisible by 3...
        # Use F=7: frame 0 free, frames 1-6 → 2 blocks of 3
        num_frames = 7
        torch.manual_seed(42)
        indices = sample_timesteps_per_block(
            0, NUM_STEPS, B, num_frames, NUM_FRAME_PER_BLOCK,
            independent_first_frame=True
        )
        assert indices.shape == (B, num_frames)
        # Frames 1-3 should be uniform, frames 4-6 should be uniform
        for b in range(B):
            assert indices[b, 1] == indices[b, 2] == indices[b, 3]
            assert indices[b, 4] == indices[b, 5] == indices[b, 6]


class TestPrepareOdeInput:
    def test_output_shapes(self, ode_latent, denoising_step_list):
        from sfp.pipelines.train_ode import prepare_ode_input, sample_timesteps_per_block

        indices = sample_timesteps_per_block(
            0, NUM_STEPS, B, F, NUM_FRAME_PER_BLOCK
        )
        noisy, timesteps = prepare_ode_input(
            ode_latent, denoising_step_list, indices
        )
        assert noisy.shape == (B, F, C, H, W)
        assert timesteps.shape == (B, F)

    def test_gather_correctness(self, denoising_step_list):
        from sfp.pipelines.train_ode import prepare_ode_input

        # Create a simple trajectory where each step has a known value
        ode = torch.arange(NUM_STEPS).float().reshape(1, NUM_STEPS, 1, 1, 1, 1)
        ode = ode.expand(1, NUM_STEPS, 2, C, H, W)

        indices = torch.tensor([[1, 2]])  # F=2
        noisy, timesteps = prepare_ode_input(ode, denoising_step_list, indices)

        # Frame 0 should have value 1.0, frame 1 should have value 2.0
        assert noisy[0, 0, 0, 0, 0].item() == pytest.approx(1.0)
        assert noisy[0, 1, 0, 0, 0].item() == pytest.approx(2.0)
        # Timesteps should be denoising_step_list[index]
        assert timesteps[0, 0].item() == 750  # denoising_step_list[1]
        assert timesteps[0, 1].item() == 500  # denoising_step_list[2]

    def test_i2v_first_frame_override(self, ode_latent, denoising_step_list):
        from sfp.pipelines.train_ode import prepare_ode_input, sample_timesteps_per_block

        indices = sample_timesteps_per_block(
            0, NUM_STEPS, B, F, NUM_FRAME_PER_BLOCK
        )
        _, timesteps = prepare_ode_input(
            ode_latent, denoising_step_list, indices, i2v=True
        )
        # First frame should always get the last step (clean = index NUM_STEPS-1)
        # denoising_step_list[-1] = 250
        assert (timesteps[:, 0] == 250).all()


# ---------------------------------------------------------------------------
# Tests for scripts/train_ode.py helpers
# ---------------------------------------------------------------------------


class TestComputeOdeLoss:
    def test_scalar_loss(self, mock_generator, ode_latent, cond_dict, denoising_step_list):
        from scripts.train_ode import compute_ode_loss

        loss, log_dict = compute_ode_loss(
            generator=mock_generator,
            ode_latent=ode_latent,
            conditional_dict=cond_dict,
            denoising_step_list=denoising_step_list,
            num_frame_per_block=NUM_FRAME_PER_BLOCK,
            independent_first_frame=False,
            i2v=False,
            device=torch.device("cpu"),
        )
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0
        assert "timestep" in log_dict
        assert "mask_ratio" in log_dict

    def test_mask_excludes_zero_timestep(self, cond_dict):
        """When all sampled indices point to the last step (timestep=250),
        the mask should be all True (250 != 0)."""
        from scripts.train_ode import compute_ode_loss

        gen = MagicMock()
        target = torch.randn(B, F, C, H, W)

        def forward_fn(noisy_image_or_video, conditional_dict, timestep_id, **kw):
            return torch.randn_like(target), target.clone()

        gen.side_effect = forward_fn
        gen.__call__ = forward_fn

        # denoising_step_list with a 0 entry to trigger masking
        step_list = torch.tensor([1000, 500, 0], dtype=torch.long)
        ode = torch.randn(B, 3, F, C, H, W)

        loss, log_dict = compute_ode_loss(
            generator=gen,
            ode_latent=ode,
            conditional_dict=cond_dict,
            denoising_step_list=step_list,
            num_frame_per_block=NUM_FRAME_PER_BLOCK,
            independent_first_frame=False,
            i2v=False,
            device=torch.device("cpu"),
        )
        # mask_ratio should be < 1.0 if any frames hit timestep=0
        assert log_dict["mask_ratio"].item() <= 1.0


class TestOdeTrainConfig:
    def test_defaults(self):
        from sfp.utils.config import OdeTrainConfig

        cfg = OdeTrainConfig()
        assert cfg.generator_name == "Wan2.1-T2V-1.3B"
        assert cfg.timestep_shift == 5.0
        assert cfg.denoising_step_list == [1000, 750, 500, 250]
        assert cfg.num_frame_per_block == 3
        assert cfg.lr == pytest.approx(2e-6)
        assert cfg.i2v is False

    def test_yaml_roundtrip(self, tmp_path):
        import draccus
        from sfp.utils.config import OdeTrainConfig

        yaml_content = """
generator_name: Wan2.1-T2V-1.3B
timestep_shift: 5.0
denoising_step_list: [1000, 750, 500, 250]
num_frame_per_block: 3
lr: 2.0e-06
data_path: /tmp/test
"""
        yaml_path = tmp_path / "test_ode.yaml"
        yaml_path.write_text(yaml_content)

        cfg = draccus.load(OdeTrainConfig, yaml_path)
        assert cfg.generator_name == "Wan2.1-T2V-1.3B"
        assert cfg.denoising_step_list == [1000, 750, 500, 250]
        assert cfg.lr == pytest.approx(2e-6)
        assert cfg.data_path == "/tmp/test"
