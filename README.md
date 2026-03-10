<p align="center">
<h1 align="center">Self Forcing Plus</h1>

Self-Forcing-Plus focuses on step distillation and CFG distillation for bidirectional models. Building upon Self-Forcing, we support 4-step T2V-14B model training and higher quality 4-step I2V-14B model training.

## News
- (2025/09) Support Wan2.2-Moe distillation! [wan22](https://github.com/GoatWu/Self-Forcing-Plus/tree/wan22)

| Model Type | Model Link |
|------------|---------------|
| Wan2.1-T2V-14B | [Huggingface](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) |
| Wan2.1-I2V-14B-480P | [Huggingface](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v) |

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

Or with pip (legacy):

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

## Quick Start

### Download Checkpoints

```bash
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir wan_models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir wan_models/Wan2.2-I2V-A14B
```

## Project Structure

```
scripts/                    # Entry points (CLI scripts)
  train_distillation.py     # DMD distillation training (functional, accelerate-based)
  infer_*.py                # Inference scripts
  demo.py                   # Demo server

sfp/                        # Core library
  model/                    # Model wrappers (WanDiffusionWrapper, WanTextEncoder, etc.)
  pipelines/                # Pipeline helpers (inference + training)
    train_dmd.py            # backward_simulate, encode_prompts, prepare_latents
    infer_*.py              # Inference pipelines
  utils/                    # Utilities (config, EMA, scheduler, data, checkpointing)
    config.py               # Typed dataclass configs (DMDTrainConfig, etc.)
    ema.py                  # EMATracker (DeepSpeed-compatible) + EMA_FSDP
    schedulers.py           # FlowMatchScheduler
  legacy/                   # Legacy class-based code (being replaced)
  trainers/                 # Legacy trainer wrappers (thin bridges)

configs/
  v2/                       # V2 typed configs (draccus dataclass-based)
    train_distillation.yaml # DMD distillation config (maps to DMDTrainConfig)
    accelerate_config.yaml  # DeepSpeed ZeRO-2, 8 GPUs, bf16
  *.yaml                    # Legacy OmegaConf configs

tests/                      # Unit and integration tests
```

## Development Workflow

This project uses a **local-edit, remote-execute** model with [just](https://github.com/casey/just):

```bash
# Set up .env with REMOTE and REMOTE_DIR
just sync                   # Sync code to remote GPU server
just test                   # Run tests remotely
just train-distillation     # Launch DMD distillation training
just rrun "..."             # Run arbitrary command on remote
```

## T2V Training

DMD training for bidirectional models does not need ODE initialization.

### Dataset Preparation

Build a text-folder dataset where each file contains a single prompt:

```
data_folder/
  1.txt
  2.txt
  ...
  N.txt
```

### DMD Distillation Training

The training script is self-contained and functional — no class hierarchies. It uses
`accelerate` for distribution (DeepSpeed ZeRO-2) and `draccus` for typed configs.

**1. Train the high-noise model**

```bash
# Via just (recommended)
just train-distillation

# Or directly
accelerate launch --config_file configs/v2/accelerate_config.yaml \
  scripts/train_distillation.py --config_path configs/v2/train_distillation.yaml
```

Key config fields (see `configs/v2/train_distillation.yaml`):

| Field | Default | Description |
|-------|---------|-------------|
| `generator_name` | `Wan2.2-T2V-A14B/high_noise_model` | Model path under `wan_models/` |
| `target` | `high_noise` | Training target region |
| `boundary_step` | `500` | Boundary between high/low noise regions |
| `denoising_step_list` | `[1000, 750]` | Backward simulation denoising steps |
| `dfake_gen_update_ratio` | `5` | Generator:critic update ratio |
| `lr` / `lr_critic` | `2e-6` / `4e-7` | Learning rates |
| `ema_weight` | `0.99` | EMA decay (starts at step 200) |

Override any field via CLI: `just train-distillation --lr 1e-6 --batch_size 2`

**2. Convert checkpoint to safetensors**

```bash
python scripts/convert_checkpoint.py \
  --input-checkpoint ./logs/dmd_distillation/checkpoint_model_002000/model.pt \
  --output-checkpoint wan_models/Wan2.2-T2V-A14B/distill_models/high_noise_model/distill_model.safetensors \
  --to-bf16
```

**3. Train the low-noise model** (requires high-noise checkpoint)

```bash
accelerate launch --config_file configs/v2/accelerate_config.yaml \
  scripts/train_distillation.py --config_path configs/v2/train_distillation_low.yaml
```

### Inference

```bash
accelerate launch scripts/infer_bidirectional_fewstep.py --config_path configs/v2/infer_bidirectional_fewstep.yaml
accelerate launch scripts/infer_causal_fewstep.py --config_path configs/v2/infer_causal_fewstep.yaml
```

## I2V Training

### Dataset Preparation

1. Generate videos using the original Wan2.1 model.

2. Generate VAE latents:
```bash
python scripts/compute_vae_latent.py \
  --input_video_folder {video_folder} \
  --output_latent_folder {latent_folder} \
  --model_name Wan2.1-T2V-14B \
  --prompt_folder {prompt_folder}
```

3. Create LMDB dataset:
```bash
python scripts/create_lmdb_14b_shards.py \
  --data_path {latent_folder} \
  --prompt_path {prompt_folder} \
  --lmdb_path {lmdb_folder}
```

### DMD Training

Same as T2V but with I2V config pointing to `Wan2.2-I2V-A14B`.

## Testing

```bash
just test                   # Run all tests on remote
just rrun "uv run pytest tests/test_train_dmd.py -v"  # Run specific tests
```

## Acknowledgements

This codebase is built on top of [CausVid](https://github.com/tianweiy/CausVid), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), and [Wan2.1](https://github.com/Wan-Video/Wan2.1).
