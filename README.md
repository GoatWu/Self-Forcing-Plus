<p align="center">
<h1 align="center">Self Forcing Plus</h1>

Self-Forcing-Plus focuses on step distillation and CFG distillation for bidirectional models. Building upon Self-Forcing, we support 4-step T2V-14B model training and higher quality 4-step I2V-14B model training.

## 🔥 News
- (2025/09) Support Wan2.2-Moe distillation! [wan22](https://github.com/GoatWu/Self-Forcing-Plus/tree/wan22)

| Model Type | Model Link |
|------------|---------------|
| Wan2.1-T2V-14B | [Huggingface](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) |
| Wan2.1-I2V-14B-480P | [Huggingface](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v) |

## Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir wan_models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir wan_models/Wan2.2-I2V-A14B
```

## T2V Training

DMD training for bidirectional models do not need ODE initialization.

### DataSet Preparation

We build the dataset in the following way, each file contains a single prompt:

```
data_folder
  |__1.txt
  |__2.txt
  ...
  |__xxx.txt
```

### DMD Training

1. Train the high_noise_model

```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/wan22_high.yaml \
--logdir logs/wan22_high \
--no_visualize \
--disable-wandb
```

2. Convert the checkpoint into .safetensors format

```
python convert_checkpoint.py --input-checkpoint ./logs/wan22_high_t2v/checkpoint_model_002000/model.pt --output-checkpoint Wan2.2-T2V-A14B/distill_models/high_noise_model/distill_model.safetensors --to-bf16
```

3. Train the low_noise_model

```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/wan22_low.yaml \
--logdir logs/wan22_low \
--no_visualize \
--disable-wandb
```

## I2V Training

### DataSet Preparation

1. Generate a series of videos using the original Wan2.1 model.

2. Generate the VAE latents.
```bash
python scripts/compute_vae_latent.py \
--input_video_folder {video_folder} \
--output_latent_folder {latent_folder} \
--model_name Wan2.1-T2V-14B \
--prompt_folder {prompt_folder}
```

3. Separate the first frame of the videos and create an lmdb dataset.
```bash
python scripts/create_lmdb_14b_shards.py \
--data_path {latent_folder} \
--prompt_path {prompt_folder} \
--lmdb_path {lmdb_folder}
```

### DMD Training

1. Train the high_noise_model

```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/wan22_high_i2v.yaml \
--logdir logs/wan22_high_i2v \
--no_visualize \
--disable-wandb
```

2. Convert the checkpoint into .safetensors format

```
python convert_checkpoint.py --input-checkpoint ./logs/wan22_high_i2v/checkpoint_model_001000/model.pt --output-checkpoint Wan2.2-I2V-A14B/distill_models/high_noise_model/distill_model.safetensors --to-bf16
```

3. Train the low_noise_model

```
torchrun --nnodes=8 --nproc_per_node=8 \
--rdzv_id=5235 \
--rdzv_backend=c10d \
--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
train.py \
--config_path configs/wan22_low_i2v.yaml \
--logdir logs/wan22_low_i2v \
--no_visualize \
--disable-wandb
```

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid), [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.
