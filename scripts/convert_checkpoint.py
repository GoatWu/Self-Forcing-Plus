from __future__ import annotations

import argparse
import gc
import os

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and save generator weights from a checkpoint."
    )
    parser.add_argument(
        "--input-checkpoint",
        type=str,
        required=True,
        help="Path to input checkpoint file",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=str,
        required=True,
        help="Path to output checkpoint file",
    )
    parser.add_argument(
        "--remove-prefix",
        type=str,
        nargs="?",
        const="model.",
        default="model.",
        help='Prefix to remove from keys (default: "model.")',
    )
    parser.add_argument(
        "--to-bf16", action="store_true", help="Convert floating tensors to bfloat16"
    )
    parser.add_argument(
        "--ema", action="store_true", help="Extract generator_ema instead of generator"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_checkpoint):
        raise FileNotFoundError(f"Input checkpoint not found: {args.input_checkpoint}")

    checkpoint = torch.load(args.input_checkpoint, map_location=torch.device("cpu"))
    key = "generator_ema" if args.ema else "generator"
    if key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain key: {key}")

    prefix = args.remove_prefix
    output = {}
    for k, v in checkpoint[key].items():
        new_key = k[len(prefix) :] if k.startswith(prefix) else k
        new_key = (
            new_key.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        if args.to_bf16 and isinstance(v, torch.Tensor) and v.is_floating_point():
            v = v.to(torch.bfloat16)
        output[new_key] = v

    del checkpoint
    gc.collect()

    if args.output_checkpoint.endswith(".safetensors"):
        save_file(output, args.output_checkpoint)
    else:
        torch.save(output, args.output_checkpoint)


if __name__ == "__main__":
    main()
