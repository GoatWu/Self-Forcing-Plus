import torch
import argparse
import os
import gc
from safetensors.torch import save_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and save the generator part from a checkpoint.')
    parser.add_argument('--input-checkpoint', type=str, required=True, help='Path to the input checkpoint file')
    parser.add_argument('--output-checkpoint', type=str, required=True, help='Path to save the output checkpoint file')
    parser.add_argument('--remove-prefix', type=str, nargs='?', const="model.", default="model.", help='Prefix to remove from keys (default: "model.")')
    parser.add_argument('--to-bf16', action='store_true', help='Convert model weights to bfloat16')
    parser.add_argument('--ema', action='store_true', help='Use EMA weights')
    args = parser.parse_args()
    
    # Extract arguments
    input_path = args.input_checkpoint
    output_path = args.output_checkpoint
    prefix_to_remove = args.remove_prefix
    convert_to_bf16 = args.to_bf16
    use_ema = args.ema
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input checkpoint file not found: {input_path}")
        return
    
    # Load the input checkpoint
    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location=torch.device('cpu'))

    model_type = "generator_ema" if use_ema else "generator"
    
    # Check if 'generator' key exists
    if model_type not in checkpoint:
        print(f"Error: The '{model_type}' key does not exist in the input checkpoint")
        return
    
    # Extract the generator
    generator = checkpoint[model_type]
    print(f"Successfully extracted '{model_type}' from input checkpoint")
    
    # Remove the specified prefix from keys
    new_generator = {}
    prefix_count = 0
    tensor_count = 0
    
    for key, value in generator.items():
        # Process key - remove prefix if needed
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]  # Remove the prefix
            prefix_count += 1
        else:
            new_key = key

        new_key = new_key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")
        
        # Convert tensor to bf16 if requested
        if convert_to_bf16 and isinstance(value, torch.Tensor) and value.is_floating_point():
            value = value.to(torch.bfloat16)
            tensor_count += 1
        
        new_generator[new_key] = value
    
    # Print processing summary
    print(f"Removed prefix '{prefix_to_remove}' from {prefix_count} keys")
    if convert_to_bf16:
        print(f"Converted {tensor_count} tensors to bfloat16")

    del checkpoint
    gc.collect()
    
    # Save the new checkpoint
    print(f"Saving generator to {output_path}...")
    
    # Choose save method based on file extension
    if output_path.endswith('.safetensors'):
        save_file(new_generator, output_path)
        print(f"Successfully saved generator to {output_path} (safetensors format)")
    elif output_path.endswith('.pt') or output_path.endswith('.pth'):
        torch.save(new_generator, output_path)
        print(f"Successfully saved generator to {output_path} (PyTorch format)")
    else:
        # Default to PyTorch format
        torch.save(new_generator, output_path)
        print(f"Successfully saved generator to {output_path} (PyTorch format - default)")

if __name__ == "__main__":
    main()
