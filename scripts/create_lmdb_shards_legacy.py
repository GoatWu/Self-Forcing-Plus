"""
python create_lmdb_14b_shards.py \
--data_path /mnt/localssd/wanx_14b_data \
--lmdb_path /mnt/localssd/wanx_14B_shift-3.0_cfg-5.0_lmdb
"""

from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os
import imageio
from PIL import Image

from sfp.legacy.utils.lmdb import store_arrays_to_lmdb, process_data_dict


def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="path to ode pairs"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True, help="path to prompt folder"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="path to video folder"
    )
    parser.add_argument("--lmdb_path", type=str, required=True, help="path to lmdb")
    parser.add_argument("--num_shards", type=int, default=16, help="num_shards")

    args = parser.parse_args()

    # figure out the maximum map size needed
    map_size = int(1e12)  # adapt to your need, set to 1TB by default
    os.makedirs(args.lmdb_path, exist_ok=True)
    # 1) Open one LMDB env per shard
    envs = []
    num_shards = args.num_shards
    for shard_id in range(num_shards):
        print("shard_id ", shard_id)
        path = os.path.join(args.lmdb_path, f"shard_{shard_id}")
        env = lmdb.open(
            path,
            map_size=map_size,
            subdir=True,  # set to True if you want a directory per env
            readonly=False,
            metasync=True,
            sync=True,
            lock=True,
            readahead=False,
            meminit=False,
        )
        envs.append(env)

    counters = [0] * num_shards
    seen_prompts = set()  # for deduplication
    neg_prompts = set()
    total_samples = 0
    all_files = []
    all_files += sorted(glob.glob(os.path.join(args.data_path, "*.pt")))
    print(f"get {len(all_files)} .pt files.")

    prompt_to_filename = {}
    if os.path.exists(args.prompt_path):
        prompt_files = glob.glob(os.path.join(args.prompt_path, "*.txt"))
        print(f"Found {len(prompt_files)} prompt files.")

        for idx, prompt_file in tqdm(enumerate(prompt_files)):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_content = f.read().strip()
                if len(prompt_content) < 300:
                    neg_prompts.add(prompt_content)
                    continue
                filename = os.path.basename(prompt_file)
                prompt_to_filename[prompt_content] = filename
            except Exception as e:
                print(f"Error reading prompt file {prompt_file}: {e}")
                continue
    else:
        print(f"Warning: Prompt path {args.prompt_path} does not exist.")

    print("start negative prompts ---------------")
    for prompt in neg_prompts:
        print(prompt)
    print("end negative prompts -----------------")

    # 2) Prepare a write transaction for each shard
    for idx, file in tqdm(enumerate(all_files)):
        try:
            data_dict = torch.load(file)
            data_dict = process_data_dict(data_dict, seen_prompts)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

        if data_dict["latents"].shape != (1, 21, 16, 60, 104):
            continue

        if len(data_dict["prompts"][0]) < 300:
            continue

        try:
            current_filename = os.path.basename(file)

            prompt_text = data_dict["prompts"][0]
            if prompt_text in prompt_to_filename:
                corresponding_filename = prompt_to_filename[prompt_text]
                video_filename = corresponding_filename.replace(".txt", ".mp4")
                video_path = os.path.join(args.video_path, video_filename)

                if os.path.exists(video_path):
                    try:
                        reader = imageio.get_reader(video_path)
                        frame = reader.get_data(0)
                        reader.close()
                        data_dict["img"] = [Image.fromarray(frame)]
                    except Exception as e:
                        print(
                            f"Warning: Cannot read first frame from {video_path}: {e}"
                        )
                        continue
                else:
                    print(f"Warning: Video file not found: {video_path}")
                    continue

            else:
                print(f"Warning: No matching file found for {current_filename}.")
                continue

        except Exception as e:
            print(f"Error processing file {current_filename}: {e}")
            continue

        shard_id = idx % num_shards
        # write to lmdb file
        store_arrays_to_lmdb(envs[shard_id], data_dict, start_index=counters[shard_id])
        counters[shard_id] += len(data_dict["prompts"])
        data_shape = data_dict["latents"].shape
        total_samples += 1

    print(len(seen_prompts))

    # save each entry's shape to lmdb
    for shard_id, env in enumerate(envs):
        with env.begin(write=True) as txn:
            for key, val in data_dict.items():
                assert len(data_shape) == 5
                array_shape = np.array(data_shape)  # val.shape)
                array_shape[0] = counters[shard_id]
                shape_key = f"{key}_shape".encode()
                print(shape_key, array_shape)
                shape_str = " ".join(map(str, array_shape))
                txn.put(shape_key, shape_str.encode())

    print(
        f"Total {len(all_files)} videos. Finished writing {total_samples} examples into {num_shards} shards under {args.lmdb_path}"
    )


if __name__ == "__main__":
    main()
