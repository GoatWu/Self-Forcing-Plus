from __future__ import annotations

import _bootstrap  # noqa: F401
import sys

import draccus

import create_lmdb_shards_legacy as legacy
from sfp.utils.config import CreateLmdbShardsConfig


@draccus.wrap()
def main(cfg: CreateLmdbShardsConfig) -> None:
    argv = [
        "create_lmdb_shards_legacy.py",
        "--data_path",
        cfg.data_path,
        "--prompt_path",
        cfg.prompt_path,
        "--video_path",
        cfg.video_path,
        "--lmdb_path",
        cfg.lmdb_path,
        "--num_shards",
        str(cfg.num_shards),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        legacy.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
