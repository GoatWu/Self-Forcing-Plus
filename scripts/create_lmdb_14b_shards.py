"""Compatibility entrypoint. Use scripts/create_lmdb_shards.py."""

import _bootstrap  # noqa: F401

from create_lmdb_shards import main


if __name__ == "__main__":
    main()
