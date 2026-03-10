from __future__ import annotations

from copy import deepcopy
from typing import Any, TYPE_CHECKING

import os

if TYPE_CHECKING:
    from easydict import EasyDict


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_train_config(config_path: str, default_config_path: str) -> "EasyDict":
    import yaml

    with open(default_config_path, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f) or {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    merged = _deep_merge(default_cfg, cfg)
    merged["config_name"] = os.path.basename(config_path).split(".")[0]
    return EasyDict(merged)
