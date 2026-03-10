from __future__ import annotations

import _bootstrap  # noqa: F401
import sys

import subprocess
import draccus

from sfp.utils.config import DemoCommandConfig


@draccus.wrap()
def main(cfg: DemoCommandConfig) -> None:
    cmd = [
        sys.executable,
        "-m",
        "sfp.legacy.demo_app",
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--checkpoint_path",
        cfg.checkpoint_path,
        "--config_path",
        cfg.legacy_config_path,
    ]
    if cfg.trt:
        cmd.append("--trt")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
