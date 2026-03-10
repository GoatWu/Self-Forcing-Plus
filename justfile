set dotenv-load := true

REMOTE := env_var_or_default("REMOTE", "dsw")
REMOTE_DIR := env_var_or_default("REMOTE_DIR", "/mnt/data/code/Self-Forcing-Plus")
UV_PYTHON_INSTALL_MIRROR := env_var_or_default("UV_PYTHON_INSTALL_MIRROR", "")

sync:
    rsync -az --delete \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.agent_logs' \
      --exclude 'wan_models' \
      --exclude 'outputs' \
      --exclude 'logs' \
      ./ {{REMOTE}}:{{REMOTE_DIR}}

rrun +args:
    just sync
    mkdir -p .agent_logs
    ssh -tt {{REMOTE}} "cd {{REMOTE_DIR}} && UV_NO_PROGRESS=1 UV_PYTHON_INSTALL_MIRROR={{UV_PYTHON_INSTALL_MIRROR}} uv sync --dev" 2>&1 | tail -5
    ssh -tt {{REMOTE}} "cd {{REMOTE_DIR}} && source .venv/bin/activate && {{args}}" 2>&1 | tee .agent_logs/last.log

test:
    just rrun "uv run pytest -q tests/"

typecheck:
    just rrun "uv run pyrefly check sfp scripts tests"

train-distillation +args='':
    just rrun "uv run accelerate launch --config_file configs/v2/accelerate_config.yaml scripts/train_distillation.py --config_path configs/v2/train_distillation.yaml {{args}}"

train-ode +args='':
    just rrun "uv run accelerate launch --config_file configs/v2/accelerate_config.yaml scripts/train_ode.py --config_path configs/v2/train_ode.yaml {{args}}"
