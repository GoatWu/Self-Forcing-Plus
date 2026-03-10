# AGENTS.md

## Execution Model

**Local-first, remote execution**: Code is edited locally, executed remotely on GPU server.

- **Never assume local GPU execution** (local Python 3.14 has no torch)
- All execution goes through `just` commands (see `justfile`)
- Never call raw `ssh` or `rsync`
- Logs are in `.agent_logs/last.log` - inspect after failures

### Available Commands

```bash
just sync                   # Sync code to remote
just rrun "..."             # Execute arbitrary command on remote
just test                   # Run pytest on remote
just typecheck              # Run pyrefly type check on remote
just train-distillation     # Launch DMD distillation training
```

### Remote Setup

Configure `.env` (gitignored):

```env
REMOTE=dsw
REMOTE_DIR=/mnt/data/code/ygao/Projects/SFP/
```

## Environment Management

Use **uv** exclusively:

```bash
uv sync --dev               # Install all deps
uv run python script.py     # Run a script
uv run pytest -q tests/     # Run tests
```

**Never use**: pip, conda, venv, poetry

## Project Architecture

### Key Directories

```
scripts/          Entry points (CLI). Each has @draccus.wrap() main()
sfp/model/        Model wrappers (WanDiffusionWrapper, etc.) — do NOT modify
sfp/pipelines/    Pipeline helpers (inference + training functions)
sfp/utils/        Shared utilities (config, EMA, scheduler, data, checkpointing)
sfp/legacy/       Legacy class-based code — being replaced file by file
configs/v2/       Typed configs (draccus dataclass-based YAML)
tests/            Unit + integration tests
```

### Key Files

| File | Purpose |
|------|---------|
| `scripts/train_distillation.py` | DMD distillation training (functional, self-contained) |
| `sfp/pipelines/train_dmd.py` | Reusable pipeline helpers (backward_simulate, encode_prompts) |
| `sfp/utils/config.py` | All typed config dataclasses (DMDTrainConfig, etc.) |
| `sfp/utils/ema.py` | EMATracker (DeepSpeed-compatible) + legacy EMA_FSDP |
| `sfp/model/wan_wrapper.py` | WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper |
| `configs/v2/accelerate_config.yaml` | DeepSpeed ZeRO-2, 8 GPUs, bf16 |

### Distribution Strategy

- **DeepSpeed ZeRO-2** via `accelerate` (not FSDP)
- Config in `configs/v2/accelerate_config.yaml`
- `accelerator.prepare()` handles wrapping; `accelerator.backward()` for loss
- EMATracker reads params directly (ZeRO-2 keeps full params replicated)

## Configuration

Use **draccus** (not Hydra):

```python
from dataclasses import dataclass
import draccus

@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 32

@draccus.wrap()
def main(cfg: TrainConfig):
    train(cfg)
```

### draccus API Reference

```python
draccus.load(cls, path)     # Load config from YAML file
draccus.encode(cfg)         # Convert config to dict (e.g. for wandb)
draccus.wrap()              # Decorator for CLI entry points
```

- Use typed dataclasses in `sfp/utils/config.py`
- Avoid runtime config mutation
- Avoid global config objects
- Avoid Hydra

## Python Design Principles

### Composition Over Inheritance

```python
# GOOD: flat functions with explicit args
def compute_dmd_loss(generator, fake_score, real_score, shape, cond, ...):
    ...

# BAD: deep class hierarchies
class DMD(SelfForcingModel(BaseModel(nn.Module))):
    ...
```

### Functional Bias

- Prefer small, pure functions over stateful classes
- Training scripts should be self-contained with flat helper functions
- Pipeline helpers in `sfp/pipelines/` for reusable operations
- Avoid hidden state, global mutation, side effects at import time

### Explicit Entry Points

Every executable module must:
- Define `main(cfg)` with `@draccus.wrap()`
- Be CLI-callable
- Avoid executing logic at import time
- Use `try: import _bootstrap` for path setup compatibility

## Type Safety

### Required Type Annotations

- All new functions and public APIs must include type annotations
- Prefer precise types over `Any`
- Use `Protocol` / structural typing when appropriate

### Type Checking

- Use **pyrefly** for static type checking: `just typecheck`
- Fix type errors introduced as part of changes
- Prefer local, minimal typing fixes over broad type loosening

### Tensor Types: jaxtyping

```python
from jaxtyping import Float
import torch

def normalize(x: Float[torch.Tensor, "b c h w"]) -> Float[torch.Tensor, "b c h w"]:
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
```

### Runtime Checks: beartype

Use `@beartype` on boundary functions (CLI, dataloaders, RPC):
- Prefer on small/medium pure functions
- Avoid in hot inner loops unless profiling shows it's acceptable

## Deep Learning Guidelines

- Typed config objects (dataclasses in `sfp/utils/config.py`)
- Explicit dependency injection (pass models/schedulers as args)
- No singleton trainers
- No registry magic
- No metaclasses
- No implicit global seeds

**Explicit > implicit**
**Small functions > giant classes**
**Composition > inheritance**

## Change Discipline

- Make minimal, targeted diffs
- Avoid unrelated refactors
- Preserve formatting and style
- Maintain backward compatibility unless explicitly instructed
- Large architectural changes require explicit justification

## Failure Handling

1. Inspect `.agent_logs/last.log`
2. Identify root cause
3. Apply minimal fix
4. Re-run verification
5. Log the change

**Never attempt speculative multi-file rewrites.**

## Verification Workflow

After making changes:

```bash
just test                   # Run all tests on remote
just typecheck              # Type check
```

If failure:
- Inspect `.agent_logs/last.log`
- Fix root cause only
- Re-run

## Prohibited Patterns

- Framework sprawl
- Implicit runtime magic
- Registry-based architecture
- Dynamic monkey patching
- Metaclass-heavy design
- Unnecessary decorators
- Global mutable state
- Introducing Hydra
- FSDP-specific APIs in new code (use DeepSpeed-compatible patterns)
