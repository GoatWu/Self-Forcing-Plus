from __future__ import annotations

import torch


class EMATracker:
    """Lightweight EMA tracker that works with any distribution backend.

    Unlike EMA_FSDP which uses FSDP-specific summon_full_params,
    this reads parameters directly — compatible with DeepSpeed ZeRO-2
    (where params are fully replicated) and non-distributed training.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._init_shadow(model)

    @torch.no_grad()
    def _init_shadow(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name] = param.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(
                    param.detach().float().cpu(), alpha=1.0 - d
                )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype, device=param.device))


__all__ = ["EMATracker"]
