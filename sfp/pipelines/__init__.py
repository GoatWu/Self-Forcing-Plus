"""Pipeline modules for inference and training.

Imports are kept lazy to avoid pulling in heavy optional dependencies
at package-level import time.
"""

__all__ = [
    "infer_causal_fewstep",
    "infer_causal_diffusion",
    "infer_bidirectional_fewstep",
    "infer_bidirectional_diffusion",
    "train_dmd",
    "train_ode",
]
