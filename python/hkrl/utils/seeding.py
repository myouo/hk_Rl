"""Deterministic seeding for reproducible training/eval.

Evaluation uses fixed seeds per task (docs/metrics.md §2); training seeds come
from TrainConfig.seed.
"""

from __future__ import annotations


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python ``random``, NumPy, and torch (CPU+CUDA).

    Set ``deterministic_torch`` for evaluation reproducibility (slower).

    TODO(phase-2): implement; import torch lazily so the package imports without it.
    """
    raise NotImplementedError
