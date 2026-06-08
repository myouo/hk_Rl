"""Deterministic seeding for reproducible training/eval.

Evaluation uses fixed seeds per task (docs/metrics.md §2); training seeds come
from TrainConfig.seed.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python ``random``, NumPy, and torch (CPU+CUDA).

    Set ``deterministic_torch`` for evaluation reproducibility (slower).
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
