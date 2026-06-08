#!/usr/bin/env python3
"""Single-process training entry point (PRD Phase 2/3).

Use ``--smoke`` to run a short random-policy episode loop for wiring checks
(no learning). Without ``--smoke`` this runs local MLP+PPO updates.

Usage:
    python scripts/train.py --config configs/train/ppo_mlp.yaml \
        --task configs/tasks/gruz_mother.yaml [--smoke]

"""

from __future__ import annotations

from hkrl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
