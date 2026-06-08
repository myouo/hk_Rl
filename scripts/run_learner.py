#!/usr/bin/env python3
"""Run the remote Learner server (GPU): batch training + checkpoint publish.

Usage:
    python scripts/run_learner.py --config configs/train/remote_learner.yaml

Interface placeholder; implementation lands in phase 6
(docs/distributed_training.md, PRD §8).
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="HKRL Learner server")
    p.add_argument("--config", required=True)
    p.add_argument("--bind", default="0.0.0.0:5600")
    p.parse_args()
    # TODO(phase-6): build model + algo + CheckpointRegistry; LearnerServer.serve().
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
