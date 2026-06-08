#!/usr/bin/env python3
"""Run a GameWorker (Game PC): local inference + rollout upload (PRD Phase 6).

Usage:
    python scripts/run_worker.py --config configs/train/remote_learner.yaml \
        --task configs/tasks/hornet_protector.yaml --learner host:5600

Interface placeholder; implementation lands in phase 6
(docs/distributed_training.md).
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="HKRL GameWorker")
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--learner", help="learner endpoint host:port")
    p.add_argument("--registry", help="checkpoint registry endpoint")
    p.parse_args()
    # TODO(phase-6): build env/model/worker; connect to learner+registry; worker.run().
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
