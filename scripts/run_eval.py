#!/usr/bin/env python3
"""Run fixed-seed, shaping-free per-boss evaluation (PRD §13, docs/metrics.md).

Usage:
    python scripts/run_eval.py --checkpoint checkpoints/v123.pt \
        --tasks configs/tasks/hornet_protector.yaml configs/tasks/mantis_lords.yaml \
        --episodes 20

Reports per-boss win rate / damage taken / time-to-kill and (optionally) a
regression diff vs a baseline. Interface placeholder; lands in phase 3+.
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description="HKRL Evaluator")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--baseline", help="optional baseline metrics JSON for regression diff")
    p.parse_args()
    # TODO(phase-3): load model; build Evaluator(model, tasks, seeds); evaluate();
    # print per-boss shaping-free metrics; optional regression_report vs baseline.
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
