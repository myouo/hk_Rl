#!/usr/bin/env python3
"""Single-process training entry point (PRD Phase 2/3).

Builds env + model + algorithm from a train config (and a task config), runs the
local sampling + update loop. Use ``--smoke`` to run a short random-policy episode
loop for wiring checks (no learning).

Usage:
    python scripts/train.py --config configs/train/ppo_mlp.yaml \
        --task configs/tasks/gruz_mother.yaml [--smoke]

This is an interface placeholder; the loop lands in phase 2/3.
"""

from __future__ import annotations

import argparse


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL training entry point")
    p.add_argument("--config", required=True, help="path to a train config YAML")
    p.add_argument("--task", help="path to a task config YAML")
    p.add_argument("--smoke", action="store_true", help="random-policy wiring check")
    p.add_argument("--steps", type=int, default=None, help="override total steps")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    # TODO(phase-2/3):
    #   cfg  = hkrl.utils.config.load_train_config(args.config)
    #   task = hkrl.utils.config.load_task_config(args.task)
    #   transport = registry.build("transport", cfg.transport.name, ...)
    #   env   = HKRLEnv(transport, task)
    #   model = registry.build("model", cfg.model.name, ...)
    #   algo  = registry.build("algo", cfg.algorithm, model, cfg)
    #   GameWorker(env, model, cfg).run(args.steps)
    raise NotImplementedError("training loop not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
