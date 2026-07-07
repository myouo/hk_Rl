"""Command-line entry points for local training and smoke runs."""

from __future__ import annotations

import argparse
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from hkrl.eval.scripted_policies import RandomPolicy
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.models import mlp as _mlp  # noqa: F401
from hkrl.models import recurrent_policy as _recurrent_policy  # noqa: F401
from hkrl.training import ppo as _ppo  # noqa: F401
from hkrl.training import recurrent_ppo as _recurrent_ppo  # noqa: F401
from hkrl.transport.factory import make_transport
from hkrl.utils.config import TrainConfig, load_task_config, load_train_config
from hkrl.utils.logging import MetricSink, make_sink
from hkrl.utils.registry import get
from hkrl.worker.game_worker import GameWorker


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKRL training entry point")
    parser.add_argument("--config", required=True, help="path to a train config YAML")
    parser.add_argument(
        "--task",
        default="configs/tasks/gruz_mother.yaml",
        help="path to a task config YAML",
    )
    parser.add_argument("--smoke", action="store_true", help="random-policy wiring check")
    parser.add_argument(
        "--host",
        default=None,
        help="override TCP env host from the train config",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="override TCP env port from the train config",
    )
    parser.add_argument("--steps", type=int, default=None, help="override total smoke steps")
    parser.add_argument("--updates", type=int, default=1, help="number of PPO updates to run")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="directory for PPO checkpoints",
    )
    parser.add_argument(
        "--metrics",
        default="runs/smoke.jsonl",
        help="metrics output path",
    )
    parser.add_argument(
        "--metrics-kind",
        choices=("jsonl", "csv"),
        default="jsonl",
        help="metrics sink backend",
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the clean reset lifecycle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    summary = run_smoke_from_args(args) if args.smoke else run_training_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_training_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)

    if cfg.algorithm not in {"ppo", "recurrent_ppo"}:
        raise ValueError(
            f"training CLI currently supports ppo/recurrent_ppo, got {cfg.algorithm!r}"
        )
    if cfg.algorithm == "ppo" and cfg.model.name != "mlp":
        raise ValueError("training CLI supports non-MLP models through recurrent_ppo")
    from hkrl.env import HKRLEnv
    from hkrl.wrappers import NormalizeObservation

    transport = _make_env_transport(cfg, args)
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    observation_space: Any = env.observation_space
    obs_dims = _obs_dims(observation_space)
    model = _build_model(
        cfg,
        obs_dims,
        enable_macro=task.action.enable_macro_actions,
        n_macros=task.action.n_macro_actions,
        max_entities=task.observation.max_entities,
    )
    worker = GameWorker(env=env, model=model, config=cfg)
    algo_cls = get("algo", cfg.algorithm)
    algo = algo_cls(model=model, config=cfg)
    sink = make_sink(args.metrics_kind, path=Path(args.metrics))

    try:
        return run_ppo_training_loop(
            worker=worker,
            algo=algo,
            sink=sink,
            updates=int(args.updates),
            checkpoint_dir=Path(args.checkpoint_dir),
            model=model,
        )
    finally:
        sink.close()
        env.close()


def run_smoke_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.config)
    task = load_task_config(args.task)

    from hkrl.env import HKRLEnv
    from hkrl.wrappers import NormalizeObservation

    transport = _make_env_transport(cfg, args)
    env = NormalizeObservation(HKRLEnv(transport=transport, task=task))
    policy = RandomPolicy(env.action_space, seed=cfg.seed)
    sink = make_sink(args.metrics_kind, path=Path(args.metrics))

    try:
        return run_random_policy_smoke(
            env=env,
            policy=policy,
            sink=sink,
            task_id=task.task_id,
            max_steps=args.steps or min(cfg.rollout_steps, 256),
            reset_timeout_s=float(args.reset_timeout),
        )
    finally:
        sink.close()
        env.close()


def run_random_policy_smoke(
    *,
    env: Any,
    policy: Any,
    sink: MetricSink,
    task_id: str,
    max_steps: int,
    reset_timeout_s: float = 30.0,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    obs, info = env.reset(options={"reset_timeout_s": reset_timeout_s})
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    for step in range(max_steps):
        action = policy.act(obs, info.get("action_mask"))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps = step + 1
        sink.log_scalar("reward", float(reward), step=steps)

        if terminated or truncated:
            break

    record = {
        "task_id": task_id,
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
    }
    sink.log_episode(record)
    sink.flush()
    return record


def _obs_dims(observation_space: Any) -> dict[str, tuple[int, ...]]:
    return {
        "global": observation_space["global"].shape,
        "player": observation_space["player"].shape,
        "entities": observation_space["entities"].shape,
        "entity_mask": observation_space["entity_mask"].shape,
    }


def _make_env_transport(cfg: TrainConfig, args: argparse.Namespace) -> Any:
    return make_transport(
        cfg,
        host=_optional_host(getattr(args, "host", None)),
        port=_optional_port(getattr(args, "port", None)),
    )


def _optional_host(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("host must not be empty")
    return value.strip()


def _optional_port(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("port must be an integer")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    return port


def _build_model(
    cfg: TrainConfig,
    obs_dims: dict[str, tuple[int, ...]],
    *,
    enable_macro: bool,
    n_macros: int,
    max_entities: int,
) -> Any:
    model_cls = get("model", cfg.model.name)
    if cfg.model.name == "mlp":
        return model_cls(
            obs_dims,
            hidden=cfg.model.rnn_hidden or 256,
            enable_macro=enable_macro,
            n_macros=n_macros,
        )
    return model_cls(
        obs_dims,
        entity_hidden=cfg.model.entity_hidden,
        attention_layers=cfg.model.attention_layers,
        attention_heads=cfg.model.attention_heads,
        rnn_type=cfg.model.rnn_type,
        rnn_hidden=cfg.model.rnn_hidden,
        enable_macro=enable_macro,
        n_macros=n_macros,
        max_entities=max_entities,
    )


def run_ppo_training_loop(
    *,
    worker: Any,
    algo: Any,
    sink: MetricSink,
    updates: int,
    checkpoint_dir: Path | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    if updates <= 0:
        raise ValueError("updates must be positive")

    last_metrics: dict[str, float] = {}
    last_checkpoint: str | None = None
    last_checkpoint_version: int | None = None
    total_steps = 0
    registry = (
        CheckpointRegistry(str(checkpoint_dir))
        if checkpoint_dir is not None and model is not None
        else None
    )
    for update in range(1, updates + 1):
        batch = worker.collect_rollout()
        metrics = algo.update(worker.buffer)
        last_metrics = {key: float(value) for key, value in metrics.items()}
        total_steps += int(np.asarray(batch.rewards).size)

        for key, value in last_metrics.items():
            sink.log_scalar(key, value, step=update)
        rollout_reward = float(np.asarray(batch.rewards, dtype=np.float32).sum())
        sink.log_episode(
            {
                "update": update,
                "rollout_steps": int(np.asarray(batch.rewards).size),
                "rollout_reward": rollout_reward,
                "policy_version": int(getattr(batch, "policy_version", 0)),
            }
        )
        if hasattr(worker, "policy_version"):
            worker.policy_version = update

        if registry is not None:
            assert model is not None
            meta = registry.publish(
                {
                    "model_state_dict": model.state_dict(),
                    "policy_version": update,
                    "update": update,
                    "step": total_steps,
                    "metrics": last_metrics,
                },
                policy_version=update,
                step=total_steps,
            )
            last_checkpoint = str(registry.resolve_path(meta))
            last_checkpoint_version = meta.version

    sink.flush()
    return {
        "updates": updates,
        "total_steps": total_steps,
        "last_metrics": last_metrics,
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_version": last_checkpoint_version,
        "policy_version": updates,
    }
