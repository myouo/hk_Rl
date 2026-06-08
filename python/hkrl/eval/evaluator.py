"""Evaluator (PRD §4.2, §13).

Runs the policy on fixed seeds/tasks, isolated from training, and reports
shaping-free metrics: per-boss win rate, damage taken, time-to-kill, invalid
action ratio, generalization, and old-task regression. Guards against the
"reward up, win rate down" failure (docs/metrics.md §2, PRD §9.4).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hkrl import protocol
from hkrl.models.base import ActorCritic
from hkrl.utils.config import TaskConfig
from hkrl.worker.game_worker import action_tensor_to_env_action


class Evaluator:
    """Deterministic per-task evaluation harness."""

    def __init__(
        self,
        model: Any,
        tasks: Sequence[TaskConfig],
        seeds: Sequence[int],
        env_factory: Callable[[TaskConfig], Any] | None = None,
        max_steps_per_episode: int = 4096,
    ) -> None:
        if not tasks:
            raise ValueError("tasks must not be empty")
        if not seeds:
            raise ValueError("seeds must not be empty")
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")

        self.model = model
        self.tasks = list(tasks)
        self.seeds = list(seeds)
        self.env_factory = env_factory
        self.max_steps_per_episode = max_steps_per_episode

    def evaluate(self, episodes_per_task: int = 20) -> dict[str, dict[str, float]]:
        """Return ``{task_id: {win_rate, damage_taken, time_to_kill, ...}}``."""
        if episodes_per_task <= 0:
            raise ValueError("episodes_per_task must be positive")
        if self.env_factory is None:
            raise ValueError("env_factory is required for evaluation")

        results: dict[str, dict[str, float]] = {}
        for task in self.tasks:
            env = self.env_factory(task)
            try:
                episodes = [
                    self._run_episode(env, self.seeds[idx % len(self.seeds)])
                    for idx in range(episodes_per_task)
                ]
            finally:
                if hasattr(env, "close"):
                    env.close()
            results[task.task_id] = _aggregate(episodes)
        return results

    def _run_episode(self, env: Any, seed: int) -> _EpisodeResult:
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        damage_taken = 0.0
        damage_dealt = 0.0
        invalid_actions = 0
        won = False
        terminated = False
        truncated = False
        steps = 0

        for step in range(self.max_steps_per_episode):
            action = self._act(env, obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps = step + 1

            event_metrics = _metrics_from_info(info)
            damage_taken += event_metrics.damage_taken
            damage_dealt += event_metrics.damage_dealt
            invalid_actions += event_metrics.invalid_actions
            won = won or event_metrics.won

            if terminated or truncated:
                break

        return _EpisodeResult(
            won=won,
            reward=total_reward,
            length=steps,
            damage_taken=damage_taken,
            damage_dealt=damage_dealt,
            invalid_action_ratio=invalid_actions / max(steps, 1),
            time_to_kill=float(steps) if won else 0.0,
            terminated=terminated,
            truncated=truncated,
        )

    def _act(self, env: Any, obs: Any, info: dict[str, Any]) -> Any:
        action_mask = info.get("action_mask")
        if isinstance(self.model, ActorCritic):
            device = _model_device(self.model)
            obs_tensor = _obs_to_tensor(obs, device)
            mask_tensor = None
            if action_mask is not None:
                mask_tensor = torch.as_tensor(
                    np.asarray(action_mask, dtype=bool)[None, :],
                    dtype=torch.bool,
                    device=device,
                )
            action, _, _, _ = self.model.act(
                obs_tensor,
                action_mask=mask_tensor,
                deterministic=True,
            )
            enable_macro = "macro" in env.action_space.spaces
            return action_tensor_to_env_action(action[0], enable_macro=enable_macro)
        return self.model.act(obs, action_mask)

    def regression_report(
        self, baseline: dict[str, dict[str, float]], current: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Per-task win-rate delta vs a baseline (catastrophic-forgetting check)."""
        task_ids = sorted(set(baseline) | set(current))
        return {
            task_id: current.get(task_id, {}).get("win_rate", 0.0)
            - baseline.get(task_id, {}).get("win_rate", 0.0)
            for task_id in task_ids
        }


@dataclass(frozen=True)
class _EpisodeResult:
    won: bool
    reward: float
    length: int
    damage_taken: float
    damage_dealt: float
    invalid_action_ratio: float
    time_to_kill: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class _EventMetrics:
    won: bool = False
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    invalid_actions: int = 0


def _metrics_from_info(info: dict[str, Any]) -> _EventMetrics:
    won = bool(info.get("won", False))
    damage_taken = float(info.get("damage_taken", 0.0))
    damage_dealt = float(info.get("damage_dealt", 0.0))
    invalid_actions = int(info.get("invalid_actions", 0))

    for event in info.get("reward_events", []):
        kind = _event_kind(event)
        amount_value = getattr(event, "amount", getattr(event, "Amount", 0.0))
        if callable(amount_value):
            amount_value = amount_value()
        amount = float(amount_value)
        if kind == protocol.RewardEventKind.BOSS_KILLED:
            won = True
        elif kind == protocol.RewardEventKind.DAMAGE_TAKEN:
            damage_taken += amount
        elif kind == protocol.RewardEventKind.DAMAGE_DEALT:
            damage_dealt += amount
        elif kind == protocol.RewardEventKind.INVALID_ACTION:
            invalid_actions += 1

    return _EventMetrics(
        won=won,
        damage_taken=damage_taken,
        damage_dealt=damage_dealt,
        invalid_actions=invalid_actions,
    )


def _event_kind(event: Any) -> protocol.RewardEventKind | None:
    kind = getattr(event, "kind", getattr(event, "Kind", None))
    if kind is None:
        return None
    if callable(kind):
        kind = kind()
    try:
        return protocol.RewardEventKind(int(kind))
    except (TypeError, ValueError):
        return None


def _aggregate(episodes: Sequence[_EpisodeResult]) -> dict[str, float]:
    wins = [episode for episode in episodes if episode.won]
    return {
        "win_rate": _mean(float(episode.won) for episode in episodes),
        "episode_reward": _mean(episode.reward for episode in episodes),
        "episode_length": _mean(float(episode.length) for episode in episodes),
        "damage_taken": _mean(episode.damage_taken for episode in episodes),
        "damage_dealt": _mean(episode.damage_dealt for episode in episodes),
        "invalid_action_ratio": _mean(episode.invalid_action_ratio for episode in episodes),
        "time_to_kill": _mean(episode.time_to_kill for episode in wins),
        "termination_rate": _mean(float(episode.terminated) for episode in episodes),
        "truncation_rate": _mean(float(episode.truncated) for episode in episodes),
    }


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _obs_to_tensor(obs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "global": torch.as_tensor(obs["global"][None, :], dtype=torch.float32, device=device),
        "player": torch.as_tensor(obs["player"][None, :], dtype=torch.float32, device=device),
        "entities": torch.as_tensor(
            obs["entities"][None, :, :], dtype=torch.float32, device=device
        ),
        "entity_mask": torch.as_tensor(
            obs["entity_mask"][None, :], dtype=torch.bool, device=device
        ),
    }


def _model_device(model: ActorCritic) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
