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
        replay_sink: Callable[[dict[str, Any]], None] | None = None,
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
        self.replay_sink = replay_sink

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
                    self._run_episode(
                        env,
                        self.seeds[idx % len(self.seeds)],
                        task=task,
                        episode_index=idx,
                    )
                    for idx in range(episodes_per_task)
                ]
            finally:
                if hasattr(env, "close"):
                    env.close()
            results[task.task_id] = _aggregate(episodes)
        return results

    def _run_episode(
        self,
        env: Any,
        seed: int,
        *,
        task: TaskConfig,
        episode_index: int,
    ) -> _EpisodeResult:
        obs, info = env.reset(seed=seed)
        rnn_state = self._initial_rnn_state()
        total_reward = 0.0
        damage_taken = 0.0
        damage_dealt = 0.0
        heal_count = 0
        heal_amount = 0.0
        invalid_actions = 0
        death_reason = 0
        won = False
        terminated = False
        truncated = False
        steps = 0

        for step in range(self.max_steps_per_episode):
            action, rnn_state = self._act(env, obs, info, rnn_state)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps = step + 1

            event_metrics = _metrics_from_info(info)
            damage_taken += event_metrics.damage_taken
            damage_dealt += event_metrics.damage_dealt
            heal_count += event_metrics.heal_count
            heal_amount += event_metrics.heal_amount
            invalid_actions += event_metrics.invalid_actions
            if event_metrics.death_reason:
                death_reason = event_metrics.death_reason
            won = won or event_metrics.won
            self._emit_replay_step(
                task=task,
                seed=seed,
                episode_index=episode_index,
                step=steps,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                event_metrics=event_metrics,
            )

            if terminated or truncated:
                break

        return _EpisodeResult(
            won=won,
            reward=total_reward,
            length=steps,
            damage_taken=damage_taken,
            damage_dealt=damage_dealt,
            heal_count=heal_count,
            heal_amount=heal_amount,
            invalid_action_ratio=invalid_actions / max(steps, 1),
            death_reason=death_reason,
            time_to_kill=float(steps) if won else 0.0,
            terminated=terminated,
            truncated=truncated,
        )

    def _initial_rnn_state(self) -> Any:
        if not isinstance(self.model, ActorCritic):
            return None
        device = _model_device(self.model)
        return self.model.initial_state(batch_size=1, device=device)

    def _act(self, env: Any, obs: Any, info: dict[str, Any], rnn_state: Any) -> tuple[Any, Any]:
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
            action, _, _, next_state = self.model.act(
                obs_tensor,
                rnn_state=rnn_state,
                action_mask=mask_tensor,
                deterministic=True,
            )
            enable_macro = "macro" in env.action_space.spaces
            return action_tensor_to_env_action(action[0], enable_macro=enable_macro), next_state
        return self.model.act(obs, action_mask), None

    def _emit_replay_step(
        self,
        *,
        task: TaskConfig,
        seed: int,
        episode_index: int,
        step: int,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        event_metrics: _EventMetrics,
    ) -> None:
        if self.replay_sink is None:
            return
        self.replay_sink(
            {
                "action": _jsonable(action),
                "damage_dealt": event_metrics.damage_dealt,
                "damage_taken": event_metrics.damage_taken,
                "death_reason": event_metrics.death_reason,
                "episode": episode_index,
                "episode_id": int(info.get("episode_id", 0)),
                "heal_amount": event_metrics.heal_amount,
                "heal_count": event_metrics.heal_count,
                "invalid_actions": event_metrics.invalid_actions,
                "reward": float(reward),
                "seed": int(seed),
                "step": int(step),
                "task_id": task.task_id,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "type": "eval_step",
                "wire_id": int(task.wire_id),
                "won": event_metrics.won,
            }
        )

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
    heal_count: int
    heal_amount: float
    invalid_action_ratio: float
    death_reason: int
    time_to_kill: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class _EventMetrics:
    won: bool = False
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    heal_count: int = 0
    heal_amount: float = 0.0
    invalid_actions: int = 0
    death_reason: int = 0


def _metrics_from_info(info: dict[str, Any]) -> _EventMetrics:
    won = bool(info.get("won", False))
    damage_taken = float(info.get("damage_taken", 0.0))
    damage_dealt = float(info.get("damage_dealt", 0.0))
    heal_count = int(info.get("heal_count", 0))
    heal_amount = float(info.get("heal_amount", 0.0))
    invalid_actions = int(info.get("invalid_actions", 0))
    death_reason = int(info.get("death_reason", 0))

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
        elif kind == protocol.RewardEventKind.HEAL:
            heal_count += 1
            heal_amount += amount
        elif kind == protocol.RewardEventKind.PLAYER_DEATH:
            death_reason = _event_aux_int(event) or 1
        elif kind == protocol.RewardEventKind.INVALID_ACTION:
            invalid_actions += 1

    return _EventMetrics(
        won=won,
        damage_taken=damage_taken,
        damage_dealt=damage_dealt,
        heal_count=heal_count,
        heal_amount=heal_amount,
        invalid_actions=invalid_actions,
        death_reason=death_reason,
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


def _event_aux_int(event: Any) -> int:
    value = getattr(event, "aux_int", getattr(event, "AuxInt", 0))
    if callable(value):
        value = value()
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _aggregate(episodes: Sequence[_EpisodeResult]) -> dict[str, float]:
    wins = [episode for episode in episodes if episode.won]
    deaths = [episode for episode in episodes if episode.death_reason]
    win_rate = _mean(float(episode.won) for episode in episodes)
    damage_taken = _mean(episode.damage_taken for episode in episodes)
    damage_dealt = _mean(episode.damage_dealt for episode in episodes)
    return {
        "win_rate": win_rate,
        "episode_reward": _mean(episode.reward for episode in episodes),
        "episode_length": _mean(float(episode.length) for episode in episodes),
        "damage_taken": damage_taken,
        "damage_dealt": damage_dealt,
        "heal_count": _mean(float(episode.heal_count) for episode in episodes),
        "heal_amount": _mean(episode.heal_amount for episode in episodes),
        "invalid_action_ratio": _mean(episode.invalid_action_ratio for episode in episodes),
        "death_rate": _mean(float(episode.death_reason != 0) for episode in episodes),
        "death_reason": _mean(float(episode.death_reason) for episode in deaths),
        "time_to_kill": _mean(episode.time_to_kill for episode in wins),
        "per_boss_win_rate": win_rate,
        "per_boss_damage_ratio": _safe_ratio(damage_taken, damage_dealt),
        "termination_rate": _mean(float(episode.terminated) for episode in episodes),
        "truncation_rate": _mean(float(episode.truncated) for episode in episodes),
    }


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
