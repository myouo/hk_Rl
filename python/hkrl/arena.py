"""Continuous boss-arena supervision with clean terminal auto-reset.

This is intentionally outside the Mod.  A terminal observation is first
preserved for rollout/evaluation, then the supervisor issues the ordinary RESET
handshake and verifies that a new episode id is returned.  No Boss state is
written or repaired between attempts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from hkrl import protocol

ArenaPolicy = Callable[[dict[str, np.ndarray], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ArenaEpisodeResult:
    episode_index: int
    episode_id: int
    next_episode_id: int
    reset_succeeded: bool
    won: bool
    died: bool
    hitless: bool
    target_met: bool
    damage_dealt: float
    damage_taken: float
    elapsed_seconds: float
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    time_limit_reached: bool


class BossArenaSupervisor:
    """Run consecutive attempts and auto-reset after every terminal outcome."""

    def __init__(
        self,
        env: Any,
        policy: ArenaPolicy,
        *,
        reset_timeout_s: float = 60.0,
        max_steps_per_episode: int = 4096,
    ) -> None:
        if reset_timeout_s <= 0.0:
            raise ValueError("reset_timeout_s must be positive")
        if max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")
        self.env = env
        self.policy = policy
        self.reset_timeout_s = reset_timeout_s
        self.max_steps_per_episode = max_steps_per_episode

    def run(self, episodes: int) -> tuple[ArenaEpisodeResult, ...]:
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        obs, info = self._reset()
        results: list[ArenaEpisodeResult] = []
        for episode_index in range(episodes):
            episode_id = int(info.get("episode_id", 0))
            total_reward = 0.0
            damage_dealt = 0.0
            damage_taken = 0.0
            won = False
            died = False
            terminated = False
            truncated = False
            time_limit_reached = False
            steps = 0

            for step in range(self.max_steps_per_episode):
                action = self.policy(obs, info)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                steps = step + 1
                for event in info.get("reward_events", ()):
                    kind = _event_kind(event)
                    amount = _event_amount(event)
                    if kind == protocol.RewardEventKind.DAMAGE_DEALT:
                        damage_dealt += amount
                    elif kind == protocol.RewardEventKind.DAMAGE_TAKEN:
                        damage_taken += amount
                    elif kind == protocol.RewardEventKind.BOSS_KILLED:
                        won = True
                    elif kind == protocol.RewardEventKind.PLAYER_DEATH:
                        died = True
                time_limit_reached = bool(info.get("time_limit_reached", False))
                if terminated or truncated:
                    break

            supervisor_limit = not (terminated or truncated)
            if supervisor_limit:
                truncated = True

            elapsed_seconds = _elapsed_seconds(obs, steps=steps, env=self.env)
            target_met = _target_met(
                env=self.env,
                won=won,
                damage_taken=damage_taken,
                elapsed_seconds=elapsed_seconds,
            )

            # Preserve the terminal result above, then immediately begin the
            # ordinary clean reset. This mirrors GameWorker's rollout behavior.
            obs, info = self._reset()
            next_episode_id = int(info.get("episode_id", 0))
            reset_succeeded = next_episode_id != episode_id
            if not reset_succeeded:
                raise RuntimeError(f"arena RESET did not advance episode_id: {episode_id}")

            results.append(
                ArenaEpisodeResult(
                    episode_index=episode_index,
                    episode_id=episode_id,
                    next_episode_id=next_episode_id,
                    reset_succeeded=reset_succeeded,
                    won=won,
                    died=died,
                    hitless=damage_taken == 0.0,
                    target_met=target_met,
                    damage_dealt=damage_dealt,
                    damage_taken=damage_taken,
                    elapsed_seconds=elapsed_seconds,
                    steps=steps,
                    total_reward=total_reward,
                    terminated=terminated,
                    truncated=truncated,
                    time_limit_reached=time_limit_reached or supervisor_limit,
                )
            )
        return tuple(results)

    def _reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self.env.reset(
            options={
                "reset_timeout_s": self.reset_timeout_s,
                "recv_timeout_s": 10.0,
            }
        )


def _target_met(
    *,
    env: Any,
    won: bool,
    damage_taken: float,
    elapsed_seconds: float,
) -> bool:
    task = _find_task(env)
    if task is None:
        return won and damage_taken == 0.0
    arena = getattr(task, "arena", None)
    objective = getattr(arena, "objective", "win")
    if objective == "win":
        return won
    target_seconds = getattr(arena, "target_kill_time_seconds", None)
    if target_seconds is None:
        target_seconds = float(getattr(task, "time_limit_seconds", float("inf")))
    return won and damage_taken == 0.0 and elapsed_seconds <= float(target_seconds)


def _elapsed_seconds(obs: dict[str, np.ndarray], *, steps: int, env: Any) -> float:
    global_state = np.asarray(obs.get("global", ()), dtype=np.float32)
    if global_state.shape[0] > 4 and float(global_state[4]) > 0.0:
        return float(global_state[4])

    task = _find_task(env)
    action_repeat = int(getattr(getattr(task, "action", None), "action_repeat", 1))
    fixed_delta = (
        float(global_state[6])
        if global_state.shape[0] > 6 and float(global_state[6]) > 0.0
        else 0.02
    )
    return float(steps * action_repeat * fixed_delta)


def _find_task(env: Any) -> Any | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "task"):
            return current.task
        current = getattr(current, "env", None)
    return None


def _event_kind(event: Any) -> protocol.RewardEventKind:
    value = getattr(event, "kind", getattr(event, "Kind", None))
    if callable(value):
        value = value()
    if value is None:
        raise ValueError("arena reward event is missing kind")
    return protocol.RewardEventKind(int(value))


def _event_amount(event: Any) -> float:
    value = getattr(event, "amount", getattr(event, "Amount", 0.0))
    if callable(value):
        value = value()
    return float(value)


__all__ = ["ArenaEpisodeResult", "ArenaPolicy", "BossArenaSupervisor"]
