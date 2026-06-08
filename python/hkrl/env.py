"""Gymnasium environment wrapping one HKRLEnvMod instance.

Implements PRD §2.1 / Phase 2 and docs/architecture.md. Composes a Transport, the
reward function, and the spaces. ``reset()`` drives the clean-lifecycle handshake
(docs/episode_lifecycle.md); ``step()`` sends an action and returns the decoded
observation, reward, termination flags, and info (including the action mask).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from hkrl.protocol import LifecycleState, StatusCode
from hkrl.reward import DefaultReward
from hkrl.transport.base import Transport
from hkrl.utils.config import TaskConfig


class HKRLEnv(gym.Env):
    """Single-env Gymnasium interface to the game.

    The real-time action loop is local (invariant #1): this env talks to the mod
    over a Transport, never to a remote learner. ``info["action_mask"]`` carries
    the per-step mask for masked policies. ``info["lifecycle_state"]`` and
    ``info["error_code"]`` surface reset health.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transport: Transport,
        task: TaskConfig,
        reward_fn: DefaultReward | None = None,
    ) -> None:
        super().__init__()
        self.transport = transport
        self.task = task
        self.reward_fn = reward_fn or DefaultReward(task.reward)
        # TODO(phase-2): build self.observation_space / self.action_space from
        # hkrl.spaces using task.observation / task.action.
        self.observation_space: gym.Space[Any]
        self.action_space: gym.Space[Any]
        self._tick_id = 0
        self._episode_id = 0

    # -- Gym API --------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Issue RESET and poll until LifecycleState.RUNNING (or an error code).

        Returns ``(obs, info)``. On a reset failure (non-OK StatusCode) raises /
        surfaces the code rather than returning a contaminated observation
        (docs/episode_lifecycle.md §4). Increments reset metrics.

        TODO(phase-2): implement RESET handshake via self.transport + protocol.
        """
        raise NotImplementedError

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Send a STEP (with action_repeat), decode the response, compute reward.

        Returns ``(obs, reward, terminated, truncated, info)`` per Gymnasium.
        ``info`` includes ``action_mask``, ``reward_events``, ``lifecycle_state``,
        ``episode_id``, ``sps`` hints.

        TODO(phase-2): encode action -> StepRequest, send, decode StepResponse,
        reward_fn(events), map terminated/truncated.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the transport. Idempotent."""
        # TODO(phase-2): self.transport.close()
        raise NotImplementedError

    # -- helpers --------------------------------------------------------------
    def _await_running(self, timeout_s: float) -> StatusCode:
        """Poll the mod with no-op STEPs until RUNNING or error/timeout.

        TODO(phase-2): loop sending Command.STEP noops, inspect lifecycle_state.
        """
        raise NotImplementedError

    @staticmethod
    def _is_terminal(lifecycle: LifecycleState) -> bool:
        return lifecycle in (LifecycleState.TERMINATING, LifecycleState.REPORT_DONE)
