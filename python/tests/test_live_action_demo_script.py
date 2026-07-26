"""Deterministic live-action driver tests that do not require Hollow Knight."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hkrl import protocol, spaces

ROOT = Path(__file__).parents[2]


def test_live_action_driver_maps_commands_and_restores_task_repeat() -> None:
    module = _load_script()
    env = _FakeEnv()
    driver = module.LiveActionDemo(env, reset_timeout_s=12.0)

    driver.reset()
    assert env.reset_options == {
        "reset_timeout_s": 12.0,
        "recv_timeout_s": 10.0,
    }

    assert driver.execute("right 9") is True
    assert env.last_action == {
        "movement_x": 2,
        "aim_y": 1,
        "buttons": {},
        "duration": 0,
        "macro": 0,
    }
    assert env.repeat_seen_by_step == 9
    assert env.task.action.action_repeat == 2

    assert driver.execute("focus 80") is True
    assert env.last_action["buttons"] == {"focus_hold": True}
    assert env.last_action["duration"] == 3
    assert env.repeat_seen_by_step == 80


def test_live_action_driver_validates_tick_range_and_close() -> None:
    module = _load_script()
    env = _FakeEnv()
    driver = module.LiveActionDemo(env, reset_timeout_s=5.0)
    driver.reset()

    with pytest.raises(ValueError, match=r"\[1, 200\]"):
        driver.execute("left 0")
    with pytest.raises(ValueError, match=r"\[1, 200\]"):
        driver.execute("left 201")
    assert driver.execute("close") is False


class _FakeEnv:
    def __init__(self) -> None:
        self.task = SimpleNamespace(action=SimpleNamespace(action_repeat=2))
        self.reset_options: dict[str, float] | None = None
        self.last_action: dict[str, Any] = {}
        self.repeat_seen_by_step = 0
        self._obs, self._info = _sample()

    def reset(self, *, options: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.reset_options = options
        return self._obs, self._info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.last_action = action
        self.repeat_seen_by_step = self.task.action.action_repeat
        return self._obs, 0.0, False, False, self._info


def _sample() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    player = np.zeros((spaces.PLAYER_FEATURE_DIMS["privileged"],), dtype=np.float32)
    player[4:6] = 9
    entity = np.zeros((1, 24), dtype=np.float32)
    entity[0, 1] = float(protocol.EntityType.BOSS)
    entity[0, 12:14] = 650
    observation = {
        "global": np.zeros((9,), dtype=np.float32),
        "player": player,
        "entities": entity,
        "entity_mask": np.ones((1,), dtype=np.int8),
    }
    info = {
        "server_tick": 1,
        "lifecycle_state": protocol.LifecycleState.RUNNING,
        "reward_events": [],
    }
    return observation, info


def _load_script() -> ModuleType:
    path = ROOT / "scripts" / "live_action_demo.py"
    spec = importlib.util.spec_from_file_location("live_action_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
