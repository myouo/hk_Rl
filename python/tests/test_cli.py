"""CLI smoke-loop tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hkrl.cli import build_argparser, run_random_policy_smoke
from hkrl.utils.logging import JsonlSink


class FakeEnv:
    def __init__(self) -> None:
        self.reset_options: dict[str, Any] | None = None
        self.actions: list[Any] = []

    def reset(self, *, options: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
        self.reset_options = options
        return {"step": 0}, {"action_mask": np.array([True, False])}

    def step(self, action: Any) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        step = len(self.actions)
        terminated = step == 2
        return {"step": step}, 1.5, terminated, False, {"action_mask": np.array([False, True])}


class FakePolicy:
    def __init__(self) -> None:
        self.masks: list[Any] = []

    def act(self, obs: Any, action_mask: Any | None = None) -> dict[str, int]:
        del obs
        self.masks.append(action_mask)
        return {"action": len(self.masks)}


def test_build_argparser_defaults_task_and_requires_config() -> None:
    parser = build_argparser()
    args = parser.parse_args(["--config", "configs/train/ppo_mlp.yaml", "--smoke"])

    assert args.task == "configs/tasks/gruz_mother.yaml"
    assert args.smoke is True


def test_run_random_policy_smoke_steps_until_done_and_logs(tmp_path: Path) -> None:
    env = FakeEnv()
    policy = FakePolicy()
    sink = JsonlSink(tmp_path / "smoke.jsonl")

    summary = run_random_policy_smoke(
        env=env,
        policy=policy,
        sink=sink,
        task_id="gruz_mother",
        max_steps=4,
        reset_timeout_s=2.0,
    )
    sink.close()

    assert env.reset_options == {"reset_timeout_s": 2.0}
    assert env.actions == [{"action": 1}, {"action": 2}]
    np.testing.assert_array_equal(policy.masks[0], np.array([True, False]))
    np.testing.assert_array_equal(policy.masks[1], np.array([False, True]))
    assert summary == {
        "task_id": "gruz_mother",
        "steps": 2,
        "total_reward": 3.0,
        "terminated": True,
        "truncated": False,
    }
    assert (tmp_path / "smoke.jsonl").read_text(encoding="utf-8").count("\n") == 3


def test_run_random_policy_smoke_rejects_non_positive_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        run_random_policy_smoke(
            env=FakeEnv(),
            policy=FakePolicy(),
            sink=JsonlSink(tmp_path / "smoke.jsonl"),
            task_id="gruz_mother",
            max_steps=0,
        )
