"""Validated live-tuning snapshot tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hkrl.utils.config import RewardWeights, TrainConfig
from hkrl.utils.live_tuning import (
    LiveTuning,
    atomic_write_json,
    effective_reward_weights,
    effective_train_config,
    load_live_tuning,
)


def test_live_tuning_resolves_from_immutable_startup_values() -> None:
    base_train = TrainConfig(
        learning_rate=3.0e-4,
        entropy_coef=0.01,
        learner={"target_kl": 0.03},
    )
    base_reward = RewardWeights(boss_damage=0.5, player_death=-100.0)
    tuning = LiveTuning(
        version=2,
        learner={
            "learning_rate": 1.0e-4,
            "entropy_coef": 0.02,
            "target_kl": "off",
        },
        reward={"boss_damage": 1.0, "player_death": -20.0},
        worker={"time_scale": 3.0},
    )

    train = effective_train_config(base_train, tuning)
    reward = effective_reward_weights(base_reward, tuning)

    assert train.learning_rate == 1.0e-4
    assert train.entropy_coef == 0.02
    assert train.learner.target_kl is None
    assert reward.boss_damage == 1.0
    assert reward.player_death == -20.0
    assert base_train.learning_rate == 3.0e-4
    assert base_reward.boss_damage == 0.5


def test_live_tuning_rejects_unknown_non_finite_and_empty_snapshots() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LiveTuning(version=1)
    with pytest.raises(ValueError, match="extra"):
        LiveTuning.model_validate({"version": 1, "learner": {"epochs": 4}})
    with pytest.raises(ValueError):
        LiveTuning(version=1, reward={"boss_damage": float("nan")})
    with pytest.raises(ValueError, match="cannot be combined"):
        LiveTuning(version=1, reset_to_base=True, reward={"boss_damage": 1.0})

    assert LiveTuning(version=1, reset_to_base=True).reset_to_base is True


def test_live_tuning_file_is_atomic_and_round_trips(tmp_path: Path) -> None:
    tuning = LiveTuning(
        version=1,
        reward={"boss_damage": 1.0},
        note="engagement adjustment",
    )
    target = tmp_path / "live_tuning.json"

    atomic_write_json(target, tuning.checkpoint_payload())

    loaded = load_live_tuning(target)
    assert loaded == tuning
    assert target.stat().st_mode & 0o077 == 0
    assert json.loads(target.read_text(encoding="utf-8"))["version"] == 1
