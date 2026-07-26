"""Tests for the authenticated live-tuning command-line client."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hkrl.utils.config import TrainConfig


def test_tune_training_merges_full_snapshot_and_unsets_one_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    posted: list[dict[str, Any]] = []
    current = {
        "version": 1,
        "reward": {"boss_damage": 1.0},
        "learner": {"entropy_coef": 0.01},
        "worker": {},
    }

    monkeypatch.setattr(module, "load_train_config", lambda _path: TrainConfig())
    monkeypatch.setattr(module, "resolve_auth_token", lambda _cfg: "secret")
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda _endpoint, path, **_kwargs: (
            current if path == "live-tuning" else {"tuning_version": 1}
        ),
    )

    def post(
        _endpoint: str,
        _path: str,
        payload: dict[str, Any],
        **_kwargs: object,
    ) -> dict[str, Any]:
        posted.append(payload)
        return {"requested_version": payload["version"]}

    monkeypatch.setattr(module, "_post_json", post)
    args = argparse.Namespace(
        config="unused.yaml",
        endpoint="http://127.0.0.1:5601/",
        set=["reward.player_death=-20", "learner.entropy_coef=0.02"],
        unset=["reward.boss_damage"],
        reset=False,
        note="reduce edge camping",
        wait_applied=0.0,
        show=False,
    )

    result = module.run_from_args(args)

    assert posted == [
        {
            "version": 2,
            "reward": {"player_death": -20.0},
            "learner": {"entropy_coef": 0.02},
            "worker": {},
            "reset_to_base": False,
            "note": "reduce edge camping",
        }
    ]
    assert result["snapshot"] == posted[0]


def test_tune_training_unsetting_last_override_resets_to_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    posted: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "load_train_config", lambda _path: TrainConfig())
    monkeypatch.setattr(module, "resolve_auth_token", lambda _cfg: None)
    monkeypatch.setattr(
        module,
        "_get_json",
        lambda _endpoint, path, **_kwargs: (
            {"version": 4, "reward": {"boss_damage": 1.0}}
            if path == "live-tuning"
            else {"tuning_version": 4}
        ),
    )
    monkeypatch.setattr(
        module,
        "_post_json",
        lambda _endpoint, _path, payload, **_kwargs: (
            posted.append(payload) or {"requested_version": 5}
        ),
    )
    args = argparse.Namespace(
        config="unused.yaml",
        endpoint="http://localhost:5601",
        set=[],
        unset=["reward.boss_damage"],
        reset=False,
        note=None,
        wait_applied=0.0,
        show=False,
    )

    module.run_from_args(args)

    assert posted[0]["version"] == 5
    assert posted[0]["reset_to_base"] is True
    assert posted[0]["reward"] == {}


def test_tune_training_rejects_non_loopback_and_unknown_fields() -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="loopback"):
        module._loopback_endpoint("http://192.0.2.4:5601")
    with pytest.raises(ValueError, match="unsupported"):
        module._parse_assignment("learner.epochs=8")
    assert module._parse_assignment("learner.target_kl=off") == (
        "learner.target_kl",
        "off",
    )


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "tune_training.py"
    spec = importlib.util.spec_from_file_location("tune_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
