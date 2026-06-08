"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hkrl.utils.config import load_train_config, load_yaml


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_yaml_composes_defaults_with_deep_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "nested" / "child.yaml"
    child.parent.mkdir()

    _write_yaml(
        base,
        {
            "algorithm": "recurrent_ppo",
            "model": {"name": "entity_attention_gru", "rnn_hidden": 256},
            "transport": {"host": "127.0.0.1", "port": 5555},
        },
    )
    _write_yaml(
        child,
        {
            "defaults": ["../base.yaml"],
            "algorithm": "ppo",
            "model": {"name": "mlp"},
        },
    )

    assert load_yaml(child) == {
        "algorithm": "ppo",
        "model": {"name": "mlp", "rnn_hidden": 256},
        "transport": {"host": "127.0.0.1", "port": 5555},
    }


def test_load_yaml_rejects_default_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    _write_yaml(first, {"defaults": ["second.yaml"], "algorithm": "ppo"})
    _write_yaml(second, {"defaults": ["first.yaml"], "algorithm": "recurrent_ppo"})

    with pytest.raises(ValueError, match="cyclic config defaults"):
        load_yaml(first)


def test_load_train_config_composes_repo_defaults() -> None:
    config = load_train_config(Path("../configs/train/ppo_mlp.yaml"))

    assert config.algorithm == "ppo"
    assert config.gamma == 0.995
    assert config.transport.port == 5555
    assert config.model.name == "mlp"


def test_load_train_config_preserves_distributed_runtime_settings() -> None:
    config = load_train_config(Path("../configs/train/remote_learner.yaml"))

    assert config.algorithm == "appo"
    assert config.learner.bind == "0.0.0.0:5600"
    assert config.learner.max_staleness == 4
    assert config.learner.checkpoint_dir == "checkpoints/"
    assert config.learner.publish_every_updates == 10
    assert config.coordinator.bind == "0.0.0.0:5610"
    assert config.coordinator.num_workers == 4
    assert config.security.bind_scope == "lan"
    assert config.security.require_token is True
