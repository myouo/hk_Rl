"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hkrl.utils.config import load_task_config, load_train_config, load_yaml, resolve_auth_token


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


def test_load_task_config_preserves_wire_id() -> None:
    gruz = load_task_config(Path("../configs/tasks/gruz_mother.yaml"))
    hornet = load_task_config(Path("../configs/tasks/hornet_protector.yaml"))
    mantis = load_task_config(Path("../configs/tasks/mantis_lords.yaml"))

    assert gruz.task_id == "gruz_mother"
    assert gruz.wire_id == 0
    assert gruz.action.n_macro_actions == 11
    assert hornet.wire_id == 1
    assert mantis.wire_id == 2


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
    assert config.security.auth_token_env == "HKRL_AUTH_TOKEN"


def test_load_train_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(
        config,
        {
            "algorithm": "ppo",
            "model": {"name": "mlp", "unknown_model_key": 1},
            "unexpected_top_level": True,
        },
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_train_config(config)


def test_load_train_config_rejects_unknown_enum_values(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    _write_yaml(config, {"algorithm": "dqn", "transport": {"name": "udp"}})

    with pytest.raises(ValueError, match="Input should be"):
        load_train_config(config)


def test_resolve_auth_token_uses_configured_environment() -> None:
    config = load_train_config(Path("../configs/train/remote_learner.yaml"))

    assert resolve_auth_token(config, {"HKRL_AUTH_TOKEN": "secret"}) == "secret"


def test_resolve_auth_token_requires_non_empty_token() -> None:
    config = load_train_config(Path("../configs/train/remote_learner.yaml"))

    with pytest.raises(ValueError, match="HKRL_AUTH_TOKEN"):
        resolve_auth_token(config, {})
    with pytest.raises(ValueError, match="HKRL_AUTH_TOKEN"):
        resolve_auth_token(config, {"HKRL_AUTH_TOKEN": ""})


def test_resolve_auth_token_returns_none_when_disabled() -> None:
    config = load_train_config(Path("../configs/train/ppo_mlp.yaml"))

    assert resolve_auth_token(config, {}) is None
