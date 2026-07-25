"""run_checkpoint_server script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def test_checkpoint_server_resolves_ssh_loopback_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.setenv("HKRL_AUTH_TOKEN", "secret")
    args = argparse.Namespace(
        config=str(Path(__file__).parents[2] / "configs/train/ssh_remote_learner.yaml"),
        bind="127.0.0.1:0",
        checkpoint_dir=str(tmp_path),
        dry_run=True,
    )

    summary, token = module.resolve_runtime(args)

    assert summary == {
        "auth_token_required": True,
        "bind": "127.0.0.1:0",
        "checkpoint_dir": str(tmp_path.resolve()),
        "dry_run": True,
        "service": "checkpoint_registry",
    }
    assert token == "secret"


def test_checkpoint_server_requires_configured_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    monkeypatch.delenv("HKRL_AUTH_TOKEN", raising=False)
    args = argparse.Namespace(
        config=str(Path(__file__).parents[2] / "configs/train/ssh_remote_learner.yaml"),
        bind="127.0.0.1:0",
        checkpoint_dir=str(tmp_path),
        dry_run=True,
    )

    with pytest.raises(ValueError, match="HKRL_AUTH_TOKEN"):
        module.resolve_runtime(args)


def _load_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts/run_checkpoint_server.py"
    spec = importlib.util.spec_from_file_location("run_checkpoint_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
