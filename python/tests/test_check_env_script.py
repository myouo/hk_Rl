"""check_env script tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hkrl import protocol


def test_check_env_pings_live_mod_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("check_env.py")
    root = Path(__file__).parents[2]
    output = tmp_path / "nested" / "env.json"
    transport = FakeTransport(host="127.0.0.2", port=6000)
    envs: list[FakeEnv] = []
    transport_args: list[tuple[str | None, int | None]] = []

    def fake_make_transport(cfg: Any, *, host: str | None, port: int | None) -> FakeTransport:
        del cfg
        transport_args.append((host, port))
        return transport

    class PatchedEnv(FakeEnv):
        def __init__(self, *, transport: FakeTransport, task: Any) -> None:
            super().__init__(transport=transport, task=task)
            envs.append(self)

    monkeypatch.setattr(module, "make_transport", fake_make_transport)
    monkeypatch.setattr(module, "HKRLEnv", PatchedEnv)
    args = argparse.Namespace(
        config=str(root / "configs/train/ppo_mlp.yaml"),
        task=str(root / "configs/tasks/gruz_mother.yaml"),
        host="127.0.0.2",
        port=6000,
        timeout=3.0,
        output=str(output),
    )

    summary = module.run_from_args(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")

    assert transport_args == [("127.0.0.2", 6000)]
    assert envs[0].ping_timeout == 3.0
    assert envs[0].closed is True
    assert summary == {
        "env_id": 0,
        "error_code": "OK",
        "host": "127.0.0.2",
        "lifecycle_state": "IDLE",
        "ok": True,
        "port": 6000,
        "schema_version": protocol.SCHEMA_VERSION,
        "server_tick": 42,
        "task_id": "gruz_mother",
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("config", "", "config"),
        ("task", "", "task"),
        ("host", "", "host"),
        ("port", 0, "port"),
        ("port", False, "port"),
        ("timeout", 0.0, "timeout"),
        ("timeout", "5", "timeout"),
        ("output", "", "output"),
    ],
)
def test_check_env_rejects_invalid_args(field: str, value: object, match: str) -> None:
    module = _load_script("check_env.py")
    args = _args(**{field: value})

    with pytest.raises(ValueError, match=match):
        module._validate_args(args)


def test_check_env_requires_tcp_config(tmp_path: Path) -> None:
    module = _load_script("check_env.py")
    config = tmp_path / "shm.yaml"
    config.write_text("transport:\n  name: shm\n", encoding="utf-8")
    args = _args(config=str(config))

    with pytest.raises(ValueError, match="tcp transport"):
        module.run_from_args(args)


class FakeTransport:
    def __init__(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = port


class FakeEnv:
    def __init__(self, *, transport: FakeTransport, task: Any) -> None:
        self.transport = transport
        self.task = task
        self.closed = False
        self.ping_timeout: float | None = None

    def ping(self, *, timeout_s: float) -> dict[str, Any]:
        self.ping_timeout = timeout_s
        return {
            "env_id": 0,
            "error_code": protocol.StatusCode.OK,
            "lifecycle_state": protocol.LifecycleState.IDLE,
            "schema_version": protocol.SCHEMA_VERSION,
            "server_tick": 42,
        }

    def close(self) -> None:
        self.closed = True


def _args(**overrides: object) -> argparse.Namespace:
    root = Path(__file__).parents[2]
    values: dict[str, object] = {
        "config": str(root / "configs/train/ppo_mlp.yaml"),
        "host": None,
        "output": None,
        "port": None,
        "task": str(root / "configs/tasks/gruz_mother.yaml"),
        "timeout": 5.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
