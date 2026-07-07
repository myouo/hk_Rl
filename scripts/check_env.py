#!/usr/bin/env python3
"""Check live HKRLEnvMod TCP connectivity without resetting the scene."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from hkrl import protocol
from hkrl.env import EnvProtocolError, HKRLEnv
from hkrl.transport.factory import make_transport
from hkrl.utils.config import load_task_config, load_train_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKRL live env connectivity check")
    parser.add_argument("--config", default="configs/train/ppo_mlp.yaml")
    parser.add_argument("--task", default="configs/tasks/gruz_mother.yaml")
    parser.add_argument(
        "--host", default=None, help="override TCP env host from config"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="override TCP env port from config"
    )
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", help="optional path to write JSON summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    if args.output:
        output = Path(_non_empty_path_like(args.output, name="output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, sort_keys=True))
    return 0 if bool(summary.get("ok", False)) else 2


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    cfg = load_train_config(args.config)
    if cfg.transport.name != "tcp":
        raise ValueError("check_env requires tcp transport")
    task = load_task_config(args.task)
    transport = make_transport(
        cfg,
        host=_optional_host(getattr(args, "host", None)),
        port=_optional_port(getattr(args, "port", None)),
    )
    env = HKRLEnv(transport=transport, task=task)
    try:
        try:
            info = env.ping(timeout_s=float(args.timeout))
        except _DIAGNOSTIC_EXCEPTIONS as exc:
            return _failure_summary(
                exc,
                host=transport.host,
                port=transport.port,
                task_id=task.task_id,
            )
        return _summary_from_info(
            info, host=transport.host, port=transport.port, task_id=task.task_id
        )
    finally:
        env.close()


def _summary_from_info(
    info: Mapping[str, Any],
    *,
    host: str,
    port: int,
    task_id: str,
) -> dict[str, Any]:
    lifecycle = info.get("lifecycle_state")
    error_code = info.get("error_code")
    return {
        "env_id": int(info.get("env_id", 0)),
        "error_code": _enum_name(error_code),
        "host": host,
        "lifecycle_state": _enum_name(lifecycle),
        "ok": error_code == protocol.StatusCode.OK,
        "port": int(port),
        "schema_version": int(info.get("schema_version", 0)),
        "server_tick": int(info.get("server_tick", 0)),
        "task_id": task_id,
    }


_DIAGNOSTIC_EXCEPTIONS = (
    ConnectionError,
    EnvProtocolError,
    OSError,
    TimeoutError,
    ValueError,
)


def _failure_summary(
    error: BaseException,
    *,
    host: str,
    port: int,
    task_id: str,
) -> dict[str, Any]:
    return {
        "env_id": 0,
        "error": str(error),
        "error_code": _failure_error_code(error),
        "error_type": type(error).__name__,
        "host": host,
        "lifecycle_state": "UNAVAILABLE",
        "ok": False,
        "port": int(port),
        "schema_version": 0,
        "server_tick": 0,
        "task_id": task_id,
    }


def _failure_error_code(error: BaseException) -> str:
    if isinstance(error, TimeoutError):
        return "TIMEOUT"
    if isinstance(error, (EnvProtocolError, ValueError)):
        return "PROTOCOL_ERROR"
    if isinstance(error, (ConnectionError, OSError)):
        return "CONNECT_FAILED"
    return "ERROR"


def _enum_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _validate_args(args: argparse.Namespace) -> None:
    _non_empty_path_like(getattr(args, "config", None), name="config")
    _non_empty_path_like(getattr(args, "task", None), name="task")
    _optional_host(getattr(args, "host", None))
    _optional_port(getattr(args, "port", None))
    timeout = getattr(args, "timeout", None)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, Real)
        or float(timeout) <= 0.0
    ):
        raise ValueError("timeout must be positive")
    _optional_non_empty_path_like(getattr(args, "output", None), name="output")


def _optional_host(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("host must not be empty")
    return value.strip()


def _optional_port(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("port must be an integer")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    return port


def _non_empty_path_like(value: Any, *, name: str) -> str | Path:
    if not isinstance(value, str | Path) or not str(value).strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_non_empty_path_like(value: Any, *, name: str) -> str | Path | None:
    if value is None:
        return None
    return _non_empty_path_like(value, name=name)


if __name__ == "__main__":
    raise SystemExit(main())
