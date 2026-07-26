#!/usr/bin/env python3
"""Serve checkpoints plus narrow live-tuning control through an SSH forward."""

from __future__ import annotations

import argparse
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from hkrl.learner.checkpoint_http import CheckpointHttpServer
from hkrl.utils.config import (
    load_train_config,
    resolve_auth_token,
    validate_bind_address,
    validate_service_auth,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HKRL checkpoint registry HTTP server")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--bind",
        default="127.0.0.1:5601",
        help="HTTP bind address; keep loopback when using SSH forwarding",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="override config.learner.checkpoint_dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/security/path without opening a listener",
    )
    return parser


def resolve_runtime(args: argparse.Namespace) -> tuple[dict[str, Any], str | None]:
    config_path = _non_empty(args.config, name="config")
    cfg = load_train_config(config_path)
    bind = validate_bind_address(
        _non_empty(args.bind, name="bind"),
        cfg.security.bind_scope,
    )
    validate_service_auth(bind, cfg)
    auth_token = resolve_auth_token(cfg)
    checkpoint_dir = (
        Path(
            _non_empty(
                args.checkpoint_dir or cfg.learner.checkpoint_dir,
                name="checkpoint_dir",
            )
        )
        .expanduser()
        .resolve()
    )
    summary = {
        "auth_token_required": cfg.security.require_token,
        "bind": bind,
        "checkpoint_dir": str(checkpoint_dir),
        "dry_run": bool(args.dry_run),
        "service": "checkpoint_registry",
    }
    return summary, auth_token


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary, auth_token = resolve_runtime(args)
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True))
        return 0

    with CheckpointHttpServer(
        summary["checkpoint_dir"],
        summary["bind"],
        auth_token=auth_token,
    ) as server:
        startup = {**summary, "endpoint": server.endpoint}
        print(json.dumps(startup, sort_keys=True), flush=True)
        with suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


def _non_empty(value: object, *, name: str) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path-like string")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
