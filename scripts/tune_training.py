#!/usr/bin/env python3
"""Submit or inspect authenticated live-training tuning snapshots.

The endpoint is the existing loopback checkpoint registry (normally reached
through SSH port 5601). Requests are versioned and applied by the learner at a
clean update boundary, then distributed to workers in a signed checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import yaml  # type: ignore[import-untyped]
from hkrl.utils.config import load_train_config, resolve_auth_token
from hkrl.utils.live_tuning import LiveTuning

_TUNABLE_PATHS = frozenset(
    {
        "reward.boss_damage",
        "reward.player_damage",
        "reward.soul_gained",
        "reward.heal_amount",
        "reward.boss_kill",
        "reward.player_death",
        "reward.time_penalty",
        "reward.invalid_action",
        "learner.learning_rate",
        "learner.entropy_coef",
        "learner.value_coef",
        "learner.clip_range",
        "learner.max_grad_norm",
        "learner.target_kl",
        "worker.time_scale",
    }
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or update HKRL live tuning")
    parser.add_argument(
        "--config",
        default="configs/train/linux_game_worker.yaml",
        help="train config used to resolve the bearer-token environment",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5601/",
        help="loopback checkpoint/control endpoint",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="set one supported field; repeat for multiple fields",
    )
    parser.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="PATH",
        help="return one supported field to its startup value; repeat as needed",
    )
    parser.add_argument(
        "--reset", action="store_true", help="remove all live overrides"
    )
    parser.add_argument(
        "--note", default=None, help="short audit note for this version"
    )
    parser.add_argument(
        "--wait-applied",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="wait until the learner reports this version applied",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show requested/applied state without changing it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    result = run_from_args(args)
    print(json.dumps(result, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = _loopback_endpoint(args.endpoint)
    wait_seconds = _non_negative_finite(args.wait_applied, name="wait_applied")
    assignments = list(args.set or [])
    removals = list(args.unset or [])
    if args.show and (assignments or removals or args.reset):
        raise ValueError("--show cannot be combined with --set, --unset, or --reset")
    if args.reset and (assignments or removals):
        raise ValueError("--reset cannot be combined with --set or --unset")

    cfg = load_train_config(Path(args.config))
    token = resolve_auth_token(cfg)
    requested = _get_json(endpoint, "live-tuning", token=token, missing_ok=True)
    status = _get_json(endpoint, "live-tuning/status", token=token, missing_ok=True)
    if args.show or (not assignments and not removals and not args.reset):
        return {
            "applied": status,
            "endpoint": endpoint,
            "requested": requested,
            "tunable_paths": sorted(_TUNABLE_PATHS),
        }

    current_version = 0 if requested is None else _version(requested)
    payload: dict[str, Any] = (
        {
            "learner": {},
            "reward": {},
            "worker": {},
            "reset_to_base": True,
        }
        if args.reset
        else _next_snapshot(requested)
    )
    for assignment in assignments:
        path, value = _parse_assignment(assignment)
        section, field = path.split(".", 1)
        section_payload = payload.setdefault(section, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f"stored live tuning section {section!r} is invalid")
        section_payload[field] = value
    for path in removals:
        section, field = _validate_path(path)
        section_payload = payload.setdefault(section, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f"stored live tuning section {section!r} is invalid")
        section_payload.pop(field, None)
    payload["version"] = current_version + 1
    payload["reset_to_base"] = bool(args.reset or not _has_overrides(payload))
    if args.note is not None:
        payload["note"] = args.note
    else:
        payload.pop("note", None)

    tuning = LiveTuning.model_validate(payload)
    response = _post_json(
        endpoint,
        "live-tuning",
        tuning.checkpoint_payload(),
        token=token,
    )
    applied = (
        _wait_for_applied(endpoint, tuning.version, token=token, timeout_s=wait_seconds)
        if wait_seconds > 0.0
        else status
    )
    return {
        "applied": applied,
        "endpoint": endpoint,
        "request": response,
        "snapshot": tuning.checkpoint_payload(),
    }


def _next_snapshot(current: dict[str, Any] | None) -> dict[str, Any]:
    if current is None:
        return {"learner": {}, "reward": {}, "worker": {}}
    snapshot = deepcopy(current)
    snapshot.pop("version", None)
    snapshot.pop("note", None)
    snapshot["reset_to_base"] = False
    for section in ("learner", "reward", "worker"):
        value = snapshot.setdefault(section, {})
        if not isinstance(value, dict):
            raise ValueError(f"stored live tuning section {section!r} is invalid")
    return snapshot


def _parse_assignment(assignment: str) -> tuple[str, Any]:
    if not isinstance(assignment, str) or "=" not in assignment:
        raise ValueError("--set values must use PATH=VALUE")
    path, raw = assignment.split("=", 1)
    path = path.strip()
    _validate_path(path)
    if not raw.strip():
        raise ValueError(f"live tuning value for {path!r} must not be empty")
    value = (
        "off"
        if path == "learner.target_kl" and raw.strip().lower() == "off"
        else yaml.safe_load(raw)
    )
    if value is None:
        raise ValueError(
            "use --reset to return all live parameters to their startup values"
        )
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f"live tuning value for {path!r} must be scalar")
    return path, value


def _validate_path(path: Any) -> tuple[str, str]:
    if not isinstance(path, str):
        raise ValueError("live tuning path must be a string")
    normalized = path.strip()
    if normalized not in _TUNABLE_PATHS:
        known = ", ".join(sorted(_TUNABLE_PATHS))
        raise ValueError(
            f"unsupported live tuning path {normalized!r}; expected one of: {known}"
        )
    section, field = normalized.split(".", 1)
    return section, field


def _has_overrides(payload: dict[str, Any]) -> bool:
    for section in ("learner", "reward", "worker"):
        value = payload.get(section)
        if isinstance(value, dict) and value:
            return True
    return False


def _wait_for_applied(
    endpoint: str,
    version: int,
    *,
    token: str | None,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        status = _get_json(
            endpoint,
            "live-tuning/status",
            token=token,
            missing_ok=True,
        )
        if status is not None and _status_version(status) >= version:
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"live tuning version {version} was not applied within {timeout_s:.1f}s"
            )
        time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))


def _get_json(
    endpoint: str,
    path: str,
    *,
    token: str | None,
    missing_ok: bool,
) -> dict[str, Any] | None:
    request = Request(urljoin(endpoint, path), method="GET")
    _add_auth(request, token)
    try:
        with urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise RuntimeError(
            f"control endpoint GET {path!r} failed with HTTP {exc.code}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"control endpoint {path!r} returned a non-object")
    return payload


def _post_json(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    *,
    token: str | None,
) -> dict[str, Any]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    request = Request(
        urljoin(endpoint, path),
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    _add_auth(request, token)
    try:
        with urlopen(request, timeout=5.0) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"control endpoint POST {path!r} failed with HTTP {exc.code}: {detail}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("control endpoint returned a non-object")
    return result


def _add_auth(request: Request, token: str | None) -> None:
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")


def _loopback_endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint must be a non-empty HTTP URL")
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("endpoint must be an HTTP(S) URL")
    host = parsed.hostname
    try:
        loopback = ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise ValueError(
            "live tuning endpoint must be loopback; use an SSH local forward"
        )
    return value.rstrip("/") + "/"


def _version(payload: dict[str, Any]) -> int:
    return int(LiveTuning.model_validate(payload).version)


def _status_version(payload: dict[str, Any]) -> int:
    value = payload.get("tuning_version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("live tuning status has an invalid tuning_version")
    return value


def _non_negative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
