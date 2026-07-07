"""Config-driven transport construction."""

from __future__ import annotations

import os
from collections.abc import Mapping

from hkrl.transport import shared_memory as _shared_memory  # noqa: F401
from hkrl.transport import tcp as _tcp  # noqa: F401
from hkrl.transport.base import Transport
from hkrl.utils.config import TrainConfig, resolve_auth_token
from hkrl.utils.registry import get

INPROCESS_SHM_ENV = "HKRL_ENABLE_INPROCESS_SHM"


def make_transport(
    config: TrainConfig,
    *,
    host: str | None = None,
    port: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> Transport:
    """Build the env transport selected by ``config.transport.name``."""
    transport_cls = get("transport", config.transport.name)

    if config.transport.name == "tcp":
        return transport_cls(
            host=host or config.transport.host,
            port=config.transport.port if port is None else port,
            auth_token=_env_transport_auth_token(config, environ),
        )

    if config.transport.name == "shm":
        if not _inprocess_shm_enabled(environ):
            raise ValueError(
                "transport.name='shm' is currently an in-process Python prototype, "
                "not a live HKRLEnvMod shared-memory transport. Use transport.name='tcp' "
                f"for live game runs, or set {INPROCESS_SHM_ENV}=1 for explicit "
                "prototype/test use."
            )
        return transport_cls(
            name=config.transport.shm_name,
            req_slots=config.transport.req_slots,
            resp_slots=config.transport.resp_slots,
        )

    raise ValueError(f"unsupported transport {config.transport.name!r}")


def _env_transport_auth_token(
    config: TrainConfig,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve optional HKRLEnvMod TCP auth for env clients.

    The mod enables TCP auth whenever its ``HKRL_AUTH_TOKEN`` environment
    variable is set, independently of the Python train config used by local
    smoke/training scripts. Sending an auth preface is harmless when mod auth is
    disabled, so env clients opportunistically send a non-empty configured token
    even when ``security.require_token`` is false.
    """
    if config.security.require_token:
        return resolve_auth_token(config, environ)

    env = os.environ if environ is None else environ
    token = env.get(config.security.auth_token_env)
    if token is None or token == "":
        return None
    return token


def _inprocess_shm_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(INPROCESS_SHM_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
