"""Config-driven transport construction."""

from __future__ import annotations

from collections.abc import Mapping

from hkrl.transport import shared_memory as _shared_memory  # noqa: F401
from hkrl.transport import tcp as _tcp  # noqa: F401
from hkrl.transport.base import Transport
from hkrl.utils.config import TrainConfig, resolve_auth_token
from hkrl.utils.registry import get


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
            auth_token=resolve_auth_token(config, environ),
        )

    if config.transport.name == "shm":
        return transport_cls(
            name=config.transport.shm_name,
            req_slots=config.transport.req_slots,
            resp_slots=config.transport.resp_slots,
        )

    raise ValueError(f"unsupported transport {config.transport.name!r}")
