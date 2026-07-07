"""Transport factory tests."""

from __future__ import annotations

import pytest
from hkrl.transport.factory import make_transport
from hkrl.transport.shared_memory import SharedMemoryTransport
from hkrl.transport.tcp import TcpTransport
from hkrl.utils.config import TrainConfig


def test_make_transport_builds_tcp_with_auth_token() -> None:
    config = TrainConfig(
        transport={"name": "tcp", "host": "127.0.0.2", "port": 5556},
        security={"require_token": True, "auth_token_env": "HKRL_TEST_TOKEN"},
    )

    transport = make_transport(config, environ={"HKRL_TEST_TOKEN": "secret"})

    assert isinstance(transport, TcpTransport)
    assert transport.host == "127.0.0.2"
    assert transport.port == 5556
    assert transport.auth_token == "secret"


def test_make_transport_applies_tcp_host_port_overrides() -> None:
    config = TrainConfig(transport={"name": "tcp", "host": "127.0.0.2", "port": 5556})

    transport = make_transport(config, host="127.0.0.3", port=6000, environ={})

    assert isinstance(transport, TcpTransport)
    assert transport.host == "127.0.0.3"
    assert transport.port == 6000
    assert transport.auth_token is None


def test_make_transport_sends_optional_env_auth_token_for_tcp_env() -> None:
    config = TrainConfig(
        transport={"name": "tcp"},
        security={"require_token": False, "auth_token_env": "HKRL_TEST_TOKEN"},
    )

    transport = make_transport(config, environ={"HKRL_TEST_TOKEN": "secret"})

    assert isinstance(transport, TcpTransport)
    assert transport.auth_token == "secret"


def test_make_transport_ignores_empty_optional_env_auth_token() -> None:
    config = TrainConfig(
        transport={"name": "tcp"},
        security={"require_token": False, "auth_token_env": "HKRL_TEST_TOKEN"},
    )

    transport = make_transport(config, environ={"HKRL_TEST_TOKEN": ""})

    assert isinstance(transport, TcpTransport)
    assert transport.auth_token is None


def test_make_transport_rejects_shared_memory_for_live_env_by_default() -> None:
    config = TrainConfig(transport={"name": "shm"})

    with pytest.raises(ValueError, match="in-process Python prototype"):
        make_transport(config, environ={})


def test_make_transport_builds_explicit_inprocess_shared_memory() -> None:
    config = TrainConfig(
        transport={
            "name": "shm",
            "shm_name": "hkrl_test",
            "req_slots": 2,
            "resp_slots": 3,
        },
        security={"require_token": True, "auth_token_env": "HKRL_TEST_TOKEN"},
    )

    transport = make_transport(config, environ={"HKRL_ENABLE_INPROCESS_SHM": "1"})

    assert isinstance(transport, SharedMemoryTransport)
    assert transport.name == "hkrl_test"
    assert transport.req_slots == 2
    assert transport.resp_slots == 3
