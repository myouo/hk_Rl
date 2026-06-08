"""Protocol-level invariants that must hold even at the skeleton stage."""

from __future__ import annotations

import pytest
from hkrl import protocol


def test_schema_version_is_positive() -> None:
    assert isinstance(protocol.SCHEMA_VERSION, int)
    assert protocol.SCHEMA_VERSION >= 1


def test_file_identifier_is_four_bytes() -> None:
    # FlatBuffers file_identifier must be exactly 4 bytes (matches hkrl.fbs).
    assert protocol.FILE_IDENTIFIER == b"HKRL"
    assert len(protocol.FILE_IDENTIFIER) == 4


def test_enum_mirrors_have_expected_members() -> None:
    assert protocol.Command.STEP == 0
    assert protocol.LifecycleState.RUNNING.name == "RUNNING"
    assert protocol.StatusCode.OK == 0
    assert protocol.EntityType.BOSS == 1


@pytest.mark.xfail(reason="encode/decode land in phase 1 (needs generated bindings)", strict=True)
def test_encode_decode_roundtrip() -> None:
    frame = protocol.encode_step_request()
    protocol.decode_step_response(frame)
