"""Wire protocol constants and (de)serialization boundary.

Implements docs/protocol.md. The message *layout* is owned by
``schema/hkrl.fbs`` and the generated bindings under ``hkrl.schema``; this module
holds the version constant, command/enum mirrors for ergonomic Python use, and
the encode/decode helpers that wrap the FlatBuffers bindings.

IMPORTANT: ``SCHEMA_VERSION`` MUST match the constant mirrored in
``mod/HKRLEnvMod/Transport/Protocol.cs``. Bump both on every schema change and
record it in CHANGELOG.md.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Mirrors the schema_version carried in every StepRequest/StepResponse and the
# C# Protocol.SCHEMA_VERSION. See schema/README.md evolution rules.
SCHEMA_VERSION: int = 1

# FlatBuffers file_identifier (must equal the one in hkrl.fbs).
FILE_IDENTIFIER: bytes = b"HKRL"


class Command(enum.IntEnum):
    """Mirror of HKRL.Command (schema/hkrl.fbs)."""

    STEP = 0
    RESET = 1
    PAUSE = 2
    RESUME = 3
    SET_TASK = 4
    SET_TIMESCALE = 5
    PING = 6


class LifecycleState(enum.IntEnum):
    """Mirror of HKRL.LifecycleState. See docs/episode_lifecycle.md."""

    IDLE = 0
    RESET_REQUESTED = 1
    FREEZE_INPUT = 2
    CLEAR_EVENTS = 3
    LOAD_SCENE = 4
    WAIT_SCENE_READY = 5
    WAIT_PLAYER_READY = 6
    WAIT_BOSS_READY = 7
    RESTORE_PLAYER_STATE = 8
    CLEAR_PROJECTILES = 9
    COUNTDOWN = 10
    RUNNING = 11
    TERMINATING = 12
    REPORT_DONE = 13
    CLEANUP = 14


class StatusCode(enum.IntEnum):
    """Mirror of HKRL.StatusCode (StepResponse.error_code)."""

    OK = 0
    RESET_TIMEOUT = 1
    SCENE_LOAD_FAILED = 2
    BOSS_NOT_FOUND = 3
    PLAYER_NOT_READY = 4
    INTERNAL_ERROR = 5
    SCHEMA_MISMATCH = 6


class EntityType(enum.IntEnum):
    """Mirror of HKRL.EntityType."""

    PLAYER = 0
    BOSS = 1
    ENEMY = 2
    PROJECTILE = 3
    HAZARD = 4
    PLATFORM = 5
    PICKUP = 6
    EFFECT = 7
    UNKNOWN = 255


class RewardEventKind(enum.IntEnum):
    """Mirror of HKRL.RewardEventKind. Payload semantics: docs/reward_design.md."""

    DAMAGE_DEALT = 0
    DAMAGE_TAKEN = 1
    HEAL = 2
    SOUL_GAINED = 3
    BOSS_KILLED = 4
    PLAYER_DEATH = 5
    SCENE_CHANGED = 6
    INVALID_ACTION = 7
    STAGGER = 8


@dataclass(slots=True)
class RewardEvent:
    """Decoded reward event (Python-friendly view of HKRL.RewardEvent)."""

    kind: RewardEventKind
    entity_id: int = 0
    amount: float = 0.0
    aux_int: int = 0
    aux_int2: int = 0


@dataclass(slots=True)
class DecodedObservation:
    """Numpy-friendly decoded observation snapshot.

    Field arrays are documented in docs/observation_schema.md. Decoders fill
    these from the zero-copy FlatBuffers buffer; normalization happens later in
    ``hkrl.spaces`` / wrappers, not here.
    """

    global_state: np.ndarray  # GlobalState features
    player_state: np.ndarray  # PlayerState features
    entities: np.ndarray  # (max_entities, entity_feat_dim)
    entity_mask: np.ndarray  # (max_entities,) bool


def encode_step_request(*args: object, **kwargs: object) -> bytes:
    """Encode a StepRequest into a length-prefixed FlatBuffers frame.

    Wraps the generated ``hkrl.schema`` builders. Framing per docs/protocol.md §1.

    TODO(phase-1): implement using generated FlatBuffers bindings + struct length
    prefix (uint32 LE).
    """
    raise NotImplementedError


def decode_step_response(frame: bytes) -> object:
    """Decode a length-prefixed StepResponse frame (zero-copy).

    Verifies ``file_identifier`` and ``schema_version`` before reading; raises on
    ``StatusCode.SCHEMA_MISMATCH`` divergence.

    TODO(phase-1): implement using generated bindings; return a typed view
    (DecodedObservation + RewardEvent[] + flags).
    """
    raise NotImplementedError
