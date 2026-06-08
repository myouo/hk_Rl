"""RolloutBatch serialization for worker->learner transfer/spooling.

The training transport can carry these bytes over TCP/gRPC/ZeroMQ later; for
local smoke tests and crash recovery, a compressed NPZ file gives a stable,
pickle-free boundary that preserves dtypes and shapes.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from hkrl.training.rollout_buffer import RolloutBatch

BATCH_FORMAT_VERSION = 2

_ARRAY_FIELDS: tuple[str, ...] = (
    "obs_global",
    "obs_player",
    "obs_entities",
    "entity_mask",
    "actions",
    "log_probs",
    "values",
    "advantages",
    "returns",
    "rewards",
    "dones",
    "truncateds",
    "action_masks",
    "prev_actions",
    "prev_rewards",
    "episode_ids",
    "task_ids",
)


def save_rollout_batch(path: str | Path, batch: RolloutBatch) -> Path:
    """Persist a RolloutBatch atomically as a compressed, pickle-free NPZ file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")

    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **_batch_payload(batch))
    tmp.replace(target)
    return target


def load_rollout_batch(path: str | Path) -> RolloutBatch:
    """Load a RolloutBatch saved by :func:`save_rollout_batch`."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        return _batch_from_npz(data)


def serialize_rollout_batch(batch: RolloutBatch) -> bytes:
    """Serialize a RolloutBatch to compressed NPZ bytes for network transfer."""
    buffer = BytesIO()
    np.savez_compressed(buffer, **_batch_payload(batch))
    return buffer.getvalue()


def deserialize_rollout_batch(payload: bytes) -> RolloutBatch:
    """Load a RolloutBatch from :func:`serialize_rollout_batch` bytes."""
    with np.load(BytesIO(payload), allow_pickle=False) as data:
        return _batch_from_npz(data)


def _batch_payload(batch: RolloutBatch) -> dict[str, Any]:
    payload: dict[str, Any] = {field: np.asarray(getattr(batch, field)) for field in _ARRAY_FIELDS}
    payload["batch_format_version"] = np.array([BATCH_FORMAT_VERSION], dtype=np.int32)
    payload["policy_version"] = np.array([batch.policy_version], dtype=np.int64)
    payload["rnn_states_present"] = np.array([batch.rnn_states is not None], dtype=np.bool_)
    payload["rnn_states"] = (
        np.asarray(batch.rnn_states)
        if batch.rnn_states is not None
        else np.empty((0,), dtype=np.float32)
    )
    return payload


def _batch_from_npz(data: np.lib.npyio.NpzFile) -> RolloutBatch:
    version = _scalar_int(data, "batch_format_version")
    if version != BATCH_FORMAT_VERSION:
        raise ValueError(
            f"unsupported RolloutBatch format version {version}; expected {BATCH_FORMAT_VERSION}"
        )

    arrays = {field: _array(data, field) for field in _ARRAY_FIELDS}
    rnn_states = _array(data, "rnn_states") if _scalar_bool(data, "rnn_states_present") else None
    policy_version = _scalar_int(data, "policy_version")

    batch = RolloutBatch(
        obs_global=arrays["obs_global"],
        obs_player=arrays["obs_player"],
        obs_entities=arrays["obs_entities"],
        entity_mask=arrays["entity_mask"],
        actions=arrays["actions"],
        log_probs=arrays["log_probs"],
        values=arrays["values"],
        advantages=arrays["advantages"],
        returns=arrays["returns"],
        rewards=arrays["rewards"],
        dones=arrays["dones"],
        truncateds=arrays["truncateds"],
        action_masks=arrays["action_masks"],
        prev_actions=arrays["prev_actions"],
        prev_rewards=arrays["prev_rewards"],
        rnn_states=rnn_states,
        episode_ids=arrays["episode_ids"],
        task_ids=arrays["task_ids"],
        policy_version=policy_version,
    )
    _validate_batch_shapes(batch)
    return batch


def _validate_batch_shapes(batch: RolloutBatch) -> None:
    rewards_shape = np.asarray(batch.rewards).shape
    if len(rewards_shape) != 2:
        raise ValueError(f"RolloutBatch rewards must have shape (time, env), got {rewards_shape}")

    expected = rewards_shape[:2]
    for field in _ARRAY_FIELDS:
        array = np.asarray(getattr(batch, field))
        if array.ndim < 2 or array.shape[:2] != expected:
            raise ValueError(
                f"RolloutBatch field {field!r} must share time/env shape {expected}, "
                f"got {array.shape}"
            )

    if batch.rnn_states is not None:
        rnn_states = np.asarray(batch.rnn_states)
        if rnn_states.ndim != 4:
            raise ValueError(
                "RolloutBatch rnn_states must have shape (time, layers, envs, hidden), "
                f"got {rnn_states.shape}"
            )
        if rnn_states.shape[0] != expected[0] or rnn_states.shape[2] != expected[1]:
            raise ValueError(
                "RolloutBatch rnn_states must align with rollout time/env shape "
                f"{expected}, got {rnn_states.shape}"
            )


def _array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        raise ValueError(f"RolloutBatch file missing field {key!r}")
    return data[key].copy()


def _scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int:
    array = _array(data, key).reshape(-1)
    if array.shape != (1,):
        raise ValueError(f"RolloutBatch field {key!r} must be scalar")
    return int(array[0])


def _scalar_bool(data: np.lib.npyio.NpzFile, key: str) -> bool:
    array = _array(data, key).reshape(-1)
    if array.shape != (1,):
        raise ValueError(f"RolloutBatch field {key!r} must be scalar")
    return bool(array[0])
