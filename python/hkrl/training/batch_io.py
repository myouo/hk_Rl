"""RolloutBatch serialization for worker->learner transfer/spooling.

The training transport can carry these bytes over TCP/gRPC/ZeroMQ later; for
local smoke tests and crash recovery, a compressed NPZ file gives a stable,
pickle-free boundary that preserves dtypes and shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hkrl.training.rollout_buffer import RolloutBatch

BATCH_FORMAT_VERSION = 1

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
    "episode_ids",
    "task_ids",
)


def save_rollout_batch(path: str | Path, batch: RolloutBatch) -> Path:
    """Persist a RolloutBatch atomically as a compressed, pickle-free NPZ file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    payload: dict[str, Any] = {field: np.asarray(getattr(batch, field)) for field in _ARRAY_FIELDS}
    payload["batch_format_version"] = np.array([BATCH_FORMAT_VERSION], dtype=np.int32)
    payload["policy_version"] = np.array([batch.policy_version], dtype=np.int64)
    payload["rnn_states_present"] = np.array([batch.rnn_states is not None], dtype=np.bool_)
    payload["rnn_states"] = (
        np.asarray(batch.rnn_states)
        if batch.rnn_states is not None
        else np.empty((0,), dtype=np.float32)
    )

    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    tmp.replace(target)
    return target


def load_rollout_batch(path: str | Path) -> RolloutBatch:
    """Load a RolloutBatch saved by :func:`save_rollout_batch`."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        version = _scalar_int(data, "batch_format_version")
        if version != BATCH_FORMAT_VERSION:
            raise ValueError(
                f"unsupported RolloutBatch format version {version}; "
                f"expected {BATCH_FORMAT_VERSION}"
            )

        arrays = {field: _array(data, field) for field in _ARRAY_FIELDS}
        rnn_states = (
            _array(data, "rnn_states") if _scalar_bool(data, "rnn_states_present") else None
        )
        policy_version = _scalar_int(data, "policy_version")

    return RolloutBatch(
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
        rnn_states=rnn_states,
        episode_ids=arrays["episode_ids"],
        task_ids=arrays["task_ids"],
        policy_version=policy_version,
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
