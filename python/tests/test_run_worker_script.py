"""run_worker script tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hkrl.learner.checkpoint_registry import CheckpointRegistry
from hkrl.training.batch_io import load_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch


def test_run_worker_dry_run_builds_summary(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.delenv("HKRL_AUTH_TOKEN", raising=False)
    registry = CheckpointRegistry(str(tmp_path / "checkpoints"))
    registry.publish({"model_state_dict": {}, "policy_version": 3}, policy_version=3, step=1)
    module = _load_script("run_worker.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        task=str(root / "configs/tasks/gruz_mother.yaml"),
        tasks=None,
        env_host="127.0.0.2",
        env_port=6001,
        learner="127.0.0.1:5600",
        registry=str(tmp_path / "checkpoints"),
        batch_dir=str(tmp_path / "batches"),
        heartbeat_jsonl=str(tmp_path / "heartbeats.jsonl"),
        worker_id="game-pc-1",
        steps=None,
        max_consecutive_failures=5,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary == {
        "algorithm": "appo",
        "auth_token_configured": False,
        "auth_token_env": "HKRL_AUTH_TOKEN",
        "auth_token_required": True,
        "batch_dir": str(tmp_path / "batches"),
        "dry_run": True,
        "enable_macro_actions": True,
        "env_host": "127.0.0.2",
        "env_port": 6001,
        "heartbeat_jsonl": str(tmp_path / "heartbeats.jsonl"),
        "learner": "127.0.0.1:5600",
        "learner_upload_enabled": True,
        "latest_checkpoint": 1,
        "max_consecutive_failures": 5,
        "model": "entity_attention_gru",
        "n_macro_actions": 11,
        "registry": str(tmp_path / "checkpoints"),
        "task_id": "gruz_mother",
        "task_ids": ["gruz_mother"],
        "worker_id": "game-pc-1",
    }


def test_run_worker_task_provider_cycles_tasks() -> None:
    module = _load_script("run_worker.py")
    tasks = [
        module.TaskConfig(task_id="a", wire_id=1, scene="A"),
        module.TaskConfig(task_id="b", wire_id=2, scene="B"),
    ]
    provider = module._make_task_provider(tasks)
    assert provider is not None

    assert provider().task_id == "a"
    assert provider().task_id == "b"
    assert provider().task_id == "a"


def test_run_worker_rejects_incompatible_task_layouts() -> None:
    module = _load_script("run_worker.py")
    tasks = [
        module.TaskConfig(
            task_id="a",
            wire_id=1,
            scene="A",
            action={"n_macro_actions": 11},
        ),
        module.TaskConfig(
            task_id="b",
            wire_id=2,
            scene="B",
            action={"n_macro_actions": 4},
        ),
    ]

    try:
        module._validate_task_layouts(tasks)
    except ValueError as exc:
        assert "n_macro_actions" in str(exc)
    else:
        raise AssertionError("expected incompatible macro layouts to fail")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("config", "", "config"),
        ("task", "", "task"),
        ("tasks", [], "tasks"),
        ("tasks", "configs/tasks/gruz_mother.yaml", "tasks"),
        ("tasks", [""], r"tasks\[0\]"),
        ("worker_id", "", "worker_id"),
        ("worker_id", "   ", "worker_id"),
        ("worker_id", None, "worker_id"),
        ("steps", 0, "steps"),
        ("steps", True, "steps"),
        ("max_consecutive_failures", -1, "max_consecutive_failures"),
        ("max_consecutive_failures", False, "max_consecutive_failures"),
        ("env_host", "", "env_host"),
        ("env_host", "   ", "env_host"),
        ("env_port", 0, "env_port"),
        ("env_port", 65536, "env_port"),
        ("env_port", False, "env_port"),
        ("learner", "", "learner endpoint"),
        ("learner", "missing-port", "host:port"),
        ("learner", "127.0.0.1:0", "port"),
        ("registry", "", "registry"),
        ("batch_dir", "", "batch_dir"),
        ("heartbeat_jsonl", "", "heartbeat_jsonl"),
    ],
)
def test_run_worker_rejects_invalid_gate_args(
    field: str,
    value: object,
    match: str,
) -> None:
    module = _load_script("run_worker.py")
    args = _worker_args(**{field: value})

    with pytest.raises(ValueError, match=match):
        module._validate_worker_args(args)


def test_run_worker_mlp_model_uses_default_hidden_when_rnn_hidden_zero() -> None:
    module = _load_script("run_worker.py")
    cfg = module.load_train_config(Path(__file__).parents[2] / "configs/train/ppo_mlp.yaml")
    model = module._build_model(
        cfg,
        {
            "global": (2,),
            "player": (3,),
            "entities": (4, 5),
            "entity_mask": (4,),
        },
        enable_macro=True,
        n_macros=11,
        max_entities=4,
    )

    assert model.trunk[0].out_features == 256


def test_run_worker_checkpoint_auth_token_only_for_http(monkeypatch: object) -> None:
    module = _load_script("run_worker.py")
    cfg = module.load_train_config(Path(__file__).parents[2] / "configs/train/remote_learner.yaml")
    monkeypatch.delenv("HKRL_AUTH_TOKEN", raising=False)

    assert module._checkpoint_auth_token(cfg, "/tmp/checkpoints") is None
    with pytest.raises(ValueError, match="HKRL_AUTH_TOKEN"):
        module._checkpoint_auth_token(cfg, "http://127.0.0.1:8000/checkpoints")

    monkeypatch.setenv("HKRL_AUTH_TOKEN", "secret")
    assert module._checkpoint_auth_token(cfg, "http://127.0.0.1:8000/checkpoints") == "secret"


def test_run_worker_batch_spooler_writes_rollout_npz(tmp_path: Path) -> None:
    module = _load_script("run_worker.py")
    written: list[str] = []
    uploader = module._make_batch_uploader(str(tmp_path), "game/pc:1", written)
    assert uploader is not None

    accepted = uploader(_sample_batch(policy_version=4))

    assert accepted is None
    assert len(written) == 1
    path = Path(written[0])
    assert path.name == "game_pc_1_00000001_v000004.npz"
    loaded = load_rollout_batch(path)
    assert loaded.policy_version == 4
    np.testing.assert_array_equal(loaded.rewards, np.array([[1.0]], dtype=np.float32))


def test_run_worker_batch_spooler_rejects_empty_batch_dir() -> None:
    module = _load_script("run_worker.py")

    with pytest.raises(ValueError, match="batch_dir"):
        module._make_batch_uploader("", "game-pc-1", [])


def test_run_worker_heartbeat_sink_writes_coordinator_jsonl(tmp_path: Path) -> None:
    module = _load_script("run_worker.py")
    written: list[None] = []
    sink = module._make_heartbeat_sink(
        str(tmp_path / "nested" / "heartbeats.jsonl"),
        "game/pc:1",
        written,
    )
    assert sink is not None

    sink({"sps": 12.5, "status": "running"})

    records = [
        json.loads(line)
        for line in (tmp_path / "nested" / "heartbeats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert written == [None]
    assert records == [
        {
            "payload": {"sps": 12.5, "status": "running"},
            "worker_id": "game_pc_1",
        }
    ]


def test_run_worker_heartbeat_sink_rejects_empty_path() -> None:
    module = _load_script("run_worker.py")

    with pytest.raises(ValueError, match="heartbeat_jsonl"):
        module._make_heartbeat_sink("", "game-pc-1", [])


def test_run_worker_batch_uploader_sends_to_learner(monkeypatch: object) -> None:
    module = _load_script("run_worker.py")
    submitted_versions: list[int] = []
    clients: list[tuple[str, str | None]] = []

    class FakeBatchClient:
        def __init__(self, endpoint: str, *, auth_token: str | None = None) -> None:
            clients.append((endpoint, auth_token))

        def submit(self, batch: RolloutBatch) -> bool:
            submitted_versions.append(batch.policy_version)
            return True

    monkeypatch.setattr(module, "BatchIntakeClient", FakeBatchClient)
    written: list[str] = []
    uploaded: list[bool] = []
    uploader = module._make_batch_uploader(
        None,
        "game-pc-1",
        written,
        learner_endpoint="127.0.0.1:5600",
        auth_token="secret",
        uploaded=uploaded,
    )
    assert uploader is not None

    accepted = uploader(_sample_batch(policy_version=7))

    assert accepted is True
    assert clients == [("127.0.0.1:5600", "secret")]
    assert written == []
    assert submitted_versions == [7]
    assert uploaded == [True]


def test_run_worker_upload_summary_counts_accepted_and_rejected() -> None:
    module = _load_script("run_worker.py")

    assert module._upload_summary([True, False, True]) == {
        "learner_accepted_batches": 2,
        "learner_rejected_batches": 1,
        "learner_submitted_batches": 3,
        "uploaded_batches": 3,
    }


def test_run_worker_upload_summary_rejects_non_boolean_acks() -> None:
    module = _load_script("run_worker.py")

    with pytest.raises(ValueError, match="indexes \\[1\\]"):
        module._upload_summary([True, None])


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker_args(**overrides: object) -> argparse.Namespace:
    root = Path(__file__).parents[2]
    values: dict[str, object] = {
        "batch_dir": None,
        "config": str(root / "configs/train/remote_learner.yaml"),
        "env_host": None,
        "env_port": None,
        "heartbeat_jsonl": None,
        "learner": None,
        "max_consecutive_failures": 3,
        "registry": None,
        "steps": None,
        "task": str(root / "configs/tasks/gruz_mother.yaml"),
        "tasks": None,
        "worker_id": "worker-0",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _sample_batch(policy_version: int) -> RolloutBatch:
    return RolloutBatch(
        obs_global=np.zeros((1, 1, 2), dtype=np.float32),
        obs_player=np.zeros((1, 1, 3), dtype=np.float32),
        obs_entities=np.zeros((1, 1, 4, 5), dtype=np.float32),
        entity_mask=np.ones((1, 1, 4), dtype=bool),
        actions=np.zeros((1, 1, 2), dtype=np.int64),
        log_probs=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        advantages=np.ones((1, 1), dtype=np.float32),
        returns=np.ones((1, 1), dtype=np.float32),
        rewards=np.ones((1, 1), dtype=np.float32),
        dones=np.zeros((1, 1), dtype=bool),
        truncateds=np.zeros((1, 1), dtype=bool),
        action_masks=np.ones((1, 1, 6), dtype=bool),
        prev_actions=np.zeros((1, 1, 2), dtype=np.int64),
        prev_rewards=np.zeros((1, 1), dtype=np.float32),
        rnn_states=None,
        episode_ids=np.ones((1, 1), dtype=np.uint64),
        task_ids=np.ones((1, 1), dtype=np.int64),
        policy_version=policy_version,
    )
