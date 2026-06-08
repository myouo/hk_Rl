"""run_coordinator script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def test_run_coordinator_builds_assignment_summary() -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[str(root / "configs/tasks/gruz_mother.yaml")],
        bind=None,
        num_workers=2,
        worker_ids=None,
        heartbeat_timeout_s=9.0,
        heartbeat_jsonl=None,
        seed=123,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary["assignments"] == {
        "worker-0": "gruz_mother",
        "worker-1": "gruz_mother",
    }
    assert summary["bind"] == "0.0.0.0:5610"
    assert summary["dry_run"] is True
    assert summary["heartbeat_timeout_s"] == 9.0
    assert summary["ingested_heartbeats"] == 0
    assert summary["num_workers"] == 2
    assert summary["task_ids"] == ["gruz_mother"]
    assert summary["task_wire_ids"] == {"gruz_mother": 0}
    assert summary["metrics"] == {
        "worker_count": 2.0,
        "active_worker_count": 2.0,
        "lost_worker_count": 0.0,
        "assigned_worker_count": 2.0,
        "sps": 0.0,
        "sps_mean": 0.0,
        "worker_crash_count": 0.0,
    }
    assert summary["workers"]["worker-0"]["assigned_task"] == "gruz_mother"


def test_run_coordinator_ingests_heartbeat_jsonl(tmp_path: Path) -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    heartbeat_jsonl = tmp_path / "heartbeats.jsonl"
    heartbeat_jsonl.write_text(
        "\n".join(
            [
                (
                    '{"worker_id":"game-pc-1","payload":{"sps":12.5,'
                    '"status":"running","worker_crash_count":1}}'
                ),
                '{"worker_id":"extra","sps":7.5,"status":"running"}',
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[str(root / "configs/tasks/gruz_mother.yaml")],
        bind="127.0.0.1:0",
        num_workers=None,
        worker_ids=["game-pc-1"],
        heartbeat_timeout_s=None,
        heartbeat_jsonl=str(heartbeat_jsonl),
        seed=0,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary["bind"] == "127.0.0.1:0"
    assert summary["assignments"] == {"game-pc-1": "gruz_mother"}
    assert summary["ingested_heartbeats"] == 2
    assert summary["metrics"]["worker_count"] == 2.0
    assert summary["metrics"]["active_worker_count"] == 2.0
    assert summary["metrics"]["sps"] == 20.0
    assert summary["metrics"]["sps_mean"] == 10.0
    assert summary["metrics"]["worker_crash_count"] == 1.0
    assert summary["workers"]["game-pc-1"]["info"]["status"] == "running"
    assert summary["workers"]["game-pc-1"]["metrics"]["sps"] == 12.5
    assert summary["workers"]["extra"]["assigned_task"] is None
    assert summary["workers"]["extra"]["info"]["source"] == "heartbeat"


def test_run_coordinator_rejects_duplicate_worker_ids() -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[str(root / "configs/tasks/gruz_mother.yaml")],
        bind=None,
        num_workers=None,
        worker_ids=["same", "same"],
        heartbeat_timeout_s=None,
        heartbeat_jsonl=None,
        seed=None,
        dry_run=True,
    )

    try:
        module.run_from_args(args)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("expected duplicate worker ids to fail")


def test_run_coordinator_rejects_wildcard_bind_for_localhost_scope(tmp_path: Path) -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    config = tmp_path / "localhost.yaml"
    config.write_text(
        "\n".join(
            [
                "coordinator:",
                "  bind: 0.0.0.0:5610",
                "security:",
                "  bind_scope: localhost",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config),
        tasks=[str(root / "configs/tasks/gruz_mother.yaml")],
        bind=None,
        num_workers=1,
        worker_ids=None,
        heartbeat_timeout_s=None,
        heartbeat_jsonl=None,
        seed=None,
        dry_run=True,
    )

    try:
        module.run_from_args(args)
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("expected localhost scope wildcard bind to fail")


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
