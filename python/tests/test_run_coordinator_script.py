"""run_coordinator script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


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
        eval_metrics=None,
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
    assert summary["eval_metrics"] is None
    assert summary["eval_winrates"] == {}
    assert summary["heartbeat_timeout_s"] == 9.0
    assert summary["ingested_heartbeats"] == 0
    assert summary["num_workers"] == 2
    assert summary["sampler_mastered_tasks"] == []
    assert summary["sampler_weights"] == {"gruz_mother": 1.0}
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
        "recovering_worker_count": 0.0,
        "worker_policy_version_min": 0.0,
        "worker_policy_version_max": 0.0,
        "worker_policy_lag_max": 0.0,
        "stale_policy_worker_count": 0.0,
        "worker_without_policy_version_count": 2.0,
        "worker_checkpoint_version_min": 0.0,
        "worker_checkpoint_version_max": 0.0,
        "worker_checkpoint_lag_max": 0.0,
        "stale_checkpoint_worker_count": 0.0,
        "worker_without_checkpoint_version_count": 2.0,
        "worker_learner_upload_accepted_batches": 0.0,
        "worker_learner_upload_failed_batches": 0.0,
        "worker_learner_upload_rejected_batches": 0.0,
        "worker_learner_upload_submitted_batches": 0.0,
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
        eval_metrics=None,
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


def test_run_coordinator_rejects_non_finite_heartbeat_jsonl_metrics(tmp_path: Path) -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    heartbeat_jsonl = tmp_path / "heartbeats.jsonl"
    heartbeat_jsonl.write_text(
        '{"worker_id":"game-pc-1","payload":{"sps":NaN,"status":"running"}}\n',
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
        eval_metrics=None,
        seed=0,
        dry_run=True,
    )

    with pytest.raises(ValueError, match=r"sps.*finite"):
        module.run_from_args(args)


def test_run_coordinator_applies_eval_metrics_to_sampler(tmp_path: Path) -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    eval_metrics = tmp_path / "eval.json"
    eval_metrics.write_text(
        (
            '{"metrics":{'
            '"gruz_mother":{"win_rate":0.9},'
            '"hornet_protector_attuned":{"win_rate":0.2}'
            "}}\n"
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[
            str(root / "configs/tasks/gruz_mother.yaml"),
            str(root / "configs/tasks/hornet_protector.yaml"),
        ],
        bind="127.0.0.1:0",
        num_workers=2,
        worker_ids=None,
        heartbeat_timeout_s=None,
        heartbeat_jsonl=None,
        eval_metrics=str(eval_metrics),
        seed=0,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary["eval_winrates"] == {
        "gruz_mother": 0.9,
        "hornet_protector_attuned": 0.2,
    }
    assert summary["sampler_mastered_tasks"] == ["gruz_mother"]
    assert summary["sampler_weights"]["gruz_mother"] == pytest.approx(0.1)
    assert summary["sampler_weights"]["hornet_protector_attuned"] == pytest.approx(0.8)


def test_run_coordinator_accepts_per_boss_win_rate_metrics(tmp_path: Path) -> None:
    module = _load_script("run_coordinator.py")
    root = Path(__file__).parents[2]
    eval_metrics = tmp_path / "eval.json"
    eval_metrics.write_text(
        '{"metrics":{"gruz_mother":{"per_boss_win_rate":0.25}}}\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[str(root / "configs/tasks/gruz_mother.yaml")],
        bind="127.0.0.1:0",
        num_workers=1,
        worker_ids=None,
        heartbeat_timeout_s=None,
        heartbeat_jsonl=None,
        eval_metrics=str(eval_metrics),
        seed=0,
        dry_run=True,
    )

    summary = module.run_from_args(args)

    assert summary["eval_winrates"] == {"gruz_mother": 0.25}
    assert summary["sampler_weights"]["gruz_mother"] == pytest.approx(0.75)


def test_run_coordinator_uses_per_boss_win_rate_when_win_rate_is_null(
    tmp_path: Path,
) -> None:
    module = _load_script("run_coordinator.py")
    eval_metrics = tmp_path / "eval.json"
    eval_metrics.write_text(
        '{"metrics":{"gruz_mother":{"win_rate":null,"per_boss_win_rate":0.5}}}\n',
        encoding="utf-8",
    )

    assert module._load_eval_winrates(str(eval_metrics)) == {"gruz_mother": 0.5}


@pytest.mark.parametrize(
    ("metrics_json", "match"),
    [
        ('{"metrics":{"gruz_mother":{"win_rate":1.2}}}\n', r"win_rate.*\[0, 1\]"),
        ('{"metrics":{"gruz_mother":{"win_rate":NaN}}}\n', r"win_rate.*finite"),
        ('{"metrics":{"gruz_mother":{"win_rate":"0.5"}}}\n', r"win_rate.*numeric"),
        ('{"metrics":{"gruz_mother":{"win_rate":true}}}\n', r"win_rate.*numeric"),
        ('{"metrics":{"gruz_mother":0.5}}\n', r"must be an object"),
        ('{"metrics":{"gruz_mother":{"episode_reward":1.0}}}\n', r"must include win_rate"),
    ],
)
def test_run_coordinator_rejects_invalid_eval_winrates(
    tmp_path: Path, metrics_json: str, match: str
) -> None:
    module = _load_script("run_coordinator.py")
    eval_metrics = tmp_path / "eval.json"
    eval_metrics.write_text(metrics_json, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        module._load_eval_winrates(str(eval_metrics))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"config": ""}, "config"),
        ({"tasks": []}, "at least one task"),
        ({"tasks": "configs/tasks/gruz_mother.yaml"}, "at least one task"),
        ({"tasks": [""]}, r"tasks\[0\]"),
        ({"bind": ""}, "bind"),
        ({"heartbeat_jsonl": ""}, "heartbeat_jsonl"),
        ({"eval_metrics": ""}, "eval_metrics"),
        ({"num_workers": 0}, "num_workers"),
        ({"num_workers": True}, "num_workers"),
        ({"heartbeat_timeout_s": 0.0}, "heartbeat_timeout_s"),
        ({"heartbeat_timeout_s": float("nan")}, "heartbeat_timeout_s"),
        ({"heartbeat_timeout_s": True}, "heartbeat_timeout_s"),
        ({"seed": 1.5}, "seed"),
        ({"seed": False}, "seed"),
        ({"worker_ids": []}, "worker ids"),
        ({"worker_ids": [""]}, "worker ids"),
        ({"worker_ids": [" worker-0"]}, "surrounding whitespace"),
        ({"worker_ids": "worker-0"}, "worker_ids"),
        ({"worker_ids": ["worker-0"], "num_workers": 1}, "cannot be combined"),
    ],
)
def test_run_coordinator_rejects_invalid_gate_args(
    overrides: dict[str, object],
    match: str,
) -> None:
    module = _load_script("run_coordinator.py")
    args = _coordinator_args(**overrides)

    with pytest.raises(ValueError, match=match):
        module.run_from_args(args)


def test_run_coordinator_eval_metrics_loader_rejects_empty_path() -> None:
    module = _load_script("run_coordinator.py")

    with pytest.raises(ValueError, match="eval_metrics"):
        module._load_eval_winrates("")


def test_run_coordinator_heartbeat_loader_rejects_empty_path() -> None:
    module = _load_script("run_coordinator.py")

    with pytest.raises(ValueError, match="heartbeat_jsonl"):
        module._ingest_heartbeat_jsonl(object(), "")


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
        eval_metrics=None,
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
        eval_metrics=None,
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


def _coordinator_args(**overrides: object) -> argparse.Namespace:
    root = Path(__file__).parents[2]
    values: dict[str, object] = {
        "config": str(root / "configs/train/remote_learner.yaml"),
        "tasks": [str(root / "configs/tasks/gruz_mother.yaml")],
        "bind": "127.0.0.1:0",
        "num_workers": None,
        "worker_ids": None,
        "heartbeat_timeout_s": None,
        "heartbeat_jsonl": None,
        "eval_metrics": None,
        "seed": 0,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)
