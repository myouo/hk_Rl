"""run_phase8_smoke script tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def test_run_phase8_smoke_builds_offline_distributed_summary(tmp_path: Path) -> None:
    module = _load_script("run_phase8_smoke.py")
    root = Path(__file__).parents[2]
    output = tmp_path / "summary.json"
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[
            str(root / "configs/tasks/gruz_mother.yaml"),
            str(root / "configs/tasks/hornet_protector.yaml"),
        ],
        work_dir=str(tmp_path / "smoke"),
        num_workers=2,
        seed=123,
        output=str(output),
    )

    summary = module.run_from_args(args)
    module._write_json(output, summary)

    assert summary["ok"] is True
    assert summary["checkpoint_versions"] == [1, 2]
    assert summary["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["worker_ids"] == ["worker-0", "worker-1"]
    assert summary["learner"]["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["learner"]["latest_checkpoint"] is None
    assert summary["worker"]["dry_run"] is True
    assert summary["worker"]["latest_checkpoint"] == 2
    assert summary["worker"]["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["coordinator"]["assignments"].keys() == {"worker-0", "worker-1"}
    assert summary["coordinator"]["ingested_heartbeats"] == 2
    assert summary["coordinator"]["metrics"]["active_worker_count"] == 2.0
    assert summary["coordinator"]["metrics"]["recovering_worker_count"] == 1.0
    assert summary["coordinator"]["metrics"]["stale_policy_worker_count"] == 1.0
    assert summary["coordinator"]["metrics"]["worker_policy_lag_max"] == 1.0
    assert Path(summary["artifacts"]["heartbeat_jsonl"]).exists()
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_run_phase8_smoke_resets_generated_work_dir(tmp_path: Path) -> None:
    module = _load_script("run_phase8_smoke.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        tasks=[
            str(root / "configs/tasks/gruz_mother.yaml"),
            str(root / "configs/tasks/hornet_protector.yaml"),
        ],
        work_dir=str(tmp_path / "smoke"),
        num_workers=2,
        seed=123,
        output=None,
    )

    first = module.run_from_args(args)
    second = module.run_from_args(args)

    assert first["checkpoint_versions"] == [1, 2]
    assert second["checkpoint_versions"] == [1, 2]
    assert second["worker"]["latest_checkpoint"] == 2


def test_run_phase8_smoke_rejects_empty_worker_count(tmp_path: Path) -> None:
    module = _load_script("run_phase8_smoke.py")
    args = argparse.Namespace(
        config="configs/train/remote_learner.yaml",
        tasks=["configs/tasks/gruz_mother.yaml"],
        work_dir=str(tmp_path),
        num_workers=0,
        seed=0,
        output=None,
    )

    try:
        module.run_from_args(args)
    except ValueError as exc:
        assert "num_workers" in str(exc)
    else:
        raise AssertionError("expected num_workers=0 to fail")


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
