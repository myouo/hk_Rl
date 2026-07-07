"""run_phase8_smoke script tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def test_run_phase8_smoke_builds_offline_distributed_summary(tmp_path: Path) -> None:
    module = _load_script("run_phase8_smoke.py")
    root = Path(__file__).parents[2]
    output = tmp_path / "summary.json"
    dashboard_html = tmp_path / "dashboard.html"
    dashboard_json = tmp_path / "dashboard.json"
    profile_json = tmp_path / "profile.json"
    profile_md = tmp_path / "profile.md"
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
        dashboard_html=str(dashboard_html),
        dashboard_json=str(dashboard_json),
        profile_json=str(profile_json),
        profile_md=str(profile_md),
    )

    summary = module.run_from_args(args)

    assert summary["ok"] is True
    assert summary["checkpoint_versions"] == [1, 2]
    assert summary["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["worker_ids"] == ["worker-0", "worker-1"]
    assert summary["learner"]["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["learner"]["latest_checkpoint"] == 1
    assert summary["learner"]["startup_checkpoint"] == 1
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
    assert "HKRL Phase 8 Dashboard" in dashboard_html.read_text(encoding="utf-8")
    assert json.loads(dashboard_json.read_text(encoding="utf-8"))["metrics"]["sps"] == 32.0
    assert json.loads(profile_json.read_text(encoding="utf-8"))["metrics"]["sps"] == 32.0
    assert "HKRL Phase 8 Profile" in profile_md.read_text(encoding="utf-8")


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


def test_run_phase8_smoke_ignores_directory_cleanup_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_phase8_smoke.py")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    calls: list[tuple[Path, bool]] = []

    def fake_rmtree(path: Path, *, ignore_errors: bool = False) -> None:
        calls.append((path, ignore_errors))

    monkeypatch.setattr(module.shutil, "rmtree", fake_rmtree)

    module._reset_generated_artifacts(tmp_path)

    assert calls == [(checkpoints, True)]


def test_run_phase8_smoke_work_dir_lock_releases(tmp_path: Path) -> None:
    module = _load_script("run_phase8_smoke.py")
    lock_path = tmp_path / ".phase8-smoke.lock"

    with module._work_dir_lock(tmp_path):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_run_phase8_smoke_work_dir_lock_times_out(tmp_path: Path) -> None:
    module = _load_script("run_phase8_smoke.py")
    (tmp_path / ".phase8-smoke.lock").write_text("other", encoding="ascii")

    with (
        pytest.raises(TimeoutError, match="work-dir lock"),
        module._work_dir_lock(tmp_path, timeout_s=0.0, poll_s=0.0),
    ):
        pass


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"config": ""}, "config"),
        ({"tasks": []}, "at least one task"),
        ({"tasks": "configs/tasks/gruz_mother.yaml"}, "at least one task"),
        ({"tasks": [""]}, r"tasks\[0\]"),
        ({"num_workers": 0}, "num_workers"),
        ({"num_workers": True}, "num_workers"),
        ({"num_workers": 1.5}, "num_workers"),
        ({"seed": None}, "seed"),
        ({"seed": False}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"work_dir": ""}, "work_dir"),
        ({"output": ""}, "output"),
        ({"dashboard_html": ""}, "dashboard_html"),
        ({"dashboard_json": ""}, "dashboard_json"),
        ({"profile_json": ""}, "profile_json"),
        ({"profile_md": ""}, "profile_md"),
    ],
)
def test_run_phase8_smoke_rejects_invalid_gate_args(
    overrides: dict[str, object],
    match: str,
) -> None:
    module = _load_script("run_phase8_smoke.py")
    args = _smoke_args(**overrides)

    with pytest.raises(ValueError, match=match):
        module.run_from_args(args)


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


def _smoke_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "config": "configs/train/remote_learner.yaml",
        "tasks": ["configs/tasks/gruz_mother.yaml"],
        "work_dir": None,
        "num_workers": 1,
        "seed": 0,
        "output": None,
        "dashboard_html": None,
        "dashboard_json": None,
        "profile_json": None,
        "profile_md": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
