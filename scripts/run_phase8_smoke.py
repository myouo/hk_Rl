#!/usr/bin/env python3
"""Run an offline Phase 8 distributed smoke check.

The smoke check does not connect to Hollow Knight or open learner/worker network
ports. It validates the config-driven distributed wiring by building the learner,
publishing a local checkpoint registry, probing that registry from a worker
dry-run, and ingesting synthetic worker heartbeats/evaluator metrics through the
coordinator monitoring path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from hkrl.learner.checkpoint_payload import validate_checkpoint_payload
from hkrl.learner.checkpoint_registry import CheckpointMeta, CheckpointRegistry
from hkrl.utils.config import load_task_config


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HKRL Phase 8 offline smoke")
    p.add_argument(
        "--config",
        default="configs/train/remote_learner.yaml",
        help="distributed train YAML",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "configs/tasks/gruz_mother.yaml",
            "configs/tasks/hornet_protector.yaml",
        ],
        help="task YAMLs used for learner/worker/coordinator wiring",
    )
    p.add_argument("--work-dir", help="directory for generated smoke artifacts")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="optional path to write the summary JSON")
    p.add_argument("--dashboard-html", help="optional path to write dashboard HTML")
    p.add_argument("--dashboard-json", help="optional path to write dashboard JSON")
    p.add_argument("--profile-json", help="optional path to write profile JSON")
    p.add_argument("--profile-md", help="optional path to write profile Markdown")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary = run_from_args(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    _validate_smoke_args(args)

    work_dir = _work_dir(getattr(args, "work_dir", None))
    with _work_dir_lock(work_dir):
        summary = _run_from_args_unlocked(args, work_dir)
        if getattr(args, "output", None):
            _write_json(Path(args.output), summary)
        _write_requested_artifacts(summary, args)
        return summary


def _run_from_args_unlocked(args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    _reset_generated_artifacts(work_dir)
    checkpoint_dir = work_dir / "checkpoints"
    heartbeat_jsonl = work_dir / "worker-heartbeats.jsonl"
    eval_metrics_json = work_dir / "eval-metrics.json"
    task_paths = [str(path) for path in args.tasks]
    tasks = [load_task_config(path) for path in task_paths]
    worker_ids = [f"worker-{idx}" for idx in range(args.num_workers)]

    run_learner = _load_script_module("run_learner.py")
    run_worker = _load_script_module("run_worker.py")
    run_coordinator = _load_script_module("run_coordinator.py")

    learner_summary = run_learner.run_from_args(
        argparse.Namespace(
            config=args.config,
            bind="127.0.0.1:0",
            batch_dir=None,
            checkpoint_dir=str(checkpoint_dir),
            disable_macro_actions=False,
            intake_count=0,
            intake_timeout_s=1.0,
            max_entities=None,
            max_staleness=None,
            n_macro_actions=None,
            publish_every_updates=None,
            serve_forever=False,
            task=None,
            tasks=task_paths,
            tier=None,
        )
    )
    checkpoint_metas = _publish_smoke_checkpoints(checkpoint_dir)
    checkpoint_versions = [meta.version for meta in checkpoint_metas]
    _write_heartbeat_jsonl(
        heartbeat_jsonl,
        worker_ids=worker_ids,
        checkpoint_metas=checkpoint_metas,
    )
    _write_eval_metrics(eval_metrics_json, [task.task_id for task in tasks])

    worker_summary = run_worker.run_from_args(
        argparse.Namespace(
            config=args.config,
            task=task_paths[0],
            tasks=task_paths,
            learner=None,
            registry=str(checkpoint_dir),
            batch_dir=str(work_dir / "batches"),
            heartbeat_jsonl=str(heartbeat_jsonl),
            worker_id=worker_ids[0],
            steps=None,
            max_consecutive_failures=3,
            dry_run=True,
        )
    )
    coordinator_summary = run_coordinator.run_from_args(
        argparse.Namespace(
            config=args.config,
            tasks=task_paths,
            bind="127.0.0.1:0",
            num_workers=None,
            worker_ids=worker_ids,
            heartbeat_timeout_s=None,
            heartbeat_jsonl=str(heartbeat_jsonl),
            eval_metrics=str(eval_metrics_json),
            seed=args.seed,
            dry_run=True,
        )
    )

    return {
        "artifacts": {
            "checkpoint_dir": str(checkpoint_dir),
            "eval_metrics": str(eval_metrics_json),
            "heartbeat_jsonl": str(heartbeat_jsonl),
            "work_dir": str(work_dir),
        },
        "checkpoint_policy_versions": [
            {"policy_version": meta.policy_version, "version": meta.version}
            for meta in checkpoint_metas
        ],
        "checkpoint_versions": checkpoint_versions,
        "config": args.config,
        "coordinator": coordinator_summary,
        "learner": learner_summary,
        "ok": True,
        "task_ids": [task.task_id for task in tasks],
        "worker": worker_summary,
        "worker_ids": worker_ids,
    }


def _publish_smoke_checkpoints(checkpoint_dir: Path) -> list[CheckpointMeta]:
    registry = CheckpointRegistry(str(checkpoint_dir))
    latest = registry.latest()
    if latest is None:
        raise RuntimeError("phase8 smoke requires learner startup checkpoint")

    source_state = validate_checkpoint_payload(
        torch.load(
            registry.resolve_path(latest),
            map_location="cpu",
            weights_only=True,
        )
    )
    metas = [latest]
    while len(metas) < 2:
        policy_version = metas[-1].policy_version + 1
        meta = registry.publish(
            {
                "model_state_dict": source_state["model_state_dict"],
                "policy_version": policy_version,
                "update": policy_version,
                "metrics": {"phase8_smoke": 1.0},
            },
            policy_version=policy_version,
            step=policy_version,
        )
        metas.append(meta)
    return metas


def _write_heartbeat_jsonl(
    path: Path,
    *,
    worker_ids: list[str],
    checkpoint_metas: list[CheckpointMeta],
) -> None:
    if not checkpoint_metas:
        raise ValueError("checkpoint_metas must not be empty")

    current_meta = checkpoint_metas[-1]
    stale_meta = checkpoint_metas[-2] if len(checkpoint_metas) > 1 else current_meta
    records: list[dict[str, Any]] = []
    for index, worker_id in enumerate(worker_ids):
        current = index == 0
        meta = current_meta if current else stale_meta
        records.append(
            {
                "payload": {
                    "checkpoint_version": meta.version,
                    "learner_upload_accepted_batches": 0,
                    "learner_upload_failed_batches": 0,
                    "learner_upload_rejected_batches": 0,
                    "learner_upload_submitted_batches": 0,
                    "policy_version": meta.policy_version,
                    "rollout_duration_s": 4.0 if current else 0.0,
                    "rollout_steps": 128 if current else 0,
                    "sps": 32.0 if current else 0.0,
                    "status": "running" if current else "recovering",
                    "worker_crash_count": 0 if current else 1,
                },
                "worker_id": worker_id,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_eval_metrics(path: Path, task_ids: list[str]) -> None:
    metrics = {
        task_id: {"win_rate": 0.9 if index == 0 else 0.2}
        for index, task_id in enumerate(task_ids)
    }
    _write_json(path, {"metrics": metrics})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_requested_artifacts(summary: dict[str, Any], args: argparse.Namespace) -> None:
    dashboard_html = getattr(args, "dashboard_html", None)
    dashboard_json = getattr(args, "dashboard_json", None)
    if dashboard_html or dashboard_json:
        from hkrl.coordinator.dashboard import build_dashboard_model, render_dashboard_html

        dashboard = build_dashboard_model(summary)
        if dashboard_html:
            _write_text(Path(dashboard_html), render_dashboard_html(dashboard))
        if dashboard_json:
            _write_json(Path(dashboard_json), dashboard)

    profile_json = getattr(args, "profile_json", None)
    profile_md = getattr(args, "profile_md", None)
    if profile_json or profile_md:
        from hkrl.coordinator.profiling import (
            build_profile_report,
            render_profile_markdown,
            report_to_json,
        )

        profile = build_profile_report(summary)
        if profile_json:
            _write_text(Path(profile_json), report_to_json(profile))
        if profile_md:
            _write_text(Path(profile_md), render_profile_markdown(profile))


def _validate_smoke_args(args: argparse.Namespace) -> None:
    _non_empty_path_like(getattr(args, "config", None), name="config")
    tasks = getattr(args, "tasks", None)
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        raise ValueError("at least one task is required")
    for index, task in enumerate(tasks):
        _non_empty_path_like(task, name=f"tasks[{index}]")

    _positive_int(getattr(args, "num_workers", None), name="num_workers")
    _integer(getattr(args, "seed", None), name="seed")
    for name in (
        "work_dir",
        "output",
        "dashboard_html",
        "dashboard_json",
        "profile_json",
        "profile_md",
    ):
        value = getattr(args, name, None)
        if value is not None:
            _non_empty_path_like(value, name=name)


def _positive_int(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _non_empty_path_like(value: Any, *, name: str) -> None:
    if not isinstance(value, str | os.PathLike):
        raise ValueError(f"{name} must be a path string")
    if not str(value).strip():
        raise ValueError(f"{name} must not be empty")


def _work_dir(path: str | None) -> Path:
    if path is None:
        return Path(tempfile.mkdtemp(prefix="hkrl_phase8_smoke_")).resolve()
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _reset_generated_artifacts(work_dir: Path) -> None:
    for dirname in ("batches", "checkpoints"):
        path = work_dir / dirname
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    for filename in ("eval-metrics.json", "worker-heartbeats.jsonl"):
        path = work_dir / filename
        if path.exists():
            path.unlink()


@contextmanager
def _work_dir_lock(
    work_dir: Path,
    *,
    timeout_s: float = 30.0,
    poll_s: float = 0.05,
) -> Iterator[None]:
    lock_path = work_dir / ".phase8-smoke.lock"
    deadline = time.monotonic() + timeout_s
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for smoke work-dir lock: {lock_path}")
            time.sleep(poll_s)

    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_script_module(name: str) -> ModuleType:
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load script module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
