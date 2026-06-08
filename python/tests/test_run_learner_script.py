"""run_learner script tests."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def test_run_learner_builds_server_summary(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    args = argparse.Namespace(
        config=str(Path(__file__).parents[2] / "configs/train/remote_learner.yaml"),
        bind="127.0.0.1:0",
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["algorithm"] == "appo"
    assert summary["bind"] == "127.0.0.1:0"
    assert summary["checkpoint_dir"] == str(tmp_path.resolve())
    assert summary["latest_checkpoint"] is None
    assert summary["model"] == "entity_attention_gru"
    assert summary["policy_version"] == 0
    assert summary["queued_batches"] == 0


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
