"""Aggregate experiment config and remote/local launcher tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.utils.experiment import load_experiment

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "configs/experiments/godhome_smart.yaml"
EAGER_MANIFEST = ROOT / "configs/experiments/godhome_smart_eager.yaml"


def test_aggregate_experiment_resolves_one_training_contract() -> None:
    experiment = load_experiment(MANIFEST)

    assert experiment.config.name == "godhome-smart-v1"
    assert experiment.project_root == ROOT
    assert experiment.remote_train.algorithm == "appo"
    assert experiment.remote_train.learner.batches_per_update == 4
    assert experiment.remote_train.learner.compile_mode == "reduce-overhead"
    assert experiment.local_train.model == experiment.remote_train.model
    assert [task.task_id for task in experiment.tasks] == [
        "gruz_mother",
        "hornet_protector_attuned",
        "mantis_lords",
    ]


def test_remote_training_plan_is_loopback_and_sequence_aware() -> None:
    module = _load_script("run_remote_training.py")
    plan = module.build_plan(load_experiment(MANIFEST))

    assert plan["learner_bind"] == "127.0.0.1:5600"
    assert plan["registry_bind"] == "127.0.0.1:5601"
    assert plan["batches_per_update"] == 4
    assert plan["sequence_length"] == 32
    assert "--serve-forever" in plan["learner_command"]


def test_eager_remote_overlay_preserves_contract_without_compile() -> None:
    experiment = load_experiment(EAGER_MANIFEST)
    plan = _load_script("run_remote_training.py").build_plan(experiment)

    assert experiment.config.name == "godhome-smart-eager-v1"
    assert experiment.remote_train.learner.device == "cuda"
    assert plan["compile_mode"] == "off"
    assert plan["model"] == "entity_attention_gru"
    assert plan["learner_bind"] == "127.0.0.1:5600"


def test_local_inference_plan_keeps_action_loop_on_game_host() -> None:
    module = _load_script("run_local_inference.py")
    plan = module.build_plan(
        load_experiment(MANIFEST),
        worker_id="test-worker",
        steps=64,
    )

    assert plan["env_endpoint"] == "127.0.0.1:5555"
    assert plan["learner_endpoint"] == "127.0.0.1:5600"
    assert plan["inference_threads"] == 1
    assert plan["checkpoint_poll_interval_s"] == 2.0
    assert plan["time_scale"] == 1.0
    assert plan["steps"] == 64
    assert plan["worker_id"] == "test-worker"
    assert "--steps" in plan["command"]
    assert "--tasks" in plan["command"]
    assert "--checkpoint-poll-interval-s" in plan["command"]
    assert "--time-scale" in plan["command"]


def test_local_inference_plan_rejects_non_positive_steps() -> None:
    module = _load_script("run_local_inference.py")

    with pytest.raises(ValueError, match="steps"):
        module.build_plan(load_experiment(MANIFEST), steps=0)


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
