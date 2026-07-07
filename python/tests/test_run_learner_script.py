"""run_learner script tests."""

from __future__ import annotations

import argparse
import importlib.util
import math
import socket
import threading
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from hkrl.learner.batch_intake import BatchIntakeClient
from hkrl.spaces import action_mask_layout, make_observation_space
from hkrl.training.batch_io import save_rollout_batch
from hkrl.training.rollout_buffer import RolloutBatch


def test_run_learner_builds_server_summary(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    args = argparse.Namespace(
        config=str(Path(__file__).parents[2] / "configs/train/remote_learner.yaml"),
        bind="127.0.0.1:0",
        batch_dir=None,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["algorithm"] == "appo"
    assert summary["accepted_batches"] == 0
    assert summary["batch_dir"] is None
    assert summary["bind"] == "127.0.0.1:0"
    assert summary["checkpoint_dir"] == str(tmp_path.resolve())
    assert summary["enable_macro_actions"] is True
    assert summary["latest_checkpoint"] is None
    assert summary["max_entities"] == 4
    assert summary["max_staleness"] == 2
    assert summary["model"] == "entity_attention_gru"
    assert summary["n_macro_actions"] == 11
    assert summary["publish_every_updates"] == 1
    assert summary["policy_version"] == 0
    assert summary["queued_batches"] == 0
    assert summary["rejected_batches"] == 0
    assert summary["submitted_batches"] == 0
    assert summary["task_ids"] == []
    assert summary["tier"] == "privileged"


def test_run_learner_ingests_batch_dir_and_updates(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "appo_mlp.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "epochs: 1",
                "minibatch_size: 2",
                "learning_rate: 0.001",
                "entropy_coef: 0.0",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
            ]
        ),
        encoding="utf-8",
    )
    batch_dir = tmp_path / "batches"
    save_rollout_batch(batch_dir / "worker_00000001_v000000.npz", _learner_batch())
    args = argparse.Namespace(
        config=str(config),
        bind="127.0.0.1:0",
        batch_dir=str(batch_dir),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["accepted_batches"] == 1
    assert summary["batch_dir"] == str(batch_dir)
    assert summary["latest_checkpoint"] == 1
    assert summary["policy_version"] == 1
    assert summary["queued_batches"] == 0
    assert summary["rejected_batches"] == 0
    assert summary["submitted_batches"] == 1


def test_run_learner_ingests_recurrent_batch_dir_and_updates(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "appo_gru.yaml"
    _write_recurrent_appo_config(config)
    batch_dir = tmp_path / "batches"
    save_rollout_batch(
        batch_dir / "worker_00000001_v000000.npz",
        _recurrent_learner_batch(rnn_hidden=16),
    )
    args = argparse.Namespace(
        config=str(config),
        bind="127.0.0.1:0",
        batch_dir=str(batch_dir),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        intake_count=0,
        intake_timeout_s=1.0,
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=True,
        n_macro_actions=0,
        serve_forever=False,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["accepted_batches"] == 1
    assert summary["enable_macro_actions"] is False
    assert summary["latest_checkpoint"] == 1
    assert summary["model"] == "entity_attention_gru"
    assert summary["policy_version"] == 1
    assert summary["queued_batches"] == 0
    assert summary["rejected_batches"] == 0
    assert summary["submitted_batches"] == 1


def test_run_learner_network_intake_accepts_recurrent_batch_and_updates(
    tmp_path: Path,
) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "appo_gru.yaml"
    _write_recurrent_appo_config(config)
    port = _unused_localhost_port()
    args = argparse.Namespace(
        config=str(config),
        bind=f"127.0.0.1:{port}",
        batch_dir=None,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        intake_count=1,
        intake_timeout_s=3.0,
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=True,
        n_macro_actions=0,
        serve_forever=False,
        task=None,
        tasks=None,
        tier="privileged",
    )
    summaries: list[dict[str, object]] = []
    errors: list[BaseException] = []

    thread = threading.Thread(
        target=_run_learner_thread,
        args=(module, args, summaries, errors),
    )
    thread.start()

    accepted = _submit_batch_when_ready(
        f"127.0.0.1:{port}",
        _recurrent_learner_batch(rnn_hidden=16),
    )
    thread.join(timeout=6.0)

    assert not thread.is_alive()
    assert errors == []
    assert accepted is True
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["accepted_batches"] == 1
    assert summary["latest_checkpoint"] == 1
    assert summary["network_accepted_batches"] == 1
    assert summary["network_submitted_batches"] == 1
    assert summary["policy_version"] == 1
    assert summary["queued_batches"] == 0
    assert summary["rejected_batches"] == 0
    assert summary["submitted_batches"] == 0


def test_run_learner_rejects_serve_forever_with_intake_count(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "appo_mlp.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config),
        bind="127.0.0.1:0",
        batch_dir=None,
        intake_count=1,
        intake_timeout_s=1.0,
        serve_forever=True,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    with pytest.raises(ValueError, match="serve-forever"):
        module.run_from_args(args)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("config", "", "config"),
        ("task", "", "task"),
        ("tasks", [], "tasks"),
        ("tasks", "configs/tasks/gruz_mother.yaml", "tasks"),
        ("tasks", [""], r"tasks\[0\]"),
        ("bind", "", "bind"),
        ("batch_dir", "", "batch_dir"),
        ("checkpoint_dir", "", "checkpoint_dir"),
        ("intake_count", -1, "intake_count"),
        ("intake_count", False, "intake_count"),
        ("intake_timeout_s", 0.0, "intake_timeout_s"),
        ("intake_timeout_s", math.nan, "intake_timeout_s"),
        ("intake_timeout_s", "1.0", "intake_timeout_s"),
        ("max_staleness", -1, "max_staleness"),
        ("max_staleness", True, "max_staleness"),
        ("publish_every_updates", 0, "publish_every_updates"),
        ("publish_every_updates", False, "publish_every_updates"),
        ("max_entities", 0, "max_entities"),
        ("max_entities", True, "max_entities"),
        ("n_macro_actions", -1, "n_macro_actions"),
        ("n_macro_actions", False, "n_macro_actions"),
    ],
)
def test_run_learner_rejects_invalid_gate_args(
    field: str,
    value: object,
    match: str,
) -> None:
    module = _load_script("run_learner.py")
    args = _learner_args(**{field: value})

    with pytest.raises(ValueError, match=match):
        module._validate_learner_args(args)


def test_run_learner_gate_rejects_serve_forever_with_intake_count() -> None:
    module = _load_script("run_learner.py")
    args = _learner_args(serve_forever=True, intake_count=1)

    with pytest.raises(ValueError, match="serve-forever"):
        module._validate_learner_args(args)


def test_run_learner_serve_forever_updates_after_accepted_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("run_learner.py")
    accepted_values = [False, True]
    fake_server = _FakeServer()

    class FakeIntake:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeIntake:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def serve_once(self) -> _FakeResult:
            if not accepted_values:
                raise KeyboardInterrupt
            return _FakeResult(accepted=accepted_values.pop(0))

    monkeypatch.setattr(module, "BatchIntakeServer", FakeIntake)
    cfg = module.load_train_config(Path(__file__).parents[2] / "configs/train/ppo_mlp.yaml")

    submitted, accepted = module._serve_network_forever(
        fake_server,
        "127.0.0.1:0",
        cfg,
        timeout_s=1.0,
    )

    assert submitted == 2
    assert accepted == 1
    assert fake_server.serve_calls == 1


def test_run_learner_uses_nested_config_defaults(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "remote.yaml"
    checkpoint_dir = tmp_path / "configured-checkpoints"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "minibatch_size: 2",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
                "learner:",
                "  bind: 127.0.0.1:9999",
                "  max_staleness: 6",
                f"  checkpoint_dir: {checkpoint_dir}",
                "  publish_every_updates: 3",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config),
        bind=None,
        batch_dir=None,
        checkpoint_dir=None,
        max_staleness=None,
        publish_every_updates=None,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    summary = module.run_from_args(args)

    assert summary["bind"] == "127.0.0.1:9999"
    assert summary["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert summary["max_staleness"] == 6
    assert summary["publish_every_updates"] == 3


def test_run_learner_rejects_wildcard_bind_for_localhost_scope(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    config = tmp_path / "localhost.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "minibatch_size: 2",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
                "learner:",
                "  bind: 0.0.0.0:5600",
                "security:",
                "  bind_scope: localhost",
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=str(config),
        bind=None,
        batch_dir=None,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    with pytest.raises(ValueError, match="loopback"):
        module.run_from_args(args)


def test_run_learner_requires_token_for_non_loopback_intake(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    args = argparse.Namespace(
        config=str(Path(__file__).parents[2] / "configs/train/ppo_mlp.yaml"),
        bind="0.0.0.0:0",
        batch_dir=None,
        intake_count=1,
        intake_timeout_s=1.0,
        serve_forever=False,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=4,
        disable_macro_actions=False,
        n_macro_actions=11,
        task=None,
        tasks=None,
        tier="privileged",
    )

    with pytest.raises(ValueError, match="require_token"):
        module.run_from_args(args)


def test_run_learner_infers_layout_from_task_configs(tmp_path: Path) -> None:
    module = _load_script("run_learner.py")
    root = Path(__file__).parents[2]
    args = argparse.Namespace(
        config=str(root / "configs/train/remote_learner.yaml"),
        bind="127.0.0.1:0",
        batch_dir=None,
        checkpoint_dir=str(tmp_path),
        max_staleness=2,
        publish_every_updates=1,
        max_entities=None,
        disable_macro_actions=False,
        n_macro_actions=None,
        task=None,
        tasks=[
            str(root / "configs/tasks/gruz_mother.yaml"),
            str(root / "configs/tasks/hornet_protector.yaml"),
        ],
        tier=None,
    )

    summary = module.run_from_args(args)

    assert summary["enable_macro_actions"] is True
    assert summary["max_entities"] == 64
    assert summary["n_macro_actions"] == 11
    assert summary["task_ids"] == ["gruz_mother", "hornet_protector_attuned"]
    assert summary["tier"] == "privileged"


def test_run_learner_rejects_incompatible_task_layouts() -> None:
    module = _load_script("run_learner.py")
    tasks = [
        module.TaskConfig(task_id="a", wire_id=1, scene="A", action={"n_macro_actions": 11}),
        module.TaskConfig(task_id="b", wire_id=2, scene="B", action={"n_macro_actions": 4}),
    ]

    with pytest.raises(ValueError, match="n_macro_actions"):
        module._validate_task_layouts(tasks)


def test_run_learner_mlp_model_uses_default_hidden_when_rnn_hidden_zero() -> None:
    module = _load_script("run_learner.py")
    cfg = module.load_train_config(Path(__file__).parents[2] / "configs/train/ppo_mlp.yaml")
    model = module._build_model(
        cfg,
        {
            "global": (2,),
            "player": (3,),
            "entities": (4, 5),
            "entity_mask": (4,),
        },
        max_entities=4,
        enable_macro=True,
        n_macros=11,
    )

    assert model.trunk[0].out_features == 256


def test_run_learner_batch_dir_submit_rejects_empty_path() -> None:
    module = _load_script("run_learner.py")

    with pytest.raises(ValueError, match="batch_dir"):
        module._submit_batch_dir(_FakeServer(), "")


def _learner_args(**overrides: object) -> argparse.Namespace:
    root = Path(__file__).parents[2]
    values: dict[str, object] = {
        "batch_dir": None,
        "bind": None,
        "checkpoint_dir": None,
        "config": str(root / "configs/train/remote_learner.yaml"),
        "intake_count": 0,
        "intake_timeout_s": 10.0,
        "max_entities": 64,
        "max_staleness": 4,
        "n_macro_actions": 11,
        "publish_every_updates": 1,
        "serve_forever": False,
        "task": None,
        "tasks": None,
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


def _write_recurrent_appo_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "epochs: 1",
                "minibatch_size: 2",
                "learning_rate: 0.001",
                "entropy_coef: 0.0",
                "model:",
                "  name: entity_attention_gru",
                "  entity_hidden: 8",
                "  attention_layers: 1",
                "  attention_heads: 2",
                "  rnn_type: gru",
                "  rnn_hidden: 16",
            ]
        ),
        encoding="utf-8",
    )


def _run_learner_thread(
    module: ModuleType,
    args: argparse.Namespace,
    summaries: list[dict[str, object]],
    errors: list[BaseException],
) -> None:
    try:
        summaries.append(module.run_from_args(args))
    except BaseException as exc:
        errors.append(exc)


def _submit_batch_when_ready(endpoint: str, batch: RolloutBatch) -> bool:
    deadline = time.monotonic() + 3.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return BatchIntakeClient(endpoint, timeout_s=0.5).submit(batch)
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"learner intake did not accept connections at {endpoint}") from last_error


def _unused_localhost_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeResult:
    def __init__(self, *, accepted: bool) -> None:
        self.accepted = accepted


class _FakeServer:
    def __init__(self) -> None:
        self.serve_calls = 0

    def serve(self) -> None:
        self.serve_calls += 1


def _learner_batch() -> RolloutBatch:
    observation_space = make_observation_space(max_entities=4, tier="privileged")
    time_steps = 4
    action_dim = 13
    mask_dim = len(action_mask_layout(enable_macro=True))
    actions = np.zeros((time_steps, 1, action_dim), dtype=np.int64)
    actions[:, :, 0] = np.arange(time_steps, dtype=np.int64).reshape(time_steps, 1) % 3
    actions[:, :, 1] = 1
    actions[:, :, 11] = 1
    actions[:, :, 12] = 0

    return RolloutBatch(
        obs_global=np.zeros((time_steps, 1, *observation_space["global"].shape), dtype=np.float32),
        obs_player=np.zeros((time_steps, 1, *observation_space["player"].shape), dtype=np.float32),
        obs_entities=np.zeros(
            (time_steps, 1, *observation_space["entities"].shape),
            dtype=np.float32,
        ),
        entity_mask=np.ones((time_steps, 1, *observation_space["entity_mask"].shape), dtype=bool),
        actions=actions,
        log_probs=np.full((time_steps, 1), -1.0, dtype=np.float32),
        values=np.zeros((time_steps, 1), dtype=np.float32),
        advantages=np.ones((time_steps, 1), dtype=np.float32),
        returns=np.ones((time_steps, 1), dtype=np.float32),
        rewards=np.ones((time_steps, 1), dtype=np.float32),
        dones=np.array([[False], [False], [False], [True]]),
        truncateds=np.zeros((time_steps, 1), dtype=bool),
        action_masks=np.ones((time_steps, 1, mask_dim), dtype=bool),
        prev_actions=np.zeros((time_steps, 1, action_dim), dtype=np.int64),
        prev_rewards=np.zeros((time_steps, 1), dtype=np.float32),
        rnn_states=None,
        episode_ids=np.ones((time_steps, 1), dtype=np.uint64),
        task_ids=np.ones((time_steps, 1), dtype=np.int64),
        policy_version=0,
    )


def _recurrent_learner_batch(*, rnn_hidden: int) -> RolloutBatch:
    observation_space = make_observation_space(max_entities=4, tier="privileged")
    time_steps = 4
    action_dim = 12
    mask_dim = len(action_mask_layout(enable_macro=False))
    actions = np.zeros((time_steps, 1, action_dim), dtype=np.int64)
    actions[:, :, 0] = np.arange(time_steps, dtype=np.int64).reshape(time_steps, 1) % 3
    actions[:, :, 1] = 1
    actions[:, :, 11] = 0

    return RolloutBatch(
        obs_global=np.zeros((time_steps, 1, *observation_space["global"].shape), dtype=np.float32),
        obs_player=np.zeros((time_steps, 1, *observation_space["player"].shape), dtype=np.float32),
        obs_entities=np.zeros(
            (time_steps, 1, *observation_space["entities"].shape),
            dtype=np.float32,
        ),
        entity_mask=np.ones((time_steps, 1, *observation_space["entity_mask"].shape), dtype=bool),
        actions=actions,
        log_probs=np.full((time_steps, 1), -1.0, dtype=np.float32),
        values=np.zeros((time_steps, 1), dtype=np.float32),
        advantages=np.arange(1, time_steps + 1, dtype=np.float32).reshape(time_steps, 1),
        returns=np.arange(1, time_steps + 1, dtype=np.float32).reshape(time_steps, 1),
        rewards=np.ones((time_steps, 1), dtype=np.float32),
        dones=np.array([[False], [False], [False], [True]]),
        truncateds=np.zeros((time_steps, 1), dtype=bool),
        action_masks=np.ones((time_steps, 1, mask_dim), dtype=bool),
        prev_actions=np.zeros((time_steps, 1, action_dim), dtype=np.int64),
        prev_rewards=np.zeros((time_steps, 1), dtype=np.float32),
        rnn_states=np.zeros((time_steps, 1, 1, rnn_hidden), dtype=np.float32),
        episode_ids=np.ones((time_steps, 1), dtype=np.uint64),
        task_ids=np.ones((time_steps, 1), dtype=np.int64),
        policy_version=0,
    )
