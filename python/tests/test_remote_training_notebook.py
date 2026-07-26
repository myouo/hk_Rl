"""Structural checks for the operator-facing remote GPU training notebook."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks/remote_gpu_training.ipynb"
SETUP_NOTEBOOK = ROOT / "notebooks/one_click_clone_setup.ipynb"
KAGGLE_TRAINING_NOTEBOOK = ROOT / "notebooks/kaggle_training.ipynb"
BOOTSTRAP = ROOT / "scripts/remote/bootstrap_learner_env.sh"


def _load_notebook(path: Path = NOTEBOOK) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _execute_notebook(
    path: Path,
    *,
    parameter_marker: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    notebook = _load_notebook(path)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(
            compile(source, f"{path.name}:cell-{index}", "exec"),
            namespace,
        )
        if parameter_marker in source:
            namespace.update(overrides)
    return namespace


def test_remote_training_notebook_is_valid_and_safe_by_default() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "hkrl-learner"
    cells = notebook["cells"]
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    assert 'MODE = "inspect"' in code
    assert "START_SERVICES = False" in code
    assert "STOP_SERVICES = False" in code
    assert "configs/train/ssh_remote_learner.yaml" in code
    assert "start_learner_stack.sh" in code
    assert "start_ssh_tunnel.ps1" in code
    assert "f'-Remote" in code
    assert "-LocalLearnerPort" in code
    assert "-LocalRegistryPort" in code
    assert "start_game_worker.ps1" in code


def test_remote_training_notebook_code_cells_compile() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile(
            "".join(cell["source"]),
            f"remote_gpu_training.ipynb:cell-{index}",
            "exec",
        )


def test_remote_training_notebook_has_tutorial_sections() -> None:
    notebook = _load_notebook()
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert heading in markdown


def test_one_click_setup_notebook_is_safe_by_default() -> None:
    notebook = _load_notebook(SETUP_NOTEBOOK)
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert "RUN_SETUP = False" in code
    assert "UPDATE_EXISTING = False" in code
    assert 'ENV_BACKEND = "auto"' in code
    assert "CREATE_AUTH_TOKEN = None" in code
    assert "RUN_KAGGLE_SMOKE = True" in code
    assert 'REPO_URL = "https://github.com/myouo/hk_Rl.git"' in code
    assert 'Path("/kaggle/working")' in code
    assert '"current_python"' in code
    assert '"scripts/run_phase8_smoke.py"' in code
    assert '"scripts/run_learner.py"' in code
    assert "synthetic_batch_code" in code
    assert '"kaggle-smoke-summary.json"' in code
    assert "torch.cuda.is_available()" in code
    assert '"--single-branch"' in code
    assert '"--ff-only"' in code
    assert '"status", "--porcelain"' in code
    assert "shutil.rmtree" not in code
    assert "git reset" not in code


def test_one_click_setup_notebook_code_cells_compile() -> None:
    notebook = _load_notebook(SETUP_NOTEBOOK)
    all_code = []

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        all_code.append("".join(cell["source"]))
        compile(
            "".join(cell["source"]),
            f"one_click_clone_setup.ipynb:cell-{index}",
            "exec",
        )

    code = "\n".join(all_code)
    embedded = code.split('synthetic_batch_code = """', maxsplit=1)[1].split(
        '""".strip()', maxsplit=1
    )[0]
    compile(embedded, "one_click_clone_setup.ipynb:synthetic_batch_code", "exec")


def test_one_click_setup_notebook_has_tutorial_sections() -> None:
    notebook = _load_notebook(SETUP_NOTEBOOK)
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert heading in markdown


def test_one_click_setup_notebook_synthetic_batch_code_executes(
    tmp_path: Path,
) -> None:
    notebook = _load_notebook(SETUP_NOTEBOOK)
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    embedded = code.split('synthetic_batch_code = """', maxsplit=1)[1].split(
        '""".strip()', maxsplit=1
    )[0]
    output = tmp_path / "synthetic.npz"
    previous_argv = sys.argv
    try:
        sys.argv = [
            "synthetic_batch_code",
            str(output),
            str(ROOT / "configs/tasks/gruz_mother.yaml"),
        ]
        exec(
            compile(
                embedded,
                "one_click_clone_setup.ipynb:synthetic_batch_code",
                "exec",
            ),
            {"__name__": "__main__"},
        )
    finally:
        sys.argv = previous_argv

    from hkrl.training.batch_io import load_rollout_batch

    batch = load_rollout_batch(output)
    assert batch.rewards.shape == (4, 1)
    assert batch.policy_version == 0
    assert set(batch.task_ids.reshape(-1).tolist()) == {0}


def test_one_click_setup_notebook_kaggle_preview_executes(tmp_path: Path) -> None:
    target_dir = tmp_path / "hk_Rl"
    namespace = _execute_notebook(
        SETUP_NOTEBOOK,
        parameter_marker="RUN_SETUP = False",
        overrides={
            "CHECK_NETWORK": False,
            "IS_KAGGLE_OVERRIDE": True,
            "TARGET_DIR": str(target_dir),
        },
    )

    assert namespace["is_kaggle"] is True
    assert namespace["environment_backend"] == "current_python"
    assert namespace["setup_summary"] is None
    assert not target_dir.exists()


def test_kaggle_training_notebook_is_valid_and_safe_by_default() -> None:
    notebook = _load_notebook(KAGGLE_TRAINING_NOTEBOOK)
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert 'MODE = "inspect"' in code
    assert 'BATCH_SOURCE_DIR = ""' in code
    assert 'CHECKPOINT_SOURCE_DIR = ""' in code
    assert "ALLOW_CPU = False" in code
    assert '"scripts/run_learner.py"' in code
    assert '"scripts/run_worker.py"' in code
    assert '"--batch-dir"' in code
    assert '"--publish-every-updates"' in code
    assert "weights_only=True" in code
    assert 'mode == "train" and not batch_audit' in code
    assert 'mode == "train" and checkpoint_source_dir is None' in code
    assert "拒绝重复训练已处理 batch" in code
    assert "make_tree_user_writable(checkpoint_dir)" in code
    assert "shutil.rmtree" not in code
    assert ".unlink(" not in code
    assert "git reset" not in code


def test_kaggle_training_notebook_code_cells_compile() -> None:
    notebook = _load_notebook(KAGGLE_TRAINING_NOTEBOOK)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile(
            "".join(cell["source"]),
            f"kaggle_training.ipynb:cell-{index}",
            "exec",
        )


def test_kaggle_training_notebook_has_tutorial_sections() -> None:
    notebook = _load_notebook(KAGGLE_TRAINING_NOTEBOOK)
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    for heading in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"):
        assert heading in markdown


def test_kaggle_training_notebook_inspect_executes_without_writes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runs"
    namespace = _execute_notebook(
        KAGGLE_TRAINING_NOTEBOOK,
        parameter_marker='MODE = "inspect"',
        overrides={
            "REPO_DIR": str(ROOT),
            "OUTPUT_ROOT": str(output_root),
            "RUN_ID": "kaggle-train-inspect-test",
        },
    )

    assert namespace["mode"] == "inspect"
    assert namespace["training_executed"] is False
    assert namespace["training_summary_path"] is None
    assert not output_root.exists()


def test_kaggle_training_notebook_smoke_executes_top_to_bottom(
    tmp_path: Path,
) -> None:
    config = tmp_path / "appo_mlp.yaml"
    config.write_text(
        "\n".join(
            [
                "algorithm: appo",
                "epochs: 1",
                "minibatch_size: 2",
                "learning_rate: 0.001",
                "model:",
                "  name: mlp",
                "  rnn_hidden: 16",
                "learner:",
                "  device: cpu",
                "  publish_every_updates: 4",
                "security:",
                "  bind_scope: localhost",
                "  require_token: false",
            ]
        ),
        encoding="utf-8",
    )
    task = tmp_path / "synthetic_task.yaml"
    task.write_text(
        "\n".join(
            [
                "task_id: synthetic_smoke",
                "wire_id: 7",
                "scene: Synthetic",
                "observation:",
                "  max_entities: 4",
                "  tier: privileged",
                "action:",
                "  enable_macro_actions: true",
                "  n_macro_actions: 11",
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"
    namespace = _execute_notebook(
        KAGGLE_TRAINING_NOTEBOOK,
        parameter_marker='MODE = "inspect"',
        overrides={
            "MODE": "smoke",
            "REPO_DIR": str(ROOT),
            "TRAIN_CONFIG": str(config),
            "TASK_CONFIGS": [str(task)],
            "OUTPUT_ROOT": str(output_root),
            "RUN_ID": "kaggle-train-smoke-test",
            "ALLOW_CPU": True,
        },
    )

    summary = namespace["export_summary"]
    assert summary["ok"] is True
    assert summary["synthetic_input"] is True
    assert summary["device"] == "cpu"
    assert summary["learner"]["accepted_batches"] == 1
    assert summary["learner"]["rejected_batches"] == 0
    assert summary["learner"]["policy_version"] == 1
    assert summary["learner"]["latest_checkpoint"] == 2
    assert summary["parameter_update"]["changed_tensors"] > 0
    assert namespace["training_summary_path"].is_file()
    assert namespace["archive_path"].is_file()


def test_remote_learner_bootstrap_is_executable_and_gpu_guarded() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert BOOTSTRAP.stat().st_mode & stat.S_IXUSR
    assert "torch.cuda.is_available()" in source
    assert "--torch-index-url" in source
    assert "--allow-cpu" in source
    assert "HKRL_CONDA_BIN" in source
    assert "environment.yml is intentionally CPU-only" in source
