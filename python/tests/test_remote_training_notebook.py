"""Structural checks for the operator-facing remote GPU training notebook."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
NOTEBOOK = ROOT / "notebooks/remote_gpu_training.ipynb"
BOOTSTRAP = ROOT / "scripts/remote/bootstrap_learner_env.sh"


def _load_notebook() -> dict[str, Any]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


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


def test_remote_learner_bootstrap_is_executable_and_gpu_guarded() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")

    assert BOOTSTRAP.stat().st_mode & stat.S_IXUSR
    assert "torch.cuda.is_available()" in source
    assert "--torch-index-url" in source
    assert "--allow-cpu" in source
    assert "environment.yml is intentionally CPU-only" in source
