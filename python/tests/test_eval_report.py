"""Fixed-seed evaluator report tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from hkrl.eval.report import build_eval_report, render_eval_report_markdown


def test_eval_report_summarizes_metrics_and_regressions() -> None:
    report = build_eval_report(_eval_payload(), min_win_rate=0.5, max_regression_drop=0.05)

    assert report["source"] == "run_eval"
    assert report["metadata"]["policy"] == "model"
    assert report["metadata"]["eval_workers"] == 2.0
    assert report["summary"] == {
        "mean_win_rate": 0.55,
        "min_win_rate": 0.4,
        "task_count": 2.0,
        "worst_regression_delta": -0.2,
    }
    assert report["tasks"][0]["task_id"] == "gruz_mother"
    assert report["tasks"][0]["regression_delta"] == -0.2
    assert [finding["code"] for finding in report["findings"]] == [
        "low_win_rate",
        "win_rate_regression",
    ]


def test_eval_report_rejects_missing_metrics() -> None:
    with pytest.raises(ValueError, match="metrics object"):
        build_eval_report({"metadata": {}})


def test_eval_report_marks_empty_metrics_critical() -> None:
    report = build_eval_report({"metrics": {}})

    assert report["summary"]["task_count"] == 0.0
    assert report["findings"] == [
        {
            "code": "no_eval_tasks",
            "message": "The evaluator report contains no task metrics.",
            "recommendation": "Run fixed-seed eval before producing release evidence.",
            "severity": "critical",
        }
    ]


def test_eval_report_uses_per_boss_win_rate_fallback() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "gruz_mother": {
                    "death_rate": 0.2,
                    "per_boss_win_rate": 0.6,
                },
                "hornet_protector_attuned": {
                    "per_boss_win_rate": 0.8,
                },
            }
        },
        min_win_rate=0.5,
    )

    assert report["summary"]["mean_win_rate"] == 0.7
    assert report["summary"]["min_win_rate"] == 0.6
    assert [task["win_rate"] for task in report["tasks"]] == [0.6, 0.8]
    assert report["findings"] == []


def test_eval_report_uses_per_boss_win_rate_when_win_rate_is_invalid() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "explicit_zero": {
                    "per_boss_win_rate": 0.9,
                    "win_rate": 0.0,
                },
                "null_win_rate": {
                    "per_boss_win_rate": 0.6,
                    "win_rate": None,
                },
            }
        }
    )

    assert [task["task_id"] for task in report["tasks"]] == ["explicit_zero", "null_win_rate"]
    assert [task["win_rate"] for task in report["tasks"]] == [0.0, 0.6]
    assert report["summary"]["mean_win_rate"] == 0.3


def test_eval_report_markdown_contains_task_table() -> None:
    markdown = render_eval_report_markdown(build_eval_report(_eval_payload()))

    assert "# HKRL Eval Report" in markdown
    assert "| Task | Win Rate | Regression Delta |" in markdown
    assert "| gruz_mother | 0.7 | -0.2 | 1.5 | 120 | 0.01 | 0.1 |" in markdown


def test_render_eval_report_script_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_script("render_eval_report.py")
    eval_json = tmp_path / "eval.json"
    report_json = tmp_path / "reports" / "eval-report.json"
    report_md = tmp_path / "reports" / "eval-report.md"
    eval_json.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    args = argparse.Namespace(
        eval_json=str(eval_json),
        max_regression_drop=0.05,
        min_win_rate=0.5,
        output_json=str(report_json),
        output_md=str(report_md),
    )

    report = module.run_from_args(args)

    assert report["summary"]["task_count"] == 2.0
    assert json.loads(report_json.read_text(encoding="utf-8"))["source"] == "run_eval"
    assert "HKRL Eval Report" in report_md.read_text(encoding="utf-8")


def _eval_payload() -> dict[str, object]:
    return {
        "metadata": {
            "checkpoint_dir": "runs/checkpoints",
            "episodes": 5,
            "eval_workers": 2,
            "model": "entity_attention_gru",
            "policy": "model",
            "seeds": [0, 1, 2],
            "task_ids": ["gruz_mother", "hornet_protector_attuned"],
        },
        "metrics": {
            "gruz_mother": {
                "damage_dealt": 10.0,
                "damage_taken": 1.5,
                "death_rate": 0.1,
                "invalid_action_ratio": 0.01,
                "per_boss_damage_ratio": 0.15,
                "per_boss_win_rate": 0.7,
                "time_to_kill": 120.0,
                "win_rate": 0.7,
            },
            "hornet_protector_attuned": {
                "damage_dealt": 4.0,
                "damage_taken": 3.0,
                "death_rate": 0.4,
                "invalid_action_ratio": 0.02,
                "per_boss_damage_ratio": 0.75,
                "per_boss_win_rate": 0.4,
                "time_to_kill": 0.0,
                "win_rate": 0.4,
            },
        },
        "regression": {
            "gruz_mother": -0.2,
            "hornet_protector_attuned": 0.1,
        },
    }


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
