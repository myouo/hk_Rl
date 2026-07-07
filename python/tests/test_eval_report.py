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
        "malformed_task_count": 0.0,
        "mean_win_rate": 0.55,
        "min_win_rate": 0.4,
        "task_count": 2.0,
        "valid_task_count": 2.0,
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


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_regression_drop": float("nan")}, "max_regression_drop"),
        ({"max_regression_drop": -0.01}, "max_regression_drop"),
        ({"max_regression_drop": True}, "max_regression_drop"),
        ({"max_regression_drop": "0.05"}, "max_regression_drop"),
        ({"min_win_rate": float("nan")}, "min_win_rate"),
        ({"min_win_rate": 1.2}, "min_win_rate"),
        ({"min_win_rate": False}, "min_win_rate"),
        ({"min_win_rate": "0.5"}, "min_win_rate"),
    ],
)
def test_eval_report_rejects_invalid_threshold_args(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_eval_report(_eval_payload(), **kwargs)


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


def test_eval_report_flags_malformed_regression_delta() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "gruz_mother": {
                    "per_boss_win_rate": 0.8,
                },
            },
            "regression": {
                "gruz_mother": "not a number",
            },
        },
        max_regression_drop=0.05,
    )

    assert report["tasks"][0]["regression_delta"] is None
    assert report["tasks"][0]["regression_valid"] is False
    assert report["summary"]["worst_regression_delta"] == 0.0
    assert report["findings"] == [
        {
            "code": "malformed_regression_delta",
            "message": "gruz_mother has a non-finite or non-numeric regression delta.",
            "recommendation": (
                "Rebuild the regression baseline comparison before using this eval report."
            ),
            "severity": "critical",
        }
    ]


def test_eval_report_flags_string_numeric_regression_delta() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "gruz_mother": {
                    "per_boss_win_rate": 0.8,
                },
            },
            "regression": {
                "gruz_mother": "0.1",
            },
        },
    )

    assert report["tasks"][0]["regression_delta"] is None
    assert report["tasks"][0]["regression_valid"] is False
    assert [finding["code"] for finding in report["findings"]] == ["malformed_regression_delta"]


def test_eval_report_flags_malformed_task_metrics() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "gruz_mother": "not an object",
            }
        },
        min_win_rate=0.5,
    )

    assert report["summary"]["task_count"] == 1.0
    assert report["summary"]["valid_task_count"] == 0.0
    assert report["summary"]["malformed_task_count"] == 1.0
    assert report["summary"]["mean_win_rate"] == 0.0
    assert report["tasks"][0]["metrics_valid"] is False
    assert report["tasks"][0]["win_rate"] == 0.0
    assert report["findings"] == [
        {
            "code": "malformed_task_metrics",
            "message": "gruz_mother has a non-object metric payload.",
            "recommendation": "Re-run fixed-seed eval and check the evaluator JSON writer.",
            "severity": "critical",
        },
        {
            "code": "no_valid_eval_tasks",
            "message": "The evaluator report contains no valid task metric rows.",
            "recommendation": "Re-run fixed-seed eval and inspect malformed task metric payloads.",
            "severity": "critical",
        },
    ]


def test_eval_report_flags_invalid_numeric_task_metric_fields() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "gruz_mother": {
                    "damage_taken": "1.5",
                    "death_rate": True,
                    "invalid_action_ratio": 1.2,
                    "per_boss_win_rate": 0.8,
                    "time_to_kill": -1.0,
                },
            }
        },
    )

    assert report["tasks"][0]["metrics_valid"] is False
    assert report["tasks"][0]["metric_error"] == "invalid_metric_fields"
    assert report["tasks"][0]["invalid_metric_fields"] == [
        "damage_taken",
        "death_rate",
        "invalid_action_ratio",
        "time_to_kill",
    ]
    assert report["summary"]["valid_task_count"] == 0.0
    assert report["findings"] == [
        {
            "code": "malformed_task_metrics",
            "message": (
                "gruz_mother has invalid numeric metric fields: "
                "damage_taken, death_rate, invalid_action_ratio, time_to_kill."
            ),
            "recommendation": (
                "Re-run fixed-seed eval and confirm task metrics are finite numeric values."
            ),
            "severity": "critical",
        },
        {
            "code": "no_valid_eval_tasks",
            "message": "The evaluator report contains no valid task metric rows.",
            "recommendation": "Re-run fixed-seed eval and inspect malformed task metric payloads.",
            "severity": "critical",
        },
    ]


def test_eval_report_keeps_mixed_valid_tasks_reportable() -> None:
    report = build_eval_report(
        {
            "metrics": {
                "gruz_mother": "not an object",
                "hornet_protector_attuned": {
                    "per_boss_win_rate": 0.6,
                },
            }
        },
        min_win_rate=0.5,
    )

    assert report["summary"]["task_count"] == 2.0
    assert report["summary"]["valid_task_count"] == 1.0
    assert report["summary"]["malformed_task_count"] == 1.0
    assert report["summary"]["mean_win_rate"] == 0.6
    assert [finding["code"] for finding in report["findings"]] == [
        "malformed_task_metrics",
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


@pytest.mark.parametrize(
    "task_metrics",
    [
        {},
        {"win_rate": 1.2},
        {"win_rate": float("nan")},
        {"win_rate": "0.5"},
        {"per_boss_win_rate": True},
    ],
)
def test_eval_report_flags_missing_or_invalid_win_rate_metrics(
    task_metrics: dict[str, object],
) -> None:
    report = build_eval_report({"metrics": {"gruz_mother": task_metrics}})

    assert report["tasks"][0]["metrics_valid"] is False
    assert report["tasks"][0]["metric_error"] == "missing_or_invalid_win_rate"
    assert report["tasks"][0]["win_rate"] == 0.0
    assert report["summary"] == {
        "malformed_task_count": 1.0,
        "mean_win_rate": 0.0,
        "min_win_rate": 0.0,
        "task_count": 1.0,
        "valid_task_count": 0.0,
        "worst_regression_delta": 0.0,
    }
    assert report["findings"] == [
        {
            "code": "malformed_task_metrics",
            "message": "gruz_mother has missing or invalid win-rate metrics.",
            "recommendation": (
                "Re-run fixed-seed eval and confirm win_rate or per_boss_win_rate is in [0, 1]."
            ),
            "severity": "critical",
        },
        {
            "code": "no_valid_eval_tasks",
            "message": "The evaluator report contains no valid task metric rows.",
            "recommendation": "Re-run fixed-seed eval and inspect malformed task metric payloads.",
            "severity": "critical",
        },
    ]


def test_eval_report_markdown_contains_task_table() -> None:
    markdown = render_eval_report_markdown(build_eval_report(_eval_payload()))

    assert "# HKRL Eval Report" in markdown
    assert "| Task | Metrics Valid | Regression Valid | Win Rate | Regression Delta |" in markdown
    assert "| gruz_mother | yes | yes | 0.7 | -0.2 | 1.5 | 120 | 0.01 | 0.1 |" in markdown


def test_eval_report_ignores_string_numeric_metadata() -> None:
    payload = _eval_payload()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["episodes"] = "5"
    metadata["eval_workers"] = "2"

    report = build_eval_report(payload)

    assert report["metadata"]["episodes"] == 0.0
    assert report["metadata"]["eval_workers"] == 0.0


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


def test_render_eval_report_script_can_fail_on_critical_after_writing_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_script("render_eval_report.py")
    eval_json = tmp_path / "eval.json"
    report_json = tmp_path / "reports" / "eval-report.json"
    report_md = tmp_path / "reports" / "eval-report.md"
    eval_json.write_text(
        json.dumps({"metrics": {"gruz_mother": "not an object"}}),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--eval-json",
            str(eval_json),
            "--output-json",
            str(report_json),
            "--output-md",
            str(report_md),
            "--fail-on-critical",
        ]
    )

    assert exit_code == 1
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert any(finding["severity"] == "critical" for finding in report["findings"])
    assert "**critical**" in report_md.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"eval_json": ""}, "eval_json"),
        ({"output_json": ""}, "output_json"),
        ({"output_md": ""}, "output_md"),
    ],
)
def test_render_eval_report_script_rejects_invalid_path_args(
    tmp_path: Path,
    overrides: dict[str, object],
    match: str,
) -> None:
    module = _load_script("render_eval_report.py")
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps(_eval_payload()), encoding="utf-8")
    args = argparse.Namespace(
        eval_json=str(eval_json),
        max_regression_drop=0.05,
        min_win_rate=0.5,
        output_json=None,
        output_md=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)

    with pytest.raises(ValueError, match=match):
        module.run_from_args(args)


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
