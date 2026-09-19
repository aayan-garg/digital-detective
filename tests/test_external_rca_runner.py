from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "external_rca"))

from runner import run_method


class Completed:
    returncode = 0
    stderr = ""
    stdout = json.dumps(
        {
            "ranks": ["checkoutservice_cpu", "currencyservice_cpu"],
            "ground_truth": "checkoutservice",
            "inject_time": 1705354562,
            "telemetry_window": {"mode": "rcaeval_oracle"},
        }
    )


def test_run_method_normalizes_rankings_and_metrics(tmp_path: Path) -> None:
    with patch("runner.subprocess.run", return_value=Completed()):
        result = run_method(
            case_id="re2ob_checkoutservice_cpu_2",
            method="BARO",
            dataset_root=tmp_path,
            python_executable=Path("python"),
        )

    assert result.success is True
    assert result.ranked_candidates == ("checkoutservice_cpu", "currencyservice_cpu")
    assert result.top_1 is True
    assert result.top_3 is True
    assert result.top_5 is True
    assert result.reciprocal_rank == 1.0
    assert result.inject_time == 1705354562


def test_run_method_surfaces_subprocess_failures(tmp_path: Path) -> None:
    class Failed:
        returncode = 1
        stderr = "missing isolated dependency"
        stdout = ""

    with patch("runner.subprocess.run", return_value=Failed()):
        result = run_method(
            case_id="re2ob_checkoutservice_cpu_2",
            method="RCD",
            dataset_root=tmp_path,
            python_executable=Path("python3.8"),
        )

    assert result.success is False
    assert result.ranked_candidates == ()
    assert result.error == "missing isolated dependency"


def test_run_method_rejects_empty_rankings(tmp_path: Path) -> None:
    class Empty:
        returncode = 0
        stderr = ""
        stdout = json.dumps(
            {
                "ranks": [],
                "ground_truth": "checkoutservice",
                "inject_time": 1705354562,
                "telemetry_window": {},
            }
        )

    with patch("runner.subprocess.run", return_value=Empty()):
        result = run_method(
            case_id="re2ob_checkoutservice_cpu_2",
            method="BARO",
            dataset_root=tmp_path,
            python_executable=Path("python"),
        )

    assert result.success is False
    assert "empty or malformed ranking" in (result.error or "")
