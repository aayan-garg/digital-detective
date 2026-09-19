"""Run the current, three-method RCAEval smoke test only."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

from runner import run_method

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "re2ob_checkoutservice_cpu_2",
    "re2ob_currencyservice_cpu_2",
    "re2ob_emailservice_cpu_2",
)
METHODS = ("BARO", "CausalRCA", "MicroRank")
COMMIT = "bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90"


def main() -> int:
    default_python = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else ROOT / ".venv_rcaeval_default" / "Scripts" / "python.exe"
    )
    dataset_root = Path.home() / ".cache" / "rcaeval_validation"
    results = []
    for case_id in CASES:
        for method in METHODS:
            result = run_method(
                case_id=case_id,
                method=method,
                dataset_root=dataset_root,
                python_executable=default_python,
            ).to_dict()
            result["status"] = "PASS" if result.pop("success") else "BLOCKED"
            if result["status"] == "BLOCKED":
                result["blocker_classification"] = _classify_blocker(result.get("error"))
            results.append(result)

    report = {
        "schema_version": "1.0.0",
        "experiment": "rcaeval_current_smoke",
        "rcaeval_version": "1.7.0",
        "rcaeval_commit": COMMIT,
        "python": sys.version,
        "platform": platform.platform(),
        "python_executable": str(default_python),
        "dataset_root": str(dataset_root),
        "benchmark_condition": {
            "inject_time": "case index and inject_time.txt",
            "window": "RCAEval standard 600 seconds before and after injection",
            "tdelta_seconds": 0,
            "causal_prefix": False,
        },
        "methods": list(METHODS),
        "case_ids": list(CASES),
        "results": results,
        "rd_status": "BLOCKED — Python 3.8 required",
        "multi_source_rcd_status": (
            "BLOCKED — Python 3.8 plus RCAEval pre-aggregated logts/tracets inputs required"
        ),
        "full_run_requested": False,
    }
    output = ROOT / "eval" / "results" / "rcaeval_smoke_current_v1.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    counts = {status: sum(item["status"] == status for item in results) for status in ("PASS", "BLOCKED")}
    print(f"Current RCAEval smoke: {counts['PASS']} PASS, {counts['BLOCKED']} BLOCKED")
    print(f"Artifact: {output}")
    return 0 if counts["BLOCKED"] == 0 else 1


def _classify_blocker(error: str | None) -> str:
    detail = error or ""
    if "timed out" in detail.lower():
        return "method_runtime"
    if "keyerror" in detail.lower() or "traceback" in detail.lower():
        return "method_code_or_input_layout"
    return "environment_or_input"


if __name__ == "__main__":
    raise SystemExit(main())
