"""Bounded BARO/Multi-source RCD screening; never runs causal baselines."""

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
    "re2ob_checkoutservice_delay_2",
    "re2ob_currencyservice_delay_2",
)
DATASET = Path.home() / ".cache" / "rcaeval_validation"
PYTHON = ROOT / ".venv_rcaeval_default" / "Scripts" / "python.exe"


def main() -> int:
    results = []
    for method, timeout in (("BARO", 30), ("Multi-source RCD", 45)):
        for case_id in CASES:
            result = run_method(
                case_id=case_id,
                method=method,
                dataset_root=DATASET,
                python_executable=PYTHON,
                timeout_seconds=timeout,
            ).to_dict()
            result["status"] = "PASS" if result.pop("success") else "BLOCKED"
            results.append(result)

    report = {
        "schema_version": "1.0.0",
        "experiment": "rca_15min_screening",
        "screening_only": True,
        "rcaeval_version": "1.7.0",
        "rcaeval_commit": "bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90",
        "python": sys.version,
        "platform": platform.platform(),
        "cases": list(CASES),
        "methods": ["BARO", "Multi-source RCD", "Digital Detective frozen RCA"],
        "results": results,
        "deferred_methods": {
            "CausalRCA": "not included in 15-minute screening; reproducibility/runtime investigation deferred.",
            "MicroRank": "not included in 15-minute screening; reproducibility/runtime investigation deferred.",
            "RCD": "not included in 15-minute screening; reproducibility/runtime investigation deferred.",
        },
        "benchmark_condition": "RCAEval/oracle: benchmark injection time; 600 seconds normal and incident windows; no Stage 9.",
        "digital_detective_reference": "Existing frozen results reused; no pipeline rerun.",
        "decision": "SCREENING ONLY — no statistical significance claimed.",
    }
    output = ROOT / "eval" / "results" / "rca_15min_screening_v1.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
