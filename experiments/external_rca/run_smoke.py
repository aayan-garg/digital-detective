"""Run the bounded external-baseline smoke test; never runs the 60-case suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval.harness import run_benchmark
from eval.manifest import BenchmarkManifest
from runner import METHODS, run_method


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / ".cache" / "rcaeval_validation")
    parser.add_argument("--python-default", type=Path, default=Path(sys.executable))
    parser.add_argument("--python-rcd", type=Path, default=Path("python3.8"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "smoke_external_rca.json")
    args = parser.parse_args()

    smoke_cases = [
        "re2ob_checkoutservice_cpu_2",
        "re2ob_currencyservice_cpu_2",
        "re2ob_emailservice_cpu_2",
    ]
    results = []
    for case_id in smoke_cases:
        for method in METHODS:
            executable = args.python_rcd if method in {"RCD", "Multi-source RCD"} else args.python_default
            results.append(
                run_method(
                    case_id=case_id,
                    method=method,
                    dataset_root=args.dataset_root,
                    python_executable=executable,
                ).to_dict()
            )

    reference = _run_digital_detective_reference(args.dataset_root, smoke_cases)
    report = {
        "schema_version": "1.0.0",
        "experiment": "external_rca_smoke",
        "case_ids": smoke_cases,
        "methods": list(METHODS),
        "rcaeval_commit": "bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90",
        "runtime": {"platform": platform.platform(), "python": sys.version},
        "dataset_root": str(args.dataset_root),
        "results": results,
        "digital_detective_reference": reference,
        "full_run_requested": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = sum(item["success"] for item in results)
    print(f"External RCA smoke: {passed}/{len(results)} method-case executions passed")
    print(f"Report: {args.output}")
    return 0 if passed == len(results) else 1


def _run_digital_detective_reference(dataset_root: Path, case_ids: list[str]) -> dict[str, object]:
    """Run the existing frozen reference through its public evaluation harness."""
    manifest = BenchmarkManifest.load(ROOT / "eval" / "manifests" / "re2_ob_all_cases.json")
    selected = tuple(case for case in manifest.cases if case.case_id in case_ids)
    smoke_manifest = BenchmarkManifest(
        manifest_id="external-rca-smoke-reference",
        description="External RCA baseline smoke reference",
        created_at=manifest.created_at,
        is_smoke_test=True,
        cases=tuple(
            case.__class__(
                case_id=case.case_id,
                dataset=case.dataset,
                system=case.system,
                fault=case.fault,
                root_cause_service=case.root_cause_service,
                repetition=case.repetition,
                partition="smoke",
                suite=case.suite,
            )
            for case in selected
        ),
        manifest_type="SMOKE_REGRESSION",
        dataset=manifest.dataset,
    )
    try:
        report = run_benchmark(
            smoke_manifest,
            dataset_root,
            methods=("fixed_equal_weight_fusion",),
            window_mode="oracle",
            repository_revision="working-tree",
            experiment_config_id="frozen-reference",
        )
        return {
            "success": True,
            "results": report.to_dict()["case_results"],
        }
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    raise SystemExit(main())
