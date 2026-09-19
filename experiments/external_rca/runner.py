"""Experiment-only orchestration for published RCAEval baselines.

This module does not implement or alter any RCA algorithm. It invokes the
checked-out RCAEval source in a caller-selected environment and normalizes its
rank output for the smoke experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RCAEVAL_ROOT = Path(__file__).resolve().parent / "RCAEval"
INVOKER = Path(__file__).resolve().parent / "invoke_method.py"

METHODS = {
    "BARO": "baro",
    "RCD": "rcd",
    "CausalRCA": "causalrca",
    "MicroRank": "microrank",
    "Multi-source RCD": "mmrcd",
}
SMOKE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class BaselineResult:
    case_id: str
    method: str
    ranked_candidates: tuple[str, ...]
    top_1: bool
    top_3: bool
    top_5: bool
    reciprocal_rank: float
    ground_truth: str | None
    inject_time: int | None
    telemetry_window: dict[str, int | str | None]
    runtime_seconds: float | None
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ranked_candidates": list(self.ranked_candidates)}


def run_method(
    *,
    case_id: str,
    method: str,
    dataset_root: Path,
    python_executable: Path,
    rcaeval_root: Path = RCAEVAL_ROOT,
    timeout_seconds: int = SMOKE_TIMEOUT_SECONDS,
) -> BaselineResult:
    """Run one published method and return a normalized result.

    Missing environments, unsupported source layouts, and method failures are
    represented as explicit unsuccessful results; they are never replaced by a
    fabricated ranking.
    """
    if method not in METHODS:
        raise ValueError(f"Unsupported RCAEval method: {method}")

    started = time.perf_counter()
    command = [
        str(python_executable),
        str(INVOKER),
        "--rcaeval-root",
        str(rcaeval_root),
        "--dataset-root",
        str(dataset_root),
        "--case-id",
        case_id,
        "--method",
        METHODS[method],
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=rcaeval_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return _failure(case_id, method, started, str(error))

    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return _failure(case_id, method, started, detail or f"exit code {completed.returncode}")

    try:
        payload = json.loads(completed.stdout)
        ranked = tuple(str(item) for item in payload["ranks"])
        if not ranked or any(not candidate.strip() for candidate in ranked):
            raise ValueError("RCAEval returned an empty or malformed ranking")
        ground_truth = str(payload["ground_truth"])
        inject_time = int(payload["inject_time"])
        window = payload["telemetry_window"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return _failure(case_id, method, started, f"Malformed invoker output: {error}")

    rank = next(
        (
            index
            for index, candidate in enumerate(ranked, 1)
            if _candidate_matches_service(candidate, ground_truth)
        ),
        None,
    )
    return BaselineResult(
        case_id=case_id,
        method=method,
        ranked_candidates=ranked,
        top_1=rank == 1,
        top_3=rank is not None and rank <= 3,
        top_5=rank is not None and rank <= 5,
        reciprocal_rank=1.0 / rank if rank is not None else 0.0,
        ground_truth=ground_truth,
        inject_time=inject_time,
        telemetry_window=window,
        runtime_seconds=elapsed,
        success=True,
    )


def _failure(case_id: str, method: str, started: float, error: str) -> BaselineResult:
    return BaselineResult(
        case_id=case_id,
        method=method,
        ranked_candidates=(),
        top_1=False,
        top_3=False,
        top_5=False,
        reciprocal_rank=0.0,
        ground_truth=None,
        inject_time=None,
        telemetry_window={},
        runtime_seconds=time.perf_counter() - started,
        success=False,
        error=error,
    )


def _candidate_matches_service(candidate: str, ground_truth: str) -> bool:
    """Apply RCAEval's service-level match to raw metric/service identifiers."""
    return candidate == ground_truth or candidate.split("_", 1)[0] == ground_truth


if __name__ == "__main__":
    print("Import run_method from the experiment smoke runner.", file=sys.stderr)
    raise SystemExit(2)
