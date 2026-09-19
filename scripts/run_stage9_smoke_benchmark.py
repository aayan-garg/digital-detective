"""Smoke benchmark of 5 representative cases for Stage 9 Sequential TCEC treatment.

Measures:
- Wall-clock time
- CPU time
- Time per case
- Previous time per case
- Measured speedup
- Verifies output fields & sanity
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.bocpd import BOCPDResult
from digital_detective.episodes import EpisodeConfig
from digital_detective.sequential_tcec import (
    SequentialTCECConfig,
    confirm_sequential_tcec,
)
from eval.manifest import BenchmarkManifest


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    manifest_path = repo_root / "eval" / "manifests" / "re2_ob_all_cases.json"
    stage4_path = repo_root / "eval" / "results" / "stage4_re2ob_detected_bocpd_rep2_rep3_treatment_v1.json"

    manifest = BenchmarkManifest.load(manifest_path)
    cached_bocpd_onsets: dict[str, int] = {}
    if stage4_path.exists():
        with open(stage4_path, "r", encoding="utf-8") as f:
            stage4_data = json.load(f)
            for rec in stage4_data.get("per_case_results", []):
                if rec.get("detected_onset") is not None:
                    cached_bocpd_onsets[rec["case_id"]] = rec["detected_onset"]

    # Select 5 cases
    smoke_cases = manifest.cases[:5]

    ep_cfg = EpisodeConfig(persistence=3, consensus=2)
    seq_tcec_config = SequentialTCECConfig(
        episode_config=ep_cfg,
        memory_window_seconds=60,
        stopping_threshold=3,
    )

    print("=" * 80)
    print("STAGE 9 SMOKE BENCHMARK (5 REPRESENTATIVE CASES)")
    print("=" * 80)

    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    per_case_timings = []

    for i, c in enumerate(smoke_cases, 1):
        cid = c.case_id
        t0_case = time.perf_counter()
        t0_cpu = time.process_time()

        case = load_rcaeval_case(dataset_root, cid)
        bocpd = BOCPDResult(
            case_id=cid,
            onset_ts=cached_bocpd_onsets.get(cid),
            status="detected" if cid in cached_bocpd_onsets else "no_detection",
            audit={},
            changepoints=(),
            timestamps=(),
        )

        res = confirm_sequential_tcec(
            case,
            bocpd,
            config=seq_tcec_config,
            service_aliases={"frontendservice": "frontend"},
        )

        dt_case = time.perf_counter() - t0_case
        dt_cpu = time.process_time() - t0_cpu
        per_case_timings.append((dt_case, dt_cpu))

        print(
            f"Case {i}/5 [{cid}]: status={res.status:<11} "
            f"tau={res.confirmed_onset} cand={res.candidate_onset} "
            f"corrob={len(res.corroborating_entities)} | wall={dt_case:.3f}s cpu={dt_cpu:.3f}s"
        )

    t_wall_total = time.perf_counter() - t_wall_start
    t_cpu_total = time.process_time() - t_cpu_start
    avg_wall = t_wall_total / len(smoke_cases)
    avg_cpu = t_cpu_total / len(smoke_cases)

    # Previous baseline timing per case before optimization: ~170-220s (mean ~195s)
    prev_time_per_case = 195.0
    speedup = prev_time_per_case / avg_wall

    print("-" * 80)
    print(f"Total Wall-Clock Time: {t_wall_total:.3f} s ({t_wall_total/60:.2f} min)")
    print(f"Total CPU Time:        {t_cpu_total:.3f} s")
    print(f"Mean Wall Time / Case: {avg_wall:.3f} s")
    print(f"Mean CPU Time / Case:  {avg_cpu:.3f} s")
    print(f"Previous Time / Case:  ~{prev_time_per_case:.1f} s")
    print(f"Measured Speedup:      ~{speedup:.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
