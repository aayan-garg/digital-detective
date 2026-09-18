#!/usr/bin/env python3
"""Run the Stage 6 experimental treatment benchmark comparing Digital Detective S_comb vs PyRCA HT.

Experiment definition:
- Population: RE2-OB Repetitions 2 and 3 (60 cases, 30 scenario families, 2 executions per family).
- Repetition 1: completely excluded from tuning and treatment selection.
- Incident windows & detection: Frozen causal BOCPD + TCEC confirmation (identical to Stage 5).
- Arm A: Existing Digital Detective S_comb
- Arm B: PyRCA Hypothesis Testing (HT) with aggregator="max", adjustment=False.
- Output artifact: eval/results/stage6_re2ob_pyrca_ht_treatment_v1.json
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import pandas as pd
import numpy as np
import pyrca

from digital_detective.anomaly import (
    detect_metric_anomalies,
    truncate_metric_anomaly_result,
)
from digital_detective.bocpd import (
    BOCPDConfig,
    detect_bocpd_onset,
)
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.pyrca_ht import (
    build_entity_telemetry_frames,
    convert_entity_graph_to_pyrca_adjacency,
    run_pyrca_ht,
)
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.tcec import (
    TCECConfig,
    confirm_topology_coherent_episode,
)
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from eval.manifest import BenchmarkManifest, ManifestCase
from eval.models import (
    aggregate_case_metrics,
    compute_case_metrics,
    IncidentWindow,
    RankedEntity,
)
from eval.universe import resolve_candidate_universe
from eval.windows import resolve_incident_window


def get_git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def compute_universe_hash(universe: tuple[str, ...]) -> str:
    return hashlib.sha256(",".join(sorted(universe)).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    print("=" * 80)
    print("STAGE 6 EXPERIMENTAL BENCHMARK: DIGITAL DETECTIVE S_COMB vs PyRCA HT")
    print("=" * 80)

    # 1. Load manifest and verify population
    manifest_path = repo_root / "eval" / "manifests" / "re2_ob_all_cases.json"
    manifest = BenchmarkManifest.load(manifest_path)
    manifest_hash = manifest.compute_hash()
    git_rev = get_git_revision()
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"Manifest ID:         {manifest.manifest_id}")
    print(f"Manifest Hash:       {manifest_hash}")
    print(f"Git Revision:        {git_rev}")
    print(f"Execution Timestamp: {now_iso}")
    print(f"Total Executions:    {len(manifest.cases)}")
    print(f"Python Version:      {sys.version.split()[0]}")
    print(f"PyRCA Version:       {getattr(pyrca, '__version__', '1.0.1')}")

    family_counts: dict[tuple[str, str, str, str], list[ManifestCase]] = {}
    for c in manifest.cases:
        family_counts.setdefault(c.scenario_family, []).append(c)

    assert len(manifest.cases) == 60, f"Expected 60 cases, got {len(manifest.cases)}"
    assert len(family_counts) == 30, f"Expected 30 families, got {len(family_counts)}"
    for fam, fcases in family_counts.items():
        assert len(fcases) == 2, f"Family {fam} has {len(fcases)} cases, expected 2"
        reps = {fc.repetition for fc in fcases}
        assert reps == {2, 3}, f"Family {fam} repetitions {reps} != {{2, 3}}"
    assert all(c.repetition != 1 for c in manifest.cases), "Repetition 1 leak detected!"
    print("Population verified: 60 executions across 30 scenario families (reps 2 & 3 strictly).")

    # 2. Experimental Configurations
    bocpd_config = BOCPDConfig(
        hazard_lambda=100.0,
        min_warmup=60,
        threshold=0.5,
        epsilon=1e-6,
    )
    tcec_config = TCECConfig(
        episode_config=EpisodeConfig(persistence=3, consensus=2)
    )
    control_det_config = {
        "normalization": "mean_std",
        "window_size": 60,
        "threshold": 3.0,
        "min_warmup": 60,
        "min_valid_history": 10,
        "epsilon": 1e-6,
    }
    ep_cfg = EpisodeConfig(persistence=3, consensus=2)

    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    methods = ["s_comb", "pyrca_ht"]

    per_case_results: list[dict[str, Any]] = []
    method_case_metrics: dict[str, list[Any]] = {m: [] for m in methods}

    # 3. Load frozen Stage 5 TCEC incident boundaries
    stage5_path = repo_root / "eval" / "results" / "stage5_tcec_treatment_v1.json"
    if not stage5_path.exists():
        raise FileNotFoundError(f"Frozen Stage 5 baseline artifact {stage5_path} not found")
    with open(stage5_path, "r", encoding="utf-8") as f:
        stage5_data = json.load(f)
    stage5_cases_by_id = {c["case_id"]: c for c in stage5_data["per_case_results"]}

    print("\nExecuting 60 benchmark cases using frozen Stage 5 confirmed incident windows...")
    t_start_all = time.perf_counter()

    for idx, c in enumerate(manifest.cases, start=1):
        t_case_start = time.perf_counter()
        cid = c.case_id
        target = c.root_cause_service
        fam_str = ":".join(c.scenario_family)

        case = load_rcaeval_case(dataset_root, cid)
        inject_time = int(case.ground_truth.values["inject_time"])
        universe = resolve_candidate_universe(c.system, case=case)
        universe_hash = compute_universe_hash(universe)
        has_traces = (case.traces is not None) and (case.traces.raw_data is not None)

        s5_case = stage5_cases_by_id[cid]
        confirmed_onset = s5_case["confirmed_onset"]
        candidate_onset = s5_case["candidate_onset"]
        is_confirmed = s5_case["is_confirmed"]
        detection_status = s5_case["detection_status"]
        cand_delay = s5_case["candidate_delay_sec"]
        conf_delay = s5_case["confirmed_delay_sec"]
        conf_latency = s5_case["confirmation_latency_sec"]
        is_false_early = s5_case["is_false_early"]
        candidate_entity = s5_case["candidate_entity"]
        confirming_entity = s5_case["confirming_entity"]

        # 4. Strictly causal data & graph preparation up to confirmed_onset
        case_methods: dict[str, Any] = {}
        metric_names = [col for col in case.metrics.provenance.original_field_names if col != "time"]

        if is_confirmed and confirmed_onset is not None:
            # Causal trace dependencies up to confirmed_onset
            if has_traces:
                causal_trace_deps = extract_trace_dependencies(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    max_timestamp=confirmed_onset,
                )
                causal_deps = [td.to_dependency() for td in causal_trace_deps]
            else:
                causal_deps = []

            causal_graph = build_entity_graph(metric_names, dependencies=causal_deps)
            adj, graph_hash = convert_entity_graph_to_pyrca_adjacency(causal_graph)

            # --- Arm A: S_comb ---
            full_det_res = detect_metric_anomalies(
                case,
                window_size=control_det_config["window_size"],
                threshold=control_det_config["threshold"],
                min_warmup=control_det_config["min_warmup"],
                min_valid_history=control_det_config["min_valid_history"],
                epsilon=control_det_config["epsilon"],
            )
            truncated_det = truncate_metric_anomaly_result(full_det_res, max_timestamp=confirmed_onset)
            ep_evidence = aggregate_entity_episodes(truncated_det, causal_graph, ep_cfg)

            s_scores = rank_with_s_comb(
                truncated_det,
                ep_evidence,
                graph=causal_graph,
                candidate_universe=universe,
            )
            s_ranking = tuple(
                RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                for r_idx, s in enumerate(s_scores, start=1)
            )

            # --- Arm B: PyRCA HT ---
            norm_df, inc_df = build_entity_telemetry_frames(
                case,
                causal_graph,
                incident_start_ts=confirmed_onset,
                incident_end_ts=confirmed_onset,
            )
            pyrca_ranking = run_pyrca_ht(
                norm_df,
                inc_df,
                adj,
                universe,
                aggregator="max",
                adjustment=False,
            )

            # Recording method results
            for m_name, ranking in [("s_comb", s_ranking), ("pyrca_ht", pyrca_ranking)]:
                cm = compute_case_metrics(ranking, target)
                method_case_metrics[m_name].append(cm)
                case_methods[m_name] = {
                    "top1": cm.top1,
                    "top3": cm.top3,
                    "top5": cm.top5,
                    "mrr": cm.mrr,
                    "ac1": cm.ac1,
                    "ac2": cm.ac2,
                    "ac3": cm.ac3,
                    "ac4": cm.ac4,
                    "ac5": cm.ac5,
                    "target_rank": cm.target_rank,
                    "predicted_root_cause": ranking[0].entity,
                    "ranking": [
                        {"candidate_id": r.entity, "score": r.score, "rank": r.rank}
                        for r in ranking
                    ],
                }

        else:
            # Unconfirmed / No Detection: fallback closed candidate universe sorted alphabetically
            sorted_fallback = sorted(universe)
            fallback_ranking = tuple(
                RankedEntity(entity=e, score=0.0, rank=r_idx)
                for r_idx, e in enumerate(sorted_fallback, start=1)
            )
            prelim_graph = build_entity_graph(metric_names, dependencies=[])
            adj, graph_hash = convert_entity_graph_to_pyrca_adjacency(prelim_graph)

            for m_name in methods:
                cm = compute_case_metrics(fallback_ranking, target)
                method_case_metrics[m_name].append(cm)
                case_methods[m_name] = {
                    "top1": cm.top1,
                    "top3": cm.top3,
                    "top5": cm.top5,
                    "mrr": cm.mrr,
                    "ac1": cm.ac1,
                    "ac2": cm.ac2,
                    "ac3": cm.ac3,
                    "ac4": cm.ac4,
                    "ac5": cm.ac5,
                    "target_rank": cm.target_rank,
                    "predicted_root_cause": fallback_ranking[0].entity,
                    "ranking": [
                        {"candidate_id": r.entity, "score": 0.0, "rank": r.rank}
                        for r in fallback_ranking
                    ],
                }

        t_elapsed = time.perf_counter() - t_case_start
        status_flag = "CONF" if is_confirmed else ("UNCONF" if detection_status == "unconfirmed" else "NODET")
        print(
            f"[{idx:02d}/60] {cid:34s} {status_flag:6s} | "
            f"S_comb rank={case_methods['s_comb']['target_rank']:2d} | "
            f"PyRCA HT rank={case_methods['pyrca_ht']['target_rank']:2d} | "
            f"({t_elapsed:.2f}s)",
            flush=True,
        )

        per_case_results.append({
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "scenario_family_str": fam_str,
            "repetition": c.repetition,
            "target": target,
            "inject_time": inject_time,
            "candidate_onset": candidate_onset,
            "confirmed_onset": confirmed_onset,
            "incident_window_start": confirmed_onset,
            "incident_window_end": confirmed_onset,
            "training_window_start": None,
            "training_window_end": confirmed_onset,
            "detection_status": detection_status,
            "is_confirmed": is_confirmed,
            "candidate_delay_sec": cand_delay,
            "confirmed_delay_sec": conf_delay,
            "confirmation_latency_sec": conf_latency,
            "is_false_early": is_false_early,
            "candidate_entity": candidate_entity,
            "confirming_entity": confirming_entity,
            "candidate_universe_hash": universe_hash,
            "graph_hash": graph_hash,
            "methods": case_methods,
        })

    # 5. Aggregate method performance
    method_aggregates: dict[str, dict[str, Any]] = {}
    for m_name in methods:
        cms = method_case_metrics[m_name]
        n_cases = len(cms)
        top1_acc = sum(1.0 for c in cms if c.top1) / n_cases
        top3_acc = sum(1.0 for c in cms if c.top3) / n_cases
        top5_acc = sum(1.0 for c in cms if c.top5) / n_cases
        mrr_val = sum(c.mrr for c in cms) / n_cases
        ac1_acc = sum(c.ac1 for c in cms) / n_cases
        ac2_acc = sum(c.ac2 for c in cms) / n_cases
        ac3_acc = sum(c.ac3 for c in cms) / n_cases
        ac4_acc = sum(c.ac4 for c in cms) / n_cases
        ac5_acc = sum(c.ac5 for c in cms) / n_cases
        avg5_acc = (ac1_acc + ac2_acc + ac3_acc + ac4_acc + ac5_acc) / 5.0
        method_aggregates[m_name] = {
            "top1_accuracy": round(top1_acc, 4),
            "top3_accuracy": round(top3_acc, 4),
            "top5_accuracy": round(top5_acc, 4),
            "mrr": round(mrr_val, 4),
            "ac1_accuracy": round(ac1_acc, 4),
            "ac2_accuracy": round(ac2_acc, 4),
            "ac3_accuracy": round(ac3_acc, 4),
            "ac4_accuracy": round(ac4_acc, 4),
            "ac5_accuracy": round(ac5_acc, 4),
            "avg5_accuracy": round(avg5_acc, 4),
        }

    total_time = time.perf_counter() - t_start_all
    print("\n" + "=" * 80)
    print("STAGE 6 BENCHMARK EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total Elapsed Time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print("\nMETHOD AGGREGATE SUMMARY (N=60, Reps 2 & 3):")
    print(f"{'Method':<20} | {'Top@1':<8} | {'Top@3':<8} | {'Top@5':<8} | {'MRR':<8} | {'Avg@5':<8}")
    print("-" * 70)
    for m_name in methods:
        agg = method_aggregates[m_name]
        print(
            f"{m_name:<20} | "
            f"{agg['top1_accuracy']*100:6.2f}% | "
            f"{agg['top3_accuracy']*100:6.2f}% | "
            f"{agg['top5_accuracy']*100:6.2f}% | "
            f"{agg['mrr']:6.4f} | "
            f"{agg['avg5_accuracy']*100:6.2f}%"
        )

    # 6. Save benchmark artifact
    output_path = repo_root / "eval" / "results" / "stage6_re2ob_pyrca_ht_treatment_v1.json"
    output_data = {
        "schema_version": "1.0.0",
        "experiment_id": "stage6_re2ob_pyrca_ht_treatment_v1",
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest_hash,
        "git_revision": git_rev,
        "execution_timestamp_utc": now_iso,
        "population": {
            "suite": "RE2-OB",
            "repetitions": [2, 3],
            "total_cases": len(manifest.cases),
            "scenario_families": len(family_counts),
        },
        "experimental_treatment": {
            "name": "pyrca_ht_vs_s_comb",
            "backend": "PyRCA HT (salesforce)",
            "pyrca_version": getattr(pyrca, "__version__", "1.0.1"),
            "python_version": sys.version.split()[0],
            "ht_config": {
                "aggregator": "max",
                "adjustment": False,
            },
            "detector": "causal_bocpd_plus_tcec",
        },
        "method_aggregates": method_aggregates,
        "per_case_results": per_case_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    artifact_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"\nTreatment artifact saved to: {output_path}")
    print(f"Artifact SHA-256 Hash:        {artifact_hash}")


if __name__ == "__main__":
    main()
