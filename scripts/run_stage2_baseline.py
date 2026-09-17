#!/usr/bin/env python3
"""Run and freeze the Stage 2 experimental baseline for Digital Detective.

Measurement-only script for RE2-OB repetitions 2 and 3 in detected window mode.
Enforces all Stage 2A causal boundaries without tuning or modifying any parameters.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import pyarrow.parquet as pq

from digital_detective.anomaly import (
    detect_metric_anomalies,
    truncate_metric_anomaly_result,
)
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence, TraceLatencyResult
from eval.baselines.random_ranker import rank_with_random
from eval.baselines.simple_rca import rank_with_simple_rca
from eval.manifest import BenchmarkManifest, ManifestCase
from eval.models import (
    aggregate_case_metrics,
    compute_case_metrics,
    MethodRankingResult,
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


def main() -> None:
    print("=" * 80)
    print("DIGITAL DETECTIVE STAGE 2 EXPERIMENTAL BASELINE (MEASUREMENT ONLY)")
    print("=" * 80)

    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    cases_file = dataset_root / "cases.parquet"
    if not cases_file.is_file():
        raise FileNotFoundError(f"cases.parquet not found in {dataset_root}")

    cases_table = pq.read_table(cases_file)
    case_rows = cases_table.to_pylist()

    # 1. Filter strictly to RE2-OB repetitions 2 and 3
    # Repetition 1 remains ONLY the frozen regression anchor and must NEVER enter the scientific baseline.
    re2_ob_rows = [
        r
        for r in case_rows
        if str(r.get("dataset", "")).upper() == "RE2-OB"
        and int(r.get("repetition", 1)) in (2, 3)
    ]

    manifest_cases: list[ManifestCase] = []
    for r in re2_ob_rows:
        cid = str(r.get("case") or r.get("case_id"))
        manifest_cases.append(
            ManifestCase(
                case_id=cid,
                dataset="RE2-OB",
                system=str(r.get("system")),
                fault=str(r.get("fault")),
                root_cause_service=str(r.get("root_cause_service")),
                repetition=int(r.get("repetition")),
                partition="all",
                suite="RE2",
            )
        )
    manifest_cases.sort(key=lambda c: c.case_id)

    manifest = BenchmarkManifest(
        manifest_id="stage2_re2ob_rep2_rep3_baseline_v1",
        manifest_type="SCIENTIFIC_BENCHMARK",
        description="Scientific baseline population for RE2-OB (repetitions 2 and 3, 60 cases)",
        created_at="2026-09-17T00:00:00Z",
        is_smoke_test=False,
        dataset="RE2-OB",
        cases=tuple(manifest_cases),
    )

    manifest_hash = manifest.compute_hash()
    git_rev = get_git_revision()
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"Manifest ID:         {manifest.manifest_id}")
    print(f"Manifest Hash:       {manifest_hash}")
    print(f"Git Revision:        {git_rev}")
    print(f"Execution Timestamp: {now_iso}")
    print(f"Total Executions:    {len(manifest.cases)}")

    # Verification: 60 executions expected, 30 scenario families expected, 2 executions per family
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
    print("Population verified: 60 executions across 30 scenario families (2 per family, reps 2 & 3).")

    # Fixed detector & episode configuration
    det_config = {
        "method": "rolling_mean_std_zscore",
        "window_size": 60,
        "threshold": 3.0,
        "min_warmup": 60,
        "min_valid_history": 30,
        "epsilon": 1e-6,
    }
    ep_cfg = EpisodeConfig(persistence=3, consensus=2)
    ep_config_dict = {"persistence": 3, "consensus": 2}
    universe_policy = "canonical_v1"
    window_mode = "detected"

    methods = ["s_comb", "trace_elevation", "fixed_equal_weight_fusion", "simple_rca", "random"]
    case_results_by_method: dict[str, list[MethodRankingResult]] = {m: [] for m in methods}
    per_case_records: list[dict[str, Any]] = []

    detection_outcomes: list[bool] = []
    false_early_outcomes: list[bool] = []
    detection_delays: list[int] = []
    onset_errors: list[int] = []

    for idx, c in enumerate(manifest.cases, start=1):
        cid = c.case_id
        case = load_rcaeval_case(dataset_root, cid)
        target = c.root_cause_service
        universe = resolve_candidate_universe(c.system, case=case)

        inject_time = int(
            getattr(case.ground_truth, "inject_time", None)
            or case.ground_truth.values.get("inject_time", 0)
        )

        has_metrics = hasattr(case, "metrics") and case.metrics is not None
        has_traces = hasattr(case, "traces") and case.traces is not None
        has_logs = hasattr(case, "logs") and case.logs is not None

        # 1. Unbounded metric anomaly detection to evaluate onset causally
        det_res = detect_metric_anomalies(
            case,
            window_size=det_config["window_size"],
            threshold=det_config["threshold"],
            min_warmup=det_config["min_warmup"],
            min_valid_history=det_config["min_valid_history"],
            epsilon=det_config["epsilon"],
        )

        # Trace dependencies for initial episode graph
        oracle_trace_deps = (
            extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
            if has_traces
            else ()
        )
        oracle_deps = [td.to_dependency() for td in oracle_trace_deps]
        oracle_graph = build_entity_graph(det_res.metric_names, dependencies=oracle_deps)

        initial_ep_evidence = aggregate_entity_episodes(det_res, oracle_graph, ep_cfg)

        # 2. Resolve detected incident window
        window = resolve_incident_window(
            case,
            ep_evidence=initial_ep_evidence,
            mode=window_mode,
            timestamps=det_res.timestamps,
        )

        is_detected = window.has_detected_window and window.onset_ts is not None
        detected_onset = window.onset_ts if is_detected else None

        # Collect detector audit metadata
        first_anom_ts = None
        anom_metrics_set = set()
        for mname, statuses in det_res.evaluation_statuses.items():
            for t_idx, st in enumerate(statuses):
                if st == "anomaly":
                    anom_metrics_set.add(mname)
                    ts = det_res.timestamps[t_idx]
                    if first_anom_ts is None or ts < first_anom_ts:
                        first_anom_ts = ts

        episode_entities = [
            ent for ent, ev in initial_ep_evidence.items() if getattr(ev, "has_episode", False)
        ]
        has_episodes = len(episode_entities) > 0

        # Detection performance metrics
        detection_outcomes.append(is_detected)
        delay: int | None = None
        is_false_early: bool | None = None

        if is_detected and detected_onset is not None:
            delay = detected_onset - inject_time
            is_false_early = detected_onset < inject_time
            false_early_outcomes.append(is_false_early)
            detection_delays.append(delay)
            onset_errors.append(delay)

        # 3. Apply Stage 2A causal boundary: truncate all evidence at detected_onset
        precomputed: dict[str, Any] = {}
        if is_detected and detected_onset is not None:
            # Metric causal truncation
            truncated_det = truncate_metric_anomaly_result(det_res, max_timestamp=detected_onset)
            precomputed["det_res"] = truncated_det

            # Trace causal dependencies bounded to detected_onset
            if has_traces:
                causal_trace_deps = extract_trace_dependencies(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    max_timestamp=detected_onset,
                )
                causal_deps = [td.to_dependency() for td in causal_trace_deps]
            else:
                causal_deps = ()

            causal_graph = build_entity_graph(truncated_det.metric_names, dependencies=causal_deps)
            precomputed["graph"] = causal_graph
            precomputed["ep_evidence"] = aggregate_entity_episodes(truncated_det, causal_graph, ep_cfg)

            if has_traces:
                precomputed["trace_lat"] = extract_trace_latency_evidence(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    expected_services=universe,
                    max_timestamp=detected_onset,
                )
        else:
            # Explicit no-detection outcome
            truncated_det = truncate_metric_anomaly_result(det_res, max_timestamp=0)
            precomputed["det_res"] = truncated_det
            precomputed["graph"] = build_entity_graph(truncated_det.metric_names, dependencies=())
            precomputed["ep_evidence"] = aggregate_entity_episodes(truncated_det, precomputed["graph"], ep_cfg)
            if has_traces:
                precomputed["trace_lat"] = TraceLatencyResult(
                    case_id=cid,
                    total_spans=0,
                    total_traces=0,
                    services=(),
                    edges=(),
                    sibling_overlap_count=0,
                    uninstrumented_services=tuple(sorted(universe)),
                    max_span_timestamp=None,
                    max_span_end_timestamp=None,
                    retained_spans=0,
                    excluded_spans=0,
                )

        # 4. Run downstream RCA methods
        case_methods_output: dict[str, Any] = {}

        for m in methods:
            t0 = time.perf_counter()
            if m == "s_comb":
                scores = rank_with_s_comb(
                    precomputed["det_res"],
                    precomputed["ep_evidence"],
                    graph=precomputed.get("graph"),
                    candidate_universe=universe,
                )
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(scores, start=1)
                )
            elif m == "trace_elevation":
                trace_lat = precomputed["trace_lat"]
                scores = rank_with_trace_elevation(trace_lat, candidate_universe=universe)
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(scores, start=1)
                )
            elif m == "fixed_equal_weight_fusion":
                s_scores = {
                    s.entity: s.score
                    for s in rank_with_s_comb(
                        precomputed["det_res"],
                        precomputed["ep_evidence"],
                        candidate_universe=universe,
                    )
                }
                max_s = max(s_scores.values()) if s_scores else 0.0
                norm_s = {k: (v / max_s if max_s > 0 else 0.0) for k, v in s_scores.items()}

                if has_traces and "trace_lat" in precomputed:
                    t_scores = {
                        s.entity: s.score
                        for s in rank_with_trace_elevation(
                            precomputed["trace_lat"], candidate_universe=universe
                        )
                    }
                    max_t = max(t_scores.values()) if t_scores else 0.0
                    norm_t = {k: (v / max_t if max_t > 0 else 0.0) for k, v in t_scores.items()}
                    fused = {k: (0.5 * norm_s[k] + 0.5 * norm_t[k]) for k in universe}
                else:
                    fused = norm_s

                sorted_ents = sorted(universe, key=lambda e: (-fused[e], e))
                ranking = tuple(
                    RankedEntity(entity=e, score=fused[e], rank=r_idx)
                    for r_idx, e in enumerate(sorted_ents, start=1)
                )
            elif m == "simple_rca":
                sim_res = rank_with_simple_rca(case, window, universe)
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(sim_res, start=1)
                )
            elif m == "random":
                rand_res = rank_with_random(universe, case_id=cid, seed_override=42 + idx)
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(rand_res, start=1)
                )
            else:
                raise ValueError(f"Unknown method: {m}")

            t_elapsed = time.perf_counter() - t0
            metrics = compute_case_metrics(ranking, target)

            res_obj = MethodRankingResult(
                case_id=cid,
                method_name=m,
                window_mode=window.mode,
                ranking=ranking,
                candidate_universe=universe,
                status="SUCCESS",
                metrics=metrics,
                runtime_sec=t_elapsed,
            )
            case_results_by_method[m].append(res_obj)

            top5_pred = [
                {"entity": r.entity, "score": round(r.score, 6), "rank": r.rank}
                for r in ranking[:5]
            ]
            case_methods_output[m] = {
                "top1": metrics.top1,
                "top3": metrics.top3,
                "top5": metrics.top5,
                "mrr": round(metrics.mrr, 6),
                "predicted_top5": top5_pred,
                "runtime_sec": round(t_elapsed, 6),
            }

        per_case_records.append({
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "scenario_family_str": f"{c.suite}:{c.system}:{c.root_cause_service}:{c.fault}",
            "repetition": c.repetition,
            "target": target,
            "inject_time": inject_time,
            "detected_onset": detected_onset,
            "detection_status": "DETECTED" if is_detected else "NO_DETECTION",
            "no_detection": not is_detected,
            "analysis_start": window.onset_ts,
            "analysis_end": window.end_ts,
            "detection_delay_sec": delay,
            "is_false_early": is_false_early,
            "detector_audit": {
                "first_anomaly_ts": first_anom_ts,
                "anomalous_metric_count": len(anom_metrics_set),
                "episode_existence": has_episodes,
                "episode_active_entity_count": len(episode_entities),
            },
            "methods": case_methods_output,
        })

    # 5. Compute Aggregate Metrics
    n_cases = len(manifest.cases)
    n_detected = sum(detection_outcomes)
    n_no_detection = n_cases - n_detected
    detection_rate = n_detected / n_cases
    no_detection_rate = n_no_detection / n_cases

    n_false_early = sum(false_early_outcomes)
    false_early_rate = n_false_early / n_cases if n_cases > 0 else 0.0

    mean_delay = sum(detection_delays) / len(detection_delays) if detection_delays else 0.0
    sorted_delays = sorted(detection_delays)
    median_delay = sorted_delays[len(sorted_delays) // 2] if sorted_delays else 0.0
    min_delay = min(detection_delays) if detection_delays else None
    max_delay = max(detection_delays) if detection_delays else None

    method_aggregates: dict[str, Any] = {}
    for m in methods:
        agg = aggregate_case_metrics(m, case_results_by_method[m], is_oracle=False)
        method_aggregates[m] = {
            "top1_accuracy": round(agg.top1_accuracy, 4),
            "top3_accuracy": round(agg.top3_accuracy, 4),
            "top5_accuracy": round(agg.top5_accuracy, 4),
            "mrr": round(agg.mrr, 4),
            "ac1": round(agg.ac1_accuracy, 4),
            "ac3": round(agg.ac3_accuracy, 4),
            "ac5": round(agg.ac5_accuracy, 4),
            "avg3": round(agg.avg3_accuracy, 4),
            "avg5": round(agg.avg5_accuracy, 4),
            "mean_runtime_sec": round(agg.mean_runtime_sec, 6),
        }

    # 6. Statistical Pairing Structure
    family_clusters: dict[str, list[dict[str, Any]]] = {}
    for rec in per_case_records:
        fam_key = rec["scenario_family_str"]
        family_clusters.setdefault(fam_key, []).append({
            "case_id": rec["case_id"],
            "repetition": rec["repetition"],
            "detected_onset": rec["detected_onset"],
            "s_comb_mrr": rec["methods"]["s_comb"]["mrr"],
            "trace_elevation_mrr": rec["methods"]["trace_elevation"]["mrr"],
            "fusion_mrr": rec["methods"]["fixed_equal_weight_fusion"]["mrr"],
        })

    pairing_summary = {
        "total_scenario_families": len(family_clusters),
        "executions_per_family": {k: len(v) for k, v in family_clusters.items()},
        "all_families_have_exactly_two_executions": all(len(v) == 2 for v in family_clusters.values()),
        "repetitions_clustered_within_family": all(
            {item["repetition"] for item in v} == {2, 3} for v in family_clusters.values()
        ),
    }

    # 7. Complete Artifact Document
    artifact = {
        "schema_version": "digital_detective_stage2_baseline_v1",
        "experiment_name": "stage2_re2ob_detected_meanstd_baseline_v1",
        "description": "Frozen Stage 2 control baseline for RE2-OB repetitions 2 & 3 in detected mode with causal boundaries",
        "run_metadata": {
            "git_revision": git_rev,
            "manifest_id": manifest.manifest_id,
            "manifest_hash": manifest_hash,
            "dataset": "RE2-OB",
            "dataset_artifact_version": "rcaeval_v1_re2_ob",
            "window_mode": window_mode,
            "detector_configuration": det_config,
            "episode_configuration": ep_config_dict,
            "candidate_universe_policy": universe_policy,
            "timestamp_iso": now_iso,
        },
        "population": {
            "total_executions": len(manifest.cases),
            "total_scenario_families": len(family_counts),
            "repetitions_represented": sorted(list({c.repetition for c in manifest.cases})),
            "repetition_counts": {
                2: sum(1 for c in manifest.cases if c.repetition == 2),
                3: sum(1 for c in manifest.cases if c.repetition == 3),
            },
            "repetition_1_excluded": all(c.repetition != 1 for c in manifest.cases),
            "case_ids": [c.case_id for c in manifest.cases],
            "scenario_families": sorted(list(family_clusters.keys())),
        },
        "detection_performance": {
            "total_cases": n_cases,
            "detected_count": n_detected,
            "no_detection_count": n_no_detection,
            "detection_rate": round(detection_rate, 4),
            "no_detection_rate": round(no_detection_rate, 4),
            "false_early_count": n_false_early,
            "false_early_detection_rate": round(false_early_rate, 4),
            "detection_delay_sec": {
                "mean": round(mean_delay, 2),
                "median": median_delay,
                "min": min_delay,
                "max": max_delay,
            },
            "onset_error_sec": {
                "mean": round(mean_delay, 2),
                "median": median_delay,
            },
        },
        "method_aggregates": method_aggregates,
        "statistical_pairing_structure": pairing_summary,
        "family_clusters": family_clusters,
        "per_case_results": per_case_records,
    }

    # Write out artifact
    results_dir = repo_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "stage2_re2ob_detected_meanstd_baseline_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 2 BASELINE EXECUTION SUMMARY")
    print(f"Artifact saved to: {out_path}")
    print("=" * 80)
    print(f"Total Cases:             {n_cases}")
    print(f"Detection Rate:          {detection_rate:.4f} ({n_detected}/{n_cases})")
    print(f"No-Detection Rate:       {no_detection_rate:.4f} ({n_no_detection}/{n_cases})")
    print(f"False-Early Rate:        {false_early_rate:.4f} ({n_false_early}/{n_cases})")
    print(f"Mean Detection Delay:    {mean_delay:.2f}s (median: {median_delay}s, min: {min_delay}s, max: {max_delay}s)")
    print("\nMethod Performance in Detected Mode (Causal Cutoff Active):")
    for m, agg in method_aggregates.items():
        print(f"  {m:26s} | Top@1: {agg['top1_accuracy']:.4f} | Top@3: {agg['top3_accuracy']:.4f} | Top@5: {agg['top5_accuracy']:.4f} | MRR: {agg['mrr']:.4f}")


if __name__ == "__main__":
    main()
