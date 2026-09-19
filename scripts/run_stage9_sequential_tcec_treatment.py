#!/usr/bin/env python3
"""Run the Stage 9 experimental treatment benchmark using Sequential TCEC.

Scientific comparison against the frozen Stage 5 TCEC control:
- Dataset: RE2-OB
- Repetitions: 2 and 3 (60 cases, 30 scenario families, 2 executions per family)
- Repetition 1: completely excluded
- Independent variable: incident onset confirmation (Sequential TCEC vs Boolean TCEC)
- Downstream RCA: IDENTICAL (S_comb, trace_elevation, fusion, simple_rca, random)
- Causal boundaries: IDENTICAL (analysis_end == confirmed_onset, no fallback to inject_time)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from digital_detective.anomaly import (
    detect_metric_anomalies,
    truncate_metric_anomaly_result,
)
from digital_detective.bocpd import (
    BOCPDConfig,
    BOCPDResult,
    detect_bocpd_onset,
)
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.sequential_tcec import (
    SequentialTCECConfig,
    SequentialTCECResult,
    confirm_sequential_tcec,
)
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence, TraceLatencyResult
from eval.baselines.random_ranker import rank_with_random
from eval.baselines.simple_rca import rank_with_simple_rca
from eval.manifest import BenchmarkManifest, ManifestCase
from eval.models import (
    aggregate_case_metrics,
    compute_case_metrics,
    IncidentWindow,
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
    print("DIGITAL DETECTIVE STAGE 9 EXPERIMENTAL TREATMENT: SEQUENTIAL TCEC")
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

    # 2. Configurations
    control_det_config = {
        "normalization": "mean_std",
        "window_size": 60,
        "threshold": 3.0,
        "min_warmup": 60,
        "min_valid_history": 10,
        "epsilon": 1e-6,
    }
    ep_cfg = EpisodeConfig(persistence=3, consensus=2)
    seq_tcec_config = SequentialTCECConfig(
        episode_config=ep_cfg,
        memory_window_seconds=60,
        stopping_threshold=3,
    )

    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))

    # Load frozen Stage 4 BOCPD candidate changepoints
    stage4_path = repo_root / "eval" / "results" / "stage4_re2ob_detected_bocpd_rep2_rep3_treatment_v1.json"
    cached_bocpd_onsets: dict[str, int] = {}
    if stage4_path.exists():
        with open(stage4_path, "r", encoding="utf-8") as f:
            stage4_data = json.load(f)
            for rec in stage4_data.get("per_case_results", []):
                if rec.get("detected_onset") is not None:
                    cached_bocpd_onsets[rec["case_id"]] = rec["detected_onset"]
        print(f"Loaded {len(cached_bocpd_onsets)} frozen BOCPD candidate onsets from Stage 4 baseline.")

    methods = [
        "s_comb",
        "trace_elevation",
        "fixed_equal_weight_fusion",
        "simple_rca",
        "random",
    ]
    method_case_metrics: dict[str, list[Any]] = {m: [] for m in methods}
    per_case_results: list[dict[str, Any]] = []

    confirmed_outcomes: list[bool] = []
    unconfirmed_outcomes: list[bool] = []
    no_det_outcomes: list[bool] = []
    candidate_delays: list[int] = []
    confirmed_delays: list[int] = []
    false_early_outcomes: list[bool] = []
    confirmation_latencies: list[int] = []

    print("\nExecuting 60 benchmark cases using Sequential TCEC...")
    t_start_all = time.perf_counter()

    for idx, c in enumerate(manifest.cases, start=1):
        t_case_start = time.perf_counter()
        cid = c.case_id
        target = c.root_cause_service
        fam_str = ":".join(c.scenario_family)

        case = load_rcaeval_case(dataset_root, cid)
        inject_time = int(
            getattr(case.ground_truth, "inject_time", None)
            or (case.ground_truth.values.get("inject_time") if hasattr(case.ground_truth, "values") else 0)
        )
        has_traces = hasattr(case, "traces") and case.traces is not None

        # 1. BOCPD candidate onset (reused exactly from Stage 4)
        if cid in cached_bocpd_onsets:
            bocpd_res = BOCPDResult(
                case_id=cid,
                onset_ts=cached_bocpd_onsets[cid],
                status="detected",
                audit={},
                changepoints=(),
                timestamps=(),
            )
        else:
            bocpd_res = detect_bocpd_onset(case, config=BOCPDConfig(min_warmup=60, threshold=0.5))

        # 2. Metric anomaly evaluation for this case (identical control parameters)
        full_det_res = detect_metric_anomalies(
            case,
            window_size=control_det_config["window_size"],
            threshold=control_det_config["threshold"],
            min_warmup=control_det_config["min_warmup"],
            min_valid_history=control_det_config["min_valid_history"],
            epsilon=control_det_config["epsilon"],
        )

        # 3. Sequential TCEC confirmation
        t_tcec_start = time.perf_counter()
        tcec_res = confirm_sequential_tcec(
            case,
            bocpd_res,
            config=seq_tcec_config,
            anomaly_result=full_det_res,
            service_aliases={"frontendservice": "frontend"},
        )
        t_tcec = time.perf_counter() - t_tcec_start

        is_confirmed = (tcec_res.status == "confirmed") and (tcec_res.confirmed_onset is not None)
        confirmed_onset = tcec_res.confirmed_onset if is_confirmed else None
        candidate_onset = tcec_res.candidate_onset

        # 3. Resolve detected incident window (using confirmed_onset as causal analysis_end)
        if is_confirmed and confirmed_onset is not None:
            window = resolve_incident_window(
                case,
                mode="detected",
                timestamps=tcec_res.timestamps,
                custom_onset_ts=confirmed_onset,
                custom_end_ts=confirmed_onset,
            )
        else:
            window = IncidentWindow(
                onset_ts=None,
                end_ts=None,
                mode="detected",
                source_description=f"sequential_tcec:{tcec_res.status}",
                has_detected_window=False,
            )

        confirmed_outcomes.append(is_confirmed)
        unconfirmed_outcomes.append(tcec_res.status == "unconfirmed")
        no_det_outcomes.append(tcec_res.status == "no_detection")

        cand_delay: int | None = None
        conf_delay: int | None = None
        is_false_early: bool | None = None
        latency: int | None = None

        if candidate_onset is not None:
            cand_delay = candidate_onset - inject_time
            candidate_delays.append(cand_delay)

        if is_confirmed and confirmed_onset is not None:
            conf_delay = confirmed_onset - inject_time
            is_false_early = confirmed_onset < inject_time
            false_early_outcomes.append(is_false_early)
            confirmed_delays.append(conf_delay)
            latency = confirmed_onset - candidate_onset if candidate_onset is not None else None
            if latency is not None:
                confirmation_latencies.append(latency)

        # 5. Standard downstream anomaly detection truncated strictly at confirmed_onset
        precomputed: dict[str, Any] = {}
        universe = resolve_candidate_universe(c.system, case=case)

        if is_confirmed and confirmed_onset is not None:
            truncated_det = truncate_metric_anomaly_result(full_det_res, max_timestamp=confirmed_onset)
            precomputed["det_res"] = truncated_det

            if has_traces:
                causal_trace_deps = extract_trace_dependencies(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    max_timestamp=confirmed_onset,
                )
                causal_deps = [td.to_dependency() for td in causal_trace_deps]
            else:
                causal_deps = []

            causal_graph = build_entity_graph(truncated_det.metric_names, dependencies=causal_deps)
            precomputed["graph"] = causal_graph
            precomputed["ep_evidence"] = aggregate_entity_episodes(truncated_det, causal_graph, ep_cfg)

            if has_traces:
                precomputed["trace_lat"] = extract_trace_latency_evidence(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    expected_services=universe,
                    max_timestamp=confirmed_onset,
                )
        else:
            min_ts = full_det_res.timestamps[0] if full_det_res.timestamps else 0
            truncated_det = truncate_metric_anomaly_result(full_det_res, max_timestamp=min_ts - 1)
            precomputed["det_res"] = truncated_det
            causal_graph = build_entity_graph(truncated_det.metric_names, dependencies=())
            precomputed["graph"] = causal_graph
            precomputed["ep_evidence"] = aggregate_entity_episodes(truncated_det, causal_graph, ep_cfg)
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
                )

        # 6. Evaluate all standard RCA methods on candidate universe (IDENTICAL to Stage 5 baseline)
        case_methods: dict[str, Any] = {}
        for m_name in methods:
            if m_name == "s_comb":
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
            elif m_name == "trace_elevation":
                trace_lat = precomputed["trace_lat"]
                scores = rank_with_trace_elevation(trace_lat, candidate_universe=universe)
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(scores, start=1)
                )
            elif m_name == "fixed_equal_weight_fusion":
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
            elif m_name == "simple_rca":
                sim_res = rank_with_simple_rca(case, window, universe)
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(sim_res, start=1)
                )
            elif m_name == "random":
                rand_res = rank_with_random(universe, case_id=cid, seed_override=42 + idx)
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(rand_res, start=1)
                )
            else:
                raise ValueError(f"Unknown method {m_name}")

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
                "ranking": [r.entity for r in ranking],
            }

        t_case_dur = time.perf_counter() - t_case_start

        case_entry = {
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "scenario_family_str": fam_str,
            "repetition": c.repetition,
            "target": target,
            "inject_time": inject_time,
            "candidate_onset": candidate_onset,
            "confirmed_onset": confirmed_onset,
            "confirmation_time": tcec_res.confirmation_time,
            "first_evidence_timestamp": tcec_res.first_evidence_timestamp,
            "detection_status": tcec_res.status,
            "is_confirmed": is_confirmed,
            "candidate_delay_sec": cand_delay,
            "confirmed_delay_sec": conf_delay,
            "confirmation_latency_sec": latency,
            "is_false_early": is_false_early,
            "candidate_entity": tcec_res.candidate_entity,
            "corroborating_entities": list(tcec_res.corroborating_entities),
            "corroborating_details": [
                {
                    "entity": d.entity,
                    "connected_to": d.connected_to,
                    "direction": d.direction,
                    "activation_timestamp": d.activation_timestamp,
                    "step_index": d.step_index,
                }
                for d in tcec_res.corroborating_details
            ],
            "state_transitions": list(tcec_res.state_transitions),
            "methods": case_methods,
            "runtime_sec": t_case_dur,
        }
        per_case_results.append(case_entry)

        if idx % 10 == 0 or idx == len(manifest.cases):
            early_str = f"early={is_false_early}" if is_confirmed else "unconfirmed"
            print(f"[{idx:02d}/60] Processed {cid} (target={target:20s} | status={tcec_res.status} | {early_str})")

    t_total = time.perf_counter() - t_start_all
    print(f"\nAll 60 executions completed in {t_total:.2f}s ({t_total/60:.2f}s/case).")

    # 6. Aggregates
    n_cases = len(manifest.cases)
    detection_rate = sum(1.0 for c in confirmed_outcomes if c) / n_cases
    false_early_rate = sum(1.0 for c in false_early_outcomes if c) / len(false_early_outcomes) if false_early_outcomes else 0.0

    mean_conf_delay = statistics.mean(confirmed_delays) if confirmed_delays else None
    median_conf_delay = statistics.median(confirmed_delays) if confirmed_delays else None
    mean_latency = statistics.mean(confirmation_latencies) if confirmation_latencies else None

    method_aggregates: dict[str, dict[str, Any]] = {}
    for m in methods:
        cms = method_case_metrics[m]
        method_aggregates[m] = {
            "top1_accuracy": round(sum(1.0 for item in cms if item.top1) / n_cases, 4),
            "top3_accuracy": round(sum(1.0 for item in cms if item.top3) / n_cases, 4),
            "top5_accuracy": round(sum(1.0 for item in cms if item.top5) / n_cases, 4),
            "mrr": round(sum(item.mrr for item in cms) / n_cases, 4),
            "ac1_accuracy": round(sum(item.ac1 for item in cms) / n_cases, 4),
            "ac2_accuracy": round(sum(item.ac2 for item in cms) / n_cases, 4),
            "ac3_accuracy": round(sum(item.ac3 for item in cms) / n_cases, 4),
            "ac4_accuracy": round(sum(item.ac4 for item in cms) / n_cases, 4),
            "ac5_accuracy": round(sum(item.ac5 for item in cms) / n_cases, 4),
        }

    output_data = {
        "schema_version": "stage9_sequential_tcec_treatment_v1",
        "experiment_id": "stage9_re2ob_sequential_tcec_treatment_v1",
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest_hash,
        "git_revision": git_rev,
        "execution_timestamp_utc": now_iso,
        "total_runtime_sec": t_total,
        "population": {
            "dataset": "RE2-OB",
            "total_cases": n_cases,
            "scenario_families": len(family_counts),
            "repetitions": [2, 3],
        },
        "treatment_config": {
            "persistence": seq_tcec_config.episode_config.persistence,
            "consensus": seq_tcec_config.episode_config.consensus,
            "stopping_threshold": seq_tcec_config.stopping_threshold,
            "memory_window_seconds": seq_tcec_config.memory_window_seconds,
        },
        "detection_performance": {
            "detection_rate": round(detection_rate, 4),
            "false_early_rate": round(false_early_rate, 4),
            "mean_confirmed_delay_sec": round(mean_conf_delay, 2) if mean_conf_delay is not None else None,
            "median_confirmed_delay_sec": round(median_conf_delay, 2) if median_conf_delay is not None else None,
            "mean_confirmation_latency_sec": round(mean_latency, 2) if mean_latency is not None else None,
        },
        "method_aggregates": method_aggregates,
        "per_case_results": per_case_results,
    }

    out_path = repo_root / "eval" / "results" / "stage9_re2ob_sequential_tcec_treatment_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    with open(out_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    print("\n" + "=" * 80)
    print("STAGE 9 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Detection Rate:            {detection_rate:.1%}")
    print(f"False-Early Rate:          {false_early_rate:.1%}")
    print(f"Mean Confirmed Delay:      {mean_conf_delay:.1f}s")
    print(f"S_comb MRR:                {method_aggregates['s_comb']['mrr']:.4f} | Top@1: {method_aggregates['s_comb']['top1_accuracy']:.4f}")
    print(f"Fixed Fusion MRR:          {method_aggregates['fixed_equal_weight_fusion']['mrr']:.4f} | Top@1: {method_aggregates['fixed_equal_weight_fusion']['top1_accuracy']:.4f}")
    print(f"Artifact Saved:            {out_path}")
    print(f"Artifact SHA-256:          {artifact_hash}")
    print("=" * 80)


if __name__ == "__main__":
    main()
