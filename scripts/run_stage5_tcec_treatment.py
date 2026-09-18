#!/usr/bin/env python3
"""Run the Stage 5 experimental treatment benchmark using Topology-Coherent Episode Confirmation (TCEC).

Scientific comparison against the frozen Stage 2 detected mean/std control and Stage 4 BOCPD treatment:
- Dataset: RE2-OB
- Repetitions: 2 and 3 (60 cases, 30 scenario families, 2 executions per family)
- Repetition 1: completely excluded
- Independent variable: incident onset confirmation (BOCPD + TCEC vs raw BOCPD vs mean/std)
- Downstream RCA: IDENTICAL (S_comb, trace_elevation, fusion, simple_rca, random)
- Causal boundaries: IDENTICAL (analysis_end == confirmed_onset, no fallback to inject_time)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
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

import pyarrow.parquet as pq

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
from digital_detective.tcec import (
    TCECConfig,
    confirm_topology_coherent_episode,
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
    print("DIGITAL DETECTIVE STAGE 5 EXPERIMENTAL TREATMENT (BOCPD + TCEC)")
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
    ep_config_dict = asdict(ep_cfg)

    universe_policy = "canonical_v1"
    methods = ["s_comb", "trace_elevation", "fixed_equal_weight_fusion", "simple_rca", "random"]
    case_results_by_method: dict[str, list[MethodRankingResult]] = {m: [] for m in methods}
    per_case_records: list[dict[str, Any]] = []

    confirmed_outcomes: list[bool] = []
    unconfirmed_outcomes: list[bool] = []
    no_det_outcomes: list[bool] = []
    false_early_outcomes: list[bool] = []

    candidate_delays: list[int] = []
    confirmed_delays: list[int] = []
    confirmation_latencies: list[int] = []

    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    cases_file = dataset_root / "cases.parquet"
    cases_table = pq.read_table(cases_file)
    cases_metadata = {r["case"]: r for r in cases_table.to_pylist()}

    # Load deterministic BOCPD candidate onsets from Stage 4 artifact if available
    bocpd_cache_file = repo_root / "eval" / "results" / "stage4_re2ob_detected_bocpd_rep2_rep3_treatment_v1.json"
    cached_bocpd_onsets: dict[str, int] = {}
    if bocpd_cache_file.exists():
        with open(bocpd_cache_file, "r", encoding="utf-8") as bf:
            s4_data = json.load(bf)
            for rec in s4_data.get("per_case_results", []):
                cached_bocpd_onsets[rec["case_id"]] = rec["detected_onset"]
        print(f"Loaded {len(cached_bocpd_onsets)} cached BOCPD candidate onsets from Stage 4.")

    print("\nExecuting BOCPD + TCEC treatment over all 60 cases...", flush=True)

    for idx, c in enumerate(manifest.cases, start=1):
        cid = c.case_id
        meta = cases_metadata[cid]
        target = meta["root_cause_service"]
        repetition = c.repetition
        inject_time = int(meta["inject_time"])

        t0 = time.perf_counter()
        case = load_rcaeval_case(dataset_root, cid)
        load_time = time.perf_counter() - t0

        has_traces = hasattr(case, "traces") and case.traces is not None

        # 1. Causal BOCPD candidate detection
        t_bocpd_start = time.perf_counter()
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
            bocpd_res = detect_bocpd_onset(case, config=bocpd_config)
        t_bocpd = time.perf_counter() - t_bocpd_start

        # 2. Causal TCEC confirmation
        t_tcec_start = time.perf_counter()
        tcec_res = confirm_topology_coherent_episode(
            case,
            bocpd_res,
            config=tcec_config,
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
                source_description=f"tcec:{tcec_res.status}",
                has_detected_window=False,
            )

        # 4. Performance metrics
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

        # 5. Standard downstream anomaly detection truncated at confirmed_onset
        precomputed: dict[str, Any] = {}
        universe = resolve_candidate_universe(c.system, case=case)

        if is_confirmed and confirmed_onset is not None:
            full_det_res = detect_metric_anomalies(
                case,
                window_size=control_det_config["window_size"],
                threshold=control_det_config["threshold"],
                min_warmup=control_det_config["min_warmup"],
                min_valid_history=control_det_config["min_valid_history"],
                epsilon=control_det_config["epsilon"],
            )
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
            # Explicit no-detection / unconfirmed outcome: zero-observation truncation
            full_det_res = detect_metric_anomalies(
                case,
                window_size=control_det_config["window_size"],
                threshold=control_det_config["threshold"],
                min_warmup=control_det_config["min_warmup"],
                min_valid_history=control_det_config["min_valid_history"],
                epsilon=control_det_config["epsilon"],
            )
            truncated_det = truncate_metric_anomaly_result(full_det_res, max_timestamp=0)
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
                )

        # 6. Evaluate all standard RCA methods on candidate universe (IDENTICAL to baseline)
        case_methods: dict[str, Any] = {}
        for m_name in methods:
            t0 = time.perf_counter()
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
                raise ValueError(f"Unknown method: {m_name}")

            t_elapsed = time.perf_counter() - t0
            m_metrics = compute_case_metrics(ranking, target)
            res_obj = MethodRankingResult(
                case_id=cid,
                method_name=m_name,
                window_mode=window.mode,
                ranking=ranking,
                candidate_universe=universe,
                status="SUCCESS",
                metrics=m_metrics,
                runtime_sec=t_elapsed,
            )
            case_results_by_method[m_name].append(res_obj)
            case_methods[m_name] = {
                "top1": m_metrics.top1,
                "top3": m_metrics.top3,
                "top5": m_metrics.top5,
                "mrr": m_metrics.mrr,
                "predicted_root_cause": ranking[0].entity if ranking else None,
                "ranking": [r.entity for r in ranking],
            }

        cand_del_str = f"{cand_delay:+d}s" if cand_delay is not None else "N/A"
        conf_del_str = f"{conf_delay:+d}s" if conf_delay is not None else "N/A"
        fe_str = "FE" if is_false_early else "OK"
        conf_ent_str = str(tcec_res.confirming_entity) if tcec_res.confirming_entity else "-"
        print(f"[{idx:02d}/60] {cid:<32} {tcec_res.status:<11} cand={cand_del_str:<7} conf={conf_del_str:<7} {fe_str:<3} conf_by={conf_ent_str:<18} fusion_mrr={case_methods['fixed_equal_weight_fusion']['mrr']:.2f}", flush=True)

        per_case_records.append({
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "scenario_family_str": f"{c.scenario_family[0]}_{c.scenario_family[1]}_{c.scenario_family[2]}_{c.scenario_family[3]}",
            "repetition": repetition,
            "target": target,
            "inject_time": inject_time,
            "candidate_onset": candidate_onset,
            "confirmed_onset": confirmed_onset,
            "detection_status": tcec_res.status,
            "is_confirmed": is_confirmed,
            "candidate_delay_sec": cand_delay,
            "confirmed_delay_sec": conf_delay,
            "confirmation_latency_sec": latency,
            "is_false_early": is_false_early,
            "candidate_entity": tcec_res.candidate_entity,
            "confirming_entity": tcec_res.confirming_entity,
            "methods": case_methods,
        })

    # 7. Compute Aggregate Benchmark Metrics
    method_aggregates: dict[str, Any] = {}
    for m_name in methods:
        agg = aggregate_case_metrics(m_name, case_results_by_method[m_name])
        method_aggregates[m_name] = asdict(agg)

    tot = len(manifest.cases)
    n_confirmed = sum(confirmed_outcomes)
    n_unconfirmed = sum(unconfirmed_outcomes)
    n_no_det = sum(no_det_outcomes)
    n_fe = sum(false_early_outcomes)

    det_performance = {
        "total_cases": tot,
        "confirmed_count": n_confirmed,
        "unconfirmed_count": n_unconfirmed,
        "no_detection_count": n_no_det,
        "detection_rate": n_confirmed / tot,
        "unconfirmed_rate": n_unconfirmed / tot,
        "no_detection_rate": n_no_det / tot,
        "false_early_count": n_fe,
        "false_early_rate": (n_fe / n_confirmed) if n_confirmed > 0 else 0.0,
        "mean_candidate_delay_sec": statistics.mean(candidate_delays) if candidate_delays else None,
        "median_candidate_delay_sec": statistics.median(candidate_delays) if candidate_delays else None,
        "mean_confirmed_delay_sec": statistics.mean(confirmed_delays) if confirmed_delays else None,
        "median_confirmed_delay_sec": statistics.median(confirmed_delays) if confirmed_delays else None,
        "p90_confirmed_delay_sec": statistics.quantiles(confirmed_delays, n=10)[8] if len(confirmed_delays) >= 10 else None,
        "mean_confirmation_latency_sec": statistics.mean(confirmation_latencies) if confirmation_latencies else None,
        "median_confirmation_latency_sec": statistics.median(confirmation_latencies) if confirmation_latencies else None,
    }

    # 8. Assemble Deliverable JSON
    treatment_report = {
        "schema_version": "stage5_tcec_treatment_v1",
        "experiment_id": "stage5_re2ob_detected_tcec_treatment_v1",
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest_hash,
        "git_revision": git_rev,
        "execution_timestamp_utc": now_iso,
        "population": {
            "dataset": "RE2-OB",
            "total_cases": tot,
            "scenario_families": len(family_counts),
            "repetitions": [2, 3],
        },
        "experimental_treatment": {
            "detector": "bocpd_plus_tcec",
            "bocpd_config": asdict(bocpd_config),
            "tcec_config": asdict(tcec_config),
            "control_baseline_artifact": "eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json",
            "bocpd_treatment_artifact": "eval/results/stage4_re2ob_detected_bocpd_rep2_rep3_treatment_v1.json",
        },
        "detection_performance": det_performance,
        "method_aggregates": method_aggregates,
        "per_case_results": per_case_records,
    }

    output_path = repo_root / "eval" / "results" / "stage5_tcec_treatment_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(treatment_report, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 5 TCEC TREATMENT BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Cases:             {tot}")
    print(f"Confirmed Detections:    {n_confirmed}/{tot} ({det_performance['detection_rate']*100:.1f}%)")
    print(f"Unconfirmed:             {n_unconfirmed}/{tot} ({det_performance['unconfirmed_rate']*100:.1f}%)")
    print(f"No Detection:            {n_no_det}/{tot} ({det_performance['no_detection_rate']*100:.1f}%)")
    print(f"False Early:             {n_fe}/{n_confirmed} ({det_performance['false_early_rate']*100:.1f}%)")
    print(f"Median Candidate Delay:  {det_performance['median_candidate_delay_sec']} s")
    print(f"Median Confirmed Delay:  {det_performance['median_confirmed_delay_sec']} s")
    print(f"P90 Confirmed Delay:     {det_performance['p90_confirmed_delay_sec']} s")
    print(f"Median Conf Latency:     {det_performance['median_confirmation_latency_sec']} s")
    print("\nRoot Cause Analysis Performance:")
    print(f"{'Method':<28} {'Top@1':<8} {'Top@3':<8} {'Top@5':<8} {'MRR':<8}")
    print("-" * 60)
    for m in methods:
        agg = method_aggregates[m]
        print(f"{m:<28} {agg['top1_accuracy']:<8.4f} {agg['top3_accuracy']:<8.4f} {agg['top5_accuracy']:<8.4f} {agg['mrr']:<8.4f}")
    print("=" * 80)
    print(f"Saved deliverable artifact to: {output_path}")


if __name__ == "__main__":
    main()
