#!/usr/bin/env python3
"""Run the Stage 10 experimental treatment: Parameter-Free Temporal-Topological Propagation Consistency.

Scientific comparison against the frozen Stage-9 Sequential TCEC RCA baseline:
- Dataset: RE2-OB
- Repetitions: 2 and 3 (60 executions, 30 scenario families, 2 executions per family)
- Repetition 1: completely excluded
- Causal prefix: strictly X_<= tau_confirm (from frozen Stage 9 confirmation)
- Control: Frozen Stage 9 RCA (0.5 S_comb + 0.5 E_elev fusion)
- Treatment: Temporal-topological propagation coverage P(c) primary, frozen S_fusion tie-break
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from digital_detective.anomaly import (
    detect_metric_anomalies,
    truncate_metric_anomaly_result,
)
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.propagation_consistency import (
    compute_propagation_coverage,
    rank_with_propagation_consistency,
)
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence, TraceLatencyResult
from eval.manifest import BenchmarkManifest
from eval.models import (
    aggregate_case_metrics,
    compute_case_metrics,
    RankedEntity,
)
from eval.universe import resolve_candidate_universe


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


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    t_start_total = time.perf_counter()

    print("=" * 80)
    print("DIGITAL DETECTIVE STAGE 10 EXPERIMENT: PROPAGATION CONSISTENCY TREATMENT")
    print("=" * 80)

    # 1. Load manifest and verify population
    manifest_path = repo_root / "eval/manifests/re2_ob_all_cases.json"
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

    assert len(manifest.cases) == 60, f"Expected 60 cases, got {len(manifest.cases)}"
    assert all(c.repetition in (2, 3) for c in manifest.cases), "Repetition leak detected!"
    print("Population verified: 60 executions across 30 scenario families (reps 2 & 3 strictly).")

    # 2. Load Stage 9 baseline artifact for confirmed onsets and control invariance
    stage9_path = repo_root / "eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json"
    assert stage9_path.exists(), f"Stage 9 results not found at {stage9_path}"
    stage9_hash = compute_file_sha256(stage9_path)
    print(f"Stage 9 Hash:        {stage9_hash}")

    with open(stage9_path, "r", encoding="utf-8") as f:
        stage9_data = json.load(f)

    stage9_cases = {c["case_id"]: c for c in stage9_data["per_case_results"]}
    assert len(stage9_cases) == 60, f"Expected 60 Stage 9 cases, got {len(stage9_cases)}"

    # 3. Configurations
    det_config = {
        "normalization": "mean_std",
        "window_size": 60,
        "threshold": 3.0,
        "min_warmup": 60,
        "min_valid_history": 10,
        "epsilon": 1e-6,
    }
    ep_cfg = EpisodeConfig(persistence=3, consensus=2)
    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))

    ctrl_metrics_list = []
    treat_metrics_list = []
    per_case_results: list[dict[str, Any]] = []

    print("\nExecuting 60-case Stage-10 benchmark...")

    for idx, mc in enumerate(manifest.cases, start=1):
        cid = mc.case_id
        fam = list(mc.scenario_family)
        fam_str = "::".join(fam)
        rep = mc.repetition

        s9_c = stage9_cases[cid]
        target = s9_c["target"]
        confirmed_onset = s9_c["confirmed_onset"]
        is_confirmed = s9_c["is_confirmed"]
        s9_ctrl_ranking = s9_c["methods"]["fixed_equal_weight_fusion"]["ranking"]

        t_case_start = time.perf_counter()

        # Load raw case
        case = load_rcaeval_case(dataset_root, cid)
        universe = resolve_candidate_universe("ob", case=case)

        # Truncated detector strictly at confirmed_onset
        full_det = detect_metric_anomalies(case, **det_config)
        has_traces = case.traces is not None and case.traces.raw_data is not None

        if is_confirmed and confirmed_onset is not None:
            trunc_det = truncate_metric_anomaly_result(full_det, max_timestamp=confirmed_onset)
            if has_traces:
                causal_trace_deps = extract_trace_dependencies(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    max_timestamp=confirmed_onset,
                )
                causal_deps = [td.to_dependency() for td in causal_trace_deps]
            else:
                causal_deps = []

            causal_graph = build_entity_graph(trunc_det.metric_names, dependencies=causal_deps)
            ep_ev = aggregate_entity_episodes(trunc_det, causal_graph, ep_cfg)

            if has_traces:
                trace_lat = extract_trace_latency_evidence(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    expected_services=universe,
                    max_timestamp=confirmed_onset,
                )
            else:
                trace_lat = None
        else:
            min_ts = full_det.timestamps[0] if full_det.timestamps else 0
            trunc_det = truncate_metric_anomaly_result(full_det, max_timestamp=min_ts - 1)
            causal_graph = build_entity_graph(trunc_det.metric_names, dependencies=())
            ep_ev = aggregate_entity_episodes(trunc_det, causal_graph, ep_cfg)
            trace_lat = None

        # Recompute frozen Stage-9 control (fixed_equal_weight_fusion)
        s_scores = {
            s.entity: s.score
            for s in rank_with_s_comb(trunc_det, ep_ev, candidate_universe=universe)
        }
        max_s = max(s_scores.values()) if s_scores else 0.0
        norm_s = {k: (v / max_s if max_s > 0 else 0.0) for k, v in s_scores.items()}

        if trace_lat is not None and trace_lat.total_spans > 0:
            t_scores = {
                s.entity: s.score
                for s in rank_with_trace_elevation(trace_lat, candidate_universe=universe)
            }
            max_t = max(t_scores.values()) if t_scores else 0.0
            norm_t = {k: (v / max_t if max_t > 0 else 0.0) for k, v in t_scores.items()}
            fused = {k: (0.5 * norm_s[k] + 0.5 * norm_t[k]) for k in universe}
        else:
            fused = norm_s

        ctrl_sorted = sorted(universe, key=lambda e: (-fused[e], e))
        ctrl_ranking = tuple(
            RankedEntity(entity=e, score=fused[e], rank=r_idx)
            for r_idx, e in enumerate(ctrl_sorted, start=1)
        )

        # Control invariance check
        recomputed_ctrl_ents = [r.entity for r in ctrl_ranking]
        assert recomputed_ctrl_ents == s9_ctrl_ranking, (
            f"Control mismatch for {cid}:\n"
            f"  Recomputed: {recomputed_ctrl_ents}\n"
            f"  Stage 9:    {s9_ctrl_ranking}"
        )

        # Compute Stage-10 Treatment
        cov_res = compute_propagation_coverage(ep_ev, causal_graph, universe)
        treat_ranking = rank_with_propagation_consistency(ctrl_ranking, cov_res.coverage)
        treat_ents = [r.entity for r in treat_ranking]

        cm_ctrl = compute_case_metrics(ctrl_ranking, target)
        cm_treat = compute_case_metrics(treat_ranking, target)
        ctrl_metrics_list.append(cm_ctrl)
        treat_metrics_list.append(cm_treat)

        # Ranking change analysis
        ctrl_ranks = {r.entity: r.rank for r in ctrl_ranking}
        treat_ranks = {r.entity: r.rank for r in treat_ranking}

        winner_changed = ctrl_ranking[0].entity != treat_ranking[0].entity
        baseline_root_rank = cm_ctrl.target_rank
        treatment_root_rank = cm_treat.target_rank
        true_root_promoted = (
            treatment_root_rank is not None
            and baseline_root_rank is not None
            and treatment_root_rank < baseline_root_rank
        )

        # Downstream symptoms: entities reachable from target in temporal propagation graph
        downstream_entities = set(cov_res.reachable_entities.get(target, ()))
        downstream_symptom_demoted = any(
            treat_ranks[e] > ctrl_ranks[e] for e in downstream_entities if e in treat_ranks
        )

        # Unrelated candidate: not target, not downstream of target, and target not downstream of it
        upstream_of_target = {
            e for e, reach in cov_res.reachable_entities.items() if target in reach
        }
        incident_related = {target} | downstream_entities | upstream_of_target
        unrelated_entities = set(universe) - incident_related
        unrelated_candidate_promoted = any(
            treat_ranks[e] < ctrl_ranks[e] for e in unrelated_entities if e in treat_ranks
        )

        t_case_dur = time.perf_counter() - t_case_start

        case_record = {
            "case_id": cid,
            "repetition": rep,
            "scenario_family": fam,
            "scenario_family_str": fam_str,
            "target": target,
            "confirmation_status": s9_c["detection_status"],
            "confirmed_onset": confirmed_onset,
            "baseline_ranking": recomputed_ctrl_ents,
            "treatment_ranking": treat_ents,
            "P_c": {c: cov_res.coverage.get(c, 0) for c in universe},
            "S_fusion_c": {c: fused[c] for c in universe},
            "winner_changed": winner_changed,
            "baseline_root_rank": baseline_root_rank,
            "treatment_root_rank": treatment_root_rank,
            "true_root_promoted": true_root_promoted,
            "downstream_symptom_demoted": downstream_symptom_demoted,
            "unrelated_candidate_promoted": unrelated_candidate_promoted,
            "count_anomalous_entities": len(cov_res.anomalous_entities),
            "count_temporal_propagation_edges": len(cov_res.temporal_edges),
            "causal_topology_timestamp_cutoff": confirmed_onset,
            "temporal_edges": list(cov_res.temporal_edges),
            "reachable_entities": {k: list(v) for k, v in cov_res.reachable_entities.items() if v},
            "runtime_sec": round(t_case_dur, 4),
            "metrics": {
                "control": cm_ctrl.to_dict(),
                "treatment": cm_treat.to_dict(),
            },
        }
        per_case_results.append(case_record)

        if idx % 10 == 0 or idx == 60:
            print(
                f"  [{idx:2d}/60] {cid:<36s} | Root: {target:<20s} | "
                f"Ctrl Rank: {str(baseline_root_rank):>2s} | Treat Rank: {str(treatment_root_rank):>2s} | "
                f"Edges: {len(cov_res.temporal_edges):2d}"
            )

    t_total = time.perf_counter() - t_start_total

    def summarize_metrics(metrics_list: list[Any]) -> dict[str, float]:
        n = len(metrics_list)
        return {
            "top1": sum(1.0 for m in metrics_list if m.top1) / n,
            "top3": sum(1.0 for m in metrics_list if m.top3) / n,
            "top5": sum(1.0 for m in metrics_list if m.top5) / n,
            "mrr": sum(m.mrr for m in metrics_list) / n,
            "ac1": sum(m.ac1 for m in metrics_list) / n,
            "ac2": sum(m.ac2 for m in metrics_list) / n,
            "ac3": sum(m.ac3 for m in metrics_list) / n,
            "ac4": sum(m.ac4 for m in metrics_list) / n,
            "ac5": sum(m.ac5 for m in metrics_list) / n,
            "avg3": sum(m.avg3 for m in metrics_list) / n,
            "avg5": sum(m.avg5 for m in metrics_list) / n,
        }

    ctrl_agg = summarize_metrics(ctrl_metrics_list)
    treat_agg = summarize_metrics(treat_metrics_list)

    print("\n" + "=" * 80)
    print("STAGE 10 SUMMARY AGGREGATES")
    print("=" * 80)
    print(f"Metric       | Control (Stage 9) | Treatment (Stage 10) | Diff")
    print(f"Top@1        | {ctrl_agg['top1']:17.4f} | {treat_agg['top1']:20.4f} | {treat_agg['top1'] - ctrl_agg['top1']:+.4f}")
    print(f"Top@3        | {ctrl_agg['top3']:17.4f} | {treat_agg['top3']:20.4f} | {treat_agg['top3'] - ctrl_agg['top3']:+.4f}")
    print(f"Top@5        | {ctrl_agg['top5']:17.4f} | {treat_agg['top5']:20.4f} | {treat_agg['top5'] - ctrl_agg['top5']:+.4f}")
    print(f"MRR          | {ctrl_agg['mrr']:17.4f} | {treat_agg['mrr']:20.4f} | {treat_agg['mrr'] - ctrl_agg['mrr']:+.4f}")
    print(f"Avg@5        | {ctrl_agg['avg5']:17.4f} | {treat_agg['avg5']:20.4f} | {treat_agg['avg5'] - ctrl_agg['avg5']:+.4f}")

    # Summary of ranking shifts
    promoted_roots = sum(1 for c in per_case_results if c["true_root_promoted"])
    demoted_roots = sum(
        1
        for c in per_case_results
        if c["treatment_root_rank"] is not None
        and c["baseline_root_rank"] is not None
        and c["treatment_root_rank"] > c["baseline_root_rank"]
    )
    unchanged_roots = sum(
        1
        for c in per_case_results
        if c["treatment_root_rank"] == c["baseline_root_rank"]
    )
    winner_changes = sum(1 for c in per_case_results if c["winner_changed"])
    demoted_symptoms = sum(1 for c in per_case_results if c["downstream_symptom_demoted"])
    promoted_unrelated = sum(1 for c in per_case_results if c["unrelated_candidate_promoted"])

    print(f"\nRanking shifts across 60 cases:")
    print(f"  True root promoted:       {promoted_roots} / 60")
    print(f"  True root demoted:        {demoted_roots} / 60")
    print(f"  True root unchanged:      {unchanged_roots} / 60")
    print(f"  Winner changed:           {winner_changes} / 60")
    print(f"  Downstream demoted:       {demoted_symptoms} / 60")
    print(f"  Unrelated promoted:       {promoted_unrelated} / 60")

    # Output artifact
    output_payload = {
        "schema_version": "1.0",
        "experiment_id": "stage10_re2ob_propagation_consistency_treatment_v1",
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest_hash,
        "stage9_hash": stage9_hash,
        "git_revision": git_rev,
        "execution_timestamp_utc": now_iso,
        "total_runtime_sec": round(t_total, 2),
        "population": {
            "dataset": "RE2-OB",
            "total_cases": len(manifest.cases),
            "scenario_families": 30,
            "repetitions": [2, 3],
        },
        "treatment_config": {
            "name": "Parameter-Free Temporal-Topological Propagation Consistency",
            "primary_key": "propagation_coverage_P_c",
            "tie_breaker": "frozen_S_fusion",
            "temporal_directionality": "tau_u < tau_v on static topology adjacency",
            "parameters": {},
        },
        "control_aggregates": ctrl_agg,
        "treatment_aggregates": treat_agg,
        "shifts_summary": {
            "true_root_promoted": promoted_roots,
            "true_root_demoted": demoted_roots,
            "true_root_unchanged": unchanged_roots,
            "winner_changed": winner_changes,
            "downstream_symptom_demoted": demoted_symptoms,
            "unrelated_candidate_promoted": promoted_unrelated,
        },
        "per_case_results": per_case_results,
    }

    output_path = repo_root / "eval/results/stage10_re2ob_propagation_consistency_treatment_v1.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    out_hash = compute_file_sha256(output_path)
    print(f"\nSaved treatment artifact to: {output_path}")
    print(f"Artifact SHA-256:             {out_hash}")
    print(f"Total runtime:                {t_total:.2f}s")


if __name__ == "__main__":
    main()
