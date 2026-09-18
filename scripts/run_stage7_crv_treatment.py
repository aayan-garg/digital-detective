#!/usr/bin/env python3
"""Run Stage 7 experimental treatment benchmark: Counterfactual Removal Validation (CRV).

Experiment definition:
- Population: RE2-OB Repetitions 2 and 3 (60 cases, 30 scenario families, 2 executions per family).
- Repetition 1: strictly excluded.
- Incident windows & detection: Frozen causal BOCPD + TCEC confirmation (Stage 5 baseline).
- Arm A (Control): S_comb ranking (frozen).
- Arm B (Treatment): CRV-shadow ranking (Top-5 reranked by counterfactual relief R_c, tie-broken by S_comb).
- Predeclared incident target: confirming_entity from Stage 5 TCEC (or "frontend" fallback).
- Output artifact: eval/results/stage7_re2ob_crv_treatment_v1.json
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
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from digital_detective.anomaly import (
    detect_metric_anomalies,
    truncate_metric_anomaly_result,
)
from digital_detective.crv import (
    RankedEntity as CRVRankedEntity,
    run_counterfactual_removal_validation,
)
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from eval.manifest import BenchmarkManifest, ManifestCase
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


def compute_universe_hash(universe: tuple[str, ...]) -> str:
    return hashlib.sha256(",".join(sorted(universe)).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    print("=" * 80)
    print("STAGE 7 EXPERIMENTAL BENCHMARK: COUNTERFACTUAL REMOVAL VALIDATION (CRV)")
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

    # 2. Experimental Configurations
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

    methods = ["s_comb", "crv_shadow"]
    method_case_metrics: dict[str, list[Any]] = {m: [] for m in methods}
    per_case_results: list[dict[str, Any]] = []

    # 3. Load frozen Stage 5 TCEC incident boundaries
    stage5_path = repo_root / "eval" / "results" / "stage5_tcec_treatment_v1.json"
    if not stage5_path.exists():
        raise FileNotFoundError(f"Frozen Stage 5 baseline artifact {stage5_path} not found")
    with open(stage5_path, "r", encoding="utf-8") as f:
        stage5_data = json.load(f)
    stage5_cases_by_id = {c["case_id"]: c for c in stage5_data["per_case_results"]}

    print("\nExecuting 60 benchmark cases using frozen Stage 5 confirmed incident windows...")
    t_start_all = time.perf_counter()

    total_top5_candidates = 0
    total_no_path_candidates = 0
    total_order_changed_cases = 0
    total_top1_changed_cases = 0

    for idx, c in enumerate(manifest.cases, start=1):
        t_case_start = time.perf_counter()
        cid = c.case_id
        ground_truth_target = c.root_cause_service
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
        confirming_entity = s5_case.get("confirming_entity") or "frontend"
        candidate_entity = s5_case.get("candidate_entity")

        # 4. Causal Graph & Detection up to confirmed_onset
        metric_names = [col for col in case.metrics.provenance.original_field_names if col != "time"]

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

        # Baseline S_comb calculation
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

        # Convert to CRVRankedEntity for CRV validator
        crv_s_ranking = [
            CRVRankedEntity(entity=r.entity, score=r.score, rank=r.rank)
            for r in s_ranking
        ]

        # Incident symptom target for CRV
        crv_target = confirming_entity if confirming_entity in causal_graph.entities else "frontend"

        # Execute Counterfactual Removal Validation
        crv_res = run_counterfactual_removal_validation(
            case,
            causal_graph,
            crv_s_ranking,
            incident_target=crv_target,
            incident_start_ts=confirmed_onset,
            incident_end_ts=confirmed_onset,
            top_k=5,
        )

        crv_shadow_ranking = tuple(
            RankedEntity(entity=r.entity, score=r.score, rank=r.rank)
            for r in crv_res.shadow_ranking
        )

        total_top5_candidates += len(crv_res.top5_candidates)
        total_no_path_candidates += len(crv_res.no_path_candidates)
        if crv_res.order_changed:
            total_order_changed_cases += 1
        if crv_res.top1_changed:
            total_top1_changed_cases += 1

        case_methods: dict[str, Any] = {}
        for m_name, ranking in [("s_comb", s_ranking), ("crv_shadow", crv_shadow_ranking)]:
            cm = compute_case_metrics(ranking, ground_truth_target)
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
                    {"entity": r.entity, "score": r.score, "rank": r.rank}
                    for r in ranking
                ],
            }

        t_case_dur = time.perf_counter() - t_case_start

        case_entry = {
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "scenario_family_str": fam_str,
            "repetition": c.repetition,
            "ground_truth_target": ground_truth_target,
            "inject_time": inject_time,
            "candidate_onset": candidate_onset,
            "confirmed_onset": confirmed_onset,
            "incident_target": crv_target,
            "candidate_entity": candidate_entity,
            "confirming_entity": confirming_entity,
            "candidate_universe_hash": universe_hash,
            "graph_hash": crv_res.graph_hash,
            "crv_validation": {
                "order_changed": crv_res.order_changed,
                "top1_changed": crv_res.top1_changed,
                "top5_candidates": list(crv_res.top5_candidates),
                "no_path_candidates": list(crv_res.no_path_candidates),
                "candidate_details": [
                    {
                        "candidate": v.candidate,
                        "original_rank": v.original_rank,
                        "relief_score": v.relief_score,
                        "has_target_path": v.has_target_path,
                        "counterfactual_rank": v.counterfactual_rank,
                        "factual_target_mean_deviation": v.factual_target_mean_deviation,
                        "cf_target_mean_deviation": v.cf_target_mean_deviation,
                    }
                    for v in crv_res.validations
                ],
            },
            "methods": case_methods,
            "runtime_sec": t_case_dur,
        }
        per_case_results.append(case_entry)

        if idx % 10 == 0 or idx == len(manifest.cases):
            print(f"[{idx:02d}/60] Processed {cid} (target={ground_truth_target}, CRV target={crv_target}, order_changed={crv_res.order_changed})")

    t_total = time.perf_counter() - t_start_all
    print(f"\nAll 60 executions completed in {t_total:.2f}s ({t_total/60*1000:.1f}ms/case).")

    # 5. Compute Aggregates
    method_aggregates: dict[str, dict[str, Any]] = {}
    for m in methods:
        cms = method_case_metrics[m]
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
        method_aggregates[m] = {
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

    crv_summary = {
        "total_executions": len(manifest.cases),
        "total_top5_candidates_evaluated": total_top5_candidates,
        "total_no_path_candidates": total_no_path_candidates,
        "no_path_candidate_fraction": total_no_path_candidates / max(1, total_top5_candidates),
        "executions_order_changed": total_order_changed_cases,
        "executions_order_changed_fraction": total_order_changed_cases / len(manifest.cases),
        "executions_top1_changed": total_top1_changed_cases,
        "executions_top1_changed_fraction": total_top1_changed_cases / len(manifest.cases),
    }

    # 6. Build final artifact
    output_data = {
        "schema_version": "stage7_crv_treatment_v1",
        "experiment_id": "stage7_re2ob_crv_shadow_treatment_v1",
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest_hash,
        "git_revision": git_rev,
        "execution_timestamp_utc": now_iso,
        "population": {
            "dataset": "RE2-OB",
            "total_cases": len(manifest.cases),
            "scenario_families": len(family_counts),
            "repetitions": [2, 3],
        },
        "experimental_treatment": {
            "method_control": "s_comb",
            "method_treatment": "crv_shadow",
            "top_k_reranked": 5,
            "structural_model": "linear_regression",
            "disturbance_intervention": "zero_candidate_disturbance",
            "relief_metric": "mean_target_z_deviation_reduction",
            "control_baseline_artifact": "eval/results/stage5_tcec_treatment_v1.json",
        },
        "crv_summary": crv_summary,
        "method_aggregates": method_aggregates,
        "per_case_results": per_case_results,
    }

    out_path = repo_root / "eval" / "results" / "stage7_re2ob_crv_treatment_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    with open(out_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    print("\n" + "=" * 80)
    print("STAGE 7 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Control (S_comb) MRR:    {method_aggregates['s_comb']['mrr']:.4f} | Top@1: {method_aggregates['s_comb']['top1_accuracy']:.4f} | Top@3: {method_aggregates['s_comb']['top3_accuracy']:.4f} | Top@5: {method_aggregates['s_comb']['top5_accuracy']:.4f}")
    print(f"Treatment (CRV)  MRR:    {method_aggregates['crv_shadow']['mrr']:.4f} | Top@1: {method_aggregates['crv_shadow']['top1_accuracy']:.4f} | Top@3: {method_aggregates['crv_shadow']['top3_accuracy']:.4f} | Top@5: {method_aggregates['crv_shadow']['top5_accuracy']:.4f}")
    print(f"Top-5 Order Changed:     {total_order_changed_cases}/60 ({crv_summary['executions_order_changed_fraction']*100:.1f}%)")
    print(f"Top-1 Changed:           {total_top1_changed_cases}/60 ({crv_summary['executions_top1_changed_fraction']*100:.1f}%)")
    print(f"No Target Path Cands:    {total_no_path_candidates}/{total_top5_candidates} ({crv_summary['no_path_candidate_fraction']*100:.1f}%)")
    print(f"Artifact Saved:          {out_path}")
    print(f"Artifact SHA-256:        {artifact_hash}")
    print("=" * 80)


if __name__ == "__main__":
    main()
