#!/usr/bin/env python3
"""Run Stage 8 experimental benchmark: Parameter-Free Trace Tie-Break for S_comb.

Experiment definition:
- Population: RE2-OB Repetitions 2 and 3 (60 cases, 30 scenario families, 2 executions per family).
- Repetition 1: strictly excluded.
- Incident windows & detection: Frozen causal BOCPD + TCEC confirmation (Stage 5 baseline).
- Arm A (Control): S_comb ranking (frozen).
- Arm B (Treatment): S_comb_trace_tiebreak ranking (exact S_comb ties resolved by descending E_elev).
- Output artifact: eval/results/stage8_re2ob_trace_tiebreak_treatment_v1.json
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
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.traces import extract_trace_latency_evidence
from digital_detective.trace_tiebreak import (
    RankedEntity as TieBreakRankedEntity,
    apply_trace_tiebreak,
    extract_inbound_elevations,
)
from eval.manifest import BenchmarkManifest, ManifestCase
from eval.models import (
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
    print("STAGE 8 EXPERIMENTAL BENCHMARK: PARAMETER-FREE TRACE TIE-BREAK FOR S_COMB")
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

    methods = ["s_comb", "s_comb_trace_tiebreak"]
    method_case_metrics: dict[str, list[Any]] = {m: [] for m in methods}
    per_case_results: list[dict[str, Any]] = []

    # 3. Load frozen Stage 5 TCEC incident boundaries and Stage 7 frozen S_comb rankings
    stage5_path = repo_root / "eval" / "results" / "stage5_tcec_treatment_v1.json"
    if not stage5_path.exists():
        raise FileNotFoundError(f"Frozen Stage 5 baseline artifact {stage5_path} not found")
    with open(stage5_path, "r", encoding="utf-8") as f:
        stage5_data = json.load(f)
    stage5_cases_by_id = {c["case_id"]: c for c in stage5_data["per_case_results"]}

    stage7_path = repo_root / "eval" / "results" / "stage7_re2ob_crv_treatment_v1.json"
    if not stage7_path.exists():
        raise FileNotFoundError(f"Frozen Stage 7 baseline artifact {stage7_path} not found")
    with open(stage7_path, "r", encoding="utf-8") as f:
        stage7_data = json.load(f)
    stage7_cases_by_id = {c["case_id"]: c for c in stage7_data["per_case_results"]}

    # Cache for inbound trace elevations
    trace_cache_path = repo_root / "eval" / "results" / ".cache_re2ob_trace_elevations.json"
    cached_elevations: dict[str, dict[str, float]] = {}
    if trace_cache_path.exists():
        try:
            with open(trace_cache_path, "r", encoding="utf-8") as f:
                cached_elevations = json.load(f)
            print(f"Loaded cached trace elevations for {len(cached_elevations)} cases.")
        except Exception:
            cached_elevations = {}

    print("\nExecuting 60 benchmark cases using frozen Stage 5 confirmed incident windows...")
    t_start_all = time.perf_counter()

    total_order_changed_cases = 0
    total_top1_changed_cases = 0
    total_top5_changed_cases = 0
    total_ties_resolved = 0
    trace_cache_modified = False

    for idx, c in enumerate(manifest.cases, start=1):
        t_case_start = time.perf_counter()
        cid = c.case_id
        target = c.root_cause_service
        fam_str = ":".join(c.scenario_family)

        s5_case = stage5_cases_by_id[cid]
        s7_case = stage7_cases_by_id[cid]
        confirmed_onset = s5_case["confirmed_onset"]
        candidate_onset = s5_case["candidate_onset"]
        inject_time = s5_case["inject_time"]
        confirming_entity = s5_case.get("confirming_entity")
        candidate_entity = s5_case.get("candidate_entity")

        # 4. Control Ranking: Reuse exact frozen S_comb ranking and scores
        s_comb_raw = s7_case["methods"]["s_comb"]["ranking"]
        s_ranking = tuple(
            RankedEntity(entity=r["entity"], score=float(r["score"]), rank=int(r["rank"]))
            for r in s_comb_raw
        )
        universe = tuple(r.entity for r in s_ranking)
        universe_hash = compute_universe_hash(universe)

        # 5. Inbound trace elevation E_elev
        if cid in cached_elevations:
            elev_map = cached_elevations[cid]
        else:
            case = load_rcaeval_case(dataset_root, cid)
            has_traces = (case.traces is not None) and (case.traces.raw_data is not None)
            if has_traces and confirmed_onset is not None:
                trace_lat = extract_trace_latency_evidence(
                    case,
                    service_aliases={"frontendservice": "frontend"},
                    expected_services=universe,
                    max_timestamp=confirmed_onset,
                )
                elev_map = extract_inbound_elevations(trace_lat, candidate_universe=universe)
            else:
                elev_map = {ent: 0.0 for ent in universe}
            cached_elevations[cid] = elev_map
            trace_cache_modified = True

        # 6. S_comb_trace_tiebreak (Treatment)
        tiebreak_res = apply_trace_tiebreak(
            s_ranking,
            trace_evidence=elev_map,
            case_id=cid,
        )

        treat_ranking = tuple(
            RankedEntity(entity=r.entity, score=r.score, rank=r.rank)
            for r in tiebreak_res.treatment_ranking
        )

        if tiebreak_res.order_changed:
            total_order_changed_cases += 1
        total_ties_resolved += tiebreak_res.ties_resolved_count

        cm_control = compute_case_metrics(s_ranking, target)
        cm_treatment = compute_case_metrics(treat_ranking, target)

        method_case_metrics["s_comb"].append(cm_control)
        method_case_metrics["s_comb_trace_tiebreak"].append(cm_treatment)

        if cm_control.top1 != cm_treatment.top1:
            total_top1_changed_cases += 1
        if cm_control.top5 != cm_treatment.top5:
            total_top5_changed_cases += 1

        case_methods: dict[str, Any] = {}
        for m_name, cm, rk in [
            ("s_comb", cm_control, s_ranking),
            ("s_comb_trace_tiebreak", cm_treatment, treat_ranking),
        ]:
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
                "predicted_root_cause": rk[0].entity,
                "ranking": [
                    {"entity": r.entity, "score": r.score, "rank": r.rank}
                    for r in rk
                ],
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
            "candidate_entity": candidate_entity,
            "confirming_entity": confirming_entity,
            "candidate_universe_hash": universe_hash,
            "trace_tiebreak": {
                "order_changed": tiebreak_res.order_changed,
                "ties_resolved_count": tiebreak_res.ties_resolved_count,
                "target_rank_control": cm_control.target_rank,
                "target_rank_treatment": cm_treatment.target_rank,
                "candidate_details": [
                    {
                        "entity": d.entity,
                        "s_comb_score": d.s_comb_score,
                        "s_comb_rank": d.s_comb_rank,
                        "e_elev": d.e_elev,
                        "final_rank": d.final_rank,
                        "was_tied": d.was_tied,
                        "rank_changed": d.rank_changed,
                    }
                    for d in tiebreak_res.candidate_details
                ],
            },
            "methods": case_methods,
            "runtime_sec": t_case_dur,
        }
        per_case_results.append(case_entry)

        if idx % 10 == 0 or idx == len(manifest.cases):
            print(f"[{idx:02d}/60] Processed {cid} (target={target:20s} | ctrl_rank={cm_control.target_rank:2d} -> treat_rank={cm_treatment.target_rank:2d})")

    t_total = time.perf_counter() - t_start_all
    print(f"\nAll 60 executions completed in {t_total:.2f}s ({t_total/60*1000:.1f}ms/case).")

    if trace_cache_modified:
        with open(trace_cache_path, "w", encoding="utf-8") as f:
            json.dump(cached_elevations, f, indent=2)
        print(f"Saved {len(cached_elevations)} trace elevations to cache: {trace_cache_path}")

    # 5. Compute Aggregates
    method_aggregates: dict[str, dict[str, Any]] = {}
    for m in methods:
        cms = method_case_metrics[m]
        n_cases = len(cms)
        top1_acc = sum(1.0 for item in cms if item.top1) / n_cases
        top3_acc = sum(1.0 for item in cms if item.top3) / n_cases
        top5_acc = sum(1.0 for item in cms if item.top5) / n_cases
        mrr_val = sum(item.mrr for item in cms) / n_cases
        ac1_acc = sum(item.ac1 for item in cms) / n_cases
        ac2_acc = sum(item.ac2 for item in cms) / n_cases
        ac3_acc = sum(item.ac3 for item in cms) / n_cases
        ac4_acc = sum(item.ac4 for item in cms) / n_cases
        ac5_acc = sum(item.ac5 for item in cms) / n_cases
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

    tiebreak_summary = {
        "total_executions": len(manifest.cases),
        "executions_order_changed": total_order_changed_cases,
        "executions_order_changed_fraction": total_order_changed_cases / len(manifest.cases),
        "executions_top1_changed": total_top1_changed_cases,
        "executions_top5_changed": total_top5_changed_cases,
        "total_ties_resolved": total_ties_resolved,
    }

    # 6. Build final artifact
    output_data = {
        "schema_version": "stage8_trace_tiebreak_treatment_v1",
        "experiment_id": "stage8_re2ob_trace_tiebreak_treatment_v1",
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
            "method_treatment": "s_comb_trace_tiebreak",
            "tiebreak_mechanism": "inbound_trace_elevation_e_elev_descending",
            "non_tied_preservation": "strict_s_comb_score_ordering",
            "control_baseline_artifact": "eval/results/stage5_tcec_treatment_v1.json",
        },
        "tiebreak_summary": tiebreak_summary,
        "method_aggregates": method_aggregates,
        "per_case_results": per_case_results,
    }

    out_path = repo_root / "eval" / "results" / "stage8_re2ob_trace_tiebreak_treatment_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    with open(out_path, "rb") as f:
        artifact_hash = hashlib.sha256(f.read()).hexdigest()

    print("\n" + "=" * 80)
    print("STAGE 8 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Control (S_comb)           MRR: {method_aggregates['s_comb']['mrr']:.4f} | Top@1: {method_aggregates['s_comb']['top1_accuracy']:.4f} | Top@3: {method_aggregates['s_comb']['top3_accuracy']:.4f} | Top@5: {method_aggregates['s_comb']['top5_accuracy']:.4f}")
    print(f"Treatment (Trace Tie-Break) MRR: {method_aggregates['s_comb_trace_tiebreak']['mrr']:.4f} | Top@1: {method_aggregates['s_comb_trace_tiebreak']['top1_accuracy']:.4f} | Top@3: {method_aggregates['s_comb_trace_tiebreak']['top3_accuracy']:.4f} | Top@5: {method_aggregates['s_comb_trace_tiebreak']['top5_accuracy']:.4f}")
    print(f"Order Changed Executions:  {total_order_changed_cases}/60 ({tiebreak_summary['executions_order_changed_fraction']*100:.1f}%)")
    print(f"Top-1 Changed:             {total_top1_changed_cases}/60")
    print(f"Top-5 Changed:             {total_top5_changed_cases}/60")
    print(f"Artifact Saved:            {out_path}")
    print(f"Artifact SHA-256:          {artifact_hash}")
    print("=" * 80)


if __name__ == "__main__":
    main()
