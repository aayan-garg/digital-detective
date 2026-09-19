#!/usr/bin/env python3
"""Smoke test for Stage 10 Propagation Consistency Treatment on 3 cases."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

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
from eval.models import compute_case_metrics, RankedEntity
from eval.universe import resolve_candidate_universe


def main() -> None:
    print("=" * 80)
    print("STAGE 10 SMOKE TEST: PROPAGATION CONSISTENCY TREATMENT")
    print("=" * 80)

    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    stage9_path = repo_root / "eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json"
    assert stage9_path.exists(), "Stage 9 results not found!"

    with open(stage9_path, "r", encoding="utf-8") as f:
        stage9_data = json.load(f)

    stage9_cases = {c["case_id"]: c for c in stage9_data["per_case_results"]}

    # Pick 3 representative cases
    smoke_ids = [
        "re2ob_checkoutservice_cpu_2",
        "re2ob_currencyservice_cpu_2",
        "re2ob_emailservice_cpu_2",
    ]

    ep_cfg = EpisodeConfig(persistence=3, consensus=2)
    det_config = {
        "normalization": "mean_std",
        "window_size": 60,
        "threshold": 3.0,
        "min_warmup": 60,
        "min_valid_history": 10,
        "epsilon": 1e-6,
    }

    for cid in smoke_ids:
        print(f"\nEvaluating case: {cid}")
        s9_c = stage9_cases[cid]
        target = s9_c["target"]
        confirmed_onset = s9_c["confirmed_onset"]
        is_confirmed = s9_c["is_confirmed"]
        s9_ctrl_ranking = s9_c["methods"]["fixed_equal_weight_fusion"]["ranking"]

        case = load_rcaeval_case(dataset_root, cid)
        universe = resolve_candidate_universe("ob", case=case)

        # Truncated detector
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
            norm_t = {k: 0.0 for k in universe}
            fused = norm_s

        ctrl_sorted = sorted(universe, key=lambda e: (-fused[e], e))
        ctrl_ranking = tuple(
            RankedEntity(entity=e, score=fused[e], rank=idx)
            for idx, e in enumerate(ctrl_sorted, start=1)
        )

        # Verify exact control invariance against Stage-9 JSON
        recomputed_ctrl_ents = [r.entity for r in ctrl_ranking]
        assert recomputed_ctrl_ents == s9_ctrl_ranking, (
            f"Control mismatch for {cid}:\n"
            f"  Recomputed: {recomputed_ctrl_ents}\n"
            f"  Stage 9:    {s9_ctrl_ranking}"
        )
        print("  Control invariance check: PASSED (100% exact match to Stage 9 reference)")

        # Compute Stage 10 Treatment
        cov_res = compute_propagation_coverage(ep_ev, causal_graph, universe)
        treat_ranking = rank_with_propagation_consistency(ctrl_ranking, cov_res.coverage)
        treat_ents = [r.entity for r in treat_ranking]

        cm_ctrl = compute_case_metrics(ctrl_ranking, target)
        cm_treat = compute_case_metrics(treat_ranking, target)

        print(f"  Target root:               {target}")
        print(f"  Anomalous entities:        {cov_res.anomalous_entities}")
        print(f"  Temporal edges:            {cov_res.temporal_edges}")
        print(f"  Propagation coverage P(c): {cov_res.coverage}")
        print(f"  Control root rank:         {cm_ctrl.target_rank} (Top1={cm_ctrl.top1}, MRR={cm_ctrl.mrr:.3f})")
        print(f"  Treatment root rank:       {cm_treat.target_rank} (Top1={cm_treat.top1}, MRR={cm_treat.mrr:.3f})")
        print(f"  Winner: Control={ctrl_ranking[0].entity}, Treatment={treat_ranking[0].entity}")
        print(f"  Treatment ranking:         {treat_ents[:5]}")

    print("\nSmoke test passed successfully!")


if __name__ == "__main__":
    main()
