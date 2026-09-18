#!/usr/bin/env python3
"""Stage 3 Diagnostic: Oracle vs. Detected Incident Window Headroom Analysis.

Evaluates RCA performance on the exact same 60 RE2-OB repetitions 2 & 3 cases
under the benchmark oracle injection-time window vs. the frozen detected-mode control.

Explicitly labeled: ORACLE — theoretical headroom only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence
from eval.baselines.random_ranker import rank_with_random
from eval.baselines.simple_rca import rank_with_simple_rca
from eval.manifest import BenchmarkManifest
from eval.models import compute_case_metrics, RankedEntity
from eval.universe import resolve_candidate_universe
from eval.windows import resolve_incident_window


def sha256_file(path: Path | str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest().upper()


def main() -> None:
    print("=" * 80)
    print("STAGE 3 DIAGNOSTIC: ORACLE VS. DETECTED INCIDENT WINDOW HEADROOM")
    print("LABEL: ORACLE — theoretical headroom only")
    print("=" * 80)

    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    ctrl_path = repo_root / "eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json"
    manifest_path = repo_root / "eval/manifests/re2_ob_all_cases.json"

    # Verify input control artifact hash
    ctrl_sha256 = sha256_file(ctrl_path)
    expected_ctrl_sha256 = "3C01E2CA28F89AA805A8109152E0B8D992F32330C3E70CDADC19A645C1110157"
    assert ctrl_sha256 == expected_ctrl_sha256, f"Control hash mismatch! Got {ctrl_sha256}"
    print(f"Verified Control Artifact SHA-256: {ctrl_sha256}")

    with open(ctrl_path, "r", encoding="utf-8") as f:
        ctrl_art = json.load(f)

    manifest = BenchmarkManifest.load(manifest_path)
    print(f"Loaded Manifest: {manifest.manifest_id} ({len(manifest.cases)} cases)")
    assert len(manifest.cases) == 60

    ctrl_cases = {c["case_id"]: c for c in ctrl_art["per_case_results"]}
    methods = ["s_comb", "trace_elevation", "fixed_equal_weight_fusion", "simple_rca", "random"]
    ep_cfg = EpisodeConfig(persistence=3, consensus=2)

    oracle_per_case: list[dict[str, Any]] = []
    paired_per_case: list[dict[str, Any]] = []

    print("\nExecuting Condition B (Oracle Window) on 60 cases...")
    t_start_all = time.perf_counter()

    for idx, c in enumerate(manifest.cases, start=1):
        cid = c.case_id
        c_ctrl = ctrl_cases[cid]
        case = load_rcaeval_case(dataset_root, cid)
        universe = resolve_candidate_universe(c.system, case=case)
        target = c.root_cause_service

        inject_time = int(case.ground_truth.values["inject_time"])
        has_traces = hasattr(case, "traces") and case.traces is not None
        has_logs = hasattr(case, "logs") and case.logs is not None

        # 1. Metric anomaly detection using mean/std (same as control)
        det_res = detect_metric_anomalies(
            case,
            window_size=60,
            threshold=3.0,
            min_warmup=60,
            min_valid_history=30,
            epsilon=1e-6,
            normalization="mean_std",
        )

        # 2. Trace dependencies (unbounded in oracle mode)
        oracle_trace_deps = (
            extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
            if has_traces
            else ()
        )
        oracle_deps = [td.to_dependency() for td in oracle_trace_deps]
        graph = build_entity_graph(det_res.metric_names, dependencies=oracle_deps)

        # 3. Episode evidence (unbounded in oracle mode)
        ep_evidence = aggregate_entity_episodes(det_res, graph, ep_cfg)

        # 4. Resolve Oracle Window
        window = resolve_incident_window(
            case,
            ep_evidence=ep_evidence,
            mode="oracle",
            timestamps=det_res.timestamps,
        )
        assert window.onset_ts == inject_time
        assert window.mode == "oracle"

        # 5. Trace latency evidence (unbounded in oracle mode)
        if has_traces:
            trace_lat = extract_trace_latency_evidence(
                case,
                service_aliases={"frontendservice": "frontend"},
                expected_services=universe,
            )
        else:
            trace_lat = None

        # 6. Rank with RCA methods
        case_methods_oracle: dict[str, Any] = {}
        for m in methods:
            t0 = time.perf_counter()
            if m == "s_comb":
                scores = rank_with_s_comb(
                    det_res,
                    ep_evidence,
                    graph=graph,
                    candidate_universe=universe,
                )
                ranking = tuple(
                    RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                    for r_idx, s in enumerate(scores, start=1)
                )
            elif m == "trace_elevation":
                if trace_lat is not None:
                    scores = rank_with_trace_elevation(trace_lat, candidate_universe=universe)
                    ranking = tuple(
                        RankedEntity(entity=s.entity, score=s.score, rank=r_idx)
                        for r_idx, s in enumerate(scores, start=1)
                    )
                else:
                    ranking = ()
            elif m == "fixed_equal_weight_fusion":
                s_scores = {
                    s.entity: s.score
                    for s in rank_with_s_comb(det_res, ep_evidence, candidate_universe=universe)
                }
                max_s = max(s_scores.values()) if s_scores else 0.0
                norm_s = {k: (v / max_s if max_s > 0 else 0.0) for k, v in s_scores.items()}

                if has_traces and trace_lat is not None:
                    t_scores = {
                        s.entity: s.score
                        for s in rank_with_trace_elevation(trace_lat, candidate_universe=universe)
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
                raise ValueError(f"Unknown method {m}")

            t_elapsed = time.perf_counter() - t0
            metrics = compute_case_metrics(ranking, target)
            case_methods_oracle[m] = {
                "top1": metrics.top1,
                "top3": metrics.top3,
                "top5": metrics.top5,
                "mrr": round(metrics.mrr, 6),
                "predicted_top5": [
                    {"entity": r.entity, "score": round(r.score, 6), "rank": r.rank}
                    for r in ranking[:5]
                ],
                "runtime_sec": round(t_elapsed, 6),
            }

        # Per-case record for Condition B
        case_rec_oracle = {
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "repetition": c.repetition,
            "target": target,
            "inject_time": inject_time,
            "window": {
                "onset_ts": window.onset_ts,
                "end_ts": window.end_ts,
                "mode": window.mode,
                "source": window.source_description,
            },
            "methods": case_methods_oracle,
        }
        oracle_per_case.append(case_rec_oracle)

        # Paired comparison record
        paired_rec = {
            "case_id": cid,
            "scenario_family": list(c.scenario_family),
            "scenario_family_str": ":".join(c.scenario_family),
            "repetition": c.repetition,
            "target": target,
            "timing": {
                "inject_time": inject_time,
                "detected_onset": c_ctrl["detected_onset"],
                "detected_window": {
                    "start": c_ctrl["analysis_start"],
                    "end": c_ctrl["analysis_end"],
                },
                "oracle_window": {
                    "start": window.onset_ts,
                    "end": window.end_ts,
                },
                "onset_difference_oracle_minus_detected_sec": window.onset_ts - c_ctrl["detected_onset"],
            },
            "method_differences": {},
        }
        for m in methods:
            m_ctrl = c_ctrl["methods"][m]
            m_ora = case_methods_oracle[m]
            paired_rec["method_differences"][m] = {
                "top1_detected": m_ctrl["top1"],
                "top1_oracle": m_ora["top1"],
                "top1_diff": int(m_ora["top1"]) - int(m_ctrl["top1"]),
                "top3_detected": m_ctrl["top3"],
                "top3_oracle": m_ora["top3"],
                "top3_diff": int(m_ora["top3"]) - int(m_ctrl["top3"]),
                "top5_detected": m_ctrl["top5"],
                "top5_oracle": m_ora["top5"],
                "top5_diff": int(m_ora["top5"]) - int(m_ctrl["top5"]),
                "mrr_detected": m_ctrl["mrr"],
                "mrr_oracle": m_ora["mrr"],
                "mrr_diff": round(m_ora["mrr"] - m_ctrl["mrr"], 6),
            }
        paired_per_case.append(paired_rec)

        print(
            f"[{idx:02d}/60] {cid:35s} | s_comb MRR: det={c_ctrl['methods']['s_comb']['mrr']:.3f} -> ora={case_methods_oracle['s_comb']['mrr']:.3f} (diff: {case_methods_oracle['s_comb']['mrr'] - c_ctrl['methods']['s_comb']['mrr']:+.3f})"
        )

    t_total = time.perf_counter() - t_start_all
    print(f"\nAll 60 cases completed in {t_total:.2f}s ({t_total/60:.2f}s/case).")

    # Aggregate Metrics Calculation
    method_aggregates: dict[str, dict[str, Any]] = {}
    for m in methods:
        det_top1 = sum(1 for c in ctrl_cases.values() if c["methods"][m]["top1"]) / 60.0
        det_top3 = sum(1 for c in ctrl_cases.values() if c["methods"][m]["top3"]) / 60.0
        det_top5 = sum(1 for c in ctrl_cases.values() if c["methods"][m]["top5"]) / 60.0
        det_mrr = sum(c["methods"][m]["mrr"] for c in ctrl_cases.values()) / 60.0

        ora_top1 = sum(1 for c in oracle_per_case if c["methods"][m]["top1"]) / 60.0
        ora_top3 = sum(1 for c in oracle_per_case if c["methods"][m]["top3"]) / 60.0
        ora_top5 = sum(1 for c in oracle_per_case if c["methods"][m]["top5"]) / 60.0
        ora_mrr = sum(c["methods"][m]["mrr"] for c in oracle_per_case) / 60.0

        diff_top1 = ora_top1 - det_top1
        diff_top3 = ora_top3 - det_top3
        diff_top5 = ora_top5 - det_top5
        diff_mrr = ora_mrr - det_mrr

        # Family-level robustness
        unique_fams = sorted(list(set(c["scenario_family_str"] for c in paired_per_case)))
        fam_diffs: list[float] = []
        pos_f = 0
        neg_f = 0
        tie_f = 0
        for f_str in unique_fams:
            f_cases = [c for c in paired_per_case if c["scenario_family_str"] == f_str]
            f_mean_diff = sum(c["method_differences"][m]["mrr_diff"] for c in f_cases) / len(f_cases)
            fam_diffs.append(f_mean_diff)
            if f_mean_diff > 1e-9:
                pos_f += 1
            elif f_mean_diff < -1e-9:
                neg_f += 1
            else:
                tie_f += 1

        method_aggregates[m] = {
            "detected": {
                "top1": round(det_top1, 4),
                "top3": round(det_top3, 4),
                "top5": round(det_top5, 4),
                "mrr": round(det_mrr, 4),
            },
            "oracle_headroom": {
                "top1": round(ora_top1, 4),
                "top3": round(ora_top3, 4),
                "top5": round(ora_top5, 4),
                "mrr": round(ora_mrr, 4),
            },
            "headroom_gain": {
                "top1_gain": round(diff_top1, 4),
                "top3_gain": round(diff_top3, 4),
                "top5_gain": round(diff_top5, 4),
                "mrr_gain": round(diff_mrr, 4),
                "relative_mrr_gain": round(diff_mrr / det_mrr, 4) if det_mrr > 0 else 0.0,
            },
            "family_direction": {
                "positive_families": pos_f,
                "negative_families": neg_f,
                "tied_families": tie_f,
                "min_family_diff": round(min(fam_diffs), 4),
                "max_family_diff": round(max(fam_diffs), 4),
                "median_family_diff": round(sorted(fam_diffs)[len(fam_diffs)//2], 4),
            },
        }

    # Complete Diagnostic Artifact
    diagnostic_artifact = {
        "schema_version": "stage3_oracle_vs_detected_headroom_v1",
        "label": "ORACLE — theoretical headroom only",
        "description": "Headroom diagnostic comparing frozen Stage 2 detected window vs. oracle injection window",
        "timestamp_iso": "2026-09-18T06:30:00Z",
        "inputs": {
            "control_artifact": "eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json",
            "control_artifact_sha256": ctrl_sha256,
            "manifest_path": "eval/manifests/re2_ob_all_cases.json",
            "manifest_hash": manifest.compute_hash(),
        },
        "population": {
            "dataset": "RE2-OB",
            "repetitions": [2, 3],
            "n_executions": 60,
            "n_scenario_families": 30,
            "executions_per_family": 2,
        },
        "window_semantics": {
            "condition_a_detected": {
                "mode": "detected",
                "onset": "earliest confirmed entity episode start timestamp (causally pre-injection)",
                "boundary": "truncate all evidence at detected_onset",
            },
            "condition_b_oracle": {
                "mode": "oracle",
                "onset": "ground-truth inject_time",
                "boundary": "full post-injection incident telemetry [inject_time, telemetry_end_ts]",
                "label": "ORACLE — theoretical headroom only",
            },
        },
        "aggregate_headroom_results": method_aggregates,
        "paired_per_case_results": paired_per_case,
        "oracle_per_case_results": oracle_per_case,
    }

    out_json = repo_root / "eval/results/stage3_oracle_vs_detected_headroom_v1.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostic_artifact, f, indent=2)
    print(f"\nSaved Stage 3 diagnostic JSON to: {out_json}")

    print("\n=== STAGE 3 HEADROOM SUMMARY ===")
    for m in methods:
        g = method_aggregates[m]["headroom_gain"]
        d = method_aggregates[m]["detected"]
        o = method_aggregates[m]["oracle_headroom"]
        fam = method_aggregates[m]["family_direction"]
        print(f"Method: {m:26s}")
        print(f"  Top@1: det={d['top1']:.4f} -> ora={o['top1']:.4f} (gain: {g['top1_gain']:+.4f})")
        print(f"  Top@3: det={d['top3']:.4f} -> ora={o['top3']:.4f} (gain: {g['top3_gain']:+.4f})")
        print(f"  Top@5: det={d['top5']:.4f} -> ora={o['top5']:.4f} (gain: {g['top5_gain']:+.4f})")
        print(f"  MRR:   det={d['mrr']:.4f} -> ora={o['mrr']:.4f} (gain: {g['mrr_gain']:+.4f}, rel: {g['relative_mrr_gain']:+.1%})")
        print(f"  Families: pos={fam['positive_families']}, neg={fam['negative_families']}, tie={fam['tied_families']}")


if __name__ == "__main__":
    main()
