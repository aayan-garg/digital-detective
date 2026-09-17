"""Unified evaluation harness for Digital Detective Stage 0 & Stage 1 research.

Orchestrates:
1. Manifest loading and partitioning.
2. Case loading with explicit modality tracking.
3. Closed deterministic candidate universe resolution.
4. Formal IncidentWindow construction (oracle vs detected).
5. Invocation of frozen baselines:
   - Random
   - SimpleRCA (Fang et al. arXiv:2510.04711 §3.1.2)
   - S_comb (frozen metric RCA)
   - Trace Elevation (frozen trace RCA)
   - Modality ablations (metric-only, trace-only, log-only)
   - Fixed equal-weight fusion
6. Full ranking storage and per-case metric evaluation (Top@k, MRR, AC@k, Avg@5).
7. Runtime tracking and modality availability/exclusion recording.
8. Stage 1 Oracle theoretical headroom measurement (per-fault-type and per-incident).
9. Versioned, machine-readable JSON serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence

from .baselines.random_ranker import rank_with_random
from .baselines.simple_rca import rank_with_simple_rca
from .manifest import BenchmarkManifest, ManifestCase
from .models import (
    AggregateMethodMetrics,
    IncidentWindow,
    MethodRankingResult,
    ModalityAvailability,
    RankedEntity,
    aggregate_case_metrics,
    compute_case_metrics,
)
from .universe import resolve_candidate_universe
from .windows import resolve_incident_window

HARNESS_SCHEMA_VERSION = "1.0.0"


@dataclass
class HeadroomAnalysis:
    """Theoretical headroom measurement comparing oracles against best fixed modality."""

    best_fixed_method: str
    best_fixed_metrics: AggregateMethodMetrics
    oracle_per_fault_metrics: AggregateMethodMetrics
    oracle_per_incident_metrics: AggregateMethodMetrics
    absolute_headroom_fault: dict[str, float]
    relative_headroom_fault: dict[str, float]
    absolute_headroom_incident: dict[str, float]
    relative_headroom_incident: dict[str, float]
    disclaimer: str = "ORACLE — theoretical headroom only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "disclaimer": self.disclaimer,
            "best_fixed_method": self.best_fixed_method,
            "best_fixed_metrics": self.best_fixed_metrics.to_dict(),
            "oracle_per_fault_metrics": self.oracle_per_fault_metrics.to_dict(),
            "oracle_per_incident_metrics": self.oracle_per_incident_metrics.to_dict(),
            "absolute_headroom_fault": {k: round(v, 4) for k, v in self.absolute_headroom_fault.items()},
            "relative_headroom_fault": {k: round(v, 4) for k, v in self.relative_headroom_fault.items()},
            "absolute_headroom_incident": {k: round(v, 4) for k, v in self.absolute_headroom_incident.items()},
            "relative_headroom_incident": {k: round(v, 4) for k, v in self.relative_headroom_incident.items()},
        }


@dataclass
class BenchmarkExecutionReport:
    """Complete, versioned benchmark execution report."""

    schema_version: str
    manifest_id: str
    manifest_description: str
    window_mode: str
    total_cases: int
    case_ids: tuple[str, ...]
    evaluated_methods: tuple[str, ...]
    method_aggregates: dict[str, AggregateMethodMetrics]
    headroom_analysis: HeadroomAnalysis | None
    case_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "manifest_description": self.manifest_description,
            "window_mode": self.window_mode,
            "total_cases": self.total_cases,
            "case_ids": list(self.case_ids),
            "evaluated_methods": list(self.evaluated_methods),
            "method_aggregates": {k: v.to_dict() for k, v in self.method_aggregates.items()},
            "headroom_analysis": self.headroom_analysis.to_dict() if self.headroom_analysis else None,
            "case_results": self.case_results,
        }

    def save(self, output_path: str | Path) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def run_ranker(
    method_name: str,
    case: Any,
    window: IncidentWindow,
    candidate_universe: tuple[str, ...],
    modality_avail: ModalityAvailability,
    precomputed: dict[str, Any],
) -> MethodRankingResult:
    """Execute a single ranker on a prepared case and return the full ranking result."""
    cid = str(case.metadata.case_id)
    t_start = time.perf_counter()

    if method_name == "random":
        ranking = rank_with_random(candidate_universe, case_id=cid)
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name="random",
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name == "simple_rca":
        ranking = rank_with_simple_rca(
            case,
            window,
            candidate_universe,
            enabled_modalities=("metrics", "traces", "logs"),
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name="simple_rca",
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name == "simple_rca_metric":
        ranking = rank_with_simple_rca(
            case,
            window,
            candidate_universe,
            enabled_modalities=("metrics",),
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name="simple_rca_metric",
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name == "simple_rca_trace":
        if not modality_avail.has_traces:
            t_elapsed = time.perf_counter() - t_start
            return MethodRankingResult(
                case_id=cid,
                method_name="simple_rca_trace",
                window_mode=window.mode,
                ranking=(),
                candidate_universe=candidate_universe,
                status="UNAVAILABLE",
                exclusion_reason="Trace modality missing for this case",
                runtime_sec=t_elapsed,
            )
        ranking = rank_with_simple_rca(
            case,
            window,
            candidate_universe,
            enabled_modalities=("traces",),
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name="simple_rca_trace",
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name == "simple_rca_log":
        if not modality_avail.has_logs:
            t_elapsed = time.perf_counter() - t_start
            return MethodRankingResult(
                case_id=cid,
                method_name="simple_rca_log",
                window_mode=window.mode,
                ranking=(),
                candidate_universe=candidate_universe,
                status="UNAVAILABLE",
                exclusion_reason="Logs modality missing for this case",
                runtime_sec=t_elapsed,
            )
        ranking = rank_with_simple_rca(
            case,
            window,
            candidate_universe,
            enabled_modalities=("logs",),
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name="simple_rca_log",
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name in {"s_comb", "metric_s_comb"}:
        det_res = precomputed["det_res"]
        ep_evidence = precomputed["ep_evidence"]
        graph = precomputed.get("graph")
        scores = rank_with_s_comb(
            det_res,
            ep_evidence,
            graph=graph,
            candidate_universe=candidate_universe,
        )
        ranking = tuple(
            RankedEntity(entity=s.entity, score=s.score, rank=idx)
            for idx, s in enumerate(scores, start=1)
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name=method_name,
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name in {"trace_elevation", "trace_only"}:
        if not modality_avail.has_traces:
            t_elapsed = time.perf_counter() - t_start
            return MethodRankingResult(
                case_id=cid,
                method_name=method_name,
                window_mode=window.mode,
                ranking=(),
                candidate_universe=candidate_universe,
                status="UNAVAILABLE",
                exclusion_reason="Trace modality missing for this case",
                runtime_sec=t_elapsed,
            )
        trace_lat = precomputed["trace_lat"]
        scores = rank_with_trace_elevation(trace_lat, candidate_universe=candidate_universe)
        ranking = tuple(
            RankedEntity(entity=s.entity, score=s.score, rank=idx)
            for idx, s in enumerate(scores, start=1)
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name=method_name,
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    if method_name == "fixed_equal_weight_fusion":
        # Combines normalized S_comb (metric) and normalized E_elev (trace)
        det_res = precomputed["det_res"]
        ep_evidence = precomputed["ep_evidence"]
        s_comb_scores = {
            s.entity: s.score
            for s in rank_with_s_comb(det_res, ep_evidence, candidate_universe=candidate_universe)
        }
        max_s = max(s_comb_scores.values()) if s_comb_scores else 0.0
        norm_s = {k: (v / max_s if max_s > 0 else 0.0) for k, v in s_comb_scores.items()}

        if modality_avail.has_traces and "trace_lat" in precomputed:
            trace_lat = precomputed["trace_lat"]
            trace_scores = {
                s.entity: s.score
                for s in rank_with_trace_elevation(trace_lat, candidate_universe=candidate_universe)
            }
            max_t = max(trace_scores.values()) if trace_scores else 0.0
            norm_t = {k: (v / max_t if max_t > 0 else 0.0) for k, v in trace_scores.items()}
            fused = {k: (0.5 * norm_s[k] + 0.5 * norm_t[k]) for k in candidate_universe}
        else:
            fused = norm_s

        sorted_ents = sorted(candidate_universe, key=lambda e: (-fused[e], e))
        ranking = tuple(
            RankedEntity(entity=e, score=fused[e], rank=idx)
            for idx, e in enumerate(sorted_ents, start=1)
        )
        t_elapsed = time.perf_counter() - t_start
        return MethodRankingResult(
            case_id=cid,
            method_name=method_name,
            window_mode=window.mode,
            ranking=ranking,
            candidate_universe=candidate_universe,
            status="SUCCESS",
            runtime_sec=t_elapsed,
        )

    raise ValueError(f"Unknown or unsupported method name: {method_name!r}")


def compute_oracle_headroom(
    cases: Sequence[ManifestCase],
    case_results_by_method: Mapping[str, list[MethodRankingResult]],
    single_modality_methods: Sequence[str] = ("s_comb", "trace_elevation"),
) -> HeadroomAnalysis:
    """Compute measurement-only Oracle headroom against the Best Fixed Modality.

    Oracles:
    1. Oracle per-fault-type: For each fault type, selects the modality that maximizes aggregate performance.
    2. Oracle per-incident: For each incident, selects whichever single modality achieves the highest reciprocal rank.

    Marked explicitly as: 'ORACLE — theoretical headroom only'.
    """
    # 1. Determine best fixed modality based on MRR
    fixed_metrics: dict[str, AggregateMethodMetrics] = {}
    for m in single_modality_methods:
        if m in case_results_by_method:
            fixed_metrics[m] = aggregate_case_metrics(m, case_results_by_method[m], is_oracle=False)

    if not fixed_metrics:
        raise ValueError("No valid single modality methods found for oracle headroom calculation.")

    best_fixed_name = max(fixed_metrics.keys(), key=lambda m: (fixed_metrics[m].mrr, fixed_metrics[m].top1_accuracy))
    best_fixed = fixed_metrics[best_fixed_name]

    # 2. Oracle per-fault-type
    # Group cases by fault type
    cases_by_fault: dict[str, list[ManifestCase]] = {}
    for c in cases:
        cases_by_fault.setdefault(c.fault, []).append(c)

    best_method_per_fault: dict[str, str] = {}
    for fault, fault_cases in cases_by_fault.items():
        fault_cids = {c.case_id for c in fault_cases}
        best_f_method = best_fixed_name
        best_f_mrr = -1.0
        for m in single_modality_methods:
            m_res = [r for r in case_results_by_method.get(m, []) if r.case_id in fault_cids and r.metrics is not None]
            if m_res:
                f_mrr = sum(r.metrics.mrr for r in m_res) / len(m_res)
                if f_mrr > best_f_mrr:
                    best_f_mrr = f_mrr
                    best_f_method = m
        best_method_per_fault[fault] = best_f_method

    # Assemble oracle per fault case results
    oracle_fault_results: list[MethodRankingResult] = []
    for c in cases:
        chosen_method = best_method_per_fault.get(c.fault, best_fixed_name)
        res_list = case_results_by_method.get(chosen_method, [])
        match = next((r for r in res_list if r.case_id == c.case_id), None)
        if match is not None:
            oracle_fault_results.append(match)

    oracle_fault_agg = aggregate_case_metrics(
        "oracle_per_fault_type",
        oracle_fault_results,
        is_oracle=True,
    )

    # 3. Oracle per-incident: choose best single modality result for each case
    oracle_incident_results: list[MethodRankingResult] = []
    for c in cases:
        best_case_res: MethodRankingResult | None = None
        best_rr = -1.0
        for m in single_modality_methods:
            res_list = case_results_by_method.get(m, [])
            match = next((r for r in res_list if r.case_id == c.case_id and r.metrics is not None), None)
            if match is not None:
                if match.metrics.mrr > best_rr:
                    best_rr = match.metrics.mrr
                    best_case_res = match
        if best_case_res is not None:
            oracle_incident_results.append(best_case_res)

    oracle_incident_agg = aggregate_case_metrics(
        "oracle_per_incident",
        oracle_incident_results,
        is_oracle=True,
    )

    # 4. Compute absolute and relative headroom
    metric_keys = ("top1_accuracy", "top3_accuracy", "top5_accuracy", "mrr", "avg5_accuracy")
    abs_fault: dict[str, float] = {}
    rel_fault: dict[str, float] = {}
    abs_inc: dict[str, float] = {}
    rel_inc: dict[str, float] = {}

    for k in metric_keys:
        bf_val = getattr(best_fixed, k)
        f_val = getattr(oracle_fault_agg, k)
        i_val = getattr(oracle_incident_agg, k)

        diff_f = f_val - bf_val
        diff_i = i_val - bf_val

        abs_fault[k] = diff_f
        abs_inc[k] = diff_i

        rel_fault[k] = (diff_f / bf_val) if bf_val > 1e-9 else 0.0
        rel_inc[k] = (diff_i / bf_val) if bf_val > 1e-9 else 0.0

    return HeadroomAnalysis(
        best_fixed_method=best_fixed_name,
        best_fixed_metrics=best_fixed,
        oracle_per_fault_metrics=oracle_fault_agg,
        oracle_per_incident_metrics=oracle_incident_agg,
        absolute_headroom_fault=abs_fault,
        relative_headroom_fault=rel_fault,
        absolute_headroom_incident=abs_inc,
        relative_headroom_incident=rel_inc,
    )


def run_benchmark(
    manifest: BenchmarkManifest,
    dataset_root: str | Path,
    methods: Sequence[str] = (
        "random",
        "simple_rca",
        "s_comb",
        "trace_elevation",
        "fixed_equal_weight_fusion",
    ),
    window_mode: str = "oracle",
    partition: str | None = None,
) -> BenchmarkExecutionReport:
    """Execute the benchmark across all cases in the manifest.

    Parameters
    ----------
    manifest:
        The BenchmarkManifest defining cases and partitions.
    dataset_root:
        Root path containing the cached RCAEval cases.
    methods:
        List of method names to evaluate.
    window_mode:
        'oracle' (ground-truth inject_time) or 'detected' (first anomaly episode ts).
    partition:
        Optional partition filter ('dev', 'val', 'held_out', 'smoke'). If None, all cases are run.

    Returns
    -------
    BenchmarkExecutionReport
        Complete versioned execution report containing rankings, metrics, and headroom analysis.
    """
    cases_to_run = manifest.get_partition(partition) if partition is not None else manifest.cases
    root_path = Path(dataset_root)

    case_results_by_method: dict[str, list[MethodRankingResult]] = {m: [] for m in methods}
    per_case_report_list: list[dict[str, Any]] = []

    for c in cases_to_run:
        cid = c.case_id
        case = load_rcaeval_case(root_path, cid)
        target = c.root_cause_service

        # 1. Modality availability
        has_metrics = hasattr(case, "metrics") and case.metrics is not None
        has_traces = hasattr(case, "traces") and case.traces is not None
        has_logs = hasattr(case, "logs") and case.logs is not None
        modality_avail = ModalityAvailability(
            has_metrics=has_metrics,
            has_traces=has_traces,
            has_logs=has_logs,
        )

        # 2. Closed candidate universe
        candidate_universe = resolve_candidate_universe(c.system, case=case)

        # 3. Precomputations (anomaly detection, trace dependencies, episodes)
        precomputed: dict[str, Any] = {}
        if has_metrics:
            det_res = detect_metric_anomalies(case)
            precomputed["det_res"] = det_res

            # Trace dependencies for graph
            if has_traces:
                trace_deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
                deps = [td.to_dependency() for td in trace_deps]
            else:
                deps = ()

            graph = build_entity_graph(det_res.metric_names, dependencies=deps)
            precomputed["graph"] = graph

            ep_config = EpisodeConfig(persistence=3, consensus=2)
            ep_evidence = aggregate_entity_episodes(det_res, graph, ep_config)
            precomputed["ep_evidence"] = ep_evidence

        if has_traces:
            trace_lat = extract_trace_latency_evidence(
                case,
                service_aliases={"frontendservice": "frontend"},
                expected_services=candidate_universe,
            )
            precomputed["trace_lat"] = trace_lat

        # 4. Incident window
        window = resolve_incident_window(
            case,
            ep_evidence=precomputed.get("ep_evidence"),
            mode=window_mode,
        )

        case_summary_entry: dict[str, Any] = {
            "case_id": cid,
            "system": c.system,
            "fault": c.fault,
            "target": target,
            "repetition": c.repetition,
            "partition": c.partition,
            "candidate_universe": list(candidate_universe),
            "window": {
                "onset_ts": window.onset_ts,
                "end_ts": window.end_ts,
                "mode": window.mode,
                "source": window.source_description,
            },
            "modalities": asdict(modality_avail),
            "methods": {},
        }

        # 5. Run requested methods
        for m in methods:
            res = run_ranker(
                method_name=m,
                case=case,
                window=window,
                candidate_universe=candidate_universe,
                modality_avail=modality_avail,
                precomputed=precomputed,
            )
            if res.status == "SUCCESS" and res.ranking:
                res.metrics = compute_case_metrics(res.ranking, target)

            case_results_by_method[m].append(res)
            case_summary_entry["methods"][m] = res.to_dict()

        per_case_report_list.append(case_summary_entry)

    # 6. Aggregate method metrics
    method_aggregates: dict[str, AggregateMethodMetrics] = {}
    for m in methods:
        method_aggregates[m] = aggregate_case_metrics(m, case_results_by_method[m], is_oracle=False)

    # 7. Compute Stage 1 Oracle Headroom if single modalities are present
    single_mods = [m for m in ("s_comb", "trace_elevation", "simple_rca_metric", "simple_rca_trace", "simple_rca_log") if m in methods]
    headroom_analysis: HeadroomAnalysis | None = None
    if len(single_mods) >= 2:
        headroom_analysis = compute_oracle_headroom(
            cases_to_run,
            case_results_by_method,
            single_modality_methods=single_mods,
        )

    return BenchmarkExecutionReport(
        schema_version=HARNESS_SCHEMA_VERSION,
        manifest_id=manifest.manifest_id,
        manifest_description=manifest.description,
        window_mode=window_mode,
        total_cases=len(cases_to_run),
        case_ids=tuple(c.case_id for c in cases_to_run),
        evaluated_methods=tuple(methods),
        method_aggregates=method_aggregates,
        headroom_analysis=headroom_analysis,
        case_results=per_case_report_list,
    )


def main() -> None:
    """CLI entrypoint for running evaluations."""
    import argparse
    import os
    import pyarrow.parquet as pq
    from .manifest import create_smoke_re2_ob_rep1_manifest

    parser = argparse.ArgumentParser(description="Digital Detective Evaluation Harness")
    parser.add_argument(
        "--dataset-root",
        default=os.path.expanduser("~/.cache/rcaeval_validation"),
        help="Root path of RCAEval cache directory",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest JSON file",
    )
    parser.add_argument(
        "--smoke-re2-ob",
        action="store_true",
        help="Run standard 30-case RE2-OB repetition-1 smoke test",
    )
    parser.add_argument(
        "--methods",
        default="random,simple_rca,s_comb,trace_elevation,fixed_equal_weight_fusion",
        help="Comma-separated list of methods to run",
    )
    parser.add_argument(
        "--window-mode",
        default="oracle",
        choices=["oracle", "detected"],
        help="Incident window mode",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional partition filter ('dev', 'val', 'held_out', 'smoke')",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save output report JSON",
    )

    args = parser.parse_args()
    root_path = Path(args.dataset_root)

    if args.manifest:
        manifest = BenchmarkManifest.load(args.manifest)
    elif args.smoke_re2_ob:
        cases_file = root_path / "cases.parquet"
        if not cases_file.is_file():
            raise FileNotFoundError(f"cases.parquet not found in {root_path}")
        cases_table = pq.read_table(cases_file)
        manifest = create_smoke_re2_ob_rep1_manifest(cases_table.to_pylist())
    else:
        cases_file = root_path / "cases.parquet"
        if not cases_file.is_file():
            raise FileNotFoundError(f"cases.parquet not found in {root_path}")
        cases_table = pq.read_table(cases_file)
        manifest = create_smoke_re2_ob_rep1_manifest(cases_table.to_pylist())

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    report = run_benchmark(
        manifest=manifest,
        dataset_root=root_path,
        methods=methods,
        window_mode=args.window_mode,
        partition=args.partition,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY ({report.manifest_id})")
    print(f"Total Cases Evaluated: {report.total_cases} | Window Mode: {report.window_mode}")
    print("=" * 60)

    for method, agg in report.method_aggregates.items():
        print(f"\nMethod: {method}")
        print(f"  Top@1: {agg.top1_accuracy:.4f} ({agg.top1_accuracy * agg.evaluated_cases:.1f}/{agg.evaluated_cases})")
        print(f"  Top@3: {agg.top3_accuracy:.4f} ({agg.top3_accuracy * agg.evaluated_cases:.1f}/{agg.evaluated_cases})")
        print(f"  Top@5: {agg.top5_accuracy:.4f} ({agg.top5_accuracy * agg.evaluated_cases:.1f}/{agg.evaluated_cases})")
        print(f"  MRR:   {agg.mrr:.4f}")
        print(f"  AC@1:  {agg.ac1_accuracy:.4f}")
        print(f"  AC@2:  {agg.ac2_accuracy:.4f}")
        print(f"  AC@3:  {agg.ac3_accuracy:.4f}")
        print(f"  AC@4:  {agg.ac4_accuracy:.4f}")
        print(f"  AC@5:  {agg.ac5_accuracy:.4f}")
        print(f"  Avg@3: {agg.avg3_accuracy:.4f}")
        print(f"  Avg@5: {agg.avg5_accuracy:.4f}")
        print(f"  Runtime (mean): {agg.mean_runtime_sec:.4f}s")
        print(f"  Abstention Rate:        {agg.abstention_rate if isinstance(agg.abstention_rate, str) else f'{agg.abstention_rate:.4f}'}")
        print(f"  False Remediation Rate: {agg.false_remediation_rate if isinstance(agg.false_remediation_rate, str) else f'{agg.false_remediation_rate:.4f}'}")
        print(f"  Recovery Success Rate:  {agg.recovery_success_rate if isinstance(agg.recovery_success_rate, str) else f'{agg.recovery_success_rate:.4f}'}")
        print(f"  Regression Rate:        {agg.regression_rate if isinstance(agg.regression_rate, str) else f'{agg.regression_rate:.4f}'}")
        print(f"  Query Efficiency:       {agg.query_efficiency if isinstance(agg.query_efficiency, str) else f'{agg.query_efficiency:.4f}'}")

    if report.headroom_analysis:
        ha = report.headroom_analysis
        print("\n" + "=" * 60)
        print("STAGE 1 HEADROOM ANALYSIS (ORACLES VS BEST FIXED MODALITY)")
        print(f"Disclaimer: {ha.disclaimer}")
        print("=" * 60)
        print(f"Best Fixed Modality: {ha.best_fixed_method}")
        print(f"  MRR: {ha.best_fixed_metrics.mrr:.4f} | Top@1: {ha.best_fixed_metrics.top1_accuracy:.4f}")
        print("Oracle Per-Fault-Type:")
        print(f"  MRR: {ha.oracle_per_fault_metrics.mrr:.4f} | Abs Headroom: {ha.absolute_headroom_fault['mrr']:+.4f} | Rel: {ha.relative_headroom_fault['mrr']:+.2%}")
        print(f"  Top@1: {ha.oracle_per_fault_metrics.top1_accuracy:.4f} | Abs Headroom: {ha.absolute_headroom_fault['top1_accuracy']:+.4f}")
        print("Oracle Per-Incident:")
        print(f"  MRR: {ha.oracle_per_incident_metrics.mrr:.4f} | Abs Headroom: {ha.absolute_headroom_incident['mrr']:+.4f} | Rel: {ha.relative_headroom_incident['mrr']:+.2%}")
        print(f"  Top@1: {ha.oracle_per_incident_metrics.top1_accuracy:.4f} | Abs Headroom: {ha.absolute_headroom_incident['top1_accuracy']:+.4f}")

    if args.output:
        report.save(args.output)
        print(f"\nSaved full report to: {args.output}")


if __name__ == "__main__":
    main()
