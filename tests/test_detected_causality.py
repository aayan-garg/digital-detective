"""Tests for detected incident-window causal cutoff and evidence isolation.

Verifies:
1. Detected window end_ts == detected_onset.
2. Detected mode never falls back to inject_time.
3. Later anomalies after detected_onset do NOT appear in downstream RCA evidence.
4. A later anomaly does not alter detected_onset, pre-cutoff evidence, or RCA decisions.
5. No-detection remains an explicit no-detection result.
6. Oracle mode still uses inject_time.
"""

from __future__ import annotations

import math
from typing import Any
import unittest

import pyarrow as pa

from digital_detective.anomaly import (
    detect_metric_anomalies,
    truncate_metric_anomaly_result,
)
from digital_detective.episodes import (
    EpisodeConfig,
    aggregate_entity_episodes,
    truncate_entity_episodes,
)
from digital_detective.rca import rank_with_s_comb
from digital_detective.telemetry import (
    CaseMetadata,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence, TraceLatencyResult
from eval.baselines.simple_rca import rank_with_simple_rca
from eval.models import IncidentWindow
from eval.windows import resolve_incident_window


def _build_synthetic_case(
    case_id: str,
    inject_time: int = 40,
    include_later_anomaly: bool = False,
) -> TelemetryCase:
    """Build synthetic telemetry case with known, reproducible anomaly timing.

    Timestamps: 0 to 150.
    Entity 'svcA': metrics 'svcA_cpu', 'svcA_mem'
    Entity 'svcB': metrics 'svcB_cpu', 'svcB_mem'

    Normal background: ~10.0 (with slight variation for non-zero variance).
    svcA anomaly: t=51..56 (both cpu and mem spike to 50.0).
                  With persistence=3, consensus=2, detection triggers at t=53.
    svcB anomaly (if include_later_anomaly): t=100..120 (both spike to 5000.0).
    """
    n = 151
    times = list(range(n))

    # Base signals with minor variation so rolling std > 0
    base = [10.0 + 0.1 * (i % 3 - 1) for i in range(n)]

    svcA_cpu = list(base)
    svcA_mem = list(base)
    svcB_cpu = list(base)
    svcB_mem = list(base)

    # First anomaly episode on svcA: t=51..60
    for t in range(51, 61):
        svcA_cpu[t] = 50.0 * (t - 50)
        svcA_mem[t] = 50.0 * (t - 50)

    # Strong later anomaly on svcB: t=100..115
    if include_later_anomaly:
        for t in range(100, 116):
            svcB_cpu[t] = 500.0 * (t - 99)
            svcB_mem[t] = 500.0 * (t - 99)

    table = pa.table({
        "time": times,
        "svcA_cpu": svcA_cpu,
        "svcA_mem": svcA_mem,
        "svcB_cpu": svcB_cpu,
        "svcB_mem": svcB_mem,
    })

    class DummyGT:
        values = {
            "root_cause_service": "svcA",
            "fault": "cpu",
            "inject_time": inject_time,
        }

    return TelemetryCase(
        metadata=CaseMetadata(case_id=case_id),
        ground_truth=DummyGT(),
        metrics=TelemetryModality(
            raw_data=table,
            provenance=ModalityProvenance(
                source_dataset="synthetic",
                source_case=case_id,
                original_format="parquet",
                original_field_names=("time", "svcA_cpu", "svcA_mem", "svcB_cpu", "svcB_mem"),
                timestamp_field="time",
            ),
        ),
    )


class TestDetectedCausality(unittest.TestCase):
    """Test suite proving causal bounding of detected-mode incident windows and RCA evidence."""

    def setUp(self) -> None:
        self.ep_config = EpisodeConfig(persistence=3, consensus=2)
        self.universe = ("svcA", "svcB")

    def test_detected_window_end_ts_equals_onset(self) -> None:
        """Requirement 1: Detected window end_ts must equal detected_onset."""
        case = _build_synthetic_case("case_test1", inject_time=40)
        det_res = detect_metric_anomalies(case, window_size=20, min_warmup=20)
        graph = build_entity_graph(det_res.metric_names)
        ep_evidence = aggregate_entity_episodes(det_res, graph, self.ep_config)

        window = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")

        self.assertTrue(window.has_detected_window)
        self.assertEqual(window.mode, "detected")
        self.assertEqual(window.onset_ts, 53)
        # Crucial check: end_ts must equal detected_onset (53), NOT final telemetry timestamp (150)
        self.assertEqual(window.end_ts, 53)
        self.assertEqual(window.end_ts, window.onset_ts)

    def test_detected_mode_never_falls_back_to_inject_time(self) -> None:
        """Requirement 2: Detected mode must never use inject_time."""
        case = _build_synthetic_case("case_test2", inject_time=999)  # inject_time far in future
        det_res = detect_metric_anomalies(case, window_size=20, min_warmup=20)
        graph = build_entity_graph(det_res.metric_names)
        ep_evidence = aggregate_entity_episodes(det_res, graph, self.ep_config)

        window = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")

        self.assertNotEqual(window.onset_ts, 999)
        self.assertNotEqual(window.end_ts, 999)
        self.assertEqual(window.onset_ts, 53)
        self.assertEqual(window.end_ts, 53)
        self.assertNotIn("oracle", window.source_description)

    def test_later_anomaly_does_not_appear_in_downstream_rca_evidence(self) -> None:
        """Requirement 3: Later anomaly after detected_onset is excluded from RCA evidence."""
        # Case with massive later anomaly on svcB at t=100..120
        case_with_later = _build_synthetic_case("case_with_later", include_later_anomaly=True)
        det_res = detect_metric_anomalies(case_with_later, window_size=20, min_warmup=20)
        graph = build_entity_graph(det_res.metric_names)
        ep_evidence = aggregate_entity_episodes(det_res, graph, self.ep_config)

        window = resolve_incident_window(case_with_later, ep_evidence=ep_evidence, mode="detected")
        detected_onset = window.onset_ts
        self.assertEqual(detected_onset, 53)

        # Apply causal cutoff as evaluation harness does
        det_res_causal = truncate_metric_anomaly_result(det_res, max_timestamp=detected_onset)
        ep_evidence_causal = aggregate_entity_episodes(det_res_causal, graph, self.ep_config)

        # 1. Check timestamps do not exceed detected_onset
        self.assertLessEqual(max(det_res_causal.timestamps), detected_onset)
        self.assertEqual(max(det_res_causal.timestamps), 53)
        for ev in ep_evidence_causal.values():
            if ev.timestamps:
                self.assertLessEqual(max(ev.timestamps), detected_onset)

        # 2. svcB later anomaly at t=100 must NOT exist in causal evidence
        self.assertFalse(ep_evidence_causal["svcB"].has_episode)
        self.assertEqual(len(ep_evidence_causal["svcB"].episodes), 0)
        self.assertEqual(ep_evidence_causal["svcB"].peak_active_metrics, 0)
        self.assertFalse(any(det_res_causal.anomalies["svcB_cpu"]))
        self.assertFalse(any(det_res_causal.anomalies["svcB_mem"]))

        # 3. Downstream S_comb RCA must rank svcA as #1 and not be misled by svcB
        ranking = rank_with_s_comb(
            det_res_causal,
            ep_evidence_causal,
            graph=graph,
            candidate_universe=self.universe,
        )
        self.assertEqual(ranking[0].entity, "svcA")
        self.assertGreater(ranking[0].score, 0.0)
        self.assertEqual(ranking[1].entity, "svcB")
        self.assertEqual(ranking[1].score, 0.0)

    def test_later_episode_does_not_alter_detection_decision(self) -> None:
        """Requirement 4: Later episode does not alter detected_onset, pre-cutoff evidence, or RCA decisions."""
        case_early_only = _build_synthetic_case("early_only", include_later_anomaly=False)
        case_with_later = _build_synthetic_case("with_later", include_later_anomaly=True)

        # Early-only case
        det1 = detect_metric_anomalies(case_early_only, window_size=20, min_warmup=20)
        graph1 = build_entity_graph(det1.metric_names)
        ep1 = aggregate_entity_episodes(det1, graph1, self.ep_config)
        w1 = resolve_incident_window(case_early_only, ep_evidence=ep1, mode="detected")

        det1_causal = truncate_metric_anomaly_result(det1, max_timestamp=w1.onset_ts)
        ep1_causal = aggregate_entity_episodes(det1_causal, graph1, self.ep_config)
        rank1 = rank_with_s_comb(det1_causal, ep1_causal, graph=graph1, candidate_universe=self.universe)

        # With-later case
        det2 = detect_metric_anomalies(case_with_later, window_size=20, min_warmup=20)
        graph2 = build_entity_graph(det2.metric_names)
        ep2 = aggregate_entity_episodes(det2, graph2, self.ep_config)
        w2 = resolve_incident_window(case_with_later, ep_evidence=ep2, mode="detected")

        det2_causal = truncate_metric_anomaly_result(det2, max_timestamp=w2.onset_ts)
        ep2_causal = aggregate_entity_episodes(det2_causal, graph2, self.ep_config)
        rank2 = rank_with_s_comb(det2_causal, ep2_causal, graph=graph2, candidate_universe=self.universe)

        # 1. Detected onset is identical
        self.assertEqual(w1.onset_ts, w2.onset_ts)
        self.assertEqual(w1.end_ts, w2.end_ts)

        # 2. Pre-cutoff episode evidence on svcA is strictly identical
        self.assertEqual(ep1_causal["svcA"].first_episode_start_ts, ep2_causal["svcA"].first_episode_start_ts)
        self.assertEqual(ep1_causal["svcA"].peak_active_metrics, ep2_causal["svcA"].peak_active_metrics)
        self.assertEqual(ep1_causal["svcA"].all_contributing_metrics, ep2_causal["svcA"].all_contributing_metrics)

        # 3. RCA decisions and scores are strictly identical
        self.assertEqual(len(rank1), len(rank2))
        for r1, r2 in zip(rank1, rank2):
            self.assertEqual(r1.entity, r2.entity)
            self.assertAlmostEqual(r1.score, r2.score, places=9)

    def test_no_detection_behavior(self) -> None:
        """Requirement 5: Flat telemetry produces an explicit no-detection result without inject_time."""
        n = 100
        table = pa.table({
            "time": list(range(n)),
            "svcA_cpu": [10.0 + 0.01 * (i % 2) for i in range(n)],
            "svcA_mem": [10.0 + 0.01 * (i % 2) for i in range(n)],
        })
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="flat_case"),
            ground_truth=None,
            metrics=TelemetryModality(
                raw_data=table,
                provenance=ModalityProvenance(
                    source_dataset="test",
                    source_case="flat_case",
                    original_format="parquet",
                    original_field_names=("time", "svcA_cpu", "svcA_mem"),
                    timestamp_field="time",
                ),
            ),
        )
        det_res = detect_metric_anomalies(case, window_size=20, min_warmup=20)
        graph = build_entity_graph(det_res.metric_names)
        ep_evidence = aggregate_entity_episodes(det_res, graph, self.ep_config)

        window = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")
        self.assertFalse(window.has_detected_window)
        self.assertIsNone(window.onset_ts)
        self.assertIn("no_detected_window", window.source_description)

    def test_oracle_mode_still_uses_inject_time(self) -> None:
        """Requirement 6: Oracle mode continues to use ground-truth inject_time."""
        case = _build_synthetic_case("case_oracle", inject_time=42)
        window = resolve_incident_window(case, mode="oracle")

        self.assertTrue(window.has_detected_window)
        self.assertEqual(window.mode, "oracle")
        self.assertEqual(window.onset_ts, 42)
        self.assertEqual(window.end_ts, 150)
        self.assertIn("oracle:inject_time=42", window.source_description)

    def test_truncate_entity_episodes_direct(self) -> None:
        """Direct test for truncate_entity_episodes helper function."""
        case = _build_synthetic_case("case_direct", include_later_anomaly=True)
        det_res = detect_metric_anomalies(case, window_size=20, min_warmup=20)
        graph = build_entity_graph(det_res.metric_names)
        ep_evidence = aggregate_entity_episodes(det_res, graph, self.ep_config)

        # Truncate directly at t=53
        truncated = truncate_entity_episodes(ep_evidence, max_timestamp=53)

        self.assertTrue(truncated["svcA"].has_episode)
        self.assertEqual(truncated["svcA"].first_episode_start_ts, 53)
        self.assertFalse(truncated["svcB"].has_episode)
        self.assertEqual(len(truncated["svcA"].timestamps), 54)  # 0..53

    def test_post_onset_trace_anomaly_cannot_alter_detected_ranking(self) -> None:
        """Invariant 1: A large trace anomaly strictly after detected_onset cannot alter detected RCA."""
        # Case A: normal traces up to onset (53)
        case_a = _build_synthetic_case("case_a")
        case_a.traces = _build_synthetic_traces(include_later_spike=False)

        # Case B: same case, but with an enormous trace latency spike on svcB at t=100
        case_b = _build_synthetic_case("case_b")
        case_b.traces = _build_synthetic_traces(include_later_spike=True)

        detected_onset = 53

        # Unbounded extraction would be altered by the future spike:
        trace_unbounded_b = extract_trace_latency_evidence(case_b, expected_services=self.universe)
        rank_unbounded_b = rank_with_trace_elevation(trace_unbounded_b, candidate_universe=self.universe)
        # Without causal bounding, the t=100 spike would incorrectly elevate svcB to rank 1
        self.assertEqual(rank_unbounded_b[0].entity, "svcB")

        # Causal extraction bounded to detected_onset:
        trace_causal_a = extract_trace_latency_evidence(case_a, expected_services=self.universe, max_timestamp=detected_onset)
        trace_causal_b = extract_trace_latency_evidence(case_b, expected_services=self.universe, max_timestamp=detected_onset)

        # 1. Check max span timestamp is bounded
        self.assertIsNotNone(trace_causal_a.max_span_timestamp)
        self.assertIsNotNone(trace_causal_b.max_span_timestamp)
        self.assertLessEqual(trace_causal_a.max_span_timestamp, detected_onset)
        self.assertLessEqual(trace_causal_b.max_span_timestamp, detected_onset)

        # 2. Causally bounded evidence is identical
        self.assertEqual(trace_causal_a.total_spans, trace_causal_b.total_spans)
        self.assertEqual(len(trace_causal_a.edges), len(trace_causal_b.edges))

        # 3. Trace elevation ranking and scores are strictly identical
        rank_a = rank_with_trace_elevation(trace_causal_a, candidate_universe=self.universe)
        rank_b = rank_with_trace_elevation(trace_causal_b, candidate_universe=self.universe)
        self.assertEqual(len(rank_a), len(rank_b))
        for ra, rb in zip(rank_a, rank_b):
            self.assertEqual(ra.entity, rb.entity)
            self.assertAlmostEqual(ra.score, rb.score, places=9)

        # 4. In causal mode, svcA is top-ranked, not svcB
        self.assertEqual(rank_a[0].entity, "svcA")
        self.assertEqual(rank_b[0].entity, "svcA")

    def test_post_onset_log_anomaly_cannot_alter_detected_ranking(self) -> None:
        """Invariant 2: A large log anomaly strictly after detected_onset cannot alter detected RCA."""
        case_a = _build_synthetic_case("case_a")
        case_a.logs = _build_synthetic_logs(include_later_errors=False)

        case_b = _build_synthetic_case("case_b")
        case_b.logs = _build_synthetic_logs(include_later_errors=True)

        detected_onset = 53
        window = IncidentWindow(onset_ts=detected_onset, end_ts=detected_onset, mode="detected")

        # Run SimpleRCA log-only
        rank_a = rank_with_simple_rca(case_a, window, self.universe, enabled_modalities=("logs",))
        rank_b = rank_with_simple_rca(case_b, window, self.universe, enabled_modalities=("logs",))

        # 1. Both rankings and scores must be strictly identical
        self.assertEqual(len(rank_a), len(rank_b))
        for ra, rb in zip(rank_a, rank_b):
            self.assertEqual(ra.entity, rb.entity)
            self.assertEqual(ra.score, rb.score)

        # 2. svcA received the error alert at onset (53), svcB received 0 alerts
        self.assertEqual(rank_a[0].entity, "svcA")
        self.assertEqual(rank_a[0].score, 1.0)
        self.assertEqual(rank_a[1].entity, "svcB")
        self.assertEqual(rank_a[1].score, 0.0)

    def test_arbitrary_post_onset_telemetry_leaves_detected_mode_identical(self) -> None:
        """Invariant 3: Adding arbitrary post-onset telemetry produces identical onset, evidence, rankings, and scores."""
        # Case 1: Early-only metrics, traces, logs
        case1 = _build_synthetic_case("case_multimodal_1", include_later_anomaly=False)
        case1.traces = _build_synthetic_traces(include_later_spike=False)
        case1.logs = _build_synthetic_logs(include_later_errors=False)

        # Case 2: Same case, with arbitrary post-onset spikes across metrics, traces, and logs (t >= 100)
        case2 = _build_synthetic_case("case_multimodal_2", include_later_anomaly=True)
        case2.traces = _build_synthetic_traces(include_later_spike=True)
        case2.logs = _build_synthetic_logs(include_later_errors=True)

        # Resolve windows
        det1 = detect_metric_anomalies(case1, window_size=20, min_warmup=20)
        graph1 = build_entity_graph(det1.metric_names)
        ep1 = aggregate_entity_episodes(det1, graph1, self.ep_config)
        w1 = resolve_incident_window(case1, ep_evidence=ep1, mode="detected")

        det2 = detect_metric_anomalies(case2, window_size=20, min_warmup=20)
        graph2 = build_entity_graph(det2.metric_names)
        ep2 = aggregate_entity_episodes(det2, graph2, self.ep_config)
        w2 = resolve_incident_window(case2, ep_evidence=ep2, mode="detected")

        # 1. Detected onset is identical
        self.assertEqual(w1.onset_ts, w2.onset_ts)
        self.assertEqual(w1.end_ts, w2.end_ts)
        self.assertEqual(w1.onset_ts, 53)

        onset = w1.onset_ts
        assert onset is not None

        # 2. Causal metric evidence is identical
        det1_causal = truncate_metric_anomaly_result(det1, max_timestamp=onset)
        det2_causal = truncate_metric_anomaly_result(det2, max_timestamp=onset)
        ep1_causal = aggregate_entity_episodes(det1_causal, graph1, self.ep_config)
        ep2_causal = aggregate_entity_episodes(det2_causal, graph2, self.ep_config)
        self.assertEqual(max(det1_causal.timestamps), max(det2_causal.timestamps))
        self.assertLessEqual(max(det1_causal.timestamps), onset)

        # 3. Causal trace evidence is identical
        t_causal_1 = extract_trace_latency_evidence(case1, expected_services=self.universe, max_timestamp=onset)
        t_causal_2 = extract_trace_latency_evidence(case2, expected_services=self.universe, max_timestamp=onset)
        self.assertEqual(t_causal_1.total_spans, t_causal_2.total_spans)
        self.assertEqual(t_causal_1.retained_spans, t_causal_2.retained_spans)
        self.assertIsNotNone(t_causal_1.max_span_timestamp)
        self.assertIsNotNone(t_causal_1.max_span_end_timestamp)
        self.assertLessEqual(t_causal_1.max_span_timestamp, onset)
        self.assertLessEqual(t_causal_2.max_span_timestamp, onset)
        self.assertLessEqual(t_causal_1.max_span_end_timestamp, onset)
        self.assertLessEqual(t_causal_2.max_span_end_timestamp, onset)
        self.assertEqual(t_causal_1.excluded_spans, 0)
        self.assertEqual(t_causal_2.excluded_spans, 2)  # p_late and c_late excluded

        # 4. S_comb rankings and scores are identical
        s_rank_1 = rank_with_s_comb(det1_causal, ep1_causal, graph=graph1, candidate_universe=self.universe)
        s_rank_2 = rank_with_s_comb(det2_causal, ep2_causal, graph=graph2, candidate_universe=self.universe)
        for r1, r2 in zip(s_rank_1, s_rank_2):
            self.assertEqual(r1.entity, r2.entity)
            self.assertAlmostEqual(r1.score, r2.score, places=9)

        # 5. Trace elevation rankings and scores are identical
        t_rank_1 = rank_with_trace_elevation(t_causal_1, candidate_universe=self.universe)
        t_rank_2 = rank_with_trace_elevation(t_causal_2, candidate_universe=self.universe)
        for r1, r2 in zip(t_rank_1, t_rank_2):
            self.assertEqual(r1.entity, r2.entity)
            self.assertAlmostEqual(r1.score, r2.score, places=9)

        # 6. SimpleRCA multimodal rankings and scores are identical
        sim_rank_1 = rank_with_simple_rca(case1, w1, self.universe)
        sim_rank_2 = rank_with_simple_rca(case2, w2, self.universe)
        for r1, r2 in zip(sim_rank_1, sim_rank_2):
            self.assertEqual(r1.entity, r2.entity)
            self.assertEqual(r1.score, r2.score)

    def test_crossing_trace_span_excluded_from_detected_mode(self) -> None:
        """Issue 1 requirement: Span starting before detected_onset but ending after detected_onset is excluded."""
        # Case A: Normal traces before onset (53)
        case_a = _build_synthetic_case("case_a")
        case_a.traces = _build_synthetic_traces(include_crossing_span=False)

        # Case B: Same case, with a span that starts before onset (t=50) but ends after onset (t=100) with large duration
        case_b = _build_synthetic_case("case_b")
        case_b.traces = _build_synthetic_traces(include_crossing_span=True)

        detected_onset = 53

        # Without causal completion cutoff, a filter on startTime <= 53 would include the crossing span:
        # p_cross/c_cross have start_time=50 <= 53, duration=50 -> end_time=100.
        # But causal completion requires span_end <= detected_onset.
        trace_causal_a = extract_trace_latency_evidence(
            case_a, expected_services=self.universe, max_timestamp=detected_onset
        )
        trace_causal_b = extract_trace_latency_evidence(
            case_b, expected_services=self.universe, max_timestamp=detected_onset
        )

        # 1. Crossing spans are excluded
        self.assertEqual(trace_causal_a.total_spans, trace_causal_b.total_spans)
        self.assertEqual(trace_causal_a.retained_spans, trace_causal_b.retained_spans)
        self.assertEqual(trace_causal_b.excluded_spans, 2)  # p_cross and c_cross excluded
        self.assertEqual(trace_causal_a.excluded_spans, 0)

        # 2. Maximum retained end timestamp is <= detected_onset
        self.assertIsNotNone(trace_causal_b.max_span_end_timestamp)
        self.assertLessEqual(trace_causal_b.max_span_end_timestamp, detected_onset)
        self.assertLessEqual(trace_causal_b.max_span_timestamp, detected_onset)

        # 3. Adding or removing this crossing span does NOT change detected-mode trace RCA ranking or scores
        rank_a = rank_with_trace_elevation(trace_causal_a, candidate_universe=self.universe)
        rank_b = rank_with_trace_elevation(trace_causal_b, candidate_universe=self.universe)
        self.assertEqual(len(rank_a), len(rank_b))
        for ra, rb in zip(rank_a, rank_b):
            self.assertEqual(ra.entity, rb.entity)
            self.assertAlmostEqual(ra.score, rb.score, places=9)

    def test_future_topology_dependency_excluded_from_detected_graph(self) -> None:
        """Issue 2 requirement: Dependency first observed after detected_onset must be absent from detected graph."""
        detected_onset = 53

        # Case 1: Before detected_onset, only frontend -> svcA is observed
        case1 = _build_synthetic_case("case_topo_causal_1")
        table1 = pa.table({
            "spanID": ["p1", "c1"],
            "parentSpanID": [None, "p1"],
            "traceID": ["t1", "t1"],
            "serviceName": ["frontend", "svcA"],
            "operationName": ["op", "op"],
            "startTime": [10, 11],
            "duration": [2, 1],
        })
        case1.traces = TelemetryModality(
            raw_data=table1,
            provenance=ModalityProvenance(
                source_dataset="synthetic",
                source_case="trace_case",
                original_format="parquet",
                original_field_names=tuple(table1.column_names),
                timestamp_field="startTime",
            ),
        )

        # Case 2: Same, but strictly AFTER detected_onset (t=100), a new dependency appears: frontend -> svcB
        case2 = _build_synthetic_case("case_topo_causal_2")
        table2 = pa.table({
            "spanID": ["p1", "c1", "p_late", "c_late"],
            "parentSpanID": [None, "p1", None, "p_late"],
            "traceID": ["t1", "t1", "t2", "t2"],
            "serviceName": ["frontend", "svcA", "frontend", "svcB"],
            "operationName": ["op", "op", "op", "op"],
            "startTime": [10, 11, 100, 101],
            "duration": [2, 1, 2, 1],
        })
        case2.traces = TelemetryModality(
            raw_data=table2,
            provenance=ModalityProvenance(
                source_dataset="synthetic",
                source_case="trace_case",
                original_format="parquet",
                original_field_names=tuple(table2.column_names),
                timestamp_field="startTime",
            ),
        )

        # In oracle / unbounded mode, case2 observes both edges:
        unbounded_deps2 = extract_trace_dependencies(case2)
        unbounded_edges2 = {(d.source, d.target) for d in unbounded_deps2}
        self.assertIn(("frontend", "svcB"), unbounded_edges2)

        # In detected mode (causally bounded to detected_onset = 53):
        causal_deps1 = extract_trace_dependencies(case1, max_timestamp=detected_onset)
        causal_deps2 = extract_trace_dependencies(case2, max_timestamp=detected_onset)

        # 1. frontend -> svcB is strictly absent from the detected-mode graph
        causal_edges1 = {(d.source, d.target) for d in causal_deps1}
        causal_edges2 = {(d.source, d.target) for d in causal_deps2}
        self.assertEqual(causal_edges1, {("frontend", "svcA")})
        self.assertEqual(causal_edges2, {("frontend", "svcA")})
        self.assertNotIn(("frontend", "svcB"), causal_edges2)

        # Build detected-mode graphs
        det1 = detect_metric_anomalies(case1, window_size=20, min_warmup=20)
        det2 = detect_metric_anomalies(case2, window_size=20, min_warmup=20)
        det1_causal = truncate_metric_anomaly_result(det1, max_timestamp=detected_onset)
        det2_causal = truncate_metric_anomaly_result(det2, max_timestamp=detected_onset)

        graph1 = build_entity_graph(det1_causal.metric_names, dependencies=[d.to_dependency() for d in causal_deps1])
        graph2 = build_entity_graph(det2_causal.metric_names, dependencies=[d.to_dependency() for d in causal_deps2])

        self.assertNotIn(
            ("frontend", "svcB"),
            {(dep.source, dep.target) for dep in graph2.dependencies},
        )

        # Run detected-mode RCA
        ep1 = aggregate_entity_episodes(det1_causal, graph1, self.ep_config)
        ep2 = aggregate_entity_episodes(det2_causal, graph2, self.ep_config)

        rank1 = rank_with_s_comb(det1_causal, ep1, graph=graph1, candidate_universe=self.universe)
        rank2 = rank_with_s_comb(det2_causal, ep2, graph=graph2, candidate_universe=self.universe)

        # 2. RCA ranking is unchanged compared with the case without the future dependency
        self.assertEqual(len(rank1), len(rank2))
        for r1, r2 in zip(rank1, rank2):
            self.assertEqual(r1.entity, r2.entity)
            # 3. No score changes are attributable to the future edge
            self.assertAlmostEqual(r1.score, r2.score, places=9)

    def test_static_topology_unchanged_by_future_telemetry(self) -> None:
        """Invariant 4: Static topology remains identical when future telemetry changes."""
        case_a = _build_synthetic_case("case_topo_a")
        case_a.traces = _build_synthetic_traces(include_later_spike=False)

        case_b = _build_synthetic_case("case_topo_b")
        case_b.traces = _build_synthetic_traces(include_later_spike=True)

        deps_a = extract_trace_dependencies(case_a)
        deps_b = extract_trace_dependencies(case_b)

        # Both observe identical caller -> callee edges: frontend -> svcA and frontend -> svcB
        edges_a = {(d.source, d.target) for d in deps_a}
        edges_b = {(d.source, d.target) for d in deps_b}
        self.assertEqual(edges_a, edges_b)
        self.assertEqual(edges_a, {("frontend", "svcA"), ("frontend", "svcB")})

    def test_no_detection_behavior_across_modalities(self) -> None:
        """Invariant 6: Explicit no-detection produces 0 evidence across all modalities."""
        n = 100
        table = pa.table({
            "time": list(range(n)),
            "svcA_cpu": [10.0] * n,
            "svcB_cpu": [10.0] * n,
        })
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="flat_multimodal"),
            metrics=TelemetryModality(
                raw_data=table,
                provenance=ModalityProvenance(
                    source_dataset="test",
                    source_case="flat",
                    original_format="parquet",
                    original_field_names=("time", "svcA_cpu", "svcB_cpu"),
                    timestamp_field="time",
                ),
            ),
            traces=_build_synthetic_traces(include_later_spike=False),
            logs=_build_synthetic_logs(include_later_errors=False),
        )
        det_res = detect_metric_anomalies(case, window_size=20, min_warmup=20)
        graph = build_entity_graph(det_res.metric_names)
        ep_evidence = aggregate_entity_episodes(det_res, graph, self.ep_config)
        window = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")

        self.assertFalse(window.has_detected_window)
        self.assertIsNone(window.onset_ts)

        # SimpleRCA handles no-detection with 0 alerts across all modalities
        sim_rank = rank_with_simple_rca(case, window, self.universe)
        for r in sim_rank:
            self.assertEqual(r.score, 0.0)


def _build_synthetic_traces(
    include_later_spike: bool = False,
    include_crossing_span: bool = False,
) -> TelemetryModality:
    span_ids = ["p1", "c1", "p2", "c2", "p3", "c3"]
    parent_ids = [None, "p1", None, "p2", None, "p3"]
    trace_ids = ["t1", "t1", "t2", "t2", "t3", "t3"]
    service_names = ["frontend", "svcA", "frontend", "svcB", "frontend", "svcA"]
    operation_names = ["op", "op", "op", "op", "op", "op"]
    start_times = [10, 11, 20, 21, 51, 51]
    durations = [2, 1, 2, 1, 1, 1]

    if include_crossing_span:
        # Span starting before detected_onset (53) but completing after detected_onset with large duration
        span_ids.extend(["p_cross", "c_cross"])
        parent_ids.extend([None, "p_cross"])
        trace_ids.extend(["t_cross", "t_cross"])
        service_names.extend(["frontend", "svcB"])
        operation_names.extend(["op", "op"])
        start_times.extend([50, 50])
        durations.extend([50, 50])  # ends at 100 > 53

    if include_later_spike:
        # Enormous spike on svcB strictly AFTER detected onset (t=100)
        span_ids.extend(["p_late", "c_late"])
        parent_ids.extend([None, "p_late"])
        trace_ids.extend(["t_late", "t_late"])
        service_names.extend(["frontend", "svcB"])
        operation_names.extend(["op", "op"])
        start_times.extend([100, 101])
        durations.extend([100000, 95000])

    table = pa.table({
        "spanID": span_ids,
        "parentSpanID": parent_ids,
        "traceID": trace_ids,
        "serviceName": service_names,
        "operationName": operation_names,
        "startTime": start_times,
        "duration": durations,
    })
    return TelemetryModality(
        raw_data=table,
        provenance=ModalityProvenance(
            source_dataset="synthetic",
            source_case="trace_case",
            original_format="parquet",
            original_field_names=tuple(table.column_names),
            timestamp_field="startTime",
        ),
    )


def _build_synthetic_logs(include_later_errors: bool = False) -> TelemetryModality:
    timestamps = [10, 20, 53]
    containers = ["frontend", "svcB", "svcA"]
    messages = [
        "info: started frontend",
        "info: heart beat svcB",
        "error: svcA failed connection",  # at detected_onset (t=53)
    ]
    if include_later_errors:
        # Massive errors on svcB strictly AFTER detected onset (t=100)
        for i in range(10):
            timestamps.append(100 + i)
            containers.append("svcB")
            messages.append(f"fatal error: svcB crashed {i}")

    table = pa.table({
        "timestamp": timestamps,
        "container_name": containers,
        "message": messages,
    })
    return TelemetryModality(
        raw_data=table,
        provenance=ModalityProvenance(
            source_dataset="synthetic",
            source_case="log_case",
            original_format="parquet",
            original_field_names=tuple(table.column_names),
            timestamp_field="timestamp",
        ),
    )


if __name__ == "__main__":
    unittest.main()
