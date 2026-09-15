"""Unit tests for observational trace attribution layer."""

from __future__ import annotations

import unittest

from digital_detective.episodes import (
    EntityEpisodeEvidence,
    EpisodeConfig,
    aggregate_entity_episodes,
)
from digital_detective.rca import (
    RCA_A_ANOMALY_ONLY,
    RCAConfig,
    rank_root_cause_entities,
)
from digital_detective.telemetry import (
    CaseMetadata,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.topology import (
    build_entity_graph,
    extract_trace_dependencies,
)
from digital_detective.traces import (
    DistributionSummary,
    EdgeLatencyEvidence,
    TraceLatencyResult,
    extract_trace_latency_evidence,
)
from digital_detective.trace_attribution import (
    EdgeAttribution,
    EntityTraceEvidence,
    TraceAttributionResult,
    compute_caller_wait_concentration,
    compute_edge_elevation,
    compute_terminal_factor,
    compute_trace_attribution,
)


class TestTraceAttributionFunctions(unittest.TestCase):
    """Test pure functions for edge elevation, caller concentration, and terminal factor."""

    def test_edge_elevation_stable(self) -> None:
        # Stable edge: P10=2.0, P90=2.5 -> (2.5 - 2.0)/2.5 = 0.5 / 2.5 = 0.20
        elev = compute_edge_elevation(p90=2.5, p10=2.0)
        self.assertAlmostEqual(elev, 0.20)

    def test_edge_elevation_elevated(self) -> None:
        # Elevated delay edge: P10=2.0, P90=200.0 -> (200 - 2)/200 = 198/200 = 0.99
        elev = compute_edge_elevation(p90=200.0, p10=2.0)
        self.assertAlmostEqual(elev, 0.99)

    def test_edge_elevation_zero_or_inverted(self) -> None:
        # P90 <= 0
        self.assertEqual(compute_edge_elevation(p90=0.0, p10=0.0), 0.0)
        self.assertEqual(compute_edge_elevation(p90=-10.0, p10=-20.0), 0.0)
        # P10 >= P90
        self.assertEqual(compute_edge_elevation(p90=10.0, p10=10.0), 0.0)
        self.assertEqual(compute_edge_elevation(p90=10.0, p10=15.0), 0.0)

    def test_caller_concentration_multi_sibling(self) -> None:
        # Caller u has 3 callees with wait times: 80s, 10s, 10s (total=100s)
        phi, is_single = compute_caller_wait_concentration(edge_wait=80.0, total_caller_wait=100.0, sibling_count=3)
        self.assertAlmostEqual(phi, 0.80)
        self.assertFalse(is_single)

    def test_caller_concentration_single_child(self) -> None:
        # Caller u has 1 callee (edge_wait=50s, total=50s)
        phi, is_single = compute_caller_wait_concentration(edge_wait=50.0, total_caller_wait=50.0, sibling_count=1)
        self.assertEqual(phi, 1.0)
        self.assertTrue(is_single)

    def test_caller_concentration_distinguishes_single_vs_multi_child_at_unity(self) -> None:
        # Multi-sibling caller where 1 callee absorbed 100% of duration (sibling_count=2)
        phi_multi, single_multi = compute_caller_wait_concentration(edge_wait=100.0, total_caller_wait=100.0, sibling_count=2)
        self.assertEqual(phi_multi, 1.0)
        self.assertFalse(single_multi)

        # Single-child caller where callee absorbs 100% trivially (sibling_count=1)
        phi_single, single_flag = compute_caller_wait_concentration(edge_wait=100.0, total_caller_wait=100.0, sibling_count=1)
        self.assertEqual(phi_single, 1.0)
        self.assertTrue(single_flag)

    def test_caller_concentration_zero_total_wait(self) -> None:
        phi, is_single = compute_caller_wait_concentration(edge_wait=0.0, total_caller_wait=0.0, sibling_count=2)
        self.assertEqual(phi, 0.0)
        self.assertFalse(is_single)

    def test_terminal_factor_leaf(self) -> None:
        # Callee has no downstream calls (median_child_covered_fraction is None)
        t_factor, is_ret_dom = compute_terminal_factor(
            median_child_covered_fraction=None,
            caller_p90=50.0,
            return_lag_p90=5.0,
        )
        self.assertEqual(t_factor, 1.0)
        self.assertFalse(is_ret_dom)

    def test_terminal_factor_propagated_victim(self) -> None:
        # Intermediate callee spent 98% of its time waiting on children
        t_factor, is_ret_dom = compute_terminal_factor(
            median_child_covered_fraction=0.98,
            caller_p90=50.0,
            return_lag_p90=2.0,  # 2.0 < 0.5 * 50.0
        )
        self.assertAlmostEqual(t_factor, 0.02)
        self.assertFalse(is_ret_dom)

    def test_terminal_factor_network_return_lag_exception(self) -> None:
        # Return lag dominates edge: return_lag_p90 = 202.0 >= 0.5 * caller_p90 (204.0)
        t_factor, is_ret_dom = compute_terminal_factor(
            median_child_covered_fraction=0.98,
            caller_p90=204.0,
            return_lag_p90=202.0,
        )
        self.assertEqual(t_factor, 1.0)
        self.assertTrue(is_ret_dom)

    def test_edge_attribution_bounded_range(self) -> None:
        # Across arbitrary inputs, attribution_score must stay bounded in [0.0, 1.0]
        for p90, p10 in [(200.0, 2.0), (10.0, 1.0), (0.0, 0.0), (50.0, 100.0)]:
            elev = compute_edge_elevation(p90, p10)
            self.assertTrue(0.0 <= elev <= 1.0)

        for wait, tot, sib in [(50.0, 100.0, 2), (100.0, 100.0, 1), (0.0, 0.0, 1)]:
            phi, _ = compute_caller_wait_concentration(wait, tot, sib)
            self.assertTrue(0.0 <= phi <= 1.0)

        for cov in [None, 0.0, 0.5, 0.98, 1.0]:
            t, _ = compute_terminal_factor(cov, caller_p90=100.0, return_lag_p90=10.0)
            self.assertTrue(0.0 <= t <= 1.0)


class TestComputeTraceAttribution(unittest.TestCase):
    """Test full case-level trace attribution computation on synthetic telemetry."""

    def test_uninstrumented_entity_neutral(self) -> None:
        # Synthetic empty trace latency result
        res = TraceLatencyResult(
            case_id="test_case",
            total_spans=0,
            total_traces=0,
            services=(),
            edges=(),
            sibling_overlap_count=0,
            uninstrumented_services=("adservice", "cartservice"),
        )
        attr_res = compute_trace_attribution(res, expected_services=["adservice", "cartservice"])
        self.assertEqual(attr_res.case_id, "test_case")
        self.assertEqual(len(attr_res.edges), 0)

        ad_ev = attr_res.get_entity("adservice")
        self.assertIsNotNone(ad_ev)
        assert ad_ev is not None
        self.assertFalse(ad_ev.has_trace_evidence)
        self.assertEqual(ad_ev.trace_delay_score, 0.0)
        self.assertIsNone(ad_ev.primary_edge)
        self.assertEqual(ad_ev.incoming_edges, ())

    def test_entity_max_incoming_aggregation(self) -> None:
        # frontend -> currency (A=0.8) and checkout -> currency (A=0.95)
        # currency should aggregate to max(0.8, 0.95) = 0.95
        dist_c1 = DistributionSummary.from_values([2.0] * 10 + [200.0] * 90)
        dist_c2 = DistributionSummary.from_values([3.0] * 10 + [206.0] * 90)
        dist_callee = DistributionSummary.from_values([0.2] * 100)
        dist_ret = DistributionSummary.from_values([204.0] * 100)
        dist_cov_zero = DistributionSummary.from_values([0.0] * 100)

        edge1 = EdgeLatencyEvidence(
            caller_service="frontend",
            callee_service="currencyservice",
            relationship_count=100,
            caller_duration_dist=dist_c1,
            callee_duration_dist=dist_callee,
            child_covered_fraction_dist=dist_cov_zero,
            caller_self_fraction_dist=dist_cov_zero,
            return_lag_dist=dist_ret,
            fraction_of_caller_time_dist=dist_cov_zero,
            total_caller_duration=10000,
            total_callee_duration=20,
        )
        edge2 = EdgeLatencyEvidence(
            caller_service="checkoutservice",
            callee_service="currencyservice",
            relationship_count=100,
            caller_duration_dist=dist_c2,
            callee_duration_dist=dist_callee,
            child_covered_fraction_dist=dist_cov_zero,
            caller_self_fraction_dist=dist_cov_zero,
            return_lag_dist=dist_ret,
            fraction_of_caller_time_dist=dist_cov_zero,
            total_caller_duration=20000,
            total_callee_duration=20,
        )

        res = TraceLatencyResult(
            case_id="case_currency",
            total_spans=200,
            total_traces=100,
            services=("frontend", "checkoutservice", "currencyservice"),
            edges=(edge1, edge2),
            sibling_overlap_count=0,
            uninstrumented_services=(),
        )

        attr_res = compute_trace_attribution(res)
        cur_ev = attr_res.get_entity("currencyservice")
        self.assertIsNotNone(cur_ev)
        assert cur_ev is not None
        self.assertTrue(cur_ev.has_trace_evidence)
        self.assertIsNotNone(cur_ev.primary_edge)
        self.assertEqual(len(cur_ev.incoming_edges), 2)
        # Score is maximum of incoming edges
        self.assertEqual(cur_ev.trace_delay_score, cur_ev.primary_edge.attribution_score)

    def test_frozen_rca_immutability(self) -> None:
        """Verify that computing trace attribution does NOT alter frozen metric RCA results."""
        # 1. Create a synthetic TelemetryCase with metric anomalies
        timestamps = list(range(1000, 1020))
        metrics_data = {
            "timestamp": timestamps * 2,
            "metric": ["frontend_latency-50"] * 20 + ["checkoutservice_cpu"] * 20,
            "value": [10.0] * 20 + [50.0] * 20,
        }
        traces_data = {
            "spanID": ["s1", "s2"],
            "parentSpanID": [None, "s1"],
            "traceID": ["t1", "t1"],
            "serviceName": ["frontend", "checkoutservice"],
            "operationName": ["op1", "op2"],
            "startTime": [1000, 1100],
            "duration": [500, 300],
        }
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="case_immutability"),
            metrics=TelemetryModality(
                raw_data=metrics_data,
                provenance=ModalityProvenance(
                    source_dataset="test", source_case="case_immutability",
                    original_format="dict", original_field_names=tuple(metrics_data.keys()),
                ),
            ),
            traces=TelemetryModality(
                raw_data=traces_data,
                provenance=ModalityProvenance(
                    source_dataset="test", source_case="case_immutability",
                    original_format="dict", original_field_names=tuple(traces_data.keys()),
                ),
            ),
        )

        # 2. Construct episode evidence and initial metric RCA ranking
        ev1 = EntityEpisodeEvidence(
            case_id="case_immutability",
            entity="checkoutservice",
            timestamps=(1000, 1001, 1002),
            active_metric_counts=(0, 2, 3),
            is_in_episode=(False, True, True),
            episodes=(),
            first_episode_start_ts=1001,
            peak_active_metrics=3,
            all_contributing_metrics=("checkoutservice_cpu",),
            has_episode=True,
        )
        ev2 = EntityEpisodeEvidence(
            case_id="case_immutability",
            entity="frontend",
            timestamps=(1000, 1001, 1002),
            active_metric_counts=(0, 1, 2),
            is_in_episode=(False, False, True),
            episodes=(),
            first_episode_start_ts=1002,
            peak_active_metrics=2,
            all_contributing_metrics=("frontend_latency-50",),
            has_episode=True,
        )
        evidence_list = [ev1, ev2]
        obs = extract_trace_dependencies(case)
        deps = [o.to_dependency() for o in obs]
        graph = build_entity_graph(
            metric_names=["frontend_latency-50", "checkoutservice_cpu"],
            dependencies=deps,
        )
        rca_config = RCA_A_ANOMALY_ONLY
        rca_res_1 = rank_root_cause_entities(evidence=evidence_list, graph=graph, config=rca_config)

        # 3. Run trace latency extraction and attribution layer
        trace_lat = extract_trace_latency_evidence(case)
        _ = compute_trace_attribution(trace_lat)

        # 4. Run metric RCA evaluation again
        rca_res_2 = rank_root_cause_entities(evidence=evidence_list, graph=graph, config=rca_config)

        # 5. Assert exact equality of metric RCA results
        self.assertEqual(len(rca_res_1), len(rca_res_2))
        for s1, s2 in zip(rca_res_1, rca_res_2):
            self.assertEqual(s1.entity, s2.entity)
            self.assertEqual(s1.score, s2.score)
            self.assertEqual(s1.r_strength, s2.r_strength)


if __name__ == "__main__":
    unittest.main()
