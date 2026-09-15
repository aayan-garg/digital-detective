"""Unit tests for trace latency evidence layer."""

from __future__ import annotations

import unittest

from digital_detective.telemetry import (
    CaseMetadata,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.traces import (
    DistributionSummary,
    SpanRecord,
    compute_interval_union,
    decompose_parent_span,
    extract_trace_latency_evidence,
    has_overlapping_intervals,
    interval_union_duration,
)


class TestIntervalArithmetic(unittest.TestCase):
    """Test interval union and overlap logic."""

    def test_interval_union_empty(self) -> None:
        self.assertEqual(compute_interval_union(()), ())
        self.assertEqual(interval_union_duration(()), 0)

    def test_interval_union_zero_length(self) -> None:
        self.assertEqual(compute_interval_union([(10, 10), (20, 15)]), ())
        self.assertEqual(interval_union_duration([(10, 10)]), 0)

    def test_interval_union_disjoint(self) -> None:
        ivs = [(10, 20), (25, 30), (40, 50)]
        self.assertEqual(compute_interval_union(ivs), ((10, 20), (25, 30), (40, 50)))
        self.assertEqual(interval_union_duration(ivs), 10 + 5 + 10)

    def test_interval_union_abutting(self) -> None:
        ivs = [(10, 20), (20, 30)]
        self.assertEqual(compute_interval_union(ivs), ((10, 30),))
        self.assertEqual(interval_union_duration(ivs), 20)

    def test_interval_union_overlapping(self) -> None:
        ivs = [(10, 25), (15, 30), (5, 12)]
        # Sorted: (5, 12), (10, 25), (15, 30) -> Merged: (5, 30)
        self.assertEqual(compute_interval_union(ivs), ((5, 30),))
        self.assertEqual(interval_union_duration(ivs), 25)

    def test_interval_union_nested(self) -> None:
        ivs = [(10, 50), (15, 25), (20, 30)]
        self.assertEqual(compute_interval_union(ivs), ((10, 50),))
        self.assertEqual(interval_union_duration(ivs), 40)

    def test_has_overlapping_intervals(self) -> None:
        self.assertFalse(has_overlapping_intervals([]))
        self.assertFalse(has_overlapping_intervals([(10, 20)]))
        self.assertFalse(has_overlapping_intervals([(10, 20), (20, 30), (30, 40)]))
        self.assertTrue(has_overlapping_intervals([(10, 25), (20, 30)]))
        self.assertTrue(has_overlapping_intervals([(10, 50), (15, 20)]))


class TestSpanDecomposition(unittest.TestCase):
    """Test parent span timing decomposition."""

    def test_parent_no_children(self) -> None:
        parent = SpanRecord(
            span_id="p1",
            parent_span_id=None,
            trace_id="t1",
            service_name="frontend",
            operation_name="op",
            start_time=1000,
            duration=500,
        )
        decomp = decompose_parent_span(parent, ())
        self.assertEqual(decomp.duration, 500)
        self.assertEqual(decomp.child_covered_duration, 0)
        self.assertEqual(decomp.self_duration, 500)
        self.assertEqual(decomp.child_covered_fraction, 0.0)
        self.assertEqual(decomp.self_fraction, 1.0)
        self.assertFalse(decomp.has_sibling_overlap)

    def test_parent_zero_duration(self) -> None:
        parent = SpanRecord(
            span_id="p0",
            parent_span_id=None,
            trace_id="t1",
            service_name="frontend",
            operation_name="op",
            start_time=1000,
            duration=0,
        )
        decomp = decompose_parent_span(parent, ())
        self.assertEqual(decomp.duration, 0)
        self.assertEqual(decomp.child_covered_duration, 0)
        self.assertEqual(decomp.self_duration, 0)
        self.assertEqual(decomp.child_covered_fraction, 0.0)
        self.assertEqual(decomp.self_fraction, 1.0)

    def test_parent_with_sequential_children(self) -> None:
        parent = SpanRecord(
            span_id="p1",
            parent_span_id=None,
            trace_id="t1",
            service_name="frontend",
            operation_name="op",
            start_time=1000,
            duration=1000,
        )
        c1 = SpanRecord(
            span_id="c1", parent_span_id="p1", trace_id="t1", service_name="cart",
            operation_name="op1", start_time=1100, duration=200
        )
        c2 = SpanRecord(
            span_id="c2", parent_span_id="p1", trace_id="t1", service_name="cur",
            operation_name="op2", start_time=1400, duration=300
        )
        decomp = decompose_parent_span(parent, [c1, c2])
        self.assertEqual(decomp.duration, 1000)
        self.assertEqual(decomp.child_covered_duration, 500)
        self.assertEqual(decomp.self_duration, 500)
        self.assertAlmostEqual(decomp.child_covered_fraction, 0.5)
        self.assertAlmostEqual(decomp.self_fraction, 0.5)
        self.assertFalse(decomp.has_sibling_overlap)

    def test_parent_with_overlapping_children(self) -> None:
        parent = SpanRecord(
            span_id="p1",
            parent_span_id=None,
            trace_id="t1",
            service_name="frontend",
            operation_name="op",
            start_time=1000,
            duration=1000,
        )
        c1 = SpanRecord(
            span_id="c1", parent_span_id="p1", trace_id="t1", service_name="cart",
            operation_name="op1", start_time=1100, duration=400  # 1100..1500
        )
        c2 = SpanRecord(
            span_id="c2", parent_span_id="p1", trace_id="t1", service_name="cur",
            operation_name="op2", start_time=1300, duration=400  # 1300..1700
        )
        decomp = decompose_parent_span(parent, [c1, c2])
        # Union of (1100, 1500) and (1300, 1700) is (1100, 1700) -> 600 us
        self.assertEqual(decomp.child_covered_duration, 600)
        self.assertEqual(decomp.self_duration, 400)
        self.assertAlmostEqual(decomp.child_covered_fraction, 0.6)
        self.assertTrue(decomp.has_sibling_overlap)

    def test_child_clipped_to_parent(self) -> None:
        parent = SpanRecord(
            span_id="p1",
            parent_span_id=None,
            trace_id="t1",
            service_name="frontend",
            operation_name="op",
            start_time=1000,
            duration=500,  # 1000..1500
        )
        c_exceeding = SpanRecord(
            span_id="c1", parent_span_id="p1", trace_id="t1", service_name="cart",
            operation_name="op1", start_time=900, duration=800  # 900..1700
        )
        decomp = decompose_parent_span(parent, [c_exceeding])
        # Clipped to (1000, 1500) -> 500 us
        self.assertEqual(decomp.child_covered_duration, 500)
        self.assertEqual(decomp.self_duration, 0)
        self.assertAlmostEqual(decomp.child_covered_fraction, 1.0)


class TestDistributionSummary(unittest.TestCase):
    """Test distribution summary calculation."""

    def test_empty(self) -> None:
        d = DistributionSummary.from_values([])
        self.assertEqual(d.count, 0)
        self.assertEqual(d.median, 0.0)

    def test_single_value(self) -> None:
        d = DistributionSummary.from_values([42.0])
        self.assertEqual(d.count, 1)
        self.assertEqual(d.min, 42.0)
        self.assertEqual(d.median, 42.0)
        self.assertEqual(d.max, 42.0)
        self.assertEqual(d.mean, 42.0)

    def test_multiple_values(self) -> None:
        d = DistributionSummary.from_values([10, 20, 30, 40, 50])
        self.assertEqual(d.count, 5)
        self.assertEqual(d.min, 10.0)
        self.assertEqual(d.p10, 10.0)  # idx = round(0.10 * 4) = 0 -> sorted[0] = 10.0
        self.assertEqual(d.median, 30.0)
        self.assertEqual(d.max, 50.0)
        self.assertEqual(d.mean, 30.0)

    def test_p10_small_samples(self) -> None:
        # N=1: 0.1 * 0 = 0 -> index 0
        d1 = DistributionSummary.from_values([100])
        self.assertEqual(d1.p10, 100.0)

        # N=2: 0.1 * 1 = 0.1 -> index 0
        d2 = DistributionSummary.from_values([10, 20])
        self.assertEqual(d2.p10, 10.0)

        # N=6: 0.1 * 5 = 0.5 -> round(0.5) = 0 (round-half-to-even) -> index 0
        d6 = DistributionSummary.from_values([10, 20, 30, 40, 50, 60])
        self.assertEqual(d6.p10, 10.0)

        # N=7: 0.1 * 6 = 0.6 -> round(0.6) = 1 -> index 1
        d7 = DistributionSummary.from_values([10, 20, 30, 40, 50, 60, 70])
        self.assertEqual(d7.p10, 20.0)

        # N=10: 0.1 * 9 = 0.9 -> round(0.9) = 1 -> index 1
        d10 = DistributionSummary.from_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(d10.p10, 2.0)


class TestExtractTraceLatencyEvidence(unittest.TestCase):
    """Test full trace latency evidence extraction on synthetic telemetry."""

    def test_no_traces_modality(self) -> None:
        case = TelemetryCase(metadata=CaseMetadata(case_id="case_no_traces"))
        res = extract_trace_latency_evidence(case, expected_services=["svc_a", "svc_b"])
        self.assertEqual(res.case_id, "case_no_traces")
        self.assertEqual(res.total_spans, 0)
        self.assertEqual(len(res.edges), 0)
        self.assertEqual(res.uninstrumented_services, ("svc_a", "svc_b"))

    def test_valid_trace_extraction(self) -> None:
        data = {
            "spanID": ["s_root", "s_client", "s_server"],
            "parentSpanID": [None, "s_root", "s_client"],
            "traceID": ["t1", "t1", "t1"],
            "serviceName": ["frontendservice", "frontendservice", "currencyservice"],
            "operationName": ["frontend", "CurrencyService/Convert", "CurrencyService/Convert"],
            "startTime": [1000, 1100, 1150],
            "duration": [1000, 400, 200],
            "statusCode": [0, 0, 0],
        }
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="test_case"),
            traces=TelemetryModality(
                raw_data=data,
                provenance=ModalityProvenance(
                    source_dataset="test",
                    source_case="test_case",
                    original_format="dict",
                    original_field_names=tuple(data.keys()),
                ),
            ),
        )
        res = extract_trace_latency_evidence(
            case,
            service_aliases={"frontendservice": "frontend"},
            expected_services=["frontend", "currencyservice", "adservice"],
        )
        self.assertEqual(res.case_id, "test_case")
        self.assertEqual(res.total_spans, 3)
        self.assertEqual(res.total_traces, 1)
        self.assertIn("frontend", res.services)
        self.assertIn("currencyservice", res.services)
        self.assertEqual(res.uninstrumented_services, ("adservice",))

        # Check observed edge: frontend -> currencyservice
        edge = res.get_edge("frontend", "currencyservice")
        self.assertNotNull = self.assertIsNotNone(edge)
        assert edge is not None
        self.assertEqual(edge.relationship_count, 1)
        self.assertEqual(edge.caller_duration_dist.median, 400.0)
        self.assertEqual(edge.callee_duration_dist.median, 200.0)
        # start lag: 1150 - 1100 = 50 us
        # return lag: (1100+400) - (1150+200) = 1500 - 1350 = 150 us
        self.assertEqual(edge.return_lag_dist.median, 150.0)

    def test_duplicate_trace_rows_tolerated(self) -> None:
        data = {
            "spanID": ["s1", "s1"],
            "parentSpanID": [None, None],
            "traceID": ["t1", "t1"],
            "serviceName": ["svc", "svc"],
            "operationName": ["op", "op"],
            "startTime": [100, 100],
            "duration": [50, 50],
        }
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="dup_case"),
            traces=TelemetryModality(
                raw_data=data,
                provenance=ModalityProvenance(
                    source_dataset="test",
                    source_case="dup_case",
                    original_format="dict",
                    original_field_names=tuple(data.keys()),
                ),
            ),
        )
        res = extract_trace_latency_evidence(case)
        self.assertEqual(res.total_spans, 1)

    def test_conflicting_duplicate_span_id_raises(self) -> None:
        data = {
            "spanID": ["s1", "s1"],
            "parentSpanID": [None, "p1"],  # conflicting parent
            "traceID": ["t1", "t1"],
            "serviceName": ["svc", "svc"],
            "operationName": ["op", "op"],
            "startTime": [100, 100],
            "duration": [50, 50],
        }
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="conflict_case"),
            traces=TelemetryModality(
                raw_data=data,
                provenance=ModalityProvenance(
                    source_dataset="test",
                    source_case="conflict_case",
                    original_format="dict",
                    original_field_names=tuple(data.keys()),
                ),
            ),
        )
        with self.assertRaises(ValueError):
            extract_trace_latency_evidence(case)


if __name__ == "__main__":
    unittest.main()
