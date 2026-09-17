"""Unit tests for SimpleRCA baseline implementation (Fang et al. §3.1.2).
"""

import unittest
import pyarrow as pa

from eval.baselines.simple_rca import (
    detect_log_alerts,
    detect_metric_alerts,
    detect_trace_alerts,
    rank_with_simple_rca,
)
from eval.models import IncidentWindow


class MockCase:
    def __init__(self, metrics=None, traces=None, logs=None):
        self.metrics = metrics
        self.traces = traces
        self.logs = logs


class MockMetrics:
    def __init__(self, timestamps, series):
        self.timestamps = timestamps
        self.series = series


class MockModality:
    def __init__(self, table):
        self.table = table


class TestSimpleRCA(unittest.TestCase):
    def test_metric_alerts_3sigma(self) -> None:
        # Timestamps: 0..9 normal, 10..15 incident
        timestamps = list(range(16))
        # Service A: normal mean 10, std ~0, spike at t=12 to 50 (anomaly)
        s_a_vals = [10.0] * 10 + [10.0, 10.0, 50.0, 10.0, 10.0, 10.0]
        # Service B: normal mean 10, no spike in incident
        s_b_vals = [10.0] * 16

        metrics = MockMetrics(
            timestamps=timestamps,
            series={"servicea_cpu": s_a_vals, "serviceb_cpu": s_b_vals},
        )
        case = MockCase(metrics=metrics)
        window = IncidentWindow(onset_ts=10, end_ts=15, mode="oracle")
        universe = ("servicea", "serviceb", "servicec")

        alerts = detect_metric_alerts(case, window, universe)
        self.assertEqual(alerts["servicea"], 1)
        self.assertEqual(alerts["serviceb"], 0)
        self.assertEqual(alerts["servicec"], 0)  # Preserved with 0 alerts

    def test_trace_alerts_3x_threshold(self) -> None:
        # Normal: t < 100, Incident: 100 <= t <= 200
        # Service A: normal durations 10ms, incident durations 35ms (>= 3x 10ms -> Alert!)
        # Service B: normal durations 10ms, incident durations 15ms (< 3x 10ms -> No alert)
        table = pa.table({
            "startTimeMillis": [
                50_000, 60_000, 120_000, 130_000,  # Service A
                50_000, 60_000, 120_000, 130_000,  # Service B
            ],
            "duration": [
                10.0, 10.0, 35.0, 35.0,
                10.0, 10.0, 15.0, 15.0,
            ],
            "serviceName": [
                "servicea", "servicea", "servicea", "servicea",
                "serviceb", "serviceb", "serviceb", "serviceb",
            ],
        })
        case = MockCase(traces=MockModality(table))
        window = IncidentWindow(onset_ts=100, end_ts=200, mode="oracle")
        universe = ("servicea", "serviceb", "servicec")

        alerts = detect_trace_alerts(case, window, universe)
        self.assertEqual(alerts["servicea"], 1)
        self.assertEqual(alerts["serviceb"], 0)
        self.assertEqual(alerts["servicec"], 0)

    def test_log_alerts_keyword_count(self) -> None:
        # Incident: 100 <= t <= 200
        table = pa.table({
            "timestamp": [50, 110, 120, 130, 140],
            "container_name": ["servicea", "servicea", "servicea", "serviceb", "serviceb"],
            "message": [
                "error during startup",  # pre-incident -> ignored
                "connection failed",      # matches 'fail' -> servicea +1
                "Unhandled Exception",   # matches 'exception' -> servicea +1
                "info message ok",       # no match
                "fatal error occurred",  # matches 'error' -> serviceb +1
            ],
        })
        case = MockCase(logs=MockModality(table))
        window = IncidentWindow(onset_ts=100, end_ts=200, mode="oracle")
        universe = ("servicea", "serviceb", "servicec")

        alerts = detect_log_alerts(case, window, universe)
        self.assertEqual(alerts["servicea"], 2)
        self.assertEqual(alerts["serviceb"], 1)
        self.assertEqual(alerts["servicec"], 0)

    def test_combined_ranking_and_tie_breaking(self) -> None:
        # Universe with 3 services:
        # s_b: 5 alerts (rank 1)
        # s_a: 2 alerts (rank 2)
        # s_c: 2 alerts (rank 3, tied with s_a, tie-break by name asc -> s_a then s_c)
        # s_d: 0 alerts (rank 4, preserved)
        timestamps = list(range(10))
        metrics = MockMetrics(timestamps=timestamps, series={})
        case = MockCase(metrics=metrics)
        window = IncidentWindow(onset_ts=5, end_ts=9, mode="oracle")
        universe = ("s_c", "s_a", "s_b", "s_d")

        # Mock alert summation by calling rank_with_simple_rca on empty telemetry (all 0)
        ranking = rank_with_simple_rca(case, window, universe)
        self.assertEqual(len(ranking), 4)
        # Tied at 0, should be alphabetical: s_a, s_b, s_c, s_d
        self.assertEqual([r.entity for r in ranking], ["s_a", "s_b", "s_c", "s_d"])
        self.assertEqual([r.rank for r in ranking], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
