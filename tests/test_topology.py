"""Unit and integration tests for dependency-aware telemetry representation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence
import unittest

import pyarrow.parquet as pq

from digital_detective.anomaly import MetricAnomalyResult
from digital_detective.topology import (
    DEFAULT_ENTITY_TYPE,
    Dependency,
    EntityAnomalyEvidence,
    EntityGraph,
    EntityNode,
    MetricIdentifier,
    aggregate_entity_anomaly_evidence,
    build_entity_graph,
    parse_rcaeval_metric_identifier,
)


def _make_anomaly_result(
    statuses: Mapping[str, Sequence[str]],
    case_id: str = "test_case",
    timestamps: Sequence[int] | None = None,
) -> MetricAnomalyResult:
    n = len(next(iter(statuses.values())))
    ts = tuple(range(n)) if timestamps is None else tuple(timestamps)
    return MetricAnomalyResult(
        case_id=case_id,
        timestamps=ts,
        metric_names=tuple(statuses.keys()),
        anomaly_scores={m: tuple(3.5 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        signed_scores={m: tuple(3.5 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        anomalies={m: tuple(s == "anomaly" for s in st) for m, st in statuses.items()},
        evaluation_statuses={m: tuple(st) for m, st in statuses.items()},
        valid_history_counts={m: tuple(60 for _ in st) for m, st in statuses.items()},
        summary={},
    )


class MetricParsingTests(unittest.TestCase):
    def test_parse_representative_service_metrics(self) -> None:
        ident_cpu = parse_rcaeval_metric_identifier("adservice_cpu")
        self.assertEqual(ident_cpu.raw_name, "adservice_cpu")
        self.assertEqual(ident_cpu.entity, "adservice")
        self.assertEqual(ident_cpu.signal, "cpu")

        ident_lat = parse_rcaeval_metric_identifier("cartservice_latency-50")
        self.assertEqual(ident_lat.raw_name, "cartservice_latency-50")
        self.assertEqual(ident_lat.entity, "cartservice")
        self.assertEqual(ident_lat.signal, "latency-50")

        ident_load = parse_rcaeval_metric_identifier("checkoutservice_load")
        self.assertEqual(ident_load.raw_name, "checkoutservice_load")
        self.assertEqual(ident_load.entity, "checkoutservice")
        self.assertEqual(ident_load.signal, "load")

    def test_parse_non_service_components(self) -> None:
        ident_redis = parse_rcaeval_metric_identifier("redis_cpu")
        self.assertEqual(ident_redis.entity, "redis")
        self.assertEqual(ident_redis.signal, "cpu")

        ident_gw = parse_rcaeval_metric_identifier("frontend-external_load")
        self.assertEqual(ident_gw.entity, "frontend-external")
        self.assertEqual(ident_gw.signal, "load")

        ident_pt = parse_rcaeval_metric_identifier("PassthroughCluster_error")
        self.assertEqual(ident_pt.entity, "PassthroughCluster")
        self.assertEqual(ident_pt.signal, "error")

        ident_main = parse_rcaeval_metric_identifier("main_mem")
        self.assertEqual(ident_main.entity, "main")
        self.assertEqual(ident_main.signal, "mem")

    def test_parse_malformed_names_raise_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing underscore"):
            parse_rcaeval_metric_identifier("adservicecpu")

        with self.assertRaisesRegex(ValueError, "empty entity or signal"):
            parse_rcaeval_metric_identifier("_cpu")

        with self.assertRaisesRegex(ValueError, "empty entity or signal"):
            parse_rcaeval_metric_identifier("adservice_")


class EntityGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = [
            "frontend_cpu",
            "frontend_latency",
            "checkoutservice_cpu",
            "paymentservice_cpu",
            "redis_mem",
            "frontend-external_load",
            "PassthroughCluster_error",
            "main_cpu",
        ]
        self.deps = [
            ("frontend-external", "frontend"),
            ("frontend", "checkoutservice"),
            ("checkoutservice", "paymentservice"),
            Dependency("frontend", "checkoutservice"),  # duplicate exact edge
        ]
        self.graph = build_entity_graph(self.metrics, self.deps)

    def test_entity_types_classified_correctly(self) -> None:
        self.assertEqual(self.graph.entities["frontend"].entity_type, "service")
        self.assertEqual(self.graph.entities["checkoutservice"].entity_type, "service")
        self.assertEqual(self.graph.entities["paymentservice"].entity_type, "service")
        self.assertEqual(self.graph.entities["redis"].entity_type, "datastore")
        self.assertEqual(self.graph.entities["frontend-external"].entity_type, "gateway")
        self.assertEqual(self.graph.entities["PassthroughCluster"].entity_type, "infrastructure")
        self.assertEqual(self.graph.entities["main"].entity_type, "process")

    def test_exact_metric_lookup(self) -> None:
        self.assertEqual(self.graph.get_entity_for_metric("frontend_cpu"), "frontend")
        self.assertEqual(self.graph.get_entity_for_metric("frontend_latency"), "frontend")
        self.assertEqual(self.graph.get_entity_for_metric("checkoutservice_cpu"), "checkoutservice")
        self.assertEqual(self.graph.get_entity_for_metric("redis_mem"), "redis")
        self.assertIsNone(self.graph.get_entity_for_metric("unmapped_metric"))

    def test_callers_and_callees(self) -> None:
        # frontend is called by frontend-external and calls checkoutservice
        self.assertEqual(self.graph.callers_of("frontend"), ("frontend-external",))
        self.assertEqual(self.graph.callees_of("frontend"), ("checkoutservice",))

        # checkoutservice is called by frontend and calls paymentservice
        self.assertEqual(self.graph.callers_of("checkoutservice"), ("frontend",))
        self.assertEqual(self.graph.callees_of("checkoutservice"), ("paymentservice",))

        # paymentservice has no callees
        self.assertEqual(self.graph.callees_of("paymentservice"), ())

    def test_deterministic_duplicate_edge_handling(self) -> None:
        # ("frontend", "checkoutservice") was included twice in self.deps
        # Must appear exactly once in self.graph.dependencies
        matches = [
            d for d in self.graph.dependencies
            if d.source == "frontend" and d.target == "checkoutservice"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(self.graph.dependencies), 3)


class EntityAnomalyEvidenceTests(unittest.TestCase):
    def test_evidence_aggregation_multiple_metrics_same_entity(self) -> None:
        # 2 metrics for checkoutservice (cpu, mem), 1 for paymentservice (cpu)
        statuses = {
            "checkoutservice_cpu": ["warmup", "normal", "anomaly", "anomaly", "normal"],
            "checkoutservice_mem": ["warmup", "normal", "normal", "anomaly", "normal"],
            "paymentservice_cpu": ["warmup", "normal", "normal", "normal", "anomaly"],
        }
        det_res = _make_anomaly_result(statuses, timestamps=[100, 101, 102, 103, 104])
        graph = build_entity_graph(list(statuses.keys()))

        evidence = aggregate_entity_anomaly_evidence(det_res, graph)

        checkout_ev = evidence["checkoutservice"]
        self.assertEqual(checkout_ev.case_id, "test_case")
        self.assertEqual(checkout_ev.entity, "checkoutservice")
        self.assertEqual(checkout_ev.timestamps, (100, 101, 102, 103, 104))
        # Step 2: 1 metric (cpu). Step 3: 2 metrics (cpu + mem).
        self.assertEqual(checkout_ev.active_metric_counts, (0, 0, 1, 2, 0))
        self.assertEqual(checkout_ev.is_anomalous, (False, False, True, True, False))
        self.assertEqual(checkout_ev.first_anomaly_ts, 102)
        self.assertEqual(checkout_ev.anomalous_metrics, ("checkoutservice_cpu", "checkoutservice_mem"))

        payment_ev = evidence["paymentservice"]
        self.assertEqual(payment_ev.active_metric_counts, (0, 0, 0, 0, 1))
        self.assertEqual(payment_ev.is_anomalous, (False, False, False, False, True))
        self.assertEqual(payment_ev.first_anomaly_ts, 104)
        self.assertEqual(payment_ev.anomalous_metrics, ("paymentservice_cpu",))

    def test_warmup_and_missing_states_do_not_count_as_anomaly(self) -> None:
        statuses = {
            "adservice_cpu": [
                "warmup",
                "missing_observation",
                "insufficient_history",
                "normal",
                "normal",
            ]
        }
        det_res = _make_anomaly_result(statuses, timestamps=[10, 11, 12, 13, 14])
        graph = build_entity_graph(list(statuses.keys()))

        evidence = aggregate_entity_anomaly_evidence(det_res, graph)
        ad_ev = evidence["adservice"]
        self.assertEqual(ad_ev.active_metric_counts, (0, 0, 0, 0, 0))
        self.assertEqual(ad_ev.is_anomalous, (False, False, False, False, False))
        self.assertIsNone(ad_ev.first_anomaly_ts)
        self.assertEqual(ad_ev.anomalous_metrics, ())

    def test_causality_strictly_preserved(self) -> None:
        # Modifying future values does not affect past active counts or first_anomaly_ts
        statuses_a = {
            "cartservice_cpu": ["warmup"] * 5 + ["anomaly", "anomaly"] + ["normal"] * 5,
        }
        statuses_b = {
            "cartservice_cpu": ["warmup"] * 5 + ["anomaly", "anomaly"] + ["anomaly"] * 5,
        }
        det_a = _make_anomaly_result(statuses_a)
        det_b = _make_anomaly_result(statuses_b)
        graph = build_entity_graph(list(statuses_a.keys()))

        ev_a = aggregate_entity_anomaly_evidence(det_a, graph)["cartservice"]
        ev_b = aggregate_entity_anomaly_evidence(det_b, graph)["cartservice"]

        self.assertEqual(ev_a.first_anomaly_ts, ev_b.first_anomaly_ts)
        self.assertEqual(ev_a.active_metric_counts[:7], ev_b.active_metric_counts[:7])


class RealDataIntegrationTests(unittest.TestCase):
    def test_all_15_re1_ob_cases_metric_columns_parse_cleanly(self) -> None:
        dataset_root = os.environ.get("RCAEval_DATASET_ROOT")
        if not dataset_root:
            raise unittest.SkipTest("RCAEval_DATASET_ROOT not set; skipping real-data integration test")

        root = Path(dataset_root)
        index_file = root / "cases.parquet"
        if not index_file.is_file():
            raise unittest.SkipTest(f"{index_file} not found; skipping real-data integration test")

        index_table = pq.read_table(index_file)
        cases = [
            r["case"] for r in index_table.to_pylist()
            if r.get("system") == "ob" and r.get("repetition") == 1 and r.get("suite") in ("RE1", "RE2")
        ]
        cases.sort()
        selected_15 = cases[:15]

        total_metrics_checked = 0
        all_raw_names_preserved: set[str] = set()

        for case_id in selected_15:
            metrics_path = root / case_id / "metrics.parquet"
            self.assertTrue(metrics_path.is_file(), f"Missing metrics file for {case_id}")

            table = pq.read_table(metrics_path)
            metric_cols = [c for c in table.column_names if c != "time"]

            # 1. Every column must parse cleanly without error
            for col in metric_cols:
                ident = parse_rcaeval_metric_identifier(col)
                self.assertEqual(ident.raw_name, col)
                self.assertTrue(len(ident.entity) > 0)
                self.assertTrue(len(ident.signal) > 0)
                all_raw_names_preserved.add(col)
                total_metrics_checked += 1

            # 2. Building entity graph must contain all raw metric names
            graph = build_entity_graph(metric_cols)
            for col in metric_cols:
                mapped_entity = graph.get_entity_for_metric(col)
                self.assertIsNotNone(mapped_entity, f"Metric {col} was not indexed in EntityGraph")
                self.assertIn(col, graph.entities[mapped_entity].metrics)

        self.assertGreater(total_metrics_checked, 0)
        # All 84 distinct metric column names observed in RE1-OB must be covered
        self.assertEqual(len(all_raw_names_preserved), 84)


if __name__ == "__main__":
    unittest.main()
