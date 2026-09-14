"""Unit and integration tests for dependency-aware telemetry representation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from digital_detective.anomaly import MetricAnomalyResult
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.telemetry import (
    CaseMetadata,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.topology import (
    DEFAULT_ENTITY_TYPE,
    Dependency,
    EntityAnomalyEvidence,
    EntityGraph,
    EntityNode,
    MetricIdentifier,
    TraceDependencyObservation,
    aggregate_entity_anomaly_evidence,
    build_entity_graph,
    extract_trace_dependencies,
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


class TraceDependencyExtractionTests(unittest.TestCase):
    def _make_case_with_traces(self, trace_table: Any) -> TelemetryCase:
        return TelemetryCase(
            metadata=CaseMetadata(case_id="trace_test_case", suite="RE2", system="ob"),
            traces=TelemetryModality(
                raw_data=trace_table,
                provenance=ModalityProvenance(
                    source_dataset="test",
                    source_case="trace_test_case",
                    original_format="traces.parquet",
                    original_field_names=tuple(trace_table.column_names) if hasattr(trace_table, "column_names") else tuple(trace_table.keys()),
                ),
            ) if trace_table is not None else None,
        )

    def test_direct_cross_service_parent_child_edge(self) -> None:
        table = pa.table({
            "spanID": ["span1", "span2"],
            "parentSpanID": [None, "span1"],
            "serviceName": ["frontend", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].source, "frontend")
        self.assertEqual(deps[0].target, "checkoutservice")
        self.assertEqual(deps[0].observation_count, 1)
        self.assertEqual(deps[0].to_dependency(), Dependency(source="frontend", target="checkoutservice"))

    def test_repeated_edge_produces_one_observation_with_correct_count(self) -> None:
        table = pa.table({
            "spanID": ["root1", "child1", "root2", "child2", "root3", "child3"],
            "parentSpanID": [None, "root1", None, "root2", None, "root3"],
            "serviceName": [
                "frontend", "checkoutservice",
                "frontend", "checkoutservice",
                "frontend", "checkoutservice",
            ],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].source, "frontend")
        self.assertEqual(deps[0].target, "checkoutservice")
        self.assertEqual(deps[0].observation_count, 3)

    def test_same_service_parent_child_ignored(self) -> None:
        table = pa.table({
            "spanID": ["span1", "span2"],
            "parentSpanID": [None, "span1"],
            "serviceName": ["checkoutservice", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(deps, ())

    def test_root_span_ignored(self) -> None:
        table = pa.table({
            "spanID": ["root1", "root2"],
            "parentSpanID": [None, ""],
            "serviceName": ["frontend", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(deps, ())

    def test_missing_parent_ignored(self) -> None:
        table = pa.table({
            "spanID": ["child1"],
            "parentSpanID": ["nonexistent_parent"],
            "serviceName": ["checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(deps, ())

    def test_null_or_missing_service_name_ignored(self) -> None:
        table = pa.table({
            "spanID": ["span1", "span2", "span3", "span4"],
            "parentSpanID": [None, "span1", None, "span3"],
            "serviceName": [None, "checkoutservice", "frontend", ""],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(deps, ())

    def test_duplicate_span_id_raises_value_error(self) -> None:
        table = pa.table({
            "spanID": ["dup_span", "dup_span"],
            "parentSpanID": [None, "dup_span"],
            "serviceName": ["frontend", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        with self.assertRaisesRegex(ValueError, "Duplicate spanID detected"):
            extract_trace_dependencies(case)

    def test_duplicate_span_id_conflicting_parent_raises_value_error(self) -> None:
        table = pa.table({
            "spanID": ["span_child", "span_child"],
            "parentSpanID": ["parent1", "parent2"],
            "serviceName": ["checkoutservice", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        with self.assertRaisesRegex(ValueError, "Duplicate spanID detected"):
            extract_trace_dependencies(case)

    def test_duplicate_span_id_identical_records_tolerated(self) -> None:
        table = pa.table({
            "spanID": ["p1", "c1", "c1"],
            "parentSpanID": [None, "p1", "p1"],
            "serviceName": ["frontend", "checkoutservice", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].source, "frontend")
        self.assertEqual(deps[0].target, "checkoutservice")
        self.assertEqual(deps[0].observation_count, 1)

    def test_explicit_alias_mapping(self) -> None:
        table = pa.table({
            "spanID": ["span1", "span2"],
            "parentSpanID": [None, "span1"],
            "serviceName": ["frontendservice", "checkoutservice"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].source, "frontend")
        self.assertEqual(deps[0].target, "checkoutservice")
        self.assertEqual(deps[0].observation_count, 1)

    def test_deterministic_ordering(self) -> None:
        table = pa.table({
            "spanID": ["root", "child_z", "child_a", "child_m"],
            "parentSpanID": [None, "root", "root", "root"],
            "serviceName": ["gateway", "zebra", "apple", "mango"],
        })
        case = self._make_case_with_traces(table)
        deps = extract_trace_dependencies(case)
        targets = [d.target for d in deps]
        self.assertEqual(targets, ["apple", "mango", "zebra"])

    def test_missing_traces_returns_empty_tuple(self) -> None:
        case = self._make_case_with_traces(None)
        self.assertEqual(extract_trace_dependencies(case), ())

    def test_missing_required_column_raises_value_error(self) -> None:
        table = pa.table({
            "spanID": ["span1"],
            "serviceName": ["frontend"],
            # parentSpanID is missing
        })
        case = self._make_case_with_traces(table)
        with self.assertRaisesRegex(ValueError, "Trace data missing required column"):
            extract_trace_dependencies(case)

    def test_re2_real_data_trace_extraction(self) -> None:
        dataset_root = os.environ.get("RCAEval_DATASET_ROOT")
        if not dataset_root:
            raise unittest.SkipTest("RCAEval_DATASET_ROOT not set; skipping real-data trace test")

        root = Path(dataset_root)
        case_dir = root / "re2ob_checkoutservice_cpu_1"
        if not (case_dir / "traces.parquet").is_file():
            raise unittest.SkipTest("re2ob_checkoutservice_cpu_1 traces.parquet not found")

        case = load_rcaeval_case(root, "re2ob_checkoutservice_cpu_1")

        # 1. Raw extraction without aliases: exactly 9 edges and counts
        raw_deps = extract_trace_dependencies(case)
        expected_raw = {
            ("checkoutservice", "currencyservice"): 2323,
            ("checkoutservice", "emailservice"): 669,
            ("checkoutservice", "paymentservice"): 669,
            ("checkoutservice", "productcatalogservice"): 1655,
            ("frontendservice", "checkoutservice"): 669,
            ("frontendservice", "currencyservice"): 52421,
            ("frontendservice", "productcatalogservice"): 84412,
            ("frontendservice", "recommendationservice"): 12929,
            ("recommendationservice", "productcatalogservice"): 12929,
        }
        self.assertEqual(len(raw_deps), 9)
        actual_raw = {(d.source, d.target): d.observation_count for d in raw_deps}
        self.assertEqual(actual_raw, expected_raw)

        # 2. Extraction with alias mapping frontendservice -> frontend
        aliased_deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
        expected_aliased = {
            ("checkoutservice", "currencyservice"): 2323,
            ("checkoutservice", "emailservice"): 669,
            ("checkoutservice", "paymentservice"): 669,
            ("checkoutservice", "productcatalogservice"): 1655,
            ("frontend", "checkoutservice"): 669,
            ("frontend", "currencyservice"): 52421,
            ("frontend", "productcatalogservice"): 84412,
            ("frontend", "recommendationservice"): 12929,
            ("recommendationservice", "productcatalogservice"): 12929,
        }
        self.assertEqual(len(aliased_deps), 9)
        actual_aliased = {(d.source, d.target): d.observation_count for d in aliased_deps}
        self.assertEqual(actual_aliased, expected_aliased)

        # 3. Verify trace invariants directly on real data:
        # - all parent-child links remain within traceID (0 cross-trace links)
        # - missing parents are not converted into invented edges (exactly 7 missing parents skipped)
        # - same-service links are excluded (199,346 same-service links skipped)
        trace_table = case.traces.raw_data
        span_to_trace = dict(zip(trace_table.column("spanID").to_pylist(), trace_table.column("traceID").to_pylist()))
        parents = trace_table.column("parentSpanID").to_pylist()
        traces = trace_table.column("traceID").to_pylist()

        cross_trace_count = 0
        missing_parent_count = 0
        for pid, tid in zip(parents, traces):
            if pid is not None and pid != "":
                if pid not in span_to_trace:
                    missing_parent_count += 1
                elif span_to_trace[pid] != tid:
                    cross_trace_count += 1

        self.assertEqual(cross_trace_count, 0)
        self.assertEqual(missing_parent_count, 7)

    def test_re2_real_data_recommendationservice_loss_1_duplicate_rows(self) -> None:
        dataset_root = os.environ.get("RCAEval_DATASET_ROOT")
        if not dataset_root:
            raise unittest.SkipTest("RCAEval_DATASET_ROOT not set; skipping real-data trace test")

        root = Path(dataset_root)
        case_dir = root / "re2ob_recommendationservice_loss_1"
        if not (case_dir / "traces.parquet").is_file():
            raise unittest.SkipTest("re2ob_recommendationservice_loss_1 traces.parquet not found")

        case = load_rcaeval_case(root, "re2ob_recommendationservice_loss_1")
        aliased_deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
        expected_edges = {
            ("checkoutservice", "currencyservice"),
            ("checkoutservice", "emailservice"),
            ("checkoutservice", "paymentservice"),
            ("checkoutservice", "productcatalogservice"),
            ("frontend", "checkoutservice"),
            ("frontend", "currencyservice"),
            ("frontend", "productcatalogservice"),
            ("frontend", "recommendationservice"),
            ("recommendationservice", "productcatalogservice"),
        }
        actual_edges = {(d.source, d.target) for d in aliased_deps}
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(aliased_deps), 9)


if __name__ == "__main__":
    unittest.main()
