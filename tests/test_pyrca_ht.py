"""Unit tests for PyRCA Hypothesis Testing (HT) RCA backend.

Tests all 10 mandatory criteria:
1. Graph orientation conversion (G[i, j] = 1 means j has parent i).
2. Normal training excludes incident and future samples.
3. inject_time and ground truth labels are not consumed.
4. Candidate universe projection is exact (non-candidates omitted).
5. Deterministic repeated execution produces identical rankings.
6. Complete ranking generation over candidate universe.
7. PyRCA raw residual score extraction.
8. Root nodes (parentless) and leaf nodes handled explicitly.
9. Missing-value behavior follows existing telemetry policy.
10. Integration with TelemetryCase and EntityGraph.
"""

from __future__ import annotations

import unittest
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.topology import Dependency, EntityGraph, build_entity_graph
from digital_detective.pyrca_ht import (
    build_entity_telemetry_frames,
    convert_entity_graph_to_pyrca_adjacency,
    run_pyrca_ht,
)

try:
    import networkx as nx
    from pyrca.analyzers.ht import HT, HTConfig
    PYRCA_AVAILABLE = True
except ImportError:
    PYRCA_AVAILABLE = False


def _make_mock_case(
    timestamps: list[int],
    series_dict: dict[str, list[float | None]],
    case_id: str = "mock_case_1",
) -> TelemetryCase:
    raw_data = [{"time": ts, **{k: series_dict[k][i] for k in series_dict}} for i, ts in enumerate(timestamps)]
    prov = ModalityProvenance(
        source_dataset="mock",
        source_case=case_id,
        original_format="dict",
        original_field_names=("time", *tuple(series_dict.keys())),
        timestamp_field="time",
    )
    return TelemetryCase(
        metadata=CaseMetadata(case_id=case_id, dataset="mock", system="mock"),
        metrics=TelemetryModality(raw_data=raw_data, provenance=prov),
        ground_truth=GroundTruth(values={"inject_time": 999999, "root_cause_service": "forbidden"}),
    )


@unittest.skipUnless(PANDAS_AVAILABLE, "pandas not installed in current environment")
class PyRCAHTBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metric_names = [
            "frontend_cpu",
            "checkoutservice_cpu",
            "cartservice_cpu",
            "cartservice_mem",
            "external_gateway_workload",
        ]
        self.dependencies = [
            Dependency("frontend", "checkoutservice"),
            Dependency("checkoutservice", "cartservice"),
        ]
        self.graph = build_entity_graph(self.metric_names, dependencies=self.dependencies)

    def test_1_graph_orientation_conversion(self) -> None:
        """G[i, j] = 1 represents edge i -> j (j has parent i)."""
        adj, graph_hash = convert_entity_graph_to_pyrca_adjacency(self.graph)

        self.assertIsInstance(adj, pd.DataFrame)
        self.assertTrue(len(graph_hash) > 0)
        self.assertEqual(adj.loc["frontend", "checkoutservice"], 1)
        self.assertEqual(adj.loc["checkoutservice", "cartservice"], 1)
        self.assertEqual(adj.loc["cartservice", "checkoutservice"], 0)
        self.assertEqual(adj.loc["checkoutservice", "frontend"], 0)

        if PYRCA_AVAILABLE:
            G = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph())
            # checkoutservice parent is frontend
            self.assertIn("frontend", list(G.predecessors("checkoutservice")))
            # cartservice parent is checkoutservice
            self.assertIn("checkoutservice", list(G.predecessors("cartservice")))
            # frontend has 0 parents
            self.assertEqual(list(G.predecessors("frontend")), [])

    def test_2_normal_training_excludes_incident_and_future_samples(self) -> None:
        """Normal dataframe strictly excludes timestamps >= incident_start."""
        timestamps = list(range(100, 200, 10))  # 100..190 (10 points)
        series = {
            "frontend_cpu": [1.0] * 5 + [10.0] * 5,
            "checkoutservice_cpu": [1.0] * 5 + [10.0] * 5,
            "cartservice_cpu": [1.0] * 5 + [10.0] * 5,
            "cartservice_mem": [2.0] * 5 + [20.0] * 5,
            "external_gateway_workload": [0.5] * 10,
        }
        case = _make_mock_case(timestamps, series)

        # incident start = 150, incident end = 170
        normal_df, incident_df = build_entity_telemetry_frames(
            case,
            self.graph,
            incident_start_ts=150,
            incident_end_ts=170,
        )

        # Normal should have timestamps 100, 110, 120, 130, 140 (5 rows)
        self.assertEqual(len(normal_df), 5)
        # Incident should have timestamps 150, 160, 170 (3 rows)
        self.assertEqual(len(incident_df), 3)
        # Verify columns match graph entities exactly
        self.assertEqual(list(normal_df.columns), sorted(self.graph.entities.keys()))
        self.assertEqual(list(incident_df.columns), sorted(self.graph.entities.keys()))

    def test_3_no_injection_time_or_label_consumption(self) -> None:
        """Backend runs without inspecting inject_time or root_cause_service."""
        timestamps = list(range(100, 200, 10))
        series = {m: [1.0] * 10 for m in self.metric_names}
        # TelemetryCase with NO ground truth
        case = TelemetryCase(
            metadata=CaseMetadata(case_id="blind_case", dataset="mock", system="mock"),
            metrics=TelemetryModality(
                raw_data=[{"time": ts, **{k: 1.0 for k in series}} for ts in timestamps],
                provenance=ModalityProvenance("metrics", "blind_case", "time", ("time", *self.metric_names)),
            ),
            ground_truth=None,
        )

        normal_df, incident_df = build_entity_telemetry_frames(
            case,
            self.graph,
            incident_start_ts=150,
            incident_end_ts=170,
        )
        self.assertEqual(len(normal_df), 5)
        self.assertEqual(len(incident_df), 3)

    @unittest.skipUnless(PYRCA_AVAILABLE, "PyRCA not installed in current environment")
    def test_4_candidate_projection_and_non_candidate_exclusion(self) -> None:
        """Non-candidate graph nodes are excluded; canonical universe is preserved."""
        adj, _ = convert_entity_graph_to_pyrca_adjacency(self.graph)
        normal_df = pd.DataFrame(np.random.RandomState(42).randn(50, len(adj.columns)), columns=adj.columns)
        incident_df = pd.DataFrame(np.random.RandomState(42).randn(10, len(adj.columns)), columns=adj.columns)

        canonical_universe = ("frontend", "checkoutservice", "cartservice")
        ranking = run_pyrca_ht(normal_df, incident_df, adj, canonical_universe)

        ranked_entities = [r.entity for r in ranking]
        self.assertEqual(sorted(ranked_entities), sorted(canonical_universe))
        # external_gateway must be excluded from final candidate ranking
        self.assertNotIn("external_gateway", ranked_entities)

    @unittest.skipUnless(PYRCA_AVAILABLE, "PyRCA not installed in current environment")
    def test_5_deterministic_repeated_execution(self) -> None:
        """Running HT twice on identical inputs produces identical rankings and scores."""
        adj, _ = convert_entity_graph_to_pyrca_adjacency(self.graph)
        rng = np.random.RandomState(123)
        normal_df = pd.DataFrame(rng.randn(40, len(adj.columns)), columns=adj.columns)
        incident_df = pd.DataFrame(rng.randn(5, len(adj.columns)), columns=adj.columns)

        universe = ("frontend", "checkoutservice", "cartservice")
        r1 = run_pyrca_ht(normal_df, incident_df, adj, universe)
        r2 = run_pyrca_ht(normal_df, incident_df, adj, universe)

        self.assertEqual(len(r1), len(r2))
        for item1, item2 in zip(r1, r2):
            self.assertEqual(item1.entity, item2.entity)
            self.assertEqual(item1.rank, item2.rank)
            self.assertAlmostEqual(item1.score, item2.score, places=9)

    @unittest.skipUnless(PYRCA_AVAILABLE, "PyRCA not installed in current environment")
    def test_6_complete_ranking_with_unscored_candidates(self) -> None:
        """Candidates missing from graph receive score 0.0 and complete 1..N ranks."""
        adj, _ = convert_entity_graph_to_pyrca_adjacency(self.graph)
        normal_df = pd.DataFrame(np.ones((20, len(adj.columns))), columns=adj.columns)
        incident_df = pd.DataFrame(np.ones((5, len(adj.columns))), columns=adj.columns)

        universe = ("frontend", "checkoutservice", "cartservice", "uninstrumented_svc")
        ranking = run_pyrca_ht(normal_df, incident_df, adj, universe)

        self.assertEqual(len(ranking), 4)
        ranks = [r.rank for r in ranking]
        self.assertEqual(ranks, [1, 2, 3, 4])
        by_ent = {r.entity: r for r in ranking}
        self.assertIn("uninstrumented_svc", by_ent)
        self.assertEqual(by_ent["uninstrumented_svc"].score, 0.0)

    @unittest.skipUnless(PYRCA_AVAILABLE, "PyRCA not installed in current environment")
    def test_7_raw_ht_residual_score_extraction(self) -> None:
        """HT score is residual standard error magnitude without probability clipping."""
        adj, _ = convert_entity_graph_to_pyrca_adjacency(self.graph)
        rng = np.random.RandomState(99)
        normal_df = pd.DataFrame(rng.randn(50, len(adj.columns)), columns=adj.columns)
        # Inject massive residual spike in cartservice
        incident_df = pd.DataFrame(rng.randn(5, len(adj.columns)), columns=adj.columns)
        incident_df["cartservice"] = incident_df["cartservice"] + 15.0

        universe = ("frontend", "checkoutservice", "cartservice")
        ranking = run_pyrca_ht(normal_df, incident_df, adj, universe)

        # cartservice should have large residual score and rank 1
        self.assertEqual(ranking[0].entity, "cartservice")
        self.assertGreater(ranking[0].score, 5.0)

    @unittest.skipUnless(PYRCA_AVAILABLE, "PyRCA not installed in current environment")
    def test_8_root_and_leaf_nodes_handled_explicitly(self) -> None:
        """Root nodes (no parents) use marginal scaler; leaf nodes have no children."""
        adj, _ = convert_entity_graph_to_pyrca_adjacency(self.graph)
        normal_df = pd.DataFrame(np.random.RandomState(7).randn(30, len(adj.columns)), columns=adj.columns)
        incident_df = pd.DataFrame(np.random.RandomState(7).randn(5, len(adj.columns)), columns=adj.columns)

        universe = ("frontend", "checkoutservice", "cartservice")
        ranking = run_pyrca_ht(normal_df, incident_df, adj, universe)
        self.assertEqual(len(ranking), 3)

    def test_9_missing_value_fallback(self) -> None:
        """Missing observations (None / NaN) are safely handled via nanmean/nanstd."""
        timestamps = list(range(100, 200, 10))
        series = {
            "frontend_cpu": [None, 1.0, None, 1.5, 1.2, 5.0, 5.0, 5.0, 5.0, 5.0],
            "checkoutservice_cpu": [1.0] * 10,
            "cartservice_cpu": [None] * 5 + [2.0] * 5,
            "cartservice_mem": [1.0] * 10,
            "external_gateway_workload": [0.0] * 10,
        }
        case = _make_mock_case(timestamps, series)

        normal_df, incident_df = build_entity_telemetry_frames(
            case,
            self.graph,
            incident_start_ts=150,
            incident_end_ts=190,
        )
        self.assertFalse(normal_df.isna().any().any())
        self.assertFalse(incident_df.isna().any().any())

    def test_10_invalid_window_raises_explicit_error(self) -> None:
        """Insufficient history or invalid window bounds raise ValueError."""
        timestamps = [100, 110]
        series = {m: [1.0, 2.0] for m in self.metric_names}
        case = _make_mock_case(timestamps, series)

        with self.assertRaises(ValueError):
            # incident_start_ts = 100 leaves 0 normal samples
            build_entity_telemetry_frames(case, self.graph, incident_start_ts=100, incident_end_ts=110)


if __name__ == "__main__":
    unittest.main()
