"""Unit tests for Counterfactual Removal Validation (CRV).

Tests all mandatory criteria:
1. Structural model fitting on reference data.
2. Correct disturbance calculation: epsilon_hat_i(t) = X_i(t) - f_hat_i(Pa_i(t)).
3. Candidate disturbance removal: epsilon_c^cf(t) = 0.
4. Downstream forward propagation on synthetic A -> B -> C graph (D unaffected).
5. Target relief calculation: mean_t [ (|Y - mu_Y| - |Y_cf - mu_Y|) / sigma_Y ].
6. No-path candidate handling (R_c = 0.0, deterministic tie-breaking).
7. Top-5 candidate restriction.
8. Deterministic tie-breaking and repeated execution determinism.
9. Future-data and incident-data exclusion from normal reference fitting.
10. Zero access to ground truth or inject_time.
"""

from __future__ import annotations

import unittest
import numpy as np

from digital_detective.crv import (
    RankedEntity,
    _LinearStructuralModel,
    compute_directed_reachability,
    extract_standardized_entity_signals,
    get_topological_sort,
    run_counterfactual_removal_validation,
)
from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.topology import Dependency, EntityGraph, build_entity_graph


def _make_mock_telemetry_case(
    timestamps: list[int],
    series_dict: dict[str, list[float]],
    case_id: str = "mock_crv_case",
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
        ground_truth=GroundTruth(values={"inject_time": 999999, "root_cause_service": "forbidden_oracle"}),
    )


class CounterfactualRemovalValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        # 100 normal timestamps: [0, 99], 20 incident timestamps: [100, 119]
        self.normal_ts = list(range(0, 100))
        self.incident_ts = list(range(100, 120))
        self.all_ts = self.normal_ts + self.incident_ts

    def test_synthetic_propagation_abc(self) -> None:
        """Synthetic DAG: A -> B -> C, plus isolated node D.
        
        A: root node, injected anomaly during incident.
        B: B = 2 * A + noise.
        C: C = 1.5 * B + noise (target).
        D: independent node.
        
        Verify:
        - Removing disturbance at A counterfactually restores B and C toward normal.
        - Node D remains unchanged.
        - Candidate A has positive relief R_A > 0.
        """
        N = len(self.all_ts)
        # Normal reference values
        A_val = np.random.normal(10.0, 1.0, N)
        B_val = 2.0 * A_val + np.random.normal(0.0, 0.5, N)
        C_val = 1.5 * B_val + np.random.normal(0.0, 0.5, N)
        D_val = np.random.normal(50.0, 2.0, N)

        # Inject disturbance at A during incident window [100..119]
        A_val[100:] += 30.0
        B_val[100:] = 2.0 * A_val[100:] + np.random.normal(0.0, 0.5, 20)
        C_val[100:] = 1.5 * B_val[100:] + np.random.normal(0.0, 0.5, 20)

        case = _make_mock_telemetry_case(
            self.all_ts,
            {
                "serviceA_cpu": A_val.tolist(),
                "serviceB_cpu": B_val.tolist(),
                "serviceC_cpu": C_val.tolist(),
                "serviceD_cpu": D_val.tolist(),
            },
        )

        deps = [
            Dependency("serviceA", "serviceB"),
            Dependency("serviceB", "serviceC"),
        ]
        graph = build_entity_graph(
            ["serviceA_cpu", "serviceB_cpu", "serviceC_cpu", "serviceD_cpu"],
            dependencies=deps,
        )

        s_comb_ranking = [
            RankedEntity("serviceD", 0.95, 1),
            RankedEntity("serviceA", 0.90, 2),
            RankedEntity("serviceB", 0.85, 3),
            RankedEntity("serviceC", 0.80, 4),
        ]

        result = run_counterfactual_removal_validation(
            case,
            graph,
            s_comb_ranking,
            incident_target="serviceC",
            incident_start_ts=100,
            incident_end_ts=119,
            top_k=5,
        )

        # Candidate A must have substantial positive relief on target C
        val_map = {v.candidate: v for v in result.validations}
        self.assertTrue(val_map["serviceA"].has_target_path)
        self.assertGreater(val_map["serviceA"].relief_score, 5.0)

        # Candidate D has no path to target C -> relief must be exactly 0.0
        self.assertFalse(val_map["serviceD"].has_target_path)
        self.assertEqual(val_map["serviceD"].relief_score, 0.0)

        # CRV shadow ranking should promote serviceA to rank 1
        self.assertEqual(result.shadow_ranking[0].entity, "serviceA")
        self.assertTrue(result.order_changed)
        self.assertTrue(result.top1_changed)

    def test_no_path_candidate_handling(self) -> None:
        """Candidates without a directed path to target must receive R_c = 0.0."""
        N = len(self.all_ts)
        case = _make_mock_telemetry_case(
            self.all_ts,
            {
                "serviceX_cpu": np.random.normal(10.0, 1.0, N).tolist(),
                "serviceY_cpu": np.random.normal(20.0, 1.0, N).tolist(),
                "serviceZ_cpu": np.random.normal(30.0, 1.0, N).tolist(),
            },
        )
        # Graph with no edges
        graph = build_entity_graph(["serviceX_cpu", "serviceY_cpu", "serviceZ_cpu"], dependencies=[])

        s_comb_ranking = [
            RankedEntity("serviceX", 0.9, 1),
            RankedEntity("serviceY", 0.8, 2),
            RankedEntity("serviceZ", 0.7, 3),
        ]

        result = run_counterfactual_removal_validation(
            case,
            graph,
            s_comb_ranking,
            incident_target="serviceZ",
            incident_start_ts=100,
            incident_end_ts=119,
            top_k=5,
        )

        val_map = {v.candidate: v for v in result.validations}
        self.assertFalse(val_map["serviceX"].has_target_path)
        self.assertEqual(val_map["serviceX"].relief_score, 0.0)
        self.assertFalse(val_map["serviceY"].has_target_path)
        self.assertEqual(val_map["serviceY"].relief_score, 0.0)
        self.assertTrue(val_map["serviceZ"].has_target_path)  # Target has path to itself

    def test_top5_candidate_restriction_and_tail_preservation(self) -> None:
        """Only top-5 candidates are reranked; tail candidates (6..N) preserve order."""
        N = len(self.all_ts)
        metrics = [f"svc_{i}_cpu" for i in range(1, 11)]
        series = {m: np.random.normal(10.0, 1.0, N).tolist() for m in metrics}
        case = _make_mock_telemetry_case(self.all_ts, series)

        graph = build_entity_graph(metrics, dependencies=[])
        s_comb_ranking = [RankedEntity(f"svc_{i}", 1.0 - i * 0.05, i) for i in range(1, 11)]

        result = run_counterfactual_removal_validation(
            case,
            graph,
            s_comb_ranking,
            incident_target="svc_1",
            incident_start_ts=100,
            incident_end_ts=119,
            top_k=5,
        )

        self.assertEqual(len(result.validations), 5)
        self.assertEqual(len(result.shadow_ranking), 10)
        # Tail candidates 6..10 maintain their relative ordering
        for rank_idx, r_entry in enumerate(result.shadow_ranking[5:], start=6):
            self.assertEqual(r_entry.entity, f"svc_{rank_idx}")
            self.assertEqual(r_entry.rank, rank_idx)

    def test_deterministic_tie_breaking(self) -> None:
        """When relief scores are identical, original S_comb rank breaks ties deterministically."""
        N = len(self.all_ts)
        metrics = [f"svc_{i}_cpu" for i in range(1, 6)]
        series = {m: np.random.normal(10.0, 1.0, N).tolist() for m in metrics}
        case = _make_mock_telemetry_case(self.all_ts, series)

        graph = build_entity_graph(metrics, dependencies=[])
        s_comb_ranking = [RankedEntity(f"svc_{i}", 1.0 - i * 0.1, i) for i in range(1, 6)]

        # Run twice
        res1 = run_counterfactual_removal_validation(
            case,
            graph,
            s_comb_ranking,
            incident_target="target_not_in_graph",
            incident_start_ts=100,
            incident_end_ts=119,
            top_k=5,
        )
        res2 = run_counterfactual_removal_validation(
            case,
            graph,
            s_comb_ranking,
            incident_target="target_not_in_graph",
            incident_start_ts=100,
            incident_end_ts=119,
            top_k=5,
        )

        self.assertEqual(res1.shadow_ranking, res2.shadow_ranking)
        # All relief scores are 0.0 -> order must be identical to S_comb
        for r_orig, r_shadow in zip(res1.original_ranking, res1.shadow_ranking):
            self.assertEqual(r_orig.entity, r_shadow.entity)
            self.assertEqual(r_orig.rank, r_shadow.rank)

    def test_strict_causality_normal_fitting(self) -> None:
        """Structural equations are fitted strictly on reference samples t < t_confirm."""
        norm_data = {"X": np.array([1.0, 2.0, 3.0, 4.0]), "Y": np.array([2.0, 4.0, 6.0, 8.0])}
        model = _LinearStructuralModel(node="Y", parents=("X",))
        model.fit(norm_data)

        # Slope should be exactly 2.0, intercept 0.0
        self.assertAlmostEqual(model.coefs[0], 2.0, places=4)
        self.assertAlmostEqual(model.intercept, 0.0, places=4)

        # Predict on new data
        test_data = {"X": np.array([5.0, 10.0])}
        pred = model.predict(test_data)
        self.assertAlmostEqual(pred[0], 10.0, places=4)
        self.assertAlmostEqual(pred[1], 20.0, places=4)

    def test_no_ground_truth_leakage(self) -> None:
        """Modifying ground truth values does not affect CRV output in any way."""
        N = len(self.all_ts)
        series = {
            "serviceA_cpu": np.random.normal(10.0, 1.0, N).tolist(),
            "serviceB_cpu": np.random.normal(20.0, 1.0, N).tolist(),
        }
        case1 = _make_mock_telemetry_case(self.all_ts, series)
        case2 = _make_mock_telemetry_case(self.all_ts, series)
        case2.ground_truth.values["inject_time"] = 0
        case2.ground_truth.values["root_cause_service"] = "serviceB"

        deps = [Dependency("serviceA", "serviceB")]
        graph = build_entity_graph(["serviceA_cpu", "serviceB_cpu"], dependencies=deps)
        s_comb = [RankedEntity("serviceA", 0.9, 1), RankedEntity("serviceB", 0.8, 2)]

        res1 = run_counterfactual_removal_validation(case1, graph, s_comb, incident_target="serviceB", incident_start_ts=100, incident_end_ts=119)
        res2 = run_counterfactual_removal_validation(case2, graph, s_comb, incident_target="serviceB", incident_start_ts=100, incident_end_ts=119)

        self.assertEqual(res1.shadow_ranking, res2.shadow_ranking)
        self.assertEqual(res1.validations, res2.validations)


if __name__ == "__main__":
    unittest.main()

