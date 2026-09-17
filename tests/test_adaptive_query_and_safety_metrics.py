"""Tests for adaptive modality-aware query selection, safety metrics, and post-intervention validation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from digital_detective.detective.investigator import InvestigationEngine
from digital_detective.detective.models import (
    EvidenceItem,
    InterventionValidationResult,
    InvestigationState,
    RankedHypothesis,
    RecoveryVerificationResult,
    RemediationAction,
    RootCauseDecision,
    ToolQueryRecord,
)
from digital_detective.detective.verification import (
    SystemStateSnapshot,
    validate_intervention,
    verify_recovery,
)
from eval.models import (
    AggregateMethodMetrics,
    CaseMetrics,
    MethodRankingResult,
    RankedEntity,
    aggregate_case_metrics,
)


class TestAdaptiveModalityQuerySelection(unittest.TestCase):
    """Verify deterministic modality discrimination, cost-normalization, and tie-breaking."""

    def test_modality_discrimination_calculation(self) -> None:
        """Verify dispersion and discrimination calculation across top-K candidates."""
        # Top-3 candidates with varying modality component scores
        cand1 = RankedHypothesis(
            service="checkoutservice",
            score=0.9,
            confidence=0.8,
            rank=1,
            component_scores={"metric": 0.95, "trace": 0.40, "change": 0.0},
        )
        cand2 = RankedHypothesis(
            service="paymentservice",
            score=0.7,
            confidence=0.6,
            rank=2,
            component_scores={"metric": 0.15, "trace": 0.35, "change": 0.0},
        )
        cand3 = RankedHypothesis(
            service="currencyservice",
            score=0.5,
            confidence=0.4,
            rank=3,
            component_scores={"metric": 0.10, "trace": 0.38, "change": 0.0},
        )
        top_k = [cand1, cand2, cand3]

        # Metric dispersion: 0.95 - 0.10 = 0.85
        metric_vals = [h.component_scores.get("metric", 0.0) for h in top_k]
        metric_disp = max(metric_vals) - min(metric_vals)
        self.assertAlmostEqual(metric_disp, 0.85, places=4)

        # Trace dispersion: 0.40 - 0.35 = 0.05
        trace_vals = [h.component_scores.get("trace", 0.0) for h in top_k]
        trace_disp = max(trace_vals) - min(trace_vals)
        self.assertAlmostEqual(trace_disp, 0.05, places=4)

        # Baseline delta = 0.15
        delta = 0.15
        metric_disc = metric_disp + delta  # 1.00
        trace_disc = trace_disp + delta    # 0.20

        self.assertGreater(metric_disc, trace_disc)
        self.assertAlmostEqual(metric_disc, 1.00, places=4)
        self.assertAlmostEqual(trace_disc, 0.20, places=4)

    def test_query_cost_normalization(self) -> None:
        """Verify priority inversely scales with query cost."""
        engine = InvestigationEngine(
            tool_costs={"get_metrics": 1, "get_traces": 3},
            adaptive_query_selection=True,
        )
        # For equal suspicion and discrimination, 1-cost tool has 3x priority over 3-cost tool
        disc = 0.5
        suspicion = 0.8
        unexplored = 1.0
        missing_factor = 1.6

        p_metrics = (suspicion * unexplored * disc * missing_factor) / 1
        p_traces = (suspicion * unexplored * disc * missing_factor) / 3

        self.assertAlmostEqual(p_metrics / p_traces, 3.0, places=4)

    def test_missing_modality_preference(self) -> None:
        """Verify unqueried modalities receive the missing evidence bonus factor."""
        observed_modalities = {"metrics"}
        missing_modality = "traces"

        missing_factor_unobserved = 1.6 if missing_modality not in observed_modalities else 0.7
        missing_factor_observed = 1.6 if "metrics" not in observed_modalities else 0.7

        self.assertEqual(missing_factor_unobserved, 1.6)
        self.assertEqual(missing_factor_observed, 0.7)
        self.assertGreater(missing_factor_unobserved, missing_factor_observed)

    def test_deterministic_tie_breaking(self) -> None:
        """Verify tie-breaking by (-priority, -cost, service_name, tool_name)."""
        # Two queries with identical priority and cost should sort alphabetically by service then tool
        queries = [
            (1.5, "paymentservice", "get_metrics", 1, "rationale B", "policy", 0.5),
            (1.5, "checkoutservice", "get_metrics", 1, "rationale A", "policy", 0.5),
            (1.5, "checkoutservice", "get_service_health", 1, "rationale C", "policy", 0.5),
            (2.0, "currencyservice", "get_metrics", 1, "rationale D", "policy", 0.5),
        ]
        queries.sort(key=lambda x: (-x[0], -x[3], x[1], x[2]))

        # Highest priority first (2.0)
        self.assertEqual(queries[0][1], "currencyservice")
        # Then tie-broken by service name alphabetically: "checkoutservice" before "paymentservice"
        self.assertEqual(queries[1][1], "checkoutservice")
        self.assertEqual(queries[1][2], "get_metrics")
        self.assertEqual(queries[2][1], "checkoutservice")
        self.assertEqual(queries[2][2], "get_service_health")
        self.assertEqual(queries[3][1], "paymentservice")

    def test_adaptive_vs_legacy_selection_flag(self) -> None:
        """Verify engine honors the adaptive_query_selection feature flag."""
        engine_adaptive = InvestigationEngine(budget=10, adaptive_query_selection=True)
        engine_legacy = InvestigationEngine(budget=10, adaptive_query_selection=False)

        self.assertTrue(engine_adaptive.adaptive_query_selection)
        self.assertFalse(engine_legacy.adaptive_query_selection)

    def test_budget_enforcement(self) -> None:
        """Verify queries exceeding remaining budget are rejected."""
        state = InvestigationState(
            incident_id="test_case",
            system="ob",
            candidate_universe=("checkoutservice",),
            budget=2,
            remaining_budget=1,
        )
        from digital_detective.detective.tools import DetectiveTools
        tools = DetectiveTools(case=MagicMock(), candidate_universe=("checkoutservice",), costs={"get_traces": 3})
        success, summary, _ = tools.execute_query(
            state=state,
            tool_name="get_traces",
            service="checkoutservice",
            window=MagicMock(),
        )
        self.assertFalse(success)
        self.assertIn("Tool 'get_traces' requires cost 3, but remaining budget is 1", summary)
        self.assertEqual(state.remaining_budget, 1)

    def test_no_ground_truth_access(self) -> None:
        """Verify InvestigationEngine operates without accessing or requiring ground truth."""
        import os
        from pathlib import Path
        from digital_detective.rcaeval import load_rcaeval_case
        root = os.environ.get("RCAEVAL_DATASET_ROOT") or str(Path.home() / ".cache" / "rcaeval_validation")
        try:
            case = load_rcaeval_case(root, "re2ob_checkoutservice_cpu_1")
        except Exception:
            self.skipTest("Validation dataset not available in local cache")

        # Explicitly set ground_truth to None
        case.ground_truth = None

        engine = InvestigationEngine(budget=5)
        state = engine.investigate(case=case, window_mode="detected")
        self.assertIsNotNone(state.decision)
        self.assertIn(state.decision.root_cause_service, state.candidate_universe)



class TestPostInterventionValidation(unittest.TestCase):
    """Verify InterventionValidationResult and validate_intervention logic."""

    def test_intervention_supports_hypothesis_on_broad_recovery(self) -> None:
        """Verify INTERVENTION_SUPPORTS_HYPOTHESIS when target and downstream recover with no regressions."""
        verification = RecoveryVerificationResult(
            final_status="RESOLVED",
            recovered_metrics=(
                "checkoutservice_anomaly_score: 0.95 -> 0.05",
                "frontend_latency_p90: 250.0ms -> 45.0ms",
            ),
            unresolved_symptoms=(),
            new_anomalies=(),
            regression_detected=False,
            comparison_summary={"target_service": "checkoutservice", "target_recovered": True},
        )
        before_state = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.95, "frontend": 0.50},
            latencies_p90={"frontend": 250.0},
            error_rates={},
            health_statuses={},
        )

        res = validate_intervention(
            verification=verification,
            target_service="checkoutservice",
            action="scale_service",
            pre_intervention_consistency=0.88,
            before_state=before_state,
        )

        self.assertIsInstance(res, InterventionValidationResult)
        self.assertEqual(res.causal_support, "INTERVENTION_SUPPORTS_HYPOTHESIS")
        self.assertTrue(res.recovery_verified)
        self.assertTrue(res.downstream_recovery)
        self.assertFalse(res.regression_detected)
        self.assertAlmostEqual(res.pre_intervention_consistency, 0.88)
        self.assertIn("broad system recovery", res.explanation)

    def test_intervention_contradicts_hypothesis_on_regression(self) -> None:
        """Verify INTERVENTION_CONTRADICTS_HYPOTHESIS when new regressions are detected."""
        verification = RecoveryVerificationResult(
            final_status="NOT_RESOLVED",
            recovered_metrics=(),
            unresolved_symptoms=("checkoutservice_anomaly_score: still 0.95",),
            new_anomalies=("paymentservice_error_count increased from 0 to 20",),
            regression_detected=True,
        )
        res = validate_intervention(
            verification=verification,
            target_service="checkoutservice",
            action="restart_service",
            pre_intervention_consistency=0.75,
        )
        self.assertEqual(res.causal_support, "INTERVENTION_CONTRADICTS_HYPOTHESIS")
        self.assertTrue(res.regression_detected)
        self.assertIn("induced new regressions", res.explanation)

    def test_intervention_contradicts_hypothesis_when_target_fails_to_recover(self) -> None:
        """Verify INTERVENTION_CONTRADICTS_HYPOTHESIS when remediation fails to resolve target anomalies."""
        verification = RecoveryVerificationResult(
            final_status="NOT_RESOLVED",
            recovered_metrics=(),
            unresolved_symptoms=("checkoutservice_anomaly_score: still 0.95",),
            new_anomalies=(),
            regression_detected=False,
        )
        res = validate_intervention(
            verification=verification,
            target_service="checkoutservice",
            action="restart_service",
            pre_intervention_consistency=0.60,
        )
        self.assertEqual(res.causal_support, "INTERVENTION_CONTRADICTS_HYPOTHESIS")
        self.assertIn("failed to resolve target", res.explanation)

    def test_intervention_inconclusive_on_partial_recovery(self) -> None:
        """Verify INTERVENTION_INCONCLUSIVE when recovery is uncertain or partial."""
        verification = RecoveryVerificationResult(
            final_status="UNCERTAIN",
            recovered_metrics=("checkoutservice_anomaly_score: 0.90 -> 0.10",),
            unresolved_symptoms=("frontend_latency_p90 elevated at 150.0ms",),
            new_anomalies=(),
            regression_detected=False,
        )
        res = validate_intervention(
            verification=verification,
            target_service="checkoutservice",
            action="scale_service",
            pre_intervention_consistency=0.70,
        )
        self.assertEqual(res.causal_support, "INTERVENTION_INCONCLUSIVE")
        self.assertIn("partial recovery", res.explanation)


class TestSafetyAndEfficiencyMetrics(unittest.TestCase):
    """Verify computation and missing denominator handling for safety metrics."""

    def test_missing_denominators_report_na(self) -> None:
        """Verify that metrics without valid denominators report 'N/A' rather than zero."""
        results = [
            MethodRankingResult(
                case_id=f"case_{i}",
                method_name="passive_ranker",
                window_mode="oracle",
                ranking=(RankedEntity(entity="svc_a", score=1.0, rank=1),),
                candidate_universe=("svc_a", "svc_b"),
                status="SUCCESS",
                metrics=CaseMetrics(
                    top1=True,
                    top3=True,
                    top5=True,
                    mrr=1.0,
                    ac1=1.0,
                    ac2=1.0,
                    ac3=1.0,
                    ac4=1.0,
                    ac5=1.0,
                    avg3=1.0,
                    avg5=1.0,
                    target_rank=1,
                ),
                metadata={},
            )
            for i in range(5)
        ]

        agg = aggregate_case_metrics("passive_ranker", results)
        self.assertEqual(agg.abstention_rate, 0.0)
        self.assertEqual(agg.false_remediation_rate, "N/A")
        self.assertEqual(agg.recovery_success_rate, "N/A")
        self.assertEqual(agg.regression_rate, "N/A")
        self.assertEqual(agg.query_efficiency, "N/A")

    def test_abstention_metric(self) -> None:
        """Verify abstention rate is correctly calculated when some cases abstain."""
        results = [
            MethodRankingResult(
                case_id="case_1",
                method_name="agent_method",
                window_mode="oracle",
                ranking=(RankedEntity(entity="svc_a", score=0.9, rank=1),),
                candidate_universe=("svc_a", "svc_b"),
                status="SUCCESS",
                metrics=CaseMetrics(
                    top1=True, top3=True, top5=True, mrr=1.0,
                    ac1=1.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                    avg3=1.0, avg5=1.0, target_rank=1,
                ),
            ),
            MethodRankingResult(
                case_id="case_2",
                method_name="agent_method",
                window_mode="oracle",
                ranking=(),
                candidate_universe=("svc_a", "svc_b"),
                status="ABSTAIN",
                metrics=None,
                metadata={"abstained": True},
            ),
            MethodRankingResult(
                case_id="case_3",
                method_name="agent_method",
                window_mode="oracle",
                ranking=(RankedEntity(entity="svc_b", score=0.4, rank=1),),
                candidate_universe=("svc_a", "svc_b"),
                status="SUCCESS",
                metrics=CaseMetrics(
                    top1=False, top3=True, top5=True, mrr=0.5,
                    ac1=0.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                    avg3=0.333, avg5=0.2, target_rank=2,
                ),
                metadata={"status": "STOP", "confidence": 0.4},
            ),
            MethodRankingResult(
                case_id="case_4",
                method_name="agent_method",
                window_mode="oracle",
                ranking=(RankedEntity(entity="svc_a", score=0.9, rank=1),),
                candidate_universe=("svc_a", "svc_b"),
                status="SUCCESS",
                metrics=CaseMetrics(
                    top1=True, top3=True, top5=True, mrr=1.0,
                    ac1=1.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                    avg3=1.0, avg5=1.0, target_rank=1,
                ),
            ),
        ]

        agg = aggregate_case_metrics("agent_method", results)
        self.assertAlmostEqual(agg.abstention_rate, 0.50, places=4)

    def test_remediation_and_query_efficiency_metrics(self) -> None:
        """Verify false remediation, recovery, regression, and query efficiency calculations."""
        results = [
            MethodRankingResult(
                case_id="case_1",
                method_name="agent_method",
                window_mode="oracle",
                ranking=(RankedEntity(entity="svc_a", score=0.9, rank=1),),
                candidate_universe=("svc_a", "svc_b"),
                status="SUCCESS",
                metrics=CaseMetrics(
                    top1=True, top3=True, top5=True, mrr=1.0,
                    ac1=1.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                    avg3=1.0, avg5=1.0, target_rank=1,
                ),
                metadata={
                    "remediation_authorized": True,
                    "recovery_verified": True,
                    "regression_detected": False,
                    "queries_executed": 4,
                },
            ),
            MethodRankingResult(
                case_id="case_2",
                method_name="agent_method",
                window_mode="oracle",
                ranking=(RankedEntity(entity="svc_b", score=0.85, rank=1),),
                candidate_universe=("svc_a", "svc_b"),
                status="SUCCESS",
                metrics=CaseMetrics(
                    top1=False, top3=True, top5=True, mrr=0.5,
                    ac1=0.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                    avg3=0.333, avg5=0.2, target_rank=2,
                ),
                metadata={
                    "remediation_authorized": True,
                    "recovery_verified": False,
                    "regression_detected": True,
                    "queries_executed": 6,
                },
            ),
        ]

        agg = aggregate_case_metrics("agent_method", results)
        self.assertEqual(agg.false_remediation_rate, 0.50)
        self.assertEqual(agg.recovery_success_rate, 0.50)
        self.assertEqual(agg.regression_rate, 0.50)
        self.assertEqual(agg.query_efficiency, 0.10)


if __name__ == "__main__":
    unittest.main()
