"""Unit tests for remediation proposal, safety gating, and simulated execution.
"""

import unittest

from digital_detective.detective.models import (
    EvidenceItem,
    InvestigationState,
    RankedHypothesis,
    RemediationAction,
    RootCauseDecision,
)
from digital_detective.detective.remediation import (
    execute_simulated_remediation,
    propose_remediation,
    validate_remediation,
)


class TestDetectiveRemediation(unittest.TestCase):
    def setUp(self) -> None:
        self.universe = ("cartservice", "checkoutservice", "frontend")
        self.state = InvestigationState(
            incident_id="inc_test",
            system="ob",
            candidate_universe=self.universe,
            budget=20,
            remaining_budget=20,
        )
        self.ev = EvidenceItem(
            service="checkoutservice",
            modality="metrics",
            signal="cpu",
            magnitude=0.9,
        )
        self.state.add_evidence(self.ev)
        self.state.current_rankings = (
            RankedHypothesis(service="checkoutservice", rank=1, score=0.85, confidence=0.85),
            RankedHypothesis(service="cartservice", rank=2, score=0.20, confidence=0.20),
        )

    def test_safety_gate_success(self) -> None:
        action = RemediationAction(
            target_service="checkoutservice",
            action_type="scale_service",
            preconditions=("service_must_be_degraded",),
            risk_level="LOW",
            rollback_action="restore_initial_replica_count",
            parameters={"replicas_delta": 2},
        )
        is_valid, reason = validate_remediation(action, self.state)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Safety validation passed.")

    def test_safety_gate_missing_rollback(self) -> None:
        action = RemediationAction(
            target_service="checkoutservice",
            action_type="scale_service",
            preconditions=(),
            risk_level="LOW",
            rollback_action="",  # Missing rollback!
        )
        is_valid, reason = validate_remediation(action, self.state)
        self.assertFalse(is_valid)
        self.assertIn("Mandatory rollback action is missing", reason)

    def test_safety_gate_unsupported_action(self) -> None:
        action = RemediationAction(
            target_service="checkoutservice",
            action_type="rm_rf_root",  # Destructive / unsupported!
            preconditions=(),
            risk_level="HIGH",
            rollback_action="none",
        )
        is_valid, reason = validate_remediation(action, self.state)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported action type", reason)

    def test_safety_gate_healthy_bystander_rejected(self) -> None:
        # frontend has zero degradation evidence
        action = RemediationAction(
            target_service="frontend",
            action_type="restart_service",
            preconditions=("service_must_be_degraded",),
            risk_level="LOW",
            rollback_action="restore_service_pod",
        )
        is_valid, reason = validate_remediation(action, self.state)
        self.assertFalse(is_valid)
        self.assertIn("Safety violation", reason)

    def test_propose_remediation_by_fault_type(self) -> None:
        decision_cpu = RootCauseDecision(
            root_cause_service="checkoutservice",
            ranked_candidates=self.state.current_rankings,
            confidence=0.9,
            budget_used=5,
            budget_remaining=15,
        )
        act_cpu = propose_remediation(decision_cpu, fault_type="cpu")
        self.assertEqual(act_cpu.action_type, "scale_service")
        self.assertEqual(act_cpu.target_service, "checkoutservice")
        self.assertTrue(act_cpu.rollback_action)

        act_mem = propose_remediation(decision_cpu, fault_type="mem")
        self.assertEqual(act_mem.action_type, "restart_service")

        act_delay = propose_remediation(decision_cpu, fault_type="delay")
        self.assertEqual(act_delay.action_type, "clear_connection_pool")

    def test_execute_simulated_remediation(self) -> None:
        action = RemediationAction(
            target_service="checkoutservice",
            action_type="scale_service",
            preconditions=("service_must_be_degraded",),
            risk_level="LOW",
            rollback_action="restore_initial_replica_count",
            parameters={"replicas_delta": 2},
        )
        res = execute_simulated_remediation(action, self.state)
        self.assertTrue(res.success)
        self.assertTrue(any("SANDBOX SIMULATION" in log for log in res.execution_log))
        self.assertEqual(self.state.remediation_result, res)
        self.assertTrue(self.state.remediation_authorized)
        self.assertEqual(self.state.status, "REMEDIATION_AUTHORIZED")

    def test_low_confidence_diagnosis_remediation_blocked(self) -> None:
        # Confidence is 0.55, threshold is 0.80
        self.state.current_rankings = (
            RankedHypothesis(service="checkoutservice", rank=1, score=0.55, confidence=0.55),
            RankedHypothesis(service="cartservice", rank=2, score=0.20, confidence=0.20),
        )
        action = RemediationAction(
            target_service="checkoutservice",
            action_type="scale_service",
            preconditions=("service_must_be_degraded",),
            risk_level="LOW",
            rollback_action="restore_initial_replica_count",
        )
        is_valid, reason = validate_remediation(action, self.state, min_confidence=0.80)
        self.assertFalse(is_valid)
        self.assertIn("Decision confidence (0.550) is below remediation authorization threshold", reason)

        res = execute_simulated_remediation(action, self.state, min_confidence=0.80)
        self.assertFalse(res.success)
        self.assertFalse(self.state.remediation_authorized)
        self.assertEqual(self.state.status, "DIAGNOSIS_COMPLETE")
        self.assertTrue(any("SAFETY GATE BLOCKED" in log for log in res.execution_log))

    def test_low_confidence_override_allowed(self) -> None:
        self.state.current_rankings = (
            RankedHypothesis(service="checkoutservice", rank=1, score=0.55, confidence=0.55),
            RankedHypothesis(service="cartservice", rank=2, score=0.20, confidence=0.20),
        )
        action = RemediationAction(
            target_service="checkoutservice",
            action_type="scale_service",
            preconditions=("service_must_be_degraded",),
            risk_level="LOW",
            rollback_action="restore_initial_replica_count",
        )
        # With allow_low_confidence=True, remediation is authorized
        res = execute_simulated_remediation(action, self.state, min_confidence=0.80, allow_low_confidence=True)
        self.assertTrue(res.success)
        self.assertTrue(self.state.remediation_authorized)
        self.assertEqual(self.state.status, "REMEDIATION_AUTHORIZED")


if __name__ == "__main__":
    unittest.main()
