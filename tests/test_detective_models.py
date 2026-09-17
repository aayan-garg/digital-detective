"""Unit tests for detective data models.
"""

import unittest

from digital_detective.detective.models import (
    CausalConsistencyResult,
    EvidenceItem,
    InvestigationState,
    RankedHypothesis,
    RecoveryVerificationResult,
    RemediationAction,
    RemediationExecutionResult,
    RootCauseDecision,
    ToolQueryRecord,
)


class TestDetectiveModels(unittest.TestCase):
    def test_investigation_state_budget_tracking(self) -> None:
        state = InvestigationState(
            incident_id="inc_1",
            system="ob",
            candidate_universe=("adservice", "cartservice"),
            budget=20,
            remaining_budget=20,
        )
        self.assertEqual(state.total_query_cost, 0)
        self.assertEqual(state.remaining_budget, 20)

        record = ToolQueryRecord(
            query_id="q1",
            tool_name="get_metrics",
            service="adservice",
            cost=1,
            timestamp=100.0,
            status="SUCCESS",
        )
        state.add_query_record(record)
        state.remaining_budget -= 1

        self.assertEqual(state.total_query_cost, 1)
        self.assertEqual(state.remaining_budget, 19)
        self.assertEqual(len(state.queries_executed), 1)

    def test_state_serialization(self) -> None:
        state = InvestigationState(
            incident_id="inc_1",
            system="ob",
            candidate_universe=("adservice", "cartservice"),
            budget=10,
            remaining_budget=10,
        )
        ev = EvidenceItem(
            service="adservice",
            modality="metrics",
            signal="cpu_spike",
            magnitude=0.9,
            confidence_contribution=0.3,
        )
        state.add_evidence(ev)
        d = state.to_dict()
        self.assertEqual(d["incident_id"], "inc_1")
        self.assertEqual(d["evidence_count"], 1)
        self.assertEqual(d["budget"], 10)


if __name__ == "__main__":
    unittest.main()
