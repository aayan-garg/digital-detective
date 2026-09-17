"""Unit tests for detective telemetry tools and query budget enforcement.
"""

import unittest

from digital_detective.detective.models import InvestigationState
from digital_detective.detective.tools import DetectiveTools


class DummyCase:
    def __init__(self):
        self.ground_truth = type("GT", (), {"values": {"root_cause_service": "cartservice", "fault": "cpu"}})()
        self.metrics = None
        self.traces = None
        self.logs = None


class TestDetectiveTools(unittest.TestCase):
    def setUp(self) -> None:
        self.case = DummyCase()
        self.universe = ("cartservice", "checkoutservice")
        self.tools = DetectiveTools(case=self.case, candidate_universe=self.universe)

    def test_query_budget_deduction(self) -> None:
        state = InvestigationState(
            incident_id="test",
            system="ob",
            candidate_universe=self.universe,
            budget=5,
            remaining_budget=5,
        )

        # get_metrics costs 1
        success, summary, evs = self.tools.execute_query(state, "get_metrics", "cartservice")
        self.assertTrue(success)
        self.assertEqual(state.remaining_budget, 4)
        self.assertEqual(len(state.queries_executed), 1)
        self.assertEqual(state.queries_executed[0].status, "SUCCESS")

        # get_recent_change costs 2
        success, summary, evs = self.tools.execute_query(state, "get_recent_change", "cartservice")
        self.assertTrue(success)
        self.assertEqual(state.remaining_budget, 2)
        self.assertEqual(state.total_query_cost, 3)

    def test_query_budget_strict_rejection(self) -> None:
        state = InvestigationState(
            incident_id="test",
            system="ob",
            candidate_universe=self.universe,
            budget=2,
            remaining_budget=2,
        )

        # get_traces costs 3 (exceeds budget 2)
        success, summary, evs = self.tools.execute_query(state, "get_traces", "cartservice")
        self.assertFalse(success)
        self.assertIn("requires cost 3, but remaining budget is 2", summary)
        self.assertEqual(state.remaining_budget, 2)  # Budget untouched
        self.assertEqual(len(state.queries_executed), 1)
        self.assertEqual(state.queries_executed[0].status, "REJECTED_BUDGET")

    def test_get_neighbors_tool(self) -> None:
        state = InvestigationState(
            incident_id="test",
            system="ob",
            candidate_universe=self.universe,
            budget=5,
            remaining_budget=5,
        )
        success, summary, evs = self.tools.execute_query(state, "get_neighbors", "cartservice")
        self.assertTrue(success)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].modality, "topology")

    def test_get_service_health_tool(self) -> None:
        state = InvestigationState(
            incident_id="test",
            system="ob",
            candidate_universe=self.universe,
            budget=5,
            remaining_budget=5,
        )
        success, summary, evs = self.tools.execute_query(state, "get_service_health", "cartservice")
        self.assertTrue(success)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].modality, "health")


if __name__ == "__main__":
    unittest.main()
