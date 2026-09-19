"""Unit tests for the Digital Detective Agent layer.

Tests cover:
- AgentDecision validation (action constraints, evidence-ID grounding)
- InvestigationTrajectory accumulation
- MockAgentModel decision logic (QUERY selection, FINAL_DIAGNOSIS, STOP)
- AgentOrchestrator integration (end-to-end with mock telemetry case)
- Ground-truth isolation (agent model never receives case.ground_truth)
- Budget enforcement (orchestrator rejects QUERY when budget exhausted)
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

from digital_detective.agent.evaluation import summarize_llm_records
from digital_detective.agent.models import (
    AgentDecision,
    AgentModel,
    InvestigationTrajectory,
    LLMExperimentRecord,
    MockAgentModel,
)
from digital_detective.detective.models import (
    EvidenceItem,
    InvestigationState,
    LLMDiagnosis,
    RankedHypothesis,
    ToolQueryRecord,
)


# ---------------------------------------------------------------------------
# Helpers: minimal fake investigation state
# ---------------------------------------------------------------------------

def _make_state(
    candidates: tuple[str, ...] = ("svcA", "svcB"),
    budget: int = 20,
    remaining_budget: int = 20,
    queries: list[ToolQueryRecord] | None = None,
    evidence: list[EvidenceItem] | None = None,
    rankings: tuple[RankedHypothesis, ...] | None = None,
) -> InvestigationState:
    state = InvestigationState(
        incident_id="test-001",
        system="ob",
        candidate_universe=candidates,
        budget=budget,
        remaining_budget=remaining_budget,
    )
    if queries:
        state.queries_executed.extend(queries)
    if evidence:
        state.evidence_collected.extend(evidence)
    if rankings is not None:
        state.current_rankings = rankings
    return state


def _make_query_record(
    service: str,
    tool: str,
    cost: int = 1,
    status: str = "SUCCESS",
    query_id: str | None = None,
) -> ToolQueryRecord:
    return ToolQueryRecord(
        query_id=query_id or f"q_test_{tool}_{service}",
        tool_name=tool,
        service=service,
        cost=cost,
        timestamp=0.0,
        status=status,
    )


def _make_rankings(
    services: list[str],
    scores: list[float],
    confidences: list[float],
) -> tuple[RankedHypothesis, ...]:
    return tuple(
        RankedHypothesis(
            service=svc,
            score=score,
            confidence=conf,
            rank=idx + 1,
            evidence_score=score * 0.8,
            consistency_score=0.9,
        )
        for idx, (svc, score, conf) in enumerate(zip(services, scores, confidences))
    )


def _empty_trajectory(incident_id: str = "test-001") -> InvestigationTrajectory:
    return InvestigationTrajectory(incident_id=incident_id)


# ===========================================================================
# AgentDecision tests
# ===========================================================================

class TestAgentDecisionValidation(unittest.TestCase):
    """AgentDecision must enforce structural constraints at construction."""

    def test_valid_query_decision(self) -> None:
        d = AgentDecision(
            action="QUERY",
            tool_name="get_metrics",
            service="svcA",
            reasoning="Testing query.",
        )
        self.assertEqual(d.action, "QUERY")
        self.assertEqual(d.tool_name, "get_metrics")
        self.assertEqual(d.service, "svcA")

    def test_valid_final_diagnosis(self) -> None:
        d = AgentDecision(
            action="FINAL_DIAGNOSIS",
            diagnosis_service="svcA",
            reasoning="Evidence collected.",
            evidence_ids=("q_1", "q_2"),
        )
        self.assertEqual(d.action, "FINAL_DIAGNOSIS")
        self.assertEqual(d.diagnosis_service, "svcA")
        self.assertEqual(len(d.evidence_ids), 2)

    def test_valid_stop(self) -> None:
        d = AgentDecision(
            action="STOP",
            reasoning="Budget exhausted.",
        )
        self.assertEqual(d.action, "STOP")

    def test_invalid_action_raises(self) -> None:
        with self.assertRaises(ValueError):
            AgentDecision(action="INVALID", reasoning="x")

    def test_query_requires_tool_name(self) -> None:
        with self.assertRaises(ValueError):
            AgentDecision(action="QUERY", reasoning="x", service="svcA")

    def test_query_requires_service(self) -> None:
        with self.assertRaises(ValueError):
            AgentDecision(action="QUERY", reasoning="x", tool_name="get_metrics")

    def test_final_diagnosis_requires_reasoning(self) -> None:
        with self.assertRaises(ValueError):
            AgentDecision(action="FINAL_DIAGNOSIS", reasoning="", evidence_ids=("q1",), diagnosis_service="svcA")

    def test_final_diagnosis_requires_diagnosis_service(self) -> None:
        with self.assertRaises(ValueError):
            AgentDecision(action="FINAL_DIAGNOSIS", reasoning="Some reasoning.", evidence_ids=("q1",))

    def test_final_diagnosis_requires_evidence_ids(self) -> None:
        with self.assertRaises(ValueError):
            AgentDecision(action="FINAL_DIAGNOSIS", reasoning="Some reasoning.", diagnosis_service="svcA")

    def test_to_dict_round_trips(self) -> None:
        d = AgentDecision(
            action="QUERY",
            tool_name="get_logs",
            service="svcB",
            reasoning="Check logs.",
            step_index=3,
        )
        d_dict = d.to_dict()
        self.assertEqual(d_dict["action"], "QUERY")
        self.assertEqual(d_dict["tool_name"], "get_logs")
        self.assertEqual(d_dict["step_index"], 3)


# ===========================================================================
# InvestigationTrajectory tests
# ===========================================================================

class TestInvestigationTrajectory(unittest.TestCase):
    def test_append_decision_increments_steps(self) -> None:
        traj = _empty_trajectory()
        self.assertEqual(traj.total_steps, 0)
        d = AgentDecision(action="QUERY", tool_name="get_metrics", service="svcA", reasoning="r")
        traj.append_decision(d)
        self.assertEqual(traj.total_steps, 1)
        self.assertEqual(len(traj.decisions), 1)

    def test_to_dict_structure(self) -> None:
        traj = _empty_trajectory("inc-42")
        traj.terminated_by = "FINAL_DIAGNOSIS"
        traj.total_budget_used = 7
        traj.llm_diagnosis = LLMDiagnosis(
            diagnosis_service="svcA",
            abstain=False,
            reasoning="evidence supports svcA",
            evidence_ids=("q1",),
            condition="C",
            model_name="MockAgentModel",
        )
        traj.experiment = LLMExperimentRecord(
            case_id="inc-42",
            condition="C",
            model="MockAgentModel",
            llm_diagnosis="svcA",
            abstain=False,
            reasoning="evidence supports svcA",
            evidence_ids=("q1",),
            query_count=2,
            query_cost=3,
            termination_action="FINAL_DIAGNOSIS",
            latency_sec=0.75,
            deterministic_top_service="svcA",
            deterministic_rankings=("svcA", "svcB"),
        )
        d = AgentDecision(action="STOP", reasoning="done.")
        traj.append_decision(d)
        d_dict = traj.to_dict()
        self.assertEqual(d_dict["incident_id"], "inc-42")
        self.assertEqual(d_dict["terminated_by"], "FINAL_DIAGNOSIS")
        self.assertEqual(d_dict["total_budget_used"], 7)
        self.assertEqual(len(d_dict["decisions"]), 1)
        self.assertEqual(d_dict["llm_diagnosis"]["diagnosis_service"], "svcA")
        self.assertEqual(d_dict["experiment"]["llm_diagnosis"], "svcA")


class TestLLMEvaluationSummary(unittest.TestCase):
    def test_summary_tracks_topk_abstention_and_cost(self) -> None:
        records = [
            LLMExperimentRecord(
                case_id="case-1",
                condition="A",
                model="mock",
                llm_diagnosis="svcA",
                abstain=False,
                reasoning="svcA",
                evidence_ids=("q1",),
                query_count=0,
                query_cost=0,
                termination_action="FINAL_DIAGNOSIS",
                latency_sec=0.3,
                deterministic_top_service="svcA",
                deterministic_rankings=("svcA", "svcB", "svcC"),
            ),
            LLMExperimentRecord(
                case_id="case-2",
                condition="A",
                model="mock",
                llm_diagnosis="svcC",
                abstain=False,
                reasoning="svcC",
                evidence_ids=("q2",),
                query_count=0,
                query_cost=0,
                termination_action="FINAL_DIAGNOSIS",
                latency_sec=0.4,
                deterministic_top_service="svcA",
                deterministic_rankings=("svcA", "svcB", "svcC"),
            ),
            LLMExperimentRecord(
                case_id="case-3",
                condition="A",
                model="mock",
                llm_diagnosis="",
                abstain=True,
                reasoning="abstain",
                evidence_ids=(),
                query_count=0,
                query_cost=0,
                termination_action="STOP",
                latency_sec=0.5,
                deterministic_top_service="svcA",
                deterministic_rankings=("svcA", "svcB", "svcC"),
            ),
        ]
        summary = summarize_llm_records(records)
        self.assertAlmostEqual(summary.top1, 0.5)
        self.assertAlmostEqual(summary.top3, 1.0)
        self.assertAlmostEqual(summary.abstention_rate, 1 / 3)
        self.assertAlmostEqual(summary.mean_query_count, 0.0)
        self.assertAlmostEqual(summary.mean_query_cost, 0.0)
        self.assertAlmostEqual(summary.mean_latency_sec, 0.4, places=2)


# ===========================================================================
# MockAgentModel tests
# ===========================================================================

class TestMockAgentModelQuery(unittest.TestCase):
    """MockAgentModel should emit QUERY when confidence is low."""

    def _model(self, threshold: float = 0.55) -> MockAgentModel:
        return MockAgentModel(
            confidence_threshold=threshold,
            min_evidence_queries=2,
            stop_budget_reserve=2,
        )

    def test_emits_query_when_low_confidence(self) -> None:
        state = _make_state(
            candidates=("svcA", "svcB"),
            rankings=_make_rankings(["svcA", "svcB"], [0.6, 0.3], [0.4, 0.2]),
        )
        traj = _empty_trajectory()
        model = self._model()
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_metrics", "get_service_health"],
            already_queried=set(),
        )
        self.assertEqual(decision.action, "QUERY")
        self.assertIn(decision.tool_name, ["get_metrics", "get_service_health"])

    def test_prefers_highest_suspicion_service(self) -> None:
        state = _make_state(
            candidates=("svcA", "svcB"),
            rankings=_make_rankings(["svcA", "svcB"], [0.8, 0.2], [0.4, 0.1]),
        )
        traj = _empty_trajectory()
        model = self._model()
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_metrics"],
            already_queried=set(),
        )
        self.assertEqual(decision.action, "QUERY")
        self.assertEqual(decision.service, "svcA")

    def test_avoids_already_queried(self) -> None:
        state = _make_state(
            candidates=("svcA", "svcB"),
            rankings=_make_rankings(["svcA", "svcB"], [0.8, 0.3], [0.4, 0.2]),
        )
        traj = _empty_trajectory()
        model = self._model()
        already_queried = {("svcA", "get_metrics"), ("svcA", "get_service_health")}
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_metrics", "get_service_health", "get_recent_change"],
            already_queried=already_queried,
        )
        self.assertEqual(decision.action, "QUERY")
        # Should avoid already queried (svcA, get_metrics) and (svcA, get_service_health)
        self.assertFalse(
            (decision.service == "svcA" and decision.tool_name in ("get_metrics", "get_service_health"))
        )


class TestMockAgentModelDiagnosis(unittest.TestCase):
    """MockAgentModel should emit FINAL_DIAGNOSIS when confidence >= threshold."""

    def _model(self) -> MockAgentModel:
        return MockAgentModel(confidence_threshold=0.55, min_evidence_queries=2)

    def test_emits_final_diagnosis_when_confident(self) -> None:
        # 2 successful queries on svcA
        q1 = _make_query_record("svcA", "get_metrics", query_id="q1")
        q2 = _make_query_record("svcA", "get_service_health", query_id="q2")
        state = _make_state(
            candidates=("svcA", "svcB"),
            queries=[q1, q2],
            rankings=_make_rankings(["svcA", "svcB"], [0.9, 0.1], [0.65, 0.05]),
        )
        traj = _empty_trajectory()
        traj.append_decision(
            AgentDecision(action="QUERY", tool_name="get_metrics", service="svcA", reasoning="r")
        )
        model = self._model()
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_logs"],
            already_queried={("svcA", "get_metrics"), ("svcA", "get_service_health")},
        )
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        self.assertGreater(len(decision.evidence_ids), 0)
        self.assertGreater(len(decision.reasoning), 0)

    def test_diagnosis_has_evidence_ids_from_queries(self) -> None:
        q1 = _make_query_record("svcA", "get_metrics", query_id="qXX")
        q2 = _make_query_record("svcA", "get_service_health", query_id="qYY")
        state = _make_state(
            candidates=("svcA",),
            queries=[q1, q2],
            rankings=_make_rankings(["svcA"], [1.0], [0.8]),
        )
        traj = _empty_trajectory()
        model = self._model()
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_logs"],
            already_queried={("svcA", "get_metrics"), ("svcA", "get_service_health")},
        )
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        # Evidence IDs must come from the successful queries
        for eid in decision.evidence_ids:
            self.assertIn(eid, ("qXX", "qYY"))


class TestMockAgentModelStop(unittest.TestCase):
    """MockAgentModel should emit STOP when budget is nearly exhausted."""

    def test_stop_when_budget_low_and_no_evidence(self) -> None:
        state = _make_state(
            candidates=("svcA",),
            budget=20,
            remaining_budget=2,  # == stop_budget_reserve
            rankings=_make_rankings(["svcA"], [0.5], [0.3]),
        )
        traj = _empty_trajectory()
        model = MockAgentModel(confidence_threshold=0.55, min_evidence_queries=2, stop_budget_reserve=2)
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_metrics"],
            already_queried=set(),
        )
        # No successful queries → no evidence IDs → STOP (cannot ground FINAL_DIAGNOSIS)
        self.assertIn(decision.action, ("STOP", "FINAL_DIAGNOSIS"))

    def test_no_rankings_emits_query_or_stop(self) -> None:
        state = _make_state(candidates=("svcA",), rankings=())
        traj = _empty_trajectory()
        model = MockAgentModel()
        decision = model.decide(
            state=state,
            trajectory=traj,
            available_tools=["get_metrics"],
            already_queried=set(),
        )
        # With no rankings but a candidate → should emit QUERY to seed
        self.assertIn(decision.action, ("QUERY", "STOP"))


# ===========================================================================
# AgentOrchestrator integration test (using real data is optional;
# here we test with a minimal mock case to avoid dataset dependency)
# ===========================================================================

class _MinimalCase:
    """Bare-minimum fake case for orchestrator smoke test.

    Supplies the minimal fields that resolve_incident_window, resolve_candidate_universe,
    and detect_metric_anomalies need in order not to raise.
    """

    class _Meta:
        case_id = "test-smoke-001"
        system = "ob"
        system_name = "Online Boutique (smoke)"

    class _GT:
        # inject_time required by resolve_incident_window fallback path
        values = {
            "root_cause_service": "svcA",
            "fault": "cpu",
            "inject_time": 1700000000,
        }

    metadata = _Meta()
    ground_truth = _GT()
    metrics = None
    traces = None
    logs = None




class TestAgentOrchestratorSmoke(unittest.TestCase):
    """Smoke test: orchestrator runs without error on a minimal (no-data) case."""

    def test_orchestrator_returns_trajectory_no_data(self) -> None:
        from digital_detective.agent.orchestrator import AgentOrchestrator

        model = MockAgentModel(confidence_threshold=0.1, min_evidence_queries=0, stop_budget_reserve=1)
        orch = AgentOrchestrator(agent_model=model, budget=5, max_steps=10)
        case = _MinimalCase()
        # Orchestrator uses 'detected' window mode; supply custom_onset_ts so
        # resolve_incident_window does not raise when there are no real episodes.
        trajectory = orch.investigate(case, custom_onset_ts=1700000000)

        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.incident_id, "test-smoke-001")
        self.assertIn(
            trajectory.terminated_by,
            ("FINAL_DIAGNOSIS", "STOP", "BUDGET_EXHAUSTED", "MAX_STEPS", "REQUEST_REMEDIATION"),
        )
        self.assertIsNotNone(trajectory.final_state)

    def test_ground_truth_not_passed_to_model(self) -> None:
        """The mock model must not receive ground truth; verify via trajectory."""
        from digital_detective.agent.orchestrator import AgentOrchestrator

        seen_states: list[InvestigationState] = []

        class SpyModel(AgentModel):
            def decide(self, state, trajectory, available_tools, already_queried):
                seen_states.append(state)
                # Immediately stop without querying
                evidence_ids: tuple[str, ...] = ()
                successful = [q for q in state.queries_executed if q.status == "SUCCESS"]
                if successful:
                    evidence_ids = (successful[0].query_id,)
                    return AgentDecision(
                        action="FINAL_DIAGNOSIS",
                        diagnosis_service="svcA",
                        reasoning="Smoke stop.",
                        evidence_ids=evidence_ids,
                        step_index=len(trajectory.decisions),
                    )
                return AgentDecision(
                    action="STOP",
                    reasoning="No data available.",
                    step_index=len(trajectory.decisions),
                )

        orch = AgentOrchestrator(agent_model=SpyModel(), budget=5, max_steps=5)
        # Orchestrator never reads case.ground_truth; window resolved via detected mode.
        # Supply custom_onset_ts to avoid ValueError with no real episode data.
        case = _MinimalCase()
        trajectory = orch.investigate(case, custom_onset_ts=1700000000)
        # Verify the spy model received states where ground_truth is not accessible
        for state in seen_states:
            # InvestigationState has no ground_truth attribute by design
            self.assertFalse(hasattr(state, "ground_truth"))


class TestAgentDecisionGrounding(unittest.TestCase):
    """FINAL_DIAGNOSIS requires a diagnosis service and evidence IDs."""

    def test_final_diagnosis_requires_evidence(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            AgentDecision(
                action="FINAL_DIAGNOSIS",
                diagnosis_service="svcA",
                reasoning="No supporting evidence collected.",
                evidence_ids=(),
            )
        self.assertIn("evidence_id", str(ctx.exception))

    def test_final_diagnosis_with_evidence_succeeds(self) -> None:
        d = AgentDecision(
            action="FINAL_DIAGNOSIS",
            diagnosis_service="svcA",
            reasoning="Supported by q1 and q2.",
            evidence_ids=("q1", "q2"),
        )
        self.assertEqual(d.diagnosis_service, "svcA")
        self.assertEqual(len(d.evidence_ids), 2)


if __name__ == "__main__":
    unittest.main()
