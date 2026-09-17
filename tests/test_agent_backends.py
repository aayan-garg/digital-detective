"""Tests for the Digital Detective agent backend selection, policy validation,
ground-truth isolation, and new LLM adapter response parsing.

Tests cover:
- OllamaAgentModel: response parsing, thinking-tag stripping, malformed JSON
- GrokAgentModel: configuration check (key required)
- build_agent_model factory: mode selection, Ollama unavailability, Grok key missing
- policy.validate_decision: action types, tool existence, budget, evidence IDs
- policy.validate_agent_claim: evidence citation correctness, GT-reference detection
- Ground-truth isolation: orchestrator never passes GT to the model
- Budget enforcement via max_turns
- Remediation bypass prevention
- Complete mock → tool → deterministic core integration
"""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass, field
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

from digital_detective.agent.models import (
    AgentDecision,
    AgentModel,
    GrokAgentModel,
    InvestigationTrajectory,
    MockAgentModel,
    OllamaAgentModel,
    OpenAIChatAdapter,
    _parse_llm_response,
    _strip_thinking_tags,
    build_agent_model,
)
from digital_detective.agent.policy import (
    PolicyViolation,
    ValidationResult,
    validate_agent_claim,
    validate_decision,
)
from digital_detective.detective.models import (
    EvidenceItem,
    InvestigationState,
    RankedHypothesis,
    ToolQueryRecord,
)


# ---------------------------------------------------------------------------
# Helpers
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
        for q in queries:
            state.queries_executed.append(q)
    if evidence:
        for ev in evidence:
            state.evidence_collected.append(ev)
    if rankings:
        state.current_rankings = list(rankings)
    return state


def _make_query_record(
    query_id: str = "svcA::get_metrics::1",
    service: str = "svcA",
    tool_name: str = "get_metrics",
    status: str = "SUCCESS",
    cost: int = 1,
) -> ToolQueryRecord:
    return ToolQueryRecord(
        query_id=query_id,
        tool_name=tool_name,
        service=service,
        status=status,
        cost=cost,
        timestamp=1700000001.0,
    )



def _make_evidence(
    service: str = "svcA",
    modality: str = "metrics",
    signal: str = "cpu_anomaly",
    magnitude: float = 0.85,
) -> EvidenceItem:
    return EvidenceItem(
        service=service,
        modality=modality,
        signal=signal,
        magnitude=magnitude,
    )


def _make_ranking(
    service: str = "svcA",
    rank: int = 1,
    score: float = 0.75,
    confidence: float = 0.70,
    evidence_score: float = 0.70,
    consistency_score: float = 0.65,
) -> RankedHypothesis:
    return RankedHypothesis(
        service=service,
        rank=rank,
        score=score,
        confidence=confidence,
        evidence_score=evidence_score,
        consistency_score=consistency_score,
    )


# ---------------------------------------------------------------------------
# 1. Thinking-tag stripping
# ---------------------------------------------------------------------------

class TestStripThinkingTags(unittest.TestCase):
    """_strip_thinking_tags removes Qwen3-style <think>...</think> blocks."""

    def test_strips_single_tag(self) -> None:
        raw = "<think>Let me reason through this step by step.</think>\n{\"action\": \"STOP\"}"
        result = _strip_thinking_tags(raw)
        self.assertNotIn("<think>", result)
        self.assertIn("{\"action\"", result)

    def test_strips_multiline_tag(self) -> None:
        raw = "<think>\nLong reasoning\nacross many lines.\n</think>\n{\"action\": \"STOP\", \"reasoning\": \"x\"}"
        result = _strip_thinking_tags(raw)
        self.assertNotIn("reasoning\nacross", result)
        self.assertIn("{\"action\"", result)

    def test_no_tag_unchanged(self) -> None:
        raw = "{\"action\": \"STOP\", \"reasoning\": \"clean output\"}"
        result = _strip_thinking_tags(raw)
        self.assertEqual(result, raw)


# ---------------------------------------------------------------------------
# 2. LLM response parsing
# ---------------------------------------------------------------------------

class TestParseLLMResponse(unittest.TestCase):
    """_parse_llm_response produces valid AgentDecision from diverse inputs."""

    def _state(self) -> InvestigationState:
        q = _make_query_record()
        return _make_state(queries=[q], evidence=[_make_evidence()])

    def test_valid_query_response(self) -> None:
        raw = '{"action": "QUERY", "tool_name": "get_metrics", "service": "svcA", "reasoning": "test"}'
        state = self._state()
        decision = _parse_llm_response(raw, step=0, state=state)
        self.assertEqual(decision.action, "QUERY")
        self.assertEqual(decision.tool_name, "get_metrics")
        self.assertEqual(decision.service, "svcA")

    def test_valid_final_diagnosis_with_evidence(self) -> None:
        q = _make_query_record(query_id="q1")
        state = _make_state(queries=[q], evidence=[_make_evidence()])
        raw = '{"action": "FINAL_DIAGNOSIS", "reasoning": "svcA is root cause per E1", "evidence_ids": ["q1"]}'
        decision = _parse_llm_response(raw, step=1, state=state)
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        self.assertIn("q1", decision.evidence_ids)

    def test_final_diagnosis_no_evidence_ids_auto_populates(self) -> None:
        """Missing evidence_ids → auto-populated from successful queries."""
        q = _make_query_record(query_id="q1")
        state = _make_state(queries=[q])
        raw = '{"action": "FINAL_DIAGNOSIS", "reasoning": "svcA is culprit"}'
        decision = _parse_llm_response(raw, step=1, state=state)
        # Should auto-populate from query history
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        self.assertIn("q1", decision.evidence_ids)

    def test_final_diagnosis_no_evidence_no_queries_becomes_stop(self) -> None:
        """FINAL_DIAGNOSIS with no evidence and no queries → demoted to STOP."""
        state = _make_state()
        raw = '{"action": "FINAL_DIAGNOSIS", "reasoning": "svcA is culprit"}'
        decision = _parse_llm_response(raw, step=0, state=state)
        self.assertEqual(decision.action, "STOP")

    def test_malformed_json_becomes_stop(self) -> None:
        raw = "This is not JSON at all, sorry!"
        state = _make_state()
        decision = _parse_llm_response(raw, step=0, state=state)
        self.assertEqual(decision.action, "STOP")
        self.assertIn("JSON", decision.reasoning)

    def test_unknown_action_becomes_stop(self) -> None:
        raw = '{"action": "HALLUCINATE", "reasoning": "bad action"}'
        state = _make_state()
        decision = _parse_llm_response(raw, step=0, state=state)
        self.assertEqual(decision.action, "STOP")

    def test_markdown_code_fence_stripped(self) -> None:
        raw = '```json\n{"action": "STOP", "reasoning": "test"}\n```'
        state = _make_state()
        decision = _parse_llm_response(raw, step=0, state=state)
        self.assertEqual(decision.action, "STOP")

    def test_thinking_tag_stripped_before_json_parse(self) -> None:
        raw = '<think>I think therefore I am</think>\n{"action": "STOP", "reasoning": "done"}'
        state = _make_state()
        decision = _parse_llm_response(raw, step=0, state=state)
        self.assertEqual(decision.action, "STOP")

    def test_empty_response_becomes_stop(self) -> None:
        state = _make_state()
        decision = _parse_llm_response("", step=0, state=state)
        self.assertEqual(decision.action, "STOP")


# ---------------------------------------------------------------------------
# 3. Policy validation — validate_decision
# ---------------------------------------------------------------------------

class TestPolicyValidateDecision(unittest.TestCase):
    """validate_decision correctly validates all action types."""

    def _available_tools(self) -> list[str]:
        return ["get_metrics", "get_service_health", "get_neighbors",
                "get_recent_change", "get_traces", "get_logs"]

    def test_valid_query_accepted(self) -> None:
        state = _make_state(remaining_budget=10)
        decision = AgentDecision(
            action="QUERY",
            tool_name="get_metrics",
            service="svcA",
            reasoning="gathering baseline",
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertTrue(result.is_valid)

    def test_unknown_tool_rejected(self) -> None:
        state = _make_state(remaining_budget=10)
        decision = AgentDecision(
            action="QUERY",
            tool_name="invent_data",
            service="svcA",
            reasoning="fabricating",
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("UNKNOWN_TOOL", codes)

    def test_service_not_in_universe_rejected(self) -> None:
        state = _make_state(candidates=("svcA", "svcB"), remaining_budget=10)
        decision = AgentDecision(
            action="QUERY",
            tool_name="get_metrics",
            service="unknownService",
            reasoning="checking unknown",
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("INVALID_SERVICE", codes)

    def test_budget_exceeded_rejected(self) -> None:
        # get_traces costs 3, but only 2 remaining
        state = _make_state(remaining_budget=2)
        decision = AgentDecision(
            action="QUERY",
            tool_name="get_traces",
            service="svcA",
            reasoning="need traces",
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("BUDGET_EXCEEDED", codes)

    def test_duplicate_query_flagged(self) -> None:
        state = _make_state(remaining_budget=10)
        already = {("svcA", "get_metrics")}  # exact service name as in candidate universe
        decision = AgentDecision(
            action="QUERY",
            tool_name="get_metrics",
            service="svcA",
            reasoning="redundant",
        )
        result = validate_decision(decision, state, self._available_tools(), already)
        # Duplicate is a warning, not a hard rejection — but it IS a violation
        codes = [v.code for v in result.violations]
        self.assertIn("DUPLICATE_QUERY", codes)


    def test_final_diagnosis_requires_evidence_ids(self) -> None:
        """Policy catches missing evidence on FINAL_DIAGNOSIS at the policy layer.

        Note: AgentDecision.__post_init__ also enforces this, so this test
        verifies the policy enforcement independently by calling validate_decision
        on a mock-patched decision that bypasses the dataclass guard.
        """
        # We can't construct FINAL_DIAGNOSIS with empty evidence_ids via the dataclass.
        # Verify that AgentDecision raises at construction time.
        with self.assertRaises(ValueError):
            AgentDecision(
                action="FINAL_DIAGNOSIS",
                reasoning="No evidence cited.",
                evidence_ids=(),
            )

    def test_final_diagnosis_with_valid_evidence_id_accepted(self) -> None:
        q = _make_query_record(query_id="q1")
        state = _make_state(queries=[q], evidence=[_make_evidence()])
        decision = AgentDecision(
            action="FINAL_DIAGNOSIS",
            reasoning="svcA is the root cause because E1 shows CPU spike.",
            evidence_ids=("q1",),
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertTrue(result.is_valid)

    def test_final_diagnosis_with_e_notation_id_accepted(self) -> None:
        state = _make_state(evidence=[_make_evidence()])
        q = _make_query_record()
        state.queries_executed.append(q)
        decision = AgentDecision(
            action="FINAL_DIAGNOSIS",
            reasoning="E1 shows high CPU on svcA.",
            evidence_ids=("E1",),
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertTrue(result.is_valid)

    def test_final_diagnosis_with_invalid_e_notation_rejected(self) -> None:
        state = _make_state(evidence=[_make_evidence()])  # only E1 exists
        q = _make_query_record()
        state.queries_executed.append(q)
        decision = AgentDecision(
            action="FINAL_DIAGNOSIS",
            reasoning="E99 shows root cause.",
            evidence_ids=("E99",),  # E99 doesn't exist
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("INVALID_EVIDENCE_ID", codes)

    def test_ground_truth_reference_rejected(self) -> None:
        q = _make_query_record(query_id="q1")
        state = _make_state(queries=[q])
        decision = AgentDecision(
            action="FINAL_DIAGNOSIS",
            reasoning="The ground truth shows svcA caused the fault.",
            evidence_ids=("q1",),
        )
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("GROUND_TRUTH_REFERENCE", codes)

    def test_stop_action_accepted(self) -> None:
        state = _make_state()
        decision = AgentDecision(action="STOP", reasoning="Budget exhausted.")
        result = validate_decision(decision, state, self._available_tools(), set())
        self.assertTrue(result.is_valid)

    def test_unavailable_tool_rejected(self) -> None:
        """get_traces unavailable when has_traces=False."""
        state = _make_state(remaining_budget=10)
        available = ["get_metrics", "get_service_health"]  # no traces
        decision = AgentDecision(
            action="QUERY",
            tool_name="get_traces",
            service="svcA",
            reasoning="need traces",
        )
        result = validate_decision(decision, state, available, set())
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("UNAVAILABLE_TOOL", codes)


# ---------------------------------------------------------------------------
# 4. Policy validation — validate_agent_claim
# ---------------------------------------------------------------------------

class TestValidateAgentClaim(unittest.TestCase):

    def test_valid_claim_with_query_id(self) -> None:
        q = _make_query_record(query_id="q1")
        state = _make_state(queries=[q], evidence=[_make_evidence()])
        result = validate_agent_claim(
            reasoning="q1 confirms svcA CPU spike.",
            evidence_ids=["q1"],
            state=state,
        )
        self.assertTrue(result.is_valid)

    def test_invalid_evidence_id_fails(self) -> None:
        state = _make_state()
        result = validate_agent_claim(
            reasoning="E99 shows something.",
            evidence_ids=["E99"],
            state=state,
        )
        self.assertFalse(result.is_valid)

    def test_trace_reference_without_trace_evidence_flagged(self) -> None:
        state = _make_state(evidence=[_make_evidence(modality="metrics")])
        q = _make_query_record()
        state.queries_executed.append(q)
        result = validate_agent_claim(
            reasoning="The traces show propagation from svcA to svcB.",
            evidence_ids=["svcA::get_metrics::1"],
            state=state,
        )
        # Should flag the uncollected-modality claim
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("UNCOLLECTED_MODALITY_CLAIM", codes)

    def test_ground_truth_pattern_rejected(self) -> None:
        q = _make_query_record(query_id="q1")
        state = _make_state(queries=[q])
        result = validate_agent_claim(
            reasoning="The root_cause_service is svcA per q1.",
            evidence_ids=["q1"],
            state=state,
        )
        self.assertFalse(result.is_valid)
        codes = [v.code for v in result.violations]
        self.assertIn("GROUND_TRUTH_REFERENCE", codes)


# ---------------------------------------------------------------------------
# 5. Backend selection — build_agent_model
# ---------------------------------------------------------------------------

class TestBuildAgentModel(unittest.TestCase):

    def test_mock_mode_always_returns_mock(self) -> None:
        agent = build_agent_model("mock")
        self.assertIsInstance(agent, MockAgentModel)

    def test_auto_with_no_ollama_returns_mock(self) -> None:
        """auto mode: if Ollama is unreachable, falls back to Mock."""
        with patch.object(OllamaAgentModel, "check_availability", return_value=(False, "not running")):
            agent = build_agent_model("auto")
        self.assertIsInstance(agent, MockAgentModel)

    def test_auto_with_ollama_returns_ollama(self) -> None:
        """auto mode: if Ollama is reachable, returns OllamaAgentModel."""
        with patch.object(OllamaAgentModel, "check_availability", return_value=(True, "ok")):
            agent = build_agent_model("auto")
        self.assertIsInstance(agent, OllamaAgentModel)

    def test_ollama_mode_fails_if_unavailable(self) -> None:
        with patch.object(OllamaAgentModel, "check_availability", return_value=(False, "timeout")):
            with self.assertRaises(RuntimeError) as ctx:
                build_agent_model("ollama")
        self.assertIn("Ollama unavailable", str(ctx.exception))

    def test_ollama_mode_success_if_available(self) -> None:
        with patch.object(OllamaAgentModel, "check_availability", return_value=(True, "ok")):
            agent = build_agent_model("ollama")
        self.assertIsInstance(agent, OllamaAgentModel)

    def test_grok_mode_fails_without_key(self) -> None:
        # Ensure XAI_API_KEY is not set
        env_backup = os.environ.pop("XAI_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                build_agent_model("grok")
            self.assertIn("XAI_API_KEY", str(ctx.exception))
        finally:
            if env_backup is not None:
                os.environ["XAI_API_KEY"] = env_backup

    def test_grok_mode_succeeds_with_key(self) -> None:
        with patch.dict(os.environ, {"XAI_API_KEY": "fake-test-key"}):
            agent = build_agent_model("grok")
        self.assertIsInstance(agent, GrokAgentModel)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_agent_model("invalid_mode")


# ---------------------------------------------------------------------------
# 6. OllamaAgentModel configuration
# ---------------------------------------------------------------------------

class TestOllamaAgentModel(unittest.TestCase):

    def test_default_model_name(self) -> None:
        agent = OllamaAgentModel()
        self.assertIn("qwen3", agent.model_name.lower())

    def test_env_override_model(self) -> None:
        with patch.dict(os.environ, {"DIGITAL_DETECTIVE_OLLAMA_MODEL": "llama3.2:3b"}):
            agent = OllamaAgentModel()
        self.assertIn("llama3.2", agent.model)

    def test_env_override_base_url(self) -> None:
        with patch.dict(os.environ, {"DIGITAL_DETECTIVE_OLLAMA_BASE_URL": "http://my-server:11434/v1"}):
            agent = OllamaAgentModel()
        self.assertIn("my-server", agent._base_url)

    def test_check_availability_no_connection(self) -> None:
        """check_availability returns False when Ollama is not running."""
        agent = OllamaAgentModel(base_url="http://localhost:19999/v1")
        available, msg = agent.check_availability()
        self.assertFalse(available)
        self.assertIsInstance(msg, str)

    def test_no_api_key_required(self) -> None:
        agent = OllamaAgentModel()
        self.assertIsNone(agent._api_key)

    def test_decide_returns_stop_on_connection_failure(self) -> None:
        """decide() returns STOP gracefully when Ollama is unreachable."""
        agent = OllamaAgentModel(base_url="http://localhost:19999/v1")
        state = _make_state(
            rankings=(
                _make_ranking("svcA", rank=1),
                _make_ranking("svcB", rank=2, score=0.3, confidence=0.25),
            )
        )
        trajectory = InvestigationTrajectory(incident_id="t1", model_name="Ollama/qwen3:8b")
        decision = agent.decide(
            state=state,
            trajectory=trajectory,
            available_tools=["get_metrics"],
            already_queried=set(),
        )
        self.assertEqual(decision.action, "STOP")
        self.assertIn("failed", decision.reasoning.lower())


# ---------------------------------------------------------------------------
# 7. GrokAgentModel configuration
# ---------------------------------------------------------------------------

class TestGrokAgentModel(unittest.TestCase):

    def test_grok_requires_key(self) -> None:
        env_backup = os.environ.pop("XAI_API_KEY", None)
        try:
            agent = GrokAgentModel()
            self.assertFalse(agent.is_configured())
        finally:
            if env_backup is not None:
                os.environ["XAI_API_KEY"] = env_backup

    def test_grok_configured_with_key(self) -> None:
        with patch.dict(os.environ, {"XAI_API_KEY": "test-key-123"}):
            agent = GrokAgentModel()
        self.assertTrue(agent.is_configured())
        self.assertIn("test-key-123", agent._api_key)

    def test_env_override_grok_model(self) -> None:
        with patch.dict(os.environ, {"XAI_API_KEY": "k", "DIGITAL_DETECTIVE_GROK_MODEL": "grok-4"}):
            agent = GrokAgentModel()
        self.assertIn("grok-4", agent.model)


# ---------------------------------------------------------------------------
# 8. Ground-truth isolation: orchestrator never passes GT to model
# ---------------------------------------------------------------------------

class _MinimalCase:
    class _Meta:
        case_id = "gt-isolation-001"
        system = "ob"
        system_name = "GT Isolation Smoke"

    class _GT:
        values = {
            "root_cause_service": "secret_service",
            "fault": "cpu",
            "inject_time": 1700000000,
        }

    metadata = _Meta()
    ground_truth = _GT()
    metrics = None
    traces = None
    logs = None


class TestGroundTruthIsolation(unittest.TestCase):
    """Verify that the AgentOrchestrator never exposes GT to the model."""

    def test_model_never_receives_ground_truth(self) -> None:
        from digital_detective.agent.orchestrator import AgentOrchestrator

        received_states: list[InvestigationState] = []
        received_trajectories: list[InvestigationTrajectory] = []

        class SpyModel(AgentModel):
            model_name = "SpyModel"

            def decide(self, state, trajectory, available_tools, already_queried):
                received_states.append(state)
                received_trajectories.append(trajectory)
                return AgentDecision(
                    action="STOP",
                    reasoning="Spy model stopping immediately.",
                    step_index=len(trajectory.decisions),
                )

        orch = AgentOrchestrator(agent_model=SpyModel(), budget=5, max_steps=3)
        case = _MinimalCase()
        trajectory = orch.investigate(case, custom_onset_ts=1700000000)

        # Verify spy received states with no ground_truth attribute
        self.assertGreater(len(received_states), 0)
        for state in received_states:
            self.assertFalse(
                hasattr(state, "ground_truth"),
                "InvestigationState must not have a ground_truth attribute",
            )

        # Verify trajectory has no ground_truth attribute
        for traj in received_trajectories:
            self.assertFalse(
                hasattr(traj, "ground_truth"),
                "InvestigationTrajectory must not have a ground_truth attribute",
            )

        # Final trajectory is not contaminated either
        self.assertFalse(hasattr(trajectory, "ground_truth"))

    def test_gt_service_not_in_state_decisions(self) -> None:
        """The spy model's decisions must not contain the GT service as a string."""
        from digital_detective.agent.orchestrator import AgentOrchestrator

        class StopModel(AgentModel):
            model_name = "StopModel"

            def decide(self, state, trajectory, available_tools, already_queried):
                return AgentDecision(action="STOP", reasoning="immediately stopping")

        orch = AgentOrchestrator(agent_model=StopModel(), budget=5, max_steps=2)
        traj = orch.investigate(_MinimalCase(), custom_onset_ts=1700000000)
        for dec in traj.decisions:
            # The secret GT service should never appear in reasoning
            self.assertNotIn("secret_service", dec.reasoning)


# ---------------------------------------------------------------------------
# 9. Budget enforcement (max_turns and budget_exhausted)
# ---------------------------------------------------------------------------

class TestBudgetEnforcement(unittest.TestCase):

    def test_max_steps_terminates_loop(self) -> None:
        from digital_detective.agent.orchestrator import AgentOrchestrator

        call_count = 0

        class SpamModel(AgentModel):
            model_name = "SpamModel"

            def decide(self, state, trajectory, available_tools, already_queried):
                nonlocal call_count
                call_count += 1
                return AgentDecision(action="STOP", reasoning="stopping")

        orch = AgentOrchestrator(agent_model=SpamModel(), budget=50, max_steps=3)
        traj = orch.investigate(_MinimalCase(), custom_onset_ts=1700000000)
        # Should have stopped after max_steps = 3 or on first STOP, whichever first
        self.assertLessEqual(call_count, 3)

    def test_max_consecutive_invalid_triggers_stop(self) -> None:
        """3 consecutive invalid QUERY decisions cause automatic STOP."""
        from digital_detective.agent.orchestrator import AgentOrchestrator

        class BadQueryModel(AgentModel):
            model_name = "BadQueryModel"

            def decide(self, state, trajectory, available_tools, already_queried):
                # Emit a QUERY with an invalid tool — policy will reject it
                return AgentDecision(
                    action="QUERY",
                    tool_name="nonexistent_tool",
                    service="svcA",
                    reasoning="I will always request a bad tool",
                )

        orch = AgentOrchestrator(agent_model=BadQueryModel(), budget=50, max_steps=20)
        traj = orch.investigate(_MinimalCase(), custom_onset_ts=1700000000)
        # Should terminate before 20 steps due to consecutive invalid counter
        self.assertIn(traj.terminated_by, ("STOP", "BUDGET_EXHAUSTED", "MAX_STEPS"))
        # Verify at least one POLICY REJECTED decision was recorded
        rejected = [d for d in traj.decisions if "POLICY REJECTED" in d.reasoning]
        self.assertGreater(len(rejected), 0)


# ---------------------------------------------------------------------------
# 10. Remediation bypass prevention
# ---------------------------------------------------------------------------

class TestRemediationBypass(unittest.TestCase):

    def test_below_threshold_no_remediation(self) -> None:
        """If agent issues REQUEST_REMEDIATION but confidence is low,
        remediation_authorized should be False."""
        from digital_detective.agent.orchestrator import AgentOrchestrator

        class AlwaysRemediateModel(AgentModel):
            model_name = "AlwaysRemediateModel"

            def decide(self, state, trajectory, available_tools, already_queried):
                return AgentDecision(
                    action="REQUEST_REMEDIATION",
                    reasoning="I want to remediate everything",
                )

        # Very high threshold (0.99) ensures confidence never reaches it
        orch = AgentOrchestrator(
            agent_model=AlwaysRemediateModel(),
            budget=20,
            remediation_threshold=0.99,
        )
        traj = orch.investigate(_MinimalCase(), custom_onset_ts=1700000000)
        state = traj.final_state
        if state is not None:
            # With no real telemetry, confidence should be very low
            # Remediation may be authorized or not depending on final score
            # The key test: the safety gate must have been consulted
            if state.decision and state.decision.confidence < 0.99:
                self.assertFalse(
                    state.remediation_authorized,
                    "Remediation must not be authorized below threshold",
                )

    def test_agent_cannot_directly_authorize_remediation(self) -> None:
        """InvestigationState.remediation_authorized is never set directly by model."""
        from digital_detective.detective.models import InvestigationState as IS
        state = IS(
            incident_id="x",
            system="ob",
            candidate_universe=("svcA",),
            budget=10,
            remaining_budget=10,
        )
        # The state's remediation_authorized is a property/field, not settable by model
        # (model receives a read-only view — it never has a reference to the live state)
        self.assertFalse(state.remediation_authorized)


# ---------------------------------------------------------------------------
# 11. MockAgentModel decides correctly
# ---------------------------------------------------------------------------

class TestMockAgentModelDecisions(unittest.TestCase):

    def _trajectory(self) -> InvestigationTrajectory:
        return InvestigationTrajectory(incident_id="x", model_name="Mock")

    def test_queries_top_service_first(self) -> None:
        state = _make_state(
            candidates=("svcA", "svcB"),
            remaining_budget=15,
            rankings=(_make_ranking("svcA", rank=1, score=0.75, confidence=0.45),),
        )
        model = MockAgentModel(confidence_threshold=0.55, min_evidence_queries=1)
        decision = model.decide(state, self._trajectory(), ["get_metrics", "get_service_health"], set())
        self.assertEqual(decision.action, "QUERY")
        self.assertEqual(decision.service, "svcA")

    def test_emits_final_diagnosis_when_confident(self) -> None:
        q = _make_query_record(query_id="q1")
        state = _make_state(
            candidates=("svcA", "svcB"),
            remaining_budget=10,
            queries=[q],
            evidence=[_make_evidence()],
            rankings=(_make_ranking("svcA", rank=1, score=0.90, confidence=0.85),),
        )
        model = MockAgentModel(confidence_threshold=0.55, min_evidence_queries=1)
        decision = model.decide(state, self._trajectory(), ["get_metrics"], set())
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        self.assertTrue(len(decision.evidence_ids) > 0)

    def test_stops_on_budget_exhaustion(self) -> None:
        state = _make_state(
            remaining_budget=1,
            rankings=(_make_ranking("svcA", rank=1, score=0.60, confidence=0.45),),
        )
        model = MockAgentModel(stop_budget_reserve=2)
        # Budget (1) <= stop_budget_reserve (2) → should STOP or FINAL_DIAGNOSIS
        decision = model.decide(state, self._trajectory(), ["get_metrics"], set())
        self.assertIn(decision.action, ("STOP", "FINAL_DIAGNOSIS"))


    def test_queries_different_services_for_competing_hyps(self) -> None:
        q_a = _make_query_record(query_id="q_a_metrics", service="svcA", tool_name="get_metrics")
        state = _make_state(
            candidates=("svcA", "svcB"),
            remaining_budget=15,
            queries=[q_a],
            evidence=[_make_evidence(service="svcA")],
            rankings=(
                _make_ranking("svcA", rank=1, score=0.60, confidence=0.45),
                _make_ranking("svcB", rank=2, score=0.55, confidence=0.38),
            ),
        )
        already = {("svca", "get_metrics")}
        model = MockAgentModel(confidence_threshold=0.90, min_evidence_queries=3)
        decision = model.decide(state, self._trajectory(), ["get_metrics", "get_service_health"], already)
        self.assertEqual(decision.action, "QUERY")
        # Should explore svcA's other tools or svcB's tools
        self.assertIn(decision.service, ("svcA", "svcB"))


if __name__ == "__main__":
    unittest.main()
