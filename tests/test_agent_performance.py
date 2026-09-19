"""Performance and reliability optimization tests for the Digital Detective Agent.

Validates:
1. Compact prompt construction (length reduction, top candidate filtering,
   compact already_queried format, and early finalization directive).
2. Model output limits (configurable max_tokens, num_predict, temperature via args & env vars).
3. Max-turn enforcement (terminating when max_steps is reached).
4. Duplicate query avoidance (orchestrator rejects duplicate tool+service calls).
5. Per-step and per-phase latency instrumentation (StepTiming recording & breakdown formatting).
6. Malformed response handling and graceful fallbacks.
7. Deterministic diagnostic fallback and consistency.
"""

from __future__ import annotations

import os
import unittest
from typing import Sequence
from unittest.mock import MagicMock, patch

from digital_detective.agent.models import (
    AgentDecision,
    AgentModel,
    InvestigationTrajectory,
    MockAgentModel,
    OllamaAgentModel,
    StepTiming,
    _parse_llm_response,
    _strip_thinking_tags,
    build_agent_model,
)
from digital_detective.agent.orchestrator import AgentOrchestrator
from digital_detective.agent.prompts import (
    check_finalization_directive,
    format_compact_already_queried,
    format_evidence_block,
    format_rankings,
    format_state_for_prompt,
)
from digital_detective.detective.models import (
    EvidenceItem,
    InvestigationState,
    RankedHypothesis,
    ToolQueryRecord,
)


# ---------------------------------------------------------------------------
# Test Helpers & Minimal Case
# ---------------------------------------------------------------------------

class _MinimalCase:
    class _Meta:
        case_id = "perf-case-001"
        system = "ob"
        system_name = "Online Boutique"

    class _GT:
        values = {
            "root_cause_service": "checkoutservice",
            "fault": "cpu",
            "inject_time": 1700000000,
        }

    metadata = _Meta()
    ground_truth = _GT()
    metrics = None
    traces = None
    logs = None


def _make_state(
    candidates: tuple[str, ...] = ("checkoutservice", "cartservice", "frontend", "redis"),
    budget: int = 20,
    remaining_budget: int = 20,
    queries: list[ToolQueryRecord] | None = None,
    evidence: list[EvidenceItem] | None = None,
    rankings: tuple[RankedHypothesis, ...] | None = None,
) -> InvestigationState:
    state = InvestigationState(
        incident_id="perf-case-001",
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
        query_id=query_id or f"q_{tool}_{service}",
        tool_name=tool,
        service=service,
        cost=cost,
        timestamp=100.0,
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
            evidence_score=score * 0.85,
            consistency_score=0.92,
        )
        for idx, (svc, score, conf) in enumerate(zip(services, scores, confidences))
    )


# ---------------------------------------------------------------------------
# 1. Compact Prompt Construction Tests
# ---------------------------------------------------------------------------

class TestCompactPromptConstruction(unittest.TestCase):
    """Verify that state formatting produces compact contexts."""

    def test_rankings_limited_to_top_three(self) -> None:
        rankings = _make_rankings(
            ["svc1", "svc2", "svc3", "svc4", "svc5"],
            [0.8, 0.6, 0.4, 0.2, 0.1],
            [0.85, 0.65, 0.45, 0.25, 0.15],
        )
        rendered = format_rankings(rankings, top_n=3)
        self.assertIn("svc1", rendered)
        self.assertIn("svc2", rendered)
        self.assertIn("svc3", rendered)
        self.assertNotIn("svc4", rendered)
        self.assertNotIn("svc5", rendered)

    def test_compact_already_queried_format(self) -> None:
        already_queried = {
            ("checkoutservice", "get_metrics"),
            ("checkoutservice", "get_traces"),
            ("cartservice", "get_metrics"),
        }
        rendered = format_compact_already_queried(already_queried)
        self.assertIn("checkoutservice: get_metrics, get_traces", rendered)
        self.assertIn("cartservice: get_metrics", rendered)

    def test_evidence_block_filtered_and_capped(self) -> None:
        # Create 12 evidence items across multiple services
        items = []
        for i in range(12):
            items.append(
                EvidenceItem(
                    service="checkoutservice" if i < 4 else f"svc{i}",
                    modality="metric",
                    signal=f"sig_{i}",
                    magnitude=1.0 + (i * 0.1),
                )
            )
        rendered = format_evidence_block(items, top_services={"checkoutservice"}, max_items=6)
        item_lines = [l for l in rendered.splitlines() if l.strip().startswith("E")]
        self.assertLessEqual(len(item_lines), 6)
        self.assertIn("checkoutservice", rendered)

    def test_check_finalization_directive(self) -> None:
        state = _make_state()
        state.current_rankings = _make_rankings(
            ["checkoutservice", "cartservice"],
            [0.55, 0.20],
            [0.60, 0.25],
        )
        state.evidence_collected.extend([
            EvidenceItem(service="checkoutservice", modality="metric", signal="cpu", magnitude=2.5),
            EvidenceItem(service="checkoutservice", modality="trace", signal="lat", magnitude=1.8),
        ])
        directive = check_finalization_directive(state)
        self.assertIsNotNone(directive)
        self.assertIn("STATUS DIRECTIVE", directive)
        self.assertIn("checkoutservice", directive)

    def test_total_prompt_size_drastically_reduced(self) -> None:
        state = _make_state()
        state.current_rankings = _make_rankings(
            ["checkoutservice", "cartservice", "frontend"],
            [0.6, 0.4, 0.2],
            [0.65, 0.45, 0.25],
        )
        state.evidence_collected.extend([
            EvidenceItem(service="checkoutservice", modality="metric", signal="cpu", magnitude=2.0),
            EvidenceItem(service="checkoutservice", modality="trace", signal="lat", magnitude=1.5),
        ])
        state.queries_executed.append(_make_query_record("checkoutservice", "get_metrics"))
        already_queried = {("checkoutservice", "get_metrics")}
        available_tools = ["get_metrics", "get_traces", "get_logs"]

        prompt = format_state_for_prompt(
            state=state,
            available_tools=available_tools,
            already_queried=already_queried,
            turn=2,
        )
        # Should be well under 1200 characters (~300 tokens)
        self.assertLess(len(prompt), 1200)
        self.assertIn("TURN 2", prompt)
        self.assertIn("TOP HYPOTHESES", prompt)
        self.assertIn("AVAILABLE TOOLS", prompt)


# ---------------------------------------------------------------------------
# 2. Output Limit & Ollama Configuration Tests
# ---------------------------------------------------------------------------

class TestOllamaConfiguration(unittest.TestCase):
    """Verify OllamaAgentModel parameter and environment variable configuration."""

    def test_default_tokens_and_temperature(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            model = OllamaAgentModel()
            self.assertEqual(model.max_tokens, 750)
            self.assertEqual(model.temperature, 0.0)
            self.assertIsNotNone(model.extra_payload)
            options = model.extra_payload.get("options", {})
            self.assertEqual(options.get("num_predict"), 750)
            self.assertEqual(options.get("temperature"), 0.0)
            self.assertEqual(options.get("num_gpu"), 20)

    def test_explicit_tokens_override(self) -> None:
        model = OllamaAgentModel(max_tokens=80, temperature=0.2, num_gpu=15)
        self.assertEqual(model.max_tokens, 80)
        self.assertEqual(model.temperature, 0.2)
        options = model.extra_payload.get("options", {})
        self.assertEqual(options.get("num_predict"), 80)
        self.assertEqual(options.get("temperature"), 0.2)
        self.assertEqual(options.get("num_gpu"), 15)

    def test_env_var_configuration(self) -> None:
        env = {
            "DIGITAL_DETECTIVE_OLLAMA_MODEL": "custom-model:latest",
            "DIGITAL_DETECTIVE_OLLAMA_MAX_TOKENS": "128",
            "DIGITAL_DETECTIVE_OLLAMA_TEMPERATURE": "0.1",
            "DIGITAL_DETECTIVE_OLLAMA_NUM_GPU": "24",
            "DIGITAL_DETECTIVE_OLLAMA_BASE_URL": "http://127.0.0.1:11434/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            model = OllamaAgentModel()
            self.assertEqual(model.model, "custom-model:latest")
            self.assertEqual(model.max_tokens, 128)
            self.assertEqual(model.temperature, 0.1)
            self.assertEqual(model._base_url, "http://127.0.0.1:11434/v1")
            options = model.extra_payload.get("options", {})
            self.assertEqual(options.get("num_predict"), 128)
            self.assertEqual(options.get("num_gpu"), 24)

    def test_build_agent_model_passes_max_tokens(self) -> None:
        with patch.object(OllamaAgentModel, "check_availability", return_value=(True, "OK")):
            agent = build_agent_model("ollama", ollama_max_tokens=96)
            self.assertIsInstance(agent, OllamaAgentModel)
            self.assertEqual(agent.max_tokens, 96)


# ---------------------------------------------------------------------------
# 3. Max-Turn Enforcement Tests
# ---------------------------------------------------------------------------

class TestMaxTurnsEnforcement(unittest.TestCase):
    """Verify that the orchestrator strictly enforces max reasoning turns."""

    def test_max_steps_stops_orchestrator(self) -> None:
        valid_services = ["cartservice", "frontend", "paymentservice", "emailservice"]

        class PerpetualAgent(AgentModel):
            model_name = "PerpetualAgent"

            def decide(self, state, trajectory, available_tools, already_queried):
                svc = valid_services[len(trajectory.decisions) % len(valid_services)]
                return AgentDecision(
                    action="QUERY",
                    tool_name="get_metrics",
                    service=svc,
                    reasoning="Need more data.",
                    step_index=len(trajectory.decisions),
                )

        orch = AgentOrchestrator(
            agent_model=PerpetualAgent(),
            budget=20,
            max_steps=3,  # strictly limit to 3 turns
        )

        with patch("digital_detective.agent.orchestrator.InvestigationEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine
            mock_state = _make_state(budget=20, remaining_budget=20)
            mock_engine.initialize_investigation.return_value = mock_state
            mock_engine.execute_query.return_value = (
                _make_query_record("cartservice", "get_metrics"),
                mock_state,
            )
            mock_engine.compute_decision.return_value = mock_state

            traj = orch.investigate(_MinimalCase(), custom_onset_ts=1700000000)

            self.assertLessEqual(traj.total_steps, 3)
            self.assertEqual(traj.terminated_by, "MAX_STEPS")


# ---------------------------------------------------------------------------
# 4. Duplicate Query Avoidance Tests
# ---------------------------------------------------------------------------

class TestDuplicateQueryAvoidance(unittest.TestCase):
    """Verify that deterministic policy rejects duplicate queries when configured."""

    def test_duplicate_query_rejected_when_disallowed(self) -> None:
        call_count = [0]

        class DupAgent(AgentModel):
            model_name = "DupAgent"

            def decide(self, state, trajectory, available_tools, already_queried):
                call_count[0] += 1
                if call_count[0] <= 2:
                    return AgentDecision(
                        action="QUERY",
                        tool_name="get_metrics",
                        service="checkoutservice",
                        reasoning="Checking cpu.",
                        step_index=len(trajectory.decisions),
                    )
                return AgentDecision(
                    action="STOP",
                    reasoning="Stopping now.",
                    step_index=len(trajectory.decisions),
                )

        orch = AgentOrchestrator(
            agent_model=DupAgent(),
            budget=20,
            max_steps=5,
            allow_duplicate_queries=False,  # default in our optimization pass
        )

        with patch("digital_detective.agent.orchestrator.InvestigationEngine") as mock_engine_cls:
            mock_engine = MagicMock()
            mock_engine_cls.return_value = mock_engine
            mock_state = _make_state(budget=20, remaining_budget=20)
            mock_engine.initialize_investigation.return_value = mock_state
            mock_engine.execute_query.return_value = (
                _make_query_record("checkoutservice", "get_metrics"),
                mock_state,
            )
            mock_engine.compute_decision.return_value = mock_state

            traj = orch.investigate(_MinimalCase(), custom_onset_ts=1700000000)

            # Check that a policy rejection for DUPLICATE_QUERY was registered
            rejected_decisions = [
                d for d in traj.decisions if "POLICY REJECTED" in d.reasoning and "DUPLICATE_QUERY" in d.reasoning
            ]
            self.assertGreaterEqual(len(rejected_decisions), 1)


# ---------------------------------------------------------------------------
# 5. Latency Tracking & StepTiming Tests
# ---------------------------------------------------------------------------

class TestLatencyTracking(unittest.TestCase):
    """Verify that StepTiming tracks each phase and formats a breakdown."""

    def test_step_timing_fields_and_aggregates(self) -> None:
        t = InvestigationTrajectory(incident_id="timing-test")
        timing1 = StepTiming(
            step_index=0,
            prompt_construction_sec=0.002,
            llm_call_sec=1.45,
            parsing_sec=0.001,
            validation_sec=0.001,
            tool_execution_sec=0.05,
            total_step_sec=1.504,
            prompt_chars=500,
            prompt_tokens_est=125,
            output_chars=80,
            output_tokens_est=20,
            evidence_count=2,
            trajectory_steps_count=0,
        )
        t.add_step_timing(timing1)

        self.assertEqual(len(t.step_timings), 1)
        self.assertAlmostEqual(t.total_llm_time_sec, 1.45, places=2)
        self.assertAlmostEqual(t.total_tool_time_sec, 0.05, places=2)

        breakdown = t.format_timing_breakdown()
        self.assertIn("LATENCY & PROMPT SIZE BREAKDOWN", breakdown)
        self.assertIn("Turn 1: LLM=1.45s", breakdown)
        self.assertIn("Prompt=500 chars (~125 tok)", breakdown)
        self.assertIn("Output=80 chars (~20 tok)", breakdown)
        self.assertIn("Total LLM inference time: 1.45s", breakdown)


# ---------------------------------------------------------------------------
# 6. Malformed Response & Fallback Handling
# ---------------------------------------------------------------------------

class TestMalformedResponseHandling(unittest.TestCase):
    """Verify parsing handles malformed responses, strips thinking tags, and auto-grounds."""

    def test_strip_thinking_tags(self) -> None:
        raw = "<think>I need to query checkoutservice cpu.</think>{\"action\": \"STOP\", \"reasoning\": \"done\"}"
        clean = _strip_thinking_tags(raw)
        self.assertNotIn("<think>", clean)
        self.assertEqual(clean, '{"action": "STOP", "reasoning": "done"}')

    def test_json_embedded_in_conversational_text(self) -> None:
        state = _make_state()
        raw = 'Okay, let\'s see. The candidate is checkoutservice. {"action": "QUERY", "tool_name": "get_metrics", "service": "checkoutservice", "reasoning": "query cpu"} Hope this helps!'
        decision = _parse_llm_response(raw, step=1, state=state)
        self.assertEqual(decision.action, "QUERY")
        self.assertEqual(decision.tool_name, "get_metrics")
        self.assertEqual(decision.service, "checkoutservice")

    def test_malformed_json_falls_back_to_stop(self) -> None:
        state = _make_state()
        decision = _parse_llm_response("Not valid JSON at all", step=1, state=state)
        self.assertEqual(decision.action, "STOP")
        self.assertIn("could not be parsed as JSON", decision.reasoning)

    def test_final_diagnosis_auto_grounds_from_queries(self) -> None:
        state = _make_state(queries=[_make_query_record("checkoutservice", "get_metrics", query_id="q_ok_1")])
        raw = '{"action": "FINAL_DIAGNOSIS", "diagnosis_service": "checkoutservice", "reasoning": "Diagnosed root cause."}'
        decision = _parse_llm_response(raw, step=1, state=state)
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        self.assertEqual(decision.evidence_ids, ("q_ok_1",))
        self.assertIn("[auto-grounded from query history]", decision.reasoning)

    def test_final_diagnosis_without_diagnosis_service_becomes_stop(self) -> None:
        state = _make_state(queries=[_make_query_record("checkoutservice", "get_metrics", query_id="q_ok_1")])
        raw = '{"action": "FINAL_DIAGNOSIS", "reasoning": "Diagnosed root cause."}'
        decision = _parse_llm_response(raw, step=1, state=state)
        self.assertEqual(decision.action, "STOP")
        self.assertIn("diagnosis service", decision.reasoning)

    def test_final_diagnosis_without_queries_becomes_stop(self) -> None:
        state = _make_state(queries=[])
        raw = '{"action": "FINAL_DIAGNOSIS", "diagnosis_service": "checkoutservice", "reasoning": "Diagnosed with zero queries."}'
        decision = _parse_llm_response(raw, step=1, state=state)
        self.assertEqual(decision.action, "STOP")
        self.assertIn("no queries completed", decision.reasoning)


# ---------------------------------------------------------------------------
# 7. Mock Backend Regressions Check
# ---------------------------------------------------------------------------

class TestMockBackendUnchanged(unittest.TestCase):
    """Verify MockAgentModel continues to operate deterministically."""

    def test_mock_agent_model_runs_normally(self) -> None:
        mock_agent = MockAgentModel(confidence_threshold=0.5, min_evidence_queries=1)
        state = _make_state(
            candidates=("checkoutservice", "cartservice"),
            queries=[_make_query_record("checkoutservice", "get_metrics", query_id="q_1")],
            rankings=_make_rankings(["checkoutservice"], [0.8], [0.85]),
        )
        traj = InvestigationTrajectory(incident_id="mock-test")
        decision = mock_agent.decide(state, traj, ["get_metrics", "get_traces"], set())
        self.assertEqual(decision.action, "FINAL_DIAGNOSIS")
        self.assertEqual(decision.evidence_ids, ("q_1",))


if __name__ == "__main__":
    unittest.main()
