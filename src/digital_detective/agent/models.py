"""Agent model interface, structured decision types, and all model implementations.

Architecture
------------
The AgentModel is an orchestrator: it decides WHICH telemetry query to issue next
(or when to stop) based on the current investigation state and collected evidence.
It is NOT an RCA engine — it never produces a root cause itself. All scoring,
evidence fusion, and causal validation remain in the deterministic core.

Available implementations
--------------------------
MockAgentModel
    Deterministic rule-based agent. No network required. Always works.

OllamaAgentModel
    Local Ollama endpoint (http://localhost:11434/v1).
    Default model: qwen3:8b (configurable via DIGITAL_DETECTIVE_OLLAMA_MODEL).
    No API key required. Requires Ollama to be running locally.

GrokAgentModel
    xAI Grok via https://api.x.ai/v1.
    Requires XAI_API_KEY environment variable.
    Default model: grok-4.3 (configurable via DIGITAL_DETECTIVE_GROK_MODEL).

OpenAIChatAdapter
    Generic OpenAI-compatible endpoint.
    Requires OPENAI_API_KEY. For development prefer OllamaAgentModel.

build_agent_model()
    Factory that selects the right implementation based on mode + environment.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from digital_detective.detective.models import InvestigationState

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured decision object
# ---------------------------------------------------------------------------

_VALID_ACTIONS = frozenset({"QUERY", "FINAL_DIAGNOSIS", "REQUEST_REMEDIATION", "STOP"})


@dataclass(frozen=True)
class AgentDecision:
    """Structured control signal emitted by the agent model each step.

    Attributes
    ----------
    action:
        One of ``'QUERY'``, ``'FINAL_DIAGNOSIS'``, ``'REQUEST_REMEDIATION'``,
        ``'STOP'``.
    tool_name:
        Required when ``action == 'QUERY'``.  Must be a valid detective tool
        name (``get_metrics``, ``get_traces``, ``get_logs``, ``get_neighbors``,
        ``get_service_health``, ``get_recent_change``).
    service:
        Target service name.  Required when ``action == 'QUERY'``.
    reasoning:
        Human-readable string documenting the agent's rationale for this step.
        Required for ``FINAL_DIAGNOSIS`` and ``STOP`` to ensure every diagnosis
        statement is grounded in collected evidence.
    evidence_ids:
        Optional list of query IDs from ``state.queries_executed`` that the
        agent cites as support for a ``FINAL_DIAGNOSIS`` or ``STOP`` decision.
        Every ``FINAL_DIAGNOSIS`` must reference at least one evidence ID.
    step_index:
        Zero-based index of this decision in the investigation trajectory.
    metadata:
        Free-form optional metadata for extended adapter use.
    """

    action: str
    reasoning: str
    tool_name: str = ""
    service: str = ""
    evidence_ids: tuple[str, ...] = ()
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"AgentDecision.action must be one of {sorted(_VALID_ACTIONS)}, got {self.action!r}"
            )
        if self.action == "QUERY" and not self.tool_name:
            raise ValueError("AgentDecision with action='QUERY' requires a non-empty tool_name.")
        if self.action == "QUERY" and not self.service:
            raise ValueError("AgentDecision with action='QUERY' requires a non-empty service.")
        if self.action == "FINAL_DIAGNOSIS" and not self.reasoning:
            raise ValueError("FINAL_DIAGNOSIS must include a non-empty reasoning string.")
        if self.action == "FINAL_DIAGNOSIS" and not self.evidence_ids:
            raise ValueError(
                "FINAL_DIAGNOSIS must reference at least one evidence_id "
                "from state.queries_executed to ground the diagnosis."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tool_name": self.tool_name,
            "service": self.service,
            "reasoning": self.reasoning,
            "evidence_ids": list(self.evidence_ids),
            "step_index": self.step_index,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Investigation trajectory and StepTiming profiling
# ---------------------------------------------------------------------------

@dataclass
class StepTiming:
    """Latency and token sizing instrumentation for a single agent reasoning step."""

    step_index: int
    prompt_construction_sec: float = 0.0
    llm_call_sec: float = 0.0
    parsing_sec: float = 0.0
    validation_sec: float = 0.0
    tool_execution_sec: float = 0.0
    total_step_sec: float = 0.0
    prompt_chars: int = 0
    prompt_tokens_est: int = 0
    output_chars: int = 0
    output_tokens_est: int = 0
    evidence_count: int = 0
    trajectory_steps_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "prompt_construction_sec": round(self.prompt_construction_sec, 4),
            "llm_call_sec": round(self.llm_call_sec, 4),
            "parsing_sec": round(self.parsing_sec, 4),
            "validation_sec": round(self.validation_sec, 4),
            "tool_execution_sec": round(self.tool_execution_sec, 4),
            "total_step_sec": round(self.total_step_sec, 4),
            "prompt_chars": self.prompt_chars,
            "prompt_tokens_est": self.prompt_tokens_est,
            "output_chars": self.output_chars,
            "output_tokens_est": self.output_tokens_est,
            "evidence_count": self.evidence_count,
            "trajectory_steps_count": self.trajectory_steps_count,
        }


@dataclass
class InvestigationTrajectory:
    """Structured, exportable record of a complete agent-driven investigation.

    Attributes
    ----------
    incident_id:
        Unique identifier of the investigated incident.
    decisions:
        Ordered list of every AgentDecision taken during the investigation.
    final_state:
        The ``InvestigationState`` after the agent has completed.
    total_steps:
        Number of agent decision steps taken.
    total_budget_used:
        Query budget consumed across all steps.
    terminated_by:
        How the loop ended: ``'FINAL_DIAGNOSIS'``, ``'STOP'``,
        ``'BUDGET_EXHAUSTED'``, or ``'MAX_STEPS'``.
    model_name:
        Name of the model that drove the investigation.
    step_timings:
        Per-step latency and prompt/output sizing measurements.
    """

    incident_id: str
    decisions: list[AgentDecision] = field(default_factory=list)
    final_state: InvestigationState | None = None
    total_steps: int = 0
    total_budget_used: int = 0
    terminated_by: str = ""
    model_name: str = "unknown"
    step_timings: list[StepTiming] = field(default_factory=list)
    total_llm_time_sec: float = 0.0
    total_tool_time_sec: float = 0.0
    total_prompt_time_sec: float = 0.0
    total_validation_time_sec: float = 0.0
    total_orchestration_sec: float = 0.0

    def append_decision(self, decision: AgentDecision) -> None:
        self.decisions.append(decision)
        self.total_steps = len(self.decisions)

    def add_step_timing(self, timing: StepTiming) -> None:
        self.step_timings.append(timing)
        self.total_llm_time_sec = sum(t.llm_call_sec for t in self.step_timings)
        self.total_tool_time_sec = sum(t.tool_execution_sec for t in self.step_timings)
        self.total_prompt_time_sec = sum(t.prompt_construction_sec for t in self.step_timings)
        self.total_validation_time_sec = sum(t.validation_sec + t.parsing_sec for t in self.step_timings)
        self.total_orchestration_sec = sum(t.total_step_sec for t in self.step_timings)

    def format_timing_breakdown(self) -> str:
        """Produce a formatted per-step latency and prompt-size breakdown."""
        lines = ["--- LATENCY & PROMPT SIZE BREAKDOWN ---"]
        for st in self.step_timings:
            lines.append(
                f"  Turn {st.step_index + 1}: LLM={st.llm_call_sec:.2f}s | "
                f"Tool={st.tool_execution_sec:.3f}s | "
                f"Prompt={st.prompt_chars} chars (~{st.prompt_tokens_est} tok) | "
                f"Output={st.output_chars} chars (~{st.output_tokens_est} tok) | "
                f"Evidence cited/available={st.evidence_count}"
            )
        lines.append(f"  Total LLM inference time: {self.total_llm_time_sec:.2f}s")
        lines.append(f"  Total Tool execution time: {self.total_tool_time_sec:.3f}s")
        lines.append(f"  Total Prompt prep time  : {self.total_prompt_time_sec:.4f}s")
        lines.append(f"  Total Validation/Parse  : {self.total_validation_time_sec:.4f}s")
        lines.append(f"  Total Orchestration time: {self.total_orchestration_sec:.2f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "model_name": self.model_name,
            "total_steps": self.total_steps,
            "total_budget_used": self.total_budget_used,
            "terminated_by": self.terminated_by,
            "decisions": [d.to_dict() for d in self.decisions],
            "final_state": self.final_state.to_dict() if self.final_state else None,
            "step_timings": [t.to_dict() for t in self.step_timings],
            "total_llm_time_sec": round(self.total_llm_time_sec, 4),
            "total_tool_time_sec": round(self.total_tool_time_sec, 4),
            "total_orchestration_sec": round(self.total_orchestration_sec, 4),
        }


# ---------------------------------------------------------------------------
# Abstract AgentModel interface
# ---------------------------------------------------------------------------

class AgentModel(abc.ABC):
    """Abstract interface for agent decision-making models.

    Implementations must produce exactly one ``AgentDecision`` per call to
    ``decide()``.  The model receives a read-only view of the current
    investigation state; it MUST NOT modify the state.

    The model must NOT be given access to ``case.ground_truth``.
    """

    # Subclasses may override this for display purposes.
    model_name: str = "AgentModel"

    @abc.abstractmethod
    def decide(
        self,
        state: InvestigationState,
        trajectory: InvestigationTrajectory,
        available_tools: Sequence[str],
        already_queried: set[tuple[str, str]],
    ) -> AgentDecision:
        """Produce the next investigation decision.

        Parameters
        ----------
        state:
            Current ``InvestigationState``.  Read-only; do not mutate.
        trajectory:
            Full history of decisions taken so far.
        available_tools:
            Tool names that are legal to call (available in this investigation).
        already_queried:
            Set of ``(service, tool_name)`` pairs already successfully queried.

        Returns
        -------
        AgentDecision
            The next step decision.
        """


# ---------------------------------------------------------------------------
# MockAgentModel — deterministic rule-based agent (no LLM, no network)
# ---------------------------------------------------------------------------

_TOOL_PRIORITY_ORDER = (
    "get_metrics",
    "get_service_health",
    "get_recent_change",
    "get_traces",
    "get_neighbors",
    "get_logs",
)


class MockAgentModel(AgentModel):
    """Deterministic rule-based agent for testing without a live LLM.

    Strategy
    --------
    1. If the top-ranked hypothesis has confidence >= ``confidence_threshold``
       and at least ``min_evidence_queries`` queries have been made on it,
       emit ``FINAL_DIAGNOSIS``.
    2. If budget is <= ``stop_budget_reserve`` units, emit ``STOP``.
    3. Otherwise, pick the highest-suspicion service that still has an
       un-queried tool (following ``_TOOL_PRIORITY_ORDER``), and emit
       ``QUERY``.

    This produces a bounded, deterministic investigation that exercises the
    full agent loop without requiring an LLM.
    """

    model_name = "MockAgentModel"

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        min_evidence_queries: int = 2,
        stop_budget_reserve: int = 2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_evidence_queries = min_evidence_queries
        self.stop_budget_reserve = stop_budget_reserve

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_queries_per_service(state: InvestigationState) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in state.queries_executed:
            if q.status == "SUCCESS":
                counts[q.service] = counts.get(q.service, 0) + 1
        return counts

    @staticmethod
    def _collect_evidence_query_ids(state: InvestigationState, target: str) -> tuple[str, ...]:
        """Return query IDs from state that produced evidence for ``target``."""
        ids = [
            q.query_id
            for q in state.queries_executed
            if q.service == target and q.status == "SUCCESS"
        ]
        # Fall back to any query if none match the target specifically.
        if not ids:
            ids = [q.query_id for q in state.queries_executed if q.status == "SUCCESS"]
        return tuple(ids[:3])  # cite at most 3

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def decide(
        self,
        state: InvestigationState,
        trajectory: InvestigationTrajectory,
        available_tools: Sequence[str],
        already_queried: set[tuple[str, str]],
    ) -> AgentDecision:
        step = len(trajectory.decisions)
        rankings = state.current_rankings
        universe = list(state.candidate_universe)

        # --- Guard: no rankings available yet → query best candidate metrics ---
        if not rankings:
            if universe:
                return AgentDecision(
                    action="QUERY",
                    tool_name="get_metrics",
                    service=universe[0],
                    reasoning="No hypothesis rankings available; seeding with metrics query.",
                    step_index=step,
                )
            return AgentDecision(
                action="STOP",
                reasoning="No candidates in universe and no rankings. Cannot investigate.",
                evidence_ids=(),
                step_index=step,
            )

        top_hyp = rankings[0]
        query_counts = self._count_queries_per_service(state)

        # --- Check for confident diagnosis ---
        enough_queries = query_counts.get(top_hyp.service, 0) >= self.min_evidence_queries
        if top_hyp.confidence >= self.confidence_threshold and enough_queries:
            evidence_ids = self._collect_evidence_query_ids(state, top_hyp.service)
            if not evidence_ids:
                # Cannot emit FINAL_DIAGNOSIS without evidence IDs → STOP instead
                return AgentDecision(
                    action="STOP",
                    reasoning=(
                        f"Top hypothesis '{top_hyp.service}' has confidence "
                        f"{top_hyp.confidence:.1%} but no successful query IDs were recorded. "
                        "Stopping to avoid ungrounded diagnosis."
                    ),
                    evidence_ids=(),
                    step_index=step,
                )
            return AgentDecision(
                action="FINAL_DIAGNOSIS",
                reasoning=(
                    f"Confidence {top_hyp.confidence:.1%} >= threshold {self.confidence_threshold:.1%} "
                    f"with {query_counts.get(top_hyp.service, 0)} supporting queries. "
                    f"Evidence score: {top_hyp.evidence_score:.3f}, "
                    f"causal consistency: {top_hyp.consistency_score:.2f}. "
                    f"Diagnosing '{top_hyp.service}' as root cause."
                ),
                evidence_ids=evidence_ids,
                step_index=step,
            )

        # --- Budget reserve: stop if almost out of budget ---
        if state.remaining_budget <= self.stop_budget_reserve:
            evidence_ids = self._collect_evidence_query_ids(state, top_hyp.service)
            if evidence_ids:
                return AgentDecision(
                    action="FINAL_DIAGNOSIS",
                    reasoning=(
                        f"Budget nearly exhausted ({state.remaining_budget} units remaining). "
                        f"Accepting top hypothesis '{top_hyp.service}' "
                        f"(confidence={top_hyp.confidence:.1%}) as best available diagnosis."
                    ),
                    evidence_ids=evidence_ids,
                    step_index=step,
                )
            return AgentDecision(
                action="STOP",
                reasoning=(
                    f"Budget nearly exhausted ({state.remaining_budget} units remaining) "
                    "and no successful queries completed. Cannot make a grounded diagnosis."
                ),
                evidence_ids=(),
                step_index=step,
            )

        # --- Pick next QUERY: highest suspicion service with an un-queried tool ---
        best_priority = -1.0
        best_service = ""
        best_tool = ""
        best_rationale = ""

        for hyp in rankings:
            svc = hyp.service
            for tool in _TOOL_PRIORITY_ORDER:
                if tool not in available_tools:
                    continue
                if (svc, tool) in already_queried:
                    continue
                from digital_detective.detective.tools import DEFAULT_TOOL_COSTS
                cost = DEFAULT_TOOL_COSTS.get(tool, 1)  # type: ignore[arg-type]
                if state.remaining_budget < cost:
                    continue

                n_queries = query_counts.get(svc, 0)
                novelty = 1.0 / (1.0 + n_queries)
                tool_weight = 1.0 / (1.0 + _TOOL_PRIORITY_ORDER.index(tool))
                priority = hyp.score * novelty * tool_weight

                if priority > best_priority:
                    best_priority = priority
                    best_service = svc
                    best_tool = tool
                    best_rationale = (
                        f"Service '{svc}' is rank-{hyp.rank} suspect "
                        f"(score={hyp.score:.3f}, confidence={hyp.confidence:.1%}); "
                        f"querying {tool} to gather {'initial' if n_queries == 0 else 'additional'} evidence."
                    )

        if best_service:
            return AgentDecision(
                action="QUERY",
                tool_name=best_tool,
                service=best_service,
                reasoning=best_rationale,
                step_index=step,
            )

        # --- Fallback: no affordable queries left → STOP or accept best ---
        evidence_ids = self._collect_evidence_query_ids(state, top_hyp.service)
        if evidence_ids:
            return AgentDecision(
                action="FINAL_DIAGNOSIS",
                reasoning=(
                    f"No further affordable queries available. "
                    f"Accepting top hypothesis '{top_hyp.service}' "
                    f"(confidence={top_hyp.confidence:.1%}) as final diagnosis."
                ),
                evidence_ids=evidence_ids,
                step_index=step,
            )
        return AgentDecision(
            action="STOP",
            reasoning="No affordable queries remain and no grounded evidence collected. Stopping.",
            evidence_ids=(),
            step_index=step,
        )


# ---------------------------------------------------------------------------
# Shared response-parsing utilities
# ---------------------------------------------------------------------------

def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3)."""
    # Strip XML-style thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _parse_llm_response(raw_text: str, step: int, state: InvestigationState) -> AgentDecision:
    """Parse the LLM JSON response into an AgentDecision.

    Handles common malformations:
    - Markdown code fences (```json ... ```)
    - <think>...</think> blocks from reasoning models
    - Missing evidence_ids on FINAL_DIAGNOSIS (auto-populated from query history)

    Falls back to STOP if the response cannot be parsed.
    """
    text = _strip_thinking_tags(raw_text)

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        data = None
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError:
                data = None

        if data is None:
            _log.warning("LLM response is not valid JSON: %s | raw=%r", exc, raw_text[:300])
            return AgentDecision(
                action="STOP",
                reasoning=f"LLM response could not be parsed as JSON: {exc}. Stopping to avoid invalid action.",
                step_index=step,
            )

    action = data.get("action", "STOP")
    if action not in {"QUERY", "FINAL_DIAGNOSIS", "REQUEST_REMEDIATION", "STOP"}:
        _log.warning("LLM returned unknown action %r; defaulting to STOP.", action)
        action = "STOP"

    reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."
    tool_name = str(data.get("tool_name", "")).strip()
    service = str(data.get("service", "")).strip()
    evidence_ids_raw = data.get("evidence_ids", [])
    if isinstance(evidence_ids_raw, list):
        evidence_ids: tuple[str, ...] = tuple(str(e) for e in evidence_ids_raw)
    else:
        evidence_ids = ()

    # For FINAL_DIAGNOSIS: auto-populate evidence_ids from query history if absent
    if action == "FINAL_DIAGNOSIS" and not evidence_ids:
        fallback_ids = tuple(
            q.query_id for q in state.queries_executed if q.status == "SUCCESS"
        )[:3]
        if not fallback_ids:
            _log.warning(
                "LLM FINAL_DIAGNOSIS has no evidence_ids and no successful queries; "
                "converting to STOP."
            )
            return AgentDecision(
                action="STOP",
                reasoning=(
                    "Agent issued FINAL_DIAGNOSIS without evidence IDs and no queries "
                    f"completed. Original reasoning: {reasoning}"
                ),
                step_index=step,
            )
        evidence_ids = fallback_ids
        reasoning = f"[auto-grounded from query history] {reasoning}"

    try:
        return AgentDecision(
            action=action,
            reasoning=reasoning,
            tool_name=tool_name,
            service=service,
            evidence_ids=evidence_ids,
            step_index=step,
        )
    except ValueError as exc:
        _log.warning("Parsed LLM decision is invalid (%s); falling back to STOP.", exc)
        return AgentDecision(
            action="STOP",
            reasoning=f"LLM decision was structurally invalid ({exc}). Stopping.",
            step_index=step,
        )


# ---------------------------------------------------------------------------
# Base class for OpenAI-compatible HTTP adapters (stdlib urllib only)
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 60  # seconds — Ollama local inference can be slow
_DEFAULT_MAX_RETRIES = 1


class _OpenAICompatibleAdapter(AgentModel):
    """Base class for models accessible via the OpenAI Chat Completions API format.

    Subclasses set:
    - ``_base_url``: e.g. "http://localhost:11434/v1" or "https://api.x.ai/v1"
    - ``_api_key``: string key or None (Ollama doesn't need one)
    - ``model``: model identifier string
    - ``model_name``: display name

    Uses only stdlib ``urllib.request`` — no third-party HTTP client needed.
    """

    model_name = "_OpenAICompatibleAdapter"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = True,
        extra_payload: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.json_mode = json_mode  # Whether to send response_format: json_object
        self.extra_payload = extra_payload
        self.max_tokens = max_tokens

        if system_prompt is None:
            from .prompts import SYSTEM_PROMPT
            self._system_prompt: str = SYSTEM_PROMPT
        else:
            self._system_prompt = system_prompt

    def check_availability(self) -> tuple[bool, str]:
        """Check if the backend is reachable.

        Returns
        -------
        (available: bool, message: str)
        """
        try:
            req = urllib.request.Request(
                url=self._base_url,
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return True, f"Reachable (HTTP {resp.status})"
        except Exception as exc:
            return False, str(exc)

    def _call_api(self, messages: list[dict]) -> str:
        """POST to the chat completions endpoint and return the content string."""
        payload_dict: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.json_mode:
            payload_dict["response_format"] = {"type": "json_object"}
        if self.max_tokens is not None:
            payload_dict["max_tokens"] = self.max_tokens
        if self.extra_payload:
            payload_dict.update(self.extra_payload)

        payload = json.dumps(payload_dict).encode("utf-8")
        url = f"{self._base_url}/chat/completions"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url=url, data=payload, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8")
                    data = json.loads(body)
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    if not content and msg.get("reasoning"):
                        content = msg["reasoning"]
                    return content
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                _log.warning(
                    "%s HTTP %d attempt %d/%d: %s",
                    self.model_name, exc.code, attempt + 1, self.max_retries + 1, error_body[:200],
                )
                last_exc = exc
                if exc.code in (400, 401, 403):
                    break  # Non-retryable
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                _log.warning(
                    "%s network error attempt %d/%d: %s",
                    self.model_name, attempt + 1, self.max_retries + 1, exc,
                )
                last_exc = exc

            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(
            f"{self.model_name} failed after {self.max_retries + 1} attempts: {last_exc}"
        )

    def decide(
        self,
        state: InvestigationState,
        trajectory: InvestigationTrajectory,
        available_tools: Sequence[str],
        already_queried: set[tuple[str, str]],
    ) -> AgentDecision:
        from .prompts import format_state_for_prompt

        step = len(trajectory.decisions)
        state_block = format_state_for_prompt(
            state=state,
            available_tools=list(available_tools),
            already_queried=already_queried,
            turn=step + 1,
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": state_block},
        ]

        try:
            t0 = time.perf_counter()
            raw = self._call_api(messages)
            llm_dt = time.perf_counter() - t0
            decision = _parse_llm_response(raw, step=step, state=state)
            total_prompt_chars = len(state_block) + len(self._system_prompt)
            decision.metadata.update({
                "llm_call_sec": round(llm_dt, 4),
                "prompt_chars": total_prompt_chars,
                "prompt_tokens_est": max(1, total_prompt_chars // 4),
                "output_chars": len(raw),
                "output_tokens_est": max(1, len(raw) // 4),
            })
            return decision
        except RuntimeError as exc:
            _log.error("%s call failed: %s. Falling back to STOP.", self.model_name, exc)
            return AgentDecision(
                action="STOP",
                reasoning=f"{self.model_name} call failed: {exc}. Stopping investigation.",
                step_index=step,
            )


# ---------------------------------------------------------------------------
# OllamaAgentModel
# ---------------------------------------------------------------------------

_OLLAMA_DEFAULT_MODEL = "qwen3:8b"
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_OLLAMA_MODEL_ENV = "DIGITAL_DETECTIVE_OLLAMA_MODEL"
_OLLAMA_BASE_URL_ENV = "DIGITAL_DETECTIVE_OLLAMA_BASE_URL"
_OLLAMA_NUM_GPU_ENV = "DIGITAL_DETECTIVE_OLLAMA_NUM_GPU"
_OLLAMA_DEFAULT_NUM_GPU = 20  # Safe default to avoid CUDA VRAM overflow on consumer GPUs
_OLLAMA_TEMPERATURE_ENV = "DIGITAL_DETECTIVE_OLLAMA_TEMPERATURE"
_OLLAMA_DEFAULT_TEMPERATURE = 0.0
_OLLAMA_MAX_TOKENS_ENV = "DIGITAL_DETECTIVE_OLLAMA_MAX_TOKENS"
_OLLAMA_DEFAULT_MAX_TOKENS = 750

# Concise prompt tailored for local models (smaller context window awareness)
_OLLAMA_SYSTEM_PROMPT = """\
You are the planning layer of Digital Detective.

You investigate microservice incidents by requesting telemetry tools.

You do not calculate RCA scores yourself.
You do not have access to ground truth.
You cannot execute remediation.
You cannot modify the query budget.
You must base factual conclusions on collected evidence IDs (E1, E2, ...).
You should investigate competing hypotheses before settling on one.
Prefer low-cost primary tools (get_metrics, get_service_health) on suspects before expensive tools.
Query suspected candidates across multiple modalities (metrics, traces, health) before concluding.
Do not diagnose a candidate based on only one modality or if evidence shows normal values.
When evidence is insufficient, request another useful tool or stop.

Respond ONLY with a single JSON object. No prose, no markdown code blocks.
Keep reasoning concise (1-2 sentences maximum). Do not write essays.

Example QUERY:
{"action": "QUERY", "tool_name": "<tool_name>", "service": "<service_name>", "reasoning": "<brief query rationale>"}

Example FINAL_DIAGNOSIS:
{"action": "FINAL_DIAGNOSIS", "reasoning": "<brief summary citing evidence>", "evidence_ids": ["E1", "E2"]}

Example STOP:
{"action": "STOP", "reasoning": "<why investigation stopped>"}
"""


class OllamaAgentModel(_OpenAICompatibleAdapter):
    """Agent model backed by a local Ollama instance.

    Uses Ollama's OpenAI-compatible endpoint (``/v1/chat/completions``).
    No API key is required. Ollama must be running locally.

    Environment variables
    ---------------------
    DIGITAL_DETECTIVE_OLLAMA_MODEL
        Model name (default: ``qwen3:8b``).
    DIGITAL_DETECTIVE_OLLAMA_BASE_URL
        Ollama base URL (default: ``http://localhost:11434/v1``).
    DIGITAL_DETECTIVE_OLLAMA_NUM_GPU
        Number of GPU layers to offload (default: ``20``). Set to empty or higher
        for larger VRAM GPUs.
    DIGITAL_DETECTIVE_OLLAMA_TEMPERATURE
        Sampling temperature (default: ``0.0``).
    DIGITAL_DETECTIVE_OLLAMA_MAX_TOKENS
        Maximum generated tokens for decision JSON (default: ``160``).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,  # Local inference can be slow on first token
        max_retries: int = 1,
        temperature: float | None = None,
        json_mode: bool = True,
        num_gpu: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        resolved_model = model or os.environ.get(_OLLAMA_MODEL_ENV, _OLLAMA_DEFAULT_MODEL)
        resolved_url = (base_url or os.environ.get(_OLLAMA_BASE_URL_ENV, _OLLAMA_DEFAULT_BASE_URL))

        env_temp = os.environ.get(_OLLAMA_TEMPERATURE_ENV)
        if temperature is not None:
            resolved_temp = float(temperature)
        elif env_temp is not None:
            try:
                resolved_temp = float(env_temp)
            except ValueError:
                resolved_temp = _OLLAMA_DEFAULT_TEMPERATURE
        else:
            resolved_temp = _OLLAMA_DEFAULT_TEMPERATURE

        env_tokens = os.environ.get(_OLLAMA_MAX_TOKENS_ENV)
        if max_tokens is not None:
            resolved_tokens = int(max_tokens)
        elif env_tokens is not None:
            try:
                resolved_tokens = int(env_tokens)
            except ValueError:
                resolved_tokens = _OLLAMA_DEFAULT_MAX_TOKENS
        else:
            resolved_tokens = _OLLAMA_DEFAULT_MAX_TOKENS

        env_num_gpu = os.environ.get(_OLLAMA_NUM_GPU_ENV)
        if num_gpu is not None:
            resolved_num_gpu: int | None = num_gpu
        elif env_num_gpu is not None:
            try:
                resolved_num_gpu = int(env_num_gpu) if env_num_gpu.strip() else None
            except ValueError:
                resolved_num_gpu = _OLLAMA_DEFAULT_NUM_GPU
        else:
            resolved_num_gpu = _OLLAMA_DEFAULT_NUM_GPU

        options: dict[str, Any] = {}
        if resolved_num_gpu is not None:
            options["num_gpu"] = resolved_num_gpu
        if resolved_tokens is not None:
            options["num_predict"] = resolved_tokens
        if resolved_temp is not None:
            options["temperature"] = resolved_temp

        extra = {"options": options} if options else None

        super().__init__(
            base_url=resolved_url,
            api_key=None,  # Ollama does not require an API key
            model=resolved_model,
            timeout=timeout,
            max_retries=max_retries,
            system_prompt=_OLLAMA_SYSTEM_PROMPT,
            temperature=resolved_temp,
            json_mode=json_mode,
            extra_payload=extra,
            max_tokens=resolved_tokens,
        )
        self.model_name = f"Ollama/{resolved_model}"

    def check_availability(self) -> tuple[bool, str]:
        """Check if Ollama is running and the configured model is available."""
        base = self._base_url.replace("/v1", "")  # e.g. http://localhost:11434
        try:
            req = urllib.request.Request(
                url=f"{base}/api/tags",
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                available_models = [m.get("name", "") for m in data.get("models", [])]
                # Check if our model is available (strip tag suffix for comparison)
                base_model = self.model.split(":")[0]
                found = any(
                    m == self.model or m.startswith(base_model)
                    for m in available_models
                )
                if found:
                    return True, f"Ollama running, model '{self.model}' available."
                else:
                    suggestion = (
                        f"Model '{self.model}' not found. "
                        f"Available: {available_models[:5]}. "
                        f"Run: ollama pull {self.model}"
                    )
                    return False, suggestion
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return False, (
                f"Ollama not reachable at {self._base_url}: {exc}. "
                "Start Ollama with: ollama serve"
            )
        except Exception as exc:
            return False, f"Ollama check failed: {exc}"


# ---------------------------------------------------------------------------
# GrokAgentModel
# ---------------------------------------------------------------------------

_GROK_DEFAULT_MODEL = "grok-3-mini"
_GROK_BASE_URL = "https://api.x.ai/v1"
_GROK_API_KEY_ENV = "XAI_API_KEY"
_GROK_MODEL_ENV = "DIGITAL_DETECTIVE_GROK_MODEL"


class GrokAgentModel(_OpenAICompatibleAdapter):
    """Agent model backed by xAI Grok.

    Requires the ``XAI_API_KEY`` environment variable.
    Model is configurable via ``DIGITAL_DETECTIVE_GROK_MODEL``.

    Environment variables
    ---------------------
    XAI_API_KEY
        Required. xAI API key.
    DIGITAL_DETECTIVE_GROK_MODEL
        Model name (default: ``grok-3-mini``).
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 60,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> None:
        api_key = os.environ.get(_GROK_API_KEY_ENV)
        resolved_model = model or os.environ.get(_GROK_MODEL_ENV, _GROK_DEFAULT_MODEL)
        super().__init__(
            base_url=_GROK_BASE_URL,
            api_key=api_key,
            model=resolved_model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            json_mode=True,
        )
        self.model_name = f"Grok/{resolved_model}"
        self._api_key_env = _GROK_API_KEY_ENV

    def is_configured(self) -> bool:
        """Return True if XAI_API_KEY is available."""
        return bool(self._api_key)


# ---------------------------------------------------------------------------
# OpenAIChatAdapter — generic OpenAI endpoint (for completeness)
# ---------------------------------------------------------------------------

_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
_OPENAI_MODEL_ENV = "OPENAI_MODEL"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIChatAdapter(_OpenAICompatibleAdapter):
    """Agent model backed by the OpenAI Chat Completions API.

    Requires the ``OPENAI_API_KEY`` environment variable.
    For local development, prefer ``OllamaAgentModel`` instead.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 30,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> None:
        api_key = os.environ.get(_OPENAI_API_KEY_ENV)
        base_url = os.environ.get(_OPENAI_BASE_URL_ENV, _OPENAI_DEFAULT_BASE_URL)
        resolved_model = model or os.environ.get(_OPENAI_MODEL_ENV, _OPENAI_DEFAULT_MODEL)
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=resolved_model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            json_mode=True,
        )
        self.model_name = f"OpenAI/{resolved_model}"

    def is_configured(self) -> bool:
        return bool(self._api_key)


# ---------------------------------------------------------------------------
# Factory: build_agent_model
# ---------------------------------------------------------------------------

def build_agent_model(
    mode: str = "auto",
    *,
    # Mock params
    mock_confidence_threshold: float = 0.55,
    mock_min_evidence_queries: int = 2,
    mock_stop_budget_reserve: int = 2,
    # Ollama params
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    ollama_timeout: int = 120,
    ollama_num_gpu: int | None = None,
    ollama_max_tokens: int | None = None,
    ollama_temperature: float | None = None,
    # Grok params
    grok_model: str | None = None,
    grok_timeout: int = 60,
    # OpenAI params (legacy)
    llm_model: str | None = None,
    llm_timeout: int = 30,
    llm_max_retries: int = 2,
    llm_temperature: float = 0.0,
) -> AgentModel:
    """Create an agent model based on ``mode`` and environment.

    Parameters
    ----------
    mode:
        ``'mock'``  — always use MockAgentModel (no network required).
        ``'ollama'`` — use local Ollama; fails gracefully if unavailable.
        ``'grok'``  — use xAI Grok; requires XAI_API_KEY.
        ``'openai'`` — use OpenAI; requires OPENAI_API_KEY.
        ``'auto'``  — try Ollama if available, else Mock.

    Raises
    ------
    ValueError
        If ``mode='grok'`` but XAI_API_KEY is not set.
        If ``mode='openai'`` but OPENAI_API_KEY is not set.
    RuntimeError
        If ``mode='ollama'`` but Ollama is not reachable.
    """
    mode = mode.lower()

    if mode == "mock":
        _log.info("Agent model: MockAgentModel.")
        return MockAgentModel(
            confidence_threshold=mock_confidence_threshold,
            min_evidence_queries=mock_min_evidence_queries,
            stop_budget_reserve=mock_stop_budget_reserve,
        )

    if mode == "ollama":
        agent = OllamaAgentModel(
            model=ollama_model,
            base_url=ollama_base_url,
            timeout=ollama_timeout,
            num_gpu=ollama_num_gpu,
            max_tokens=ollama_max_tokens,
            temperature=ollama_temperature,
        )
        available, msg = agent.check_availability()
        if not available:
            raise RuntimeError(
                f"Ollama unavailable: {msg}\n"
                "Start Ollama and pull the configured model, or use --model mock."
            )
        _log.info("Agent model: %s", agent.model_name)
        return agent

    if mode == "grok":
        if not os.environ.get(_GROK_API_KEY_ENV):
            raise ValueError(
                f"mode='grok' requires {_GROK_API_KEY_ENV} environment variable. "
                "Set it or use --model mock."
            )
        agent_g = GrokAgentModel(model=grok_model, timeout=grok_timeout)
        _log.info("Agent model: %s", agent_g.model_name)
        return agent_g

    if mode in ("openai", "llm"):
        if not os.environ.get(_OPENAI_API_KEY_ENV):
            raise ValueError(
                f"mode='openai' requires {_OPENAI_API_KEY_ENV} environment variable."
            )
        agent_o = OpenAIChatAdapter(
            model=llm_model,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
            temperature=llm_temperature,
        )
        _log.info("Agent model: %s", agent_o.model_name)
        return agent_o

    if mode == "auto":
        # Try Ollama first; fall back to Mock
        candidate = OllamaAgentModel(
            model=ollama_model,
            base_url=ollama_base_url,
            num_gpu=ollama_num_gpu,
            max_tokens=ollama_max_tokens,
            temperature=ollama_temperature,
        )
        available, msg = candidate.check_availability()
        if available:
            _log.info("Agent model: %s (auto-selected).", candidate.model_name)
            return candidate
        _log.warning(
            "Ollama unavailable (%s); falling back to MockAgentModel. "
            "Use --model ollama to require Ollama.",
            msg,
        )
        return MockAgentModel(
            confidence_threshold=mock_confidence_threshold,
            min_evidence_queries=mock_min_evidence_queries,
            stop_budget_reserve=mock_stop_budget_reserve,
        )

    raise ValueError(
        f"Unknown mode {mode!r}. Expected one of: 'mock', 'ollama', 'grok', 'openai', 'auto'."
    )
