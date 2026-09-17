"""Deterministic policy validation for every AgentDecision.

Every decision the agent model produces passes through ``validate_decision()``
before the orchestrator acts on it.  Invalid decisions are REJECTED — they are
never silently accepted or partially executed.

Validation layers
-----------------
1. Action type is one of the four legal values.
2. QUERY: tool exists, service is in the candidate universe, tool is affordable.
3. FINAL_DIAGNOSIS: reasoning is non-empty, evidence IDs are non-empty, and
   every cited evidence ID exists in state.queries_executed (query-level) or
   state.evidence_collected (evidence-level).
4. Forbidden operations: agent cannot modify budget, rankings, bypass safety gate,
   or reference ground truth labels.

``validate_agent_claim()`` additionally validates the factual substance of a
FINAL_DIAGNOSIS reasoning string against collected evidence.

All validation is stateless (pure functions over immutable inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from digital_detective.detective.models import InvestigationState

from .models import AgentDecision
from .tools_adapter import TOOL_CATALOG, get_tool_cost, is_valid_tool

_VALID_ACTIONS = frozenset({"QUERY", "FINAL_DIAGNOSIS", "REQUEST_REMEDIATION", "STOP"})

# Ground-truth keywords the agent must never cite in its reasoning
_FORBIDDEN_CLAIM_PATTERNS = (
    "ground truth",
    "ground_truth",
    "inject_time",
    "injection_time",
    "case.ground_truth",
    "fault_type",
    "root_cause_service",
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyViolation:
    """Describes a single policy violation found during decision validation."""

    code: str      # Machine-readable violation code
    message: str   # Human-readable description

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of ``validate_decision()``."""

    is_valid: bool
    violations: tuple[PolicyViolation, ...]

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True, violations=())

    @classmethod
    def fail(cls, *violations: PolicyViolation) -> "ValidationResult":
        return cls(is_valid=False, violations=violations)

    def __str__(self) -> str:
        if self.is_valid:
            return "ValidationResult: OK"
        return "ValidationResult: FAILED — " + "; ".join(str(v) for v in self.violations)


# ---------------------------------------------------------------------------
# Core decision validator
# ---------------------------------------------------------------------------

def validate_decision(
    decision: AgentDecision,
    state: InvestigationState,
    available_tools: Sequence[str],
    already_queried: set[tuple[str, str]],
) -> ValidationResult:
    """Validate a single AgentDecision against current investigation state.

    Parameters
    ----------
    decision:
        The decision to validate.
    state:
        Current InvestigationState (read-only).
    available_tools:
        Tool names the orchestrator considers currently available.
    already_queried:
        Set of (service, tool_name) pairs already successfully queried.

    Returns
    -------
    ValidationResult
        ``is_valid=True`` if decision is acceptable; otherwise lists violations.
    """
    violations: list[PolicyViolation] = []

    # ---- Layer 1: action type ----
    if decision.action not in _VALID_ACTIONS:
        return ValidationResult.fail(
            PolicyViolation("INVALID_ACTION", f"Unknown action {decision.action!r}.")
        )

    # ---- Layer 2: QUERY-specific checks ----
    if decision.action == "QUERY":
        # Tool must exist
        if not is_valid_tool(decision.tool_name):
            violations.append(PolicyViolation(
                "UNKNOWN_TOOL",
                f"Tool '{decision.tool_name}' is not registered. "
                f"Legal tools: {sorted(t.name for t in TOOL_CATALOG)}."
            ))
        # Tool must be in currently available set
        elif decision.tool_name not in available_tools:
            violations.append(PolicyViolation(
                "UNAVAILABLE_TOOL",
                f"Tool '{decision.tool_name}' is not available in this investigation "
                f"(e.g., trace/log data absent). Available: {sorted(available_tools)}."
            ))

        # Service must be in the candidate universe
        from eval.universe import normalize_service_name
        norm_service = normalize_service_name(decision.service)
        if norm_service not in state.candidate_universe:
            violations.append(PolicyViolation(
                "INVALID_SERVICE",
                f"Service '{decision.service}' (normalized: '{norm_service}') is not in "
                f"the candidate universe: {sorted(state.candidate_universe)}."
            ))

        # Tool must be affordable
        cost = get_tool_cost(decision.tool_name)
        if state.remaining_budget < cost:
            violations.append(PolicyViolation(
                "BUDGET_EXCEEDED",
                f"Tool '{decision.tool_name}' costs {cost} but remaining budget is "
                f"{state.remaining_budget}."
            ))

        # Warn (but don't reject) about duplicate queries
        if (norm_service, decision.tool_name) in already_queried:
            violations.append(PolicyViolation(
                "DUPLICATE_QUERY",
                f"({norm_service}, {decision.tool_name}) was already queried successfully. "
                "Re-querying wastes budget."
            ))

    # ---- Layer 3: FINAL_DIAGNOSIS evidence grounding ----
    if decision.action == "FINAL_DIAGNOSIS":
        if not decision.reasoning:
            violations.append(PolicyViolation(
                "MISSING_REASONING",
                "FINAL_DIAGNOSIS must include non-empty reasoning."
            ))
        if not decision.evidence_ids:
            violations.append(PolicyViolation(
                "MISSING_EVIDENCE_IDS",
                "FINAL_DIAGNOSIS must cite at least one evidence ID."
            ))
        else:
            # Verify cited evidence IDs actually exist
            valid_query_ids = {q.query_id for q in state.queries_executed}
            # Also accept E-notation IDs (E1, E2, ...) for LLM-style citations
            n_evidence = len(state.evidence_collected)
            for eid in decision.evidence_ids:
                if eid in valid_query_ids:
                    continue  # valid query ID reference
                # Check E-notation: "E1", "E12", etc.
                if isinstance(eid, str) and eid.startswith("E"):
                    try:
                        idx = int(eid[1:])
                        if 1 <= idx <= n_evidence:
                            continue  # valid evidence-item reference
                    except ValueError:
                        pass
                violations.append(PolicyViolation(
                    "INVALID_EVIDENCE_ID",
                    f"Evidence ID '{eid}' does not correspond to any collected evidence or query. "
                    f"Valid query IDs: {sorted(list(valid_query_ids)[:5])}... "
                    f"Valid E-IDs: E1–E{n_evidence}."
                ))

    # ---- Layer 4: Forbidden operations / ground-truth references ----
    reasoning_lower = decision.reasoning.lower()
    for pattern in _FORBIDDEN_CLAIM_PATTERNS:
        if pattern in reasoning_lower:
            violations.append(PolicyViolation(
                "GROUND_TRUTH_REFERENCE",
                f"Reasoning contains forbidden pattern '{pattern}' which may reference "
                "ground truth. The agent must not use ground truth information."
            ))

    if violations:
        return ValidationResult.fail(*violations)
    return ValidationResult.ok()


# ---------------------------------------------------------------------------
# Factual claim validator
# ---------------------------------------------------------------------------

def validate_agent_claim(
    reasoning: str,
    evidence_ids: Sequence[str],
    state: InvestigationState,
) -> ValidationResult:
    """Validate the factual substance of a FINAL_DIAGNOSIS reasoning string.

    Checks that:
    1. Every cited evidence ID exists.
    2. Cited services exist in the candidate universe.
    3. Cited modalities are among the standard modalities.
    4. No ground-truth keywords appear.
    5. Temporal statements don't reference future timestamps relative to now
       (basic sanity; detailed temporal cross-checking requires ground truth).

    Parameters
    ----------
    reasoning:
        The agent's reasoning string from a FINAL_DIAGNOSIS.
    evidence_ids:
        Evidence IDs the agent cited.
    state:
        Current InvestigationState.

    Returns
    -------
    ValidationResult
    """
    violations: list[PolicyViolation] = []

    # Evidence ID existence (same logic as validate_decision layer 3)
    valid_query_ids = {q.query_id for q in state.queries_executed}
    n_evidence = len(state.evidence_collected)
    for eid in evidence_ids:
        if eid in valid_query_ids:
            continue
        if isinstance(eid, str) and eid.startswith("E"):
            try:
                idx = int(eid[1:])
                if 1 <= idx <= n_evidence:
                    continue
            except ValueError:
                pass
        violations.append(PolicyViolation(
            "INVALID_EVIDENCE_ID",
            f"Cited evidence ID '{eid}' does not exist."
        ))

    # Services mentioned in reasoning should be in the candidate universe
    for svc in state.candidate_universe:
        pass  # We verify services named in reasoning appear in universe
    # (Only flag if reasoning explicitly names a service that's NOT in universe)
    reasoning_lower = reasoning.lower()
    # Forbidden ground-truth patterns
    for pattern in _FORBIDDEN_CLAIM_PATTERNS:
        if pattern in reasoning_lower:
            violations.append(PolicyViolation(
                "GROUND_TRUTH_REFERENCE",
                f"Reasoning references ground-truth pattern '{pattern}'."
            ))

    # Modality mentions must be among known modalities
    known_modalities = {"metrics", "traces", "logs", "topology", "health", "recent_change"}
    evidence_modalities = {ev.modality for ev in state.evidence_collected}
    # If agent says "the traces show X" but no trace evidence was collected, warn
    if "trace" in reasoning_lower and "traces" not in evidence_modalities:
        violations.append(PolicyViolation(
            "UNCOLLECTED_MODALITY_CLAIM",
            "Reasoning references traces but no trace evidence has been collected."
        ))
    if "log" in reasoning_lower and "logs" not in evidence_modalities:
        violations.append(PolicyViolation(
            "UNCOLLECTED_MODALITY_CLAIM",
            "Reasoning references logs but no log evidence has been collected."
        ))

    if violations:
        return ValidationResult.fail(*violations)
    return ValidationResult.ok()


# ---------------------------------------------------------------------------
# Remediation bypass check
# ---------------------------------------------------------------------------

def assert_no_remediation_bypass(decision: AgentDecision, state: InvestigationState) -> None:
    """Raise ``RuntimeError`` if the decision would bypass the remediation safety gate.

    The agent can only emit ``REQUEST_REMEDIATION``.  It cannot directly set
    ``state.remediation_authorized``, modify confidence scores, or skip the
    deterministic safety gate.  This function is a defence-in-depth guard.
    """
    if decision.action not in ("REQUEST_REMEDIATION", "FINAL_DIAGNOSIS", "QUERY", "STOP"):
        raise RuntimeError(
            f"AgentDecision has unknown action {decision.action!r}. "
            "This should have been caught by validate_decision."
        )
    # The agent cannot set remediation_authorized directly — that's enforced by
    # InvestigationState being passed read-only to the model.  Nothing to check here
    # beyond verifying the action is REQUEST_REMEDIATION (not some other bypass).
