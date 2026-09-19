"""System prompt and state serialization for the Digital Detective agent.

Keeps all prompt logic isolated in one module:
  - ``SYSTEM_PROMPT``: the fixed role + constraint declaration
  - ``format_state_for_prompt()``: compact structured context for each turn
  - ``format_evidence_block()``: stable E-ID evidence catalog
  - ``format_tool_result()``: per-turn tool result summary

Design constraints
------------------
* Do NOT dump raw telemetry (Parquet columns, metric arrays) into the prompt.
* Every piece of evidence has a stable EID (E1, E2, ...) assigned at collection time.
* The agent sees only ranked hypotheses, causal-consistency scores, and evidence summaries.
* Ground truth is never included.
"""

from __future__ import annotations

from typing import Sequence

from digital_detective.detective.models import (
    EvidenceItem,
    InvestigationState,
    RankedHypothesis,
)


# ---------------------------------------------------------------------------
# Fixed system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the planning layer of Digital Detective, an autonomous incident-investigation system.

ROLE
----
You are an orchestrator. You decide WHICH telemetry queries to issue and WHEN to stop.
You do NOT calculate anomaly scores, causal scores, or rankings.
You do NOT have access to ground truth about the incident.
You do NOT execute or authorize remediation.
You MUST base every factual claim in your reasoning on supplied evidence IDs (E1, E2, ...).

INVESTIGATION FLOW
------------------
1. Inspect the current hypothesis ranking produced by the deterministic scoring engine.
2. Identify the strongest uncertainty: which high-ranked service has gaps in evidence?
3. Select the telemetry query that best resolves that uncertainty.
4. Observe the returned evidence.
5. Reassess competing candidates against the new evidence.
6. Query another modality when evidence across candidates is still ambiguous.
7. Issue FINAL_DIAGNOSIS only when you can cite specific evidence IDs supporting the decision.
8. If evidence remains weak or budget is low, issue STOP rather than hallucinate certainty.

CONSTRAINTS
-----------
- You may only call tools listed in AVAILABLE TOOLS.
- Every query deducts budget. You cannot increase or reset the budget.
- You must not re-query a (service, tool) pair you have already queried successfully.
- FINAL_DIAGNOSIS must cite at least one evidence ID (e.g., "E3, E7, E12").
- Do not invent metric values, latencies, or timestamps. Only cite collected evidence.
- Investigate competing hypotheses before settling on the top-ranked service.

OUTPUT FORMAT
-------------
Respond with a single JSON object and nothing else:

For QUERY:
{
  "action": "QUERY",
  "tool_name": "<tool>",
  "service": "<service>",
  "reasoning": "<why this query reduces uncertainty>"
}

For FINAL_DIAGNOSIS:
{
  "action": "FINAL_DIAGNOSIS",
  "diagnosis_service": "<service>",
  "reasoning": "<evidence-grounded explanation citing E-IDs>",
  "evidence_ids": ["E3", "E7", "E12"]
}

For REQUEST_REMEDIATION:
{
  "action": "REQUEST_REMEDIATION",
  "reasoning": "<why remediation is warranted>"
}

For STOP:
{
  "action": "STOP",
  "reasoning": "<why investigation cannot continue or conclude>"
}

Do not include any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Compact Evidence, Ranking, and State Formatting
# ---------------------------------------------------------------------------

def assign_evidence_ids(evidence_items: list[EvidenceItem]) -> dict[int, EvidenceItem]:
    """Return a mapping {eid_int: EvidenceItem} for display and citation."""
    return {i + 1: ev for i, ev in enumerate(evidence_items)}


def format_evidence_block(evidence_items: list[EvidenceItem], top_services: set[str] | None = None, max_items: int = 8) -> str:
    """Format relevant collected evidence concisely for the prompt.

    Only displays evidence for top candidates or non-trivial signals,
    prioritizing top candidates, avoiding massive telemetry dumps in the prompt.
    """
    if not evidence_items:
        return "COLLECTED EVIDENCE: (none yet)"

    lines = ["COLLECTED EVIDENCE:"]
    top_candidates = []
    other_candidates = []
    for i, ev in enumerate(evidence_items, start=1):
        if top_services and ev.service in top_services:
            top_candidates.append((i, ev))
        elif ev.magnitude > 0.1:
            other_candidates.append((i, ev))

    # Prioritize top candidates, then fill remaining quota with other candidates
    selected = top_candidates[-max_items:]
    remaining = max_items - len(selected)
    if remaining > 0 and other_candidates:
        selected = other_candidates[-remaining:] + selected

    for i, ev in selected:
        lines.append(
            f"  E{i}: {ev.service} [{ev.modality}] {ev.signal}={ev.magnitude:.2f}"
        )
    return "\n".join(lines)


def format_rankings(
    rankings: Sequence[RankedHypothesis],
    already_queried: set[tuple[str, str]] | None = None,
    top_n: int = 3,
) -> str:
    """Format only top candidate hypothesis rankings concisely for the prompt."""
    if not rankings:
        return "TOP HYPOTHESES: (none)"
    lines = ["TOP HYPOTHESES:"]
    queried_services = {svc for svc, _ in (already_queried or set())}
    for hyp in rankings[:top_n]:
        status_tag = ""
        if already_queried is not None:
            status_tag = " [queried]" if hyp.service in queried_services else " [unexamined]"
        lines.append(
            f"  #{hyp.rank} {hyp.service} score={hyp.score:.3f} conf={hyp.confidence:.1%} cc={hyp.consistency_score:.2f}{status_tag}"
        )
    return "\n".join(lines)


def format_causal_findings(state: InvestigationState) -> str:
    """Format concise causal consistency findings."""
    dec = state.decision
    if dec is not None and dec.causal_consistency is not None:
        cc = dec.causal_consistency
        status = "PASSED" if cc.passed else "FAILED"
        return f"CAUSAL STATUS: {cc.target_service} {status} (overall={cc.consistency_score:.2f}, topo={cc.topology_score:.2f})"
    if state.current_rankings:
        top = state.current_rankings[0]
        return f"CAUSAL STATUS: top candidate '{top.service}' cc={top.consistency_score:.2f}"
    return "CAUSAL STATUS: (pending)"


def check_finalization_directive(
    state: InvestigationState,
    already_queried: set[tuple[str, str]] | None = None,
) -> str | None:
    """Check if deterministic evidence fusion has reached high confidence and stability.

    Signals to the agent when evidence is sufficient to finalize, or when
    unexamined close competitors must be queried first to compare hypotheses.
    """
    if not state.current_rankings:
        return None
    top = state.current_rankings[0]
    runner_up = state.current_rankings[1] if len(state.current_rankings) > 1 else None
    runner_up_score = runner_up.score if runner_up else 0.0
    margin = top.score - runner_up_score

    # Distinct anomalous modalities observed for top candidate
    top_modalities = {
        ev.modality for ev in state.evidence_collected
        if ev.service == top.service and ev.magnitude > 0.1
    }

    cc_passed = top.consistency_score >= 0.70
    if state.decision and state.decision.causal_consistency:
        cc_passed = state.decision.causal_consistency.passed

    # Services queried so far
    services_queried = {svc for svc, _ in (already_queried or set())}
    if not services_queried:
        services_queried = {q.service for q in state.queries_executed if q.status == "SUCCESS"}

    # If a close competing hypothesis (margin < 0.10) has not been queried yet,
    # the top candidate is not stable; guide the agent to investigate the competitor.
    runner_up_unexamined = (
        runner_up is not None
        and margin < 0.10
        and runner_up.service not in services_queried
    )

    if runner_up_unexamined and len(services_queried) < 3:
        return (
            f"STATUS DIRECTIVE: Competing candidate #{runner_up.rank} '{runner_up.service}' "
            f"is close in score (margin={margin:.3f}) and unexamined. "
            f"Query '{runner_up.service}' to resolve uncertainty before concluding."
        )

    # Stability requires that:
    # 1. At least 2 anomalous modalities observed on top candidate
    # 2. Causal consistency passed
    # 3. Not blocked by an unexamined close competitor
    is_stable = (
        len(top_modalities) >= 2
        and not runner_up_unexamined
        and (margin >= 0.05 or len(services_queried) >= 2)
    )

    if is_stable and cc_passed:
        return (
            f"STATUS DIRECTIVE: Strong, consistent evidence collected for #{top.rank} '{top.service}' "
            f"({len(top_modalities)} modalities observed, cc={top.consistency_score:.2f}, margin={margin:.3f}). "
            f"Evidence is sufficient. Issue FINAL_DIAGNOSIS citing relevant evidence IDs (e.g. ['E1', 'E2'])."
        )
    return None


def format_compact_already_queried(already_queried: set[tuple[str, str]]) -> str:
    """Format already executed (service, tool) pairs grouped by service."""
    if not already_queried:
        return "(none)"
    by_service: dict[str, list[str]] = {}
    for svc, tool in sorted(already_queried):
        by_service.setdefault(svc, []).append(tool)
    parts = [f"{svc}: {', '.join(tools)}" for svc, tools in by_service.items()]
    return "; ".join(parts)


def format_state_for_prompt(
    state: InvestigationState,
    available_tools: Sequence[str],
    already_queried: set[tuple[str, str]],
    turn: int,
) -> str:
    """Produce a compact structured context block for injection into the LLM prompt.

    Strictly minimizes tokens while retaining decision-relevant signals.
    Ground truth is never included.
    """
    top_services = {h.service for h in state.current_rankings[:3]} if state.current_rankings else None

    # Compact tool list with costs
    tool_costs = {
        "get_metrics": 1,
        "get_service_health": 1,
        "get_recent_change": 2,
        "get_traces": 3,
        "get_neighbors": 1,
        "get_logs": 2,
    }
    tools_str = ", ".join(f"{t}({tool_costs.get(t, 1)})" for t in available_tools)

    sections = [
        f"=== INCIDENT: {state.incident_id} | TURN {turn} | Budget Remaining: {state.remaining_budget}/{state.budget} ===",
        f"AVAILABLE TOOLS: {tools_str}",
        f"ALREADY QUERIED (DO NOT REPEAT): {format_compact_already_queried(already_queried)}",
        format_rankings(state.current_rankings, already_queried=already_queried, top_n=3),
        format_evidence_block(state.evidence_collected, top_services=top_services, max_items=6),
        format_causal_findings(state),
    ]

    directive = check_finalization_directive(state, already_queried=already_queried)
    if directive:
        sections.append(f"\n{directive}")

    sections.append("=== END STATE ===")
    return "\n".join(sections)


def format_rag_context_block(snippets: list[str]) -> str:
    """Wrap retrieved operational knowledge for injection into the investigator prompt.

    Returns an empty string when no snippets are available so callers can
    safely concatenate without conditional guards.

    The block is labelled reference-only so the LLM does not treat it as
    deterministic RCA evidence; it must not override RCA scores, Model B
    rankings, or safety decisions.
    """
    if not snippets:
        return ""
    body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(snippets, 1))
    return (
        "\n\nOPERATIONAL KNOWLEDGE (RAG \u2013 REFERENCE ONLY):\n"
        "The following passages are retrieved reference material. "
        "They do NOT override deterministic RCA scores, Model B rankings, "
        "or safety decisions. Use them only as supporting context.\n\n"
        + body
    )
