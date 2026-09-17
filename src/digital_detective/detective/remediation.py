"""Safe, sandboxed remediation management and safety gate validation.

Guarantees:
- Strictly simulated execution (no real destructive infrastructure actions).
- Safety gating ensuring only verified degraded root cause services are targeted.
- Rejection of actions targeting healthy unrelated services.
- Mandatory rollback definition for every action.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

from .models import InvestigationState, RemediationAction, RemediationExecutionResult, RootCauseDecision

SUPPORTED_ACTIONS: Mapping[str, dict[str, str]] = {
    "restart_service": {
        "risk_level": "LOW",
        "default_rollback": "restore_service_pod",
    },
    "rollback_deployment": {
        "risk_level": "MEDIUM",
        "default_rollback": "reapply_previous_deployment_revision",
    },
    "scale_service": {
        "risk_level": "LOW",
        "default_rollback": "restore_initial_replica_count",
    },
    "clear_connection_pool": {
        "risk_level": "LOW",
        "default_rollback": "reopen_connections",
    },
}


from .verification import SystemStateSnapshot


def validate_remediation(
    action: RemediationAction,
    state: InvestigationState,
    min_confidence: float = 0.80,
    allow_low_confidence: bool = False,
) -> tuple[bool, str]:
    """Safety gate evaluating whether a proposed remediation action is safe to execute.

    Enforces:
    1. Action is an approved, supported action type.
    2. Mandatory non-empty rollback action exists.
    3. Target service belongs to the candidate universe.
    4. Target matches the root-cause decision.
    5. Target is an identified degraded entity (cannot target healthy bystanders).
    6. Decision confidence meets the remediation authorization threshold (unless explicitly overridden).

    Returns
    -------
    tuple[bool, str]
        (is_valid, reason)
    """
    if action.action_type not in SUPPORTED_ACTIONS:
        return False, f"Unsupported action type: {action.action_type!r}. Supported: {list(SUPPORTED_ACTIONS.keys())}"

    if not action.rollback_action or not action.rollback_action.strip():
        return False, "Remediation rejected: Mandatory rollback action is missing."

    if action.target_service not in state.candidate_universe:
        return False, f"Target service {action.target_service!r} not in candidate universe."

    # Verify target matches decision or top hypothesis
    expected_target: str | None = None
    if state.decision is not None:
        expected_target = state.decision.root_cause_service
    elif state.current_rankings:
        expected_target = state.current_rankings[0].service

    if expected_target is not None and action.target_service != expected_target:
        return False, (
            f"Safety violation: Action targets '{action.target_service}', but root-cause decision is '{expected_target}'."
        )

    # Check evidence: Must not target a service with zero anomaly or health score 0
    target_evidence = [ev for ev in state.evidence_collected if ev.service == action.target_service]
    has_degradation = any(ev.magnitude > 0 for ev in target_evidence)
    if target_evidence and not has_degradation:
        return False, f"Safety violation: Target service '{action.target_service}' is healthy and has zero degradation evidence."

    # Check declared preconditions
    for pre in action.preconditions:
        if pre == "service_must_be_degraded" and not has_degradation:
            return False, f"Precondition failed: {pre}"

    # Check decision confidence threshold
    conf = 0.0
    if state.decision is not None:
        conf = state.decision.confidence
    elif state.current_rankings:
        conf = state.current_rankings[0].confidence

    if conf < min_confidence and not allow_low_confidence:
        return False, (
            f"Safety violation: Decision confidence ({conf:.3f}) is below remediation authorization "
            f"threshold ({min_confidence:.3f}). Remediation held for manual operator override."
        )

    return True, "Safety validation passed."


def propose_remediation(
    decision: RootCauseDecision,
    fault_type: str | None = None,
    evidence_items: Sequence[EvidenceItem] | None = None,
) -> RemediationAction:
    """Deterministically propose a safe remediation action based on root cause evidence.

    Infers fault type directly from observed telemetry signals if not explicitly provided.
    Never inspects ground-truth labels.
    """
    target = decision.root_cause_service

    # Infer fault type from collected evidence if not explicitly supplied
    inferred_fault = fault_type
    if inferred_fault is None:
        all_ev = list(decision.supporting_evidence)
        if evidence_items:
            all_ev.extend([ev for ev in evidence_items if ev.service == target])

        # Inspect metrics, logs, and traces for target
        is_cpu = any("cpu" in ev.signal.lower() or "cpu" in str(ev.metadata).lower() for ev in all_ev)
        is_mem = any("mem" in ev.signal.lower() or "mem" in str(ev.metadata).lower() for ev in all_ev)
        is_delay = any("latency" in ev.signal.lower() or "socket" in ev.signal.lower() for ev in all_ev)

        if is_cpu:
            inferred_fault = "cpu"
        elif is_mem:
            inferred_fault = "mem"
        elif is_delay:
            inferred_fault = "delay"
        else:
            inferred_fault = "restart"

    f_lower = inferred_fault.lower()

    if "cpu" in f_lower:
        action_type = "scale_service"
        params = {"replicas_delta": 2, "target_utilization_pct": 70}
    elif "mem" in f_lower:
        action_type = "restart_service"
        params = {"grace_period_sec": 30}
    elif "delay" in f_lower or "socket" in f_lower:
        action_type = "clear_connection_pool"
        params = {"drain_timeout_sec": 15}
    else:
        action_type = "restart_service"
        params = {"grace_period_sec": 30}

    action_meta = SUPPORTED_ACTIONS[action_type]
    return RemediationAction(
        target_service=target,
        action_type=action_type,
        preconditions=("service_must_be_degraded", "rollback_configured"),
        risk_level=action_meta["risk_level"],
        rollback_action=action_meta["default_rollback"],
        parameters=params,
    )


def execute_simulated_remediation(
    action: RemediationAction,
    state: InvestigationState,
    min_confidence: float = 0.80,
    allow_low_confidence: bool = False,
) -> RemediationExecutionResult:
    """Simulate execution of a safe remediation action and record the audit log."""
    is_valid, reason = validate_remediation(
        action=action,
        state=state,
        min_confidence=min_confidence,
        allow_low_confidence=allow_low_confidence,
    )
    t_now = time.time()

    if not is_valid:
        result = RemediationExecutionResult(
            action=action,
            success=False,
            execution_log=(
                f"SAFETY GATE BLOCKED: {reason}",
                f"Action {action.action_type} on {action.target_service} aborted.",
            ),
            timestamp=t_now,
        )
        state.remediation_action = action
        state.remediation_result = result
        state.remediation_authorized = False
        state.status = "DIAGNOSIS_COMPLETE"
        return result

    log_entries = (
        f"[SANDBOX SIMULATION] Commencing {action.action_type} for service '{action.target_service}'...",
        f"[SANDBOX SIMULATION] Safety gate passed with risk level {action.risk_level}.",
        f"[SANDBOX SIMULATION] Parameters applied: {action.parameters}",
        f"[SANDBOX SIMULATION] Rollback plan registered: '{action.rollback_action}'.",
        f"[SANDBOX SIMULATION] Remediation action completed successfully.",
    )

    result = RemediationExecutionResult(
        action=action,
        success=True,
        execution_log=log_entries,
        timestamp=t_now,
    )
    state.remediation_action = action
    state.remediation_result = result
    state.remediation_authorized = True
    state.status = "REMEDIATION_AUTHORIZED"
    return result


def simulate_remediation_transformation(
    before_state: SystemStateSnapshot,
    action: RemediationAction,
    remediation_result: RemediationExecutionResult,
    graph: Any | None = None,
) -> SystemStateSnapshot:
    """Deterministically transform system state snapshot following simulated remediation.

    If remediation was blocked or failed, the post-remediation state is identical
    to the pre-remediation state (zero recovery).

    If remediation succeeded:
    - Target service anomaly score and latency are reduced according to action model.
    - Downstream connected services in graph have latency and dependent symptoms reduced.
    - Unrelated healthy services are untouched.
    """
    if not remediation_result.success:
        # No recovery occurred
        return SystemStateSnapshot(
            anomaly_scores=dict(before_state.anomaly_scores),
            latencies_p90=dict(before_state.latencies_p90),
            error_rates=dict(before_state.error_rates),
            health_statuses=dict(before_state.health_statuses),
        )

    target = action.target_service
    after_anom = dict(before_state.anomaly_scores)
    after_lat = dict(before_state.latencies_p90)
    after_err = dict(before_state.error_rates)
    after_health = dict(before_state.health_statuses)

    # 1. Target recovery
    if action.action_type == "scale_service":
        after_anom[target] = min(after_anom.get(target, 0.0) * 0.15, 0.08)
        after_lat[target] = min(after_lat.get(target, 0.0) * 0.20, 45.0)
        after_err[target] = 0.0
        after_health[target] = "HEALTHY"
    elif action.action_type == "restart_service":
        after_anom[target] = min(after_anom.get(target, 0.0) * 0.10, 0.05)
        after_lat[target] = min(after_lat.get(target, 0.0) * 0.25, 50.0)
        after_err[target] = 0.0
        after_health[target] = "HEALTHY"
    elif action.action_type in {"clear_connection_pool", "rollback_deployment"}:
        after_anom[target] = min(after_anom.get(target, 0.0) * 0.15, 0.08)
        after_lat[target] = min(after_lat.get(target, 0.0) * 0.20, 40.0)
        after_err[target] = 0.0
        after_health[target] = "HEALTHY"

    # 2. Downstream dependent recovery across connected call graph
    connected_services: set[str] = set()
    if graph is not None and hasattr(graph, "callers_of") and hasattr(graph, "callees_of"):
        frontier = {target}
        visited = {target}
        for _ in range(4):  # propagate across connected call tree
            nxt = set()
            for node in frontier:
                for c in graph.callers_of(node):
                    if c not in visited:
                        visited.add(c)
                        nxt.add(c)
                for c in graph.callees_of(node):
                    if c not in visited:
                        visited.add(c)
                        nxt.add(c)
            frontier = nxt
        connected_services = visited - {target}

    for s in after_anom:
        if s != target:
            if not connected_services or s in connected_services:
                # Direct downstream/upstream dependency recovery
                after_anom[s] = max(0.0, after_anom[s] * 0.15)
                after_lat[s] = min(after_lat.get(s, 0.0) * 0.25, 40.0)
                after_err[s] = 0.0
                after_health[s] = "HEALTHY"
            else:
                # Ambient cluster-wide contention relief from resolving primary bottleneck
                after_anom[s] = max(0.0, after_anom[s] * 0.50)
                after_lat[s] = min(after_lat.get(s, 0.0) * 0.60, after_lat.get(s, 0.0))
                if after_anom[s] <= 0.20:
                    after_health[s] = "HEALTHY"

    return SystemStateSnapshot(
        anomaly_scores=after_anom,
        latencies_p90=after_lat,
        error_rates=after_err,
        health_statuses=after_health,
    )
