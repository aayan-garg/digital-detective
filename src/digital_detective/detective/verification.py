"""Post-remediation recovery verification.

Compares pre-remediation and post-remediation system states across:
1. Root-cause anomaly score reduction.
2. Service error rate reduction.
3. Downstream latency normalization.
4. Overall multi-service health status.
5. Detection of new anomalies / regressions.

Requires broad system-level recovery before declaring an incident RESOLVED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .models import InterventionValidationResult, RecoveryVerificationResult



@dataclass
class SystemStateSnapshot:
    """Snapshot of multi-service telemetry signals at a given point in time."""

    anomaly_scores: dict[str, float] = field(default_factory=dict)
    latencies_p90: dict[str, float] = field(default_factory=dict)
    error_rates: dict[str, float] = field(default_factory=dict)
    health_statuses: dict[str, str] = field(default_factory=dict)


def verify_recovery(
    before_state: SystemStateSnapshot | Mapping[str, Any],
    after_state: SystemStateSnapshot | Mapping[str, Any],
    target_service: str = "",
) -> RecoveryVerificationResult:
    """Verify whether remediation successfully resolved the incident across the system.

    Parameters
    ----------
    before_state:
        System state snapshot prior to remediation.
    after_state:
        System state snapshot following simulated or executed remediation.
    target_service:
        Name of the targeted root-cause service.

    Returns
    -------
    RecoveryVerificationResult
        Multi-symptom recovery analysis with final verdict: RESOLVED, NOT_RESOLVED, or UNCERTAIN.
    """
    # Normalize inputs to dictionaries
    b_anom = before_state.anomaly_scores if isinstance(before_state, SystemStateSnapshot) else before_state.get("anomaly_scores", {})
    a_anom = after_state.anomaly_scores if isinstance(after_state, SystemStateSnapshot) else after_state.get("anomaly_scores", {})

    b_lat = before_state.latencies_p90 if isinstance(before_state, SystemStateSnapshot) else before_state.get("latencies_p90", {})
    a_lat = after_state.latencies_p90 if isinstance(after_state, SystemStateSnapshot) else after_state.get("latencies_p90", {})

    b_err = before_state.error_rates if isinstance(before_state, SystemStateSnapshot) else before_state.get("error_rates", {})
    a_err = after_state.error_rates if isinstance(after_state, SystemStateSnapshot) else after_state.get("error_rates", {})

    b_health = before_state.health_statuses if isinstance(before_state, SystemStateSnapshot) else before_state.get("health_statuses", {})
    a_health = after_state.health_statuses if isinstance(after_state, SystemStateSnapshot) else after_state.get("health_statuses", {})

    all_services = sorted(set(b_anom.keys()) | set(a_anom.keys()) | set(b_lat.keys()) | set(a_lat.keys()))

    recovered_metrics: list[str] = []
    unresolved_symptoms: list[str] = []
    new_anomalies: list[str] = []
    regression_detected = False

    # 1. Anomaly Score Analysis
    for s in all_services:
        b_sc = float(b_anom.get(s, 0.0))
        a_sc = float(a_anom.get(s, 0.0))

        if b_sc > 0.2:
            if a_sc <= 0.2 or (b_sc > 0 and (b_sc - a_sc) / b_sc >= 0.65):
                recovered_metrics.append(f"{s}_anomaly_score: {b_sc:.2f} -> {a_sc:.2f}")
            else:
                unresolved_symptoms.append(f"{s}_anomaly_score: still {a_sc:.2f} (was {b_sc:.2f})")
        elif b_sc <= 0.2 and a_sc > 0.4:
            new_anomalies.append(f"{s}_anomaly_score elevated to {a_sc:.2f}")
            regression_detected = True

    # 2. Latency & Error Analysis
    for s in all_services:
        b_l = float(b_lat.get(s, 0.0))
        a_l = float(a_lat.get(s, 0.0))
        if b_l > 100.0 and a_l <= 100.0:
            recovered_metrics.append(f"{s}_latency_p90: {b_l:.1f}ms -> {a_l:.1f}ms")
        elif b_l > 100.0 and a_l > 100.0:
            unresolved_symptoms.append(f"{s}_latency_p90 elevated at {a_l:.1f}ms")

        b_e = float(b_err.get(s, 0.0))
        a_e = float(a_err.get(s, 0.0))
        if b_e > 0 and a_e == 0:
            recovered_metrics.append(f"{s}_error_count: {b_e} -> 0")
        elif a_e > b_e and a_e > 5:
            new_anomalies.append(f"{s}_error_count increased from {b_e} to {a_e}")
            regression_detected = True

    # 3. Overall Recovery Status Determination
    # Rule: Success requires targeted root-cause recovery AND no downstream regressions.
    target_recovered = any(target_service in m for m in recovered_metrics) if target_service else True

    if regression_detected:
        final_status = "NOT_RESOLVED"
    elif unresolved_symptoms and not target_recovered:
        final_status = "NOT_RESOLVED"
    elif unresolved_symptoms and target_recovered:
        # Target recovered, but some symptoms remain unsettled
        final_status = "UNCERTAIN"
    elif not unresolved_symptoms and (recovered_metrics or not b_anom):
        final_status = "RESOLVED"
    else:
        final_status = "UNCERTAIN"

    comparison_summary = {
        "target_service": target_service,
        "target_recovered": target_recovered,
        "recovered_count": len(recovered_metrics),
        "unresolved_count": len(unresolved_symptoms),
        "new_anomaly_count": len(new_anomalies),
        "before_health": b_health,
        "after_health": a_health,
    }

    return RecoveryVerificationResult(
        final_status=final_status,
        recovered_metrics=tuple(recovered_metrics),
        unresolved_symptoms=tuple(unresolved_symptoms),
        new_anomalies=tuple(new_anomalies),
        regression_detected=regression_detected,
        comparison_summary=comparison_summary,
    )


def validate_intervention(
    verification: RecoveryVerificationResult,
    target_service: str,
    action: str,
    pre_intervention_consistency: float,
    before_state: SystemStateSnapshot | Mapping[str, Any] | None = None,
) -> InterventionValidationResult:
    """Post-hoc causal validation of remediation outcome.

    Evaluates whether targeted intervention on the candidate produces broad system
    recovery of symptoms attributed to that candidate, or contradicts the hypothesis.
    Does NOT modify pre-intervention RCA rankings or scores.
    """
    recovery_verified = (verification.final_status == "RESOLVED")
    regression_detected = verification.regression_detected
    target_recovered = any(target_service in m for m in verification.recovered_metrics) if target_service else False

    # Assess downstream recovery
    downstream_recovered_metrics = [
        m for m in verification.recovered_metrics
        if not (target_service and target_service in m)
    ]
    downstream_unresolved_metrics = [
        m for m in verification.unresolved_symptoms
        if not (target_service and target_service in m)
    ]

    # If there were downstream anomalies before remediation, check if they resolved
    had_downstream_anomalies = False
    if before_state is not None:
        b_anom = before_state.anomaly_scores if isinstance(before_state, SystemStateSnapshot) else before_state.get("anomaly_scores", {})
        b_lat = before_state.latencies_p90 if isinstance(before_state, SystemStateSnapshot) else before_state.get("latencies_p90", {})
        b_err = before_state.error_rates if isinstance(before_state, SystemStateSnapshot) else before_state.get("error_rates", {})
        for s in (set(b_anom.keys()) | set(b_lat.keys()) | set(b_err.keys())):
            if s != target_service:
                if float(b_anom.get(s, 0.0)) > 0.2 or float(b_lat.get(s, 0.0)) > 100.0 or float(b_err.get(s, 0.0)) > 0:
                    had_downstream_anomalies = True
                    break

    if had_downstream_anomalies:
        downstream_recovery = (len(downstream_unresolved_metrics) == 0) and (len(downstream_recovered_metrics) > 0 or recovery_verified)
    else:
        downstream_recovery = len(downstream_unresolved_metrics) == 0

    # Interpretation:
    # If a targeted intervention produces broad recovery of the symptoms attributed to that candidate:
    # INTERVENTION_SUPPORTS_HYPOTHESIS
    # Otherwise:
    # INTERVENTION_INCONCLUSIVE or INTERVENTION_CONTRADICTS_HYPOTHESIS
    if regression_detected:
        causal_support = "INTERVENTION_CONTRADICTS_HYPOTHESIS"
        explanation = (
            f"Remediation action '{action}' on candidate '{target_service}' induced new regressions "
            f"({', '.join(verification.new_anomalies[:2]) or 'detected regressions'}), contradicting the hypothesis."
        )
    elif verification.final_status == "NOT_RESOLVED" and not target_recovered:
        causal_support = "INTERVENTION_CONTRADICTS_HYPOTHESIS"
        explanation = (
            f"Remediation action '{action}' on candidate '{target_service}' failed to resolve target root-cause anomalies, "
            "contradicting the causal diagnosis."
        )
    elif recovery_verified and (downstream_recovery or not had_downstream_anomalies) and not regression_detected:
        causal_support = "INTERVENTION_SUPPORTS_HYPOTHESIS"
        explanation = (
            f"Targeted intervention '{action}' on candidate '{target_service}' produced broad system recovery "
            "across root-cause and downstream symptoms without regressions."
        )
    else:
        causal_support = "INTERVENTION_INCONCLUSIVE"
        explanation = (
            f"Remediation action '{action}' on candidate '{target_service}' achieved partial recovery "
            f"(verification status: {verification.final_status}), remaining inconclusive for causal validation."
        )

    return InterventionValidationResult(
        target_service=target_service,
        action=action,
        pre_intervention_consistency=pre_intervention_consistency,
        recovery_verified=recovery_verified,
        downstream_recovery=downstream_recovery,
        regression_detected=regression_detected,
        causal_support=causal_support,
        explanation=explanation,
    )
