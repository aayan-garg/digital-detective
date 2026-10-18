"""Incident-aware query generation from observed telemetry evidence.

Constructs grounded queries for operational runbook retrieval using observed
service anomalies, candidate root causes, call-graph propagation, and log signatures
without leaking ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class IncidentQueryPackage:
    primary_query: str
    remediation_query: str
    metric_query: str
    target_service: Optional[str] = None
    target_category: Optional[str] = None
    observed_symptoms: Sequence[str] = ()


def build_incident_aware_query(
    leading_candidates: Sequence[str],
    anomalous_metrics: Sequence[str] = (),
    trace_propagation: Sequence[str] = (),
    log_signatures: Sequence[str] = (),
    system_name: str = "microservices",
) -> IncidentQueryPackage:
    """Build grounded multi-aspect queries from observed incident telemetry.

    Parameters
    ----------
    leading_candidates:
        Top ranked root-cause candidate services (e.g. from RCA / Model B).
    anomalous_metrics:
        Observed anomalous metric names (e.g. 'checkoutservice cpu', 'latency 90').
    trace_propagation:
        Observed trace caller -> callee call degradation edges.
    log_signatures:
        Key log event or template snippets observed during incident onset.
    system_name:
        Application context name.
    """
    top_svc = leading_candidates[0] if leading_candidates else "service"
    top_candidates_str = ", ".join(leading_candidates[:3])

    # Infer primary category from anomalous metrics and log signatures
    category = "service-health"
    all_text = " ".join(anomalous_metrics).lower() + " " + " ".join(log_signatures).lower()

    if "cpu" in all_text:
        category = "cpu"
    elif "mem" in all_text or "memory" in all_text or "oom" in all_text:
        category = "memory"
    elif "disk" in all_text or "storage" in all_text:
        category = "disk"
    elif "delay" in all_text or "latency" in all_text or "timeout" in all_text:
        category = "networking"
    elif "loss" in all_text or "socket" in all_text:
        category = "networking"
    elif "grpc" in all_text or "status" in all_text:
        category = "grpc"

    symptoms: list[str] = []
    if anomalous_metrics:
        symptoms.append(f"anomalous metrics: {', '.join(anomalous_metrics[:4])}")
    if trace_propagation:
        symptoms.append(f"degraded calls: {', '.join(trace_propagation[:3])}")
    if log_signatures:
        symptoms.append(f"error logs: {', '.join(log_signatures[:2])}")

    symptom_str = "; ".join(symptoms) if symptoms else "degraded performance"

    # 1. Primary diagnostic query
    primary_query = (
        f"Microservice incident in {system_name} on {top_svc} ({top_candidates_str}). "
        f"Observed condition: {symptom_str}. "
        f"Troubleshooting procedures, diagnostic commands, and failure triage."
    )

    # 2. Remediation and rollback query
    remediation_query = (
        f"Remediation, restart, scaling, or rollback runbook for {top_svc} {category} degradation. "
        f"Verification steps and safety prerequisites."
    )

    # 3. Metric and infrastructure query
    metric_query = (
        f"Kubernetes and Prometheus operations for {category} saturation on {top_svc}. "
        f"Resource limits, health checks, and service dependencies."
    )

    return IncidentQueryPackage(
        primary_query=primary_query,
        remediation_query=remediation_query,
        metric_query=metric_query,
        target_service=top_svc,
        target_category=category,
        observed_symptoms=tuple(symptoms),
    )
