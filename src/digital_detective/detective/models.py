"""Shared data models for Digital Detective investigation engine.

Captures investigation state, tool query auditing, structured evidence,
hypothesis ranking, causal consistency, safe remediation, and recovery verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any


@dataclass(frozen=True)
class ToolQueryRecord:
    """Audit record for a telemetry tool invocation under a query budget."""

    query_id: str
    tool_name: str
    service: str
    cost: int
    timestamp: float
    status: str  # "SUCCESS" | "REJECTED_BUDGET" | "ERROR"
    parameters: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceItem:
    """A discrete piece of structured telemetry evidence."""

    service: str
    modality: str  # "metrics" | "traces" | "logs" | "topology" | "health" | "recent_change"
    signal: str
    magnitude: float
    timestamp: int | None = None
    source: str = ""
    confidence_contribution: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedHypothesis:
    """Candidate entity with composite suspicion score and rank.

    Attributes
    ----------
    service:
        Name of the candidate microservice entity.
    score:
        Composite suspicion score combining evidence_score and consistency_score.
    confidence:
        Decision confidence bounded in [0, 1].
    rank:
        1-based rank position in current candidate universe ordering.
    evidence_score:
        Direct normalized multi-modal evidence score sum before consistency multiplier.
    consistency_score:
        Causal-consistency validation score in [0, 1].
    component_scores:
        Per-modality min-max normalized signal contributions.
    """

    service: str
    score: float
    confidence: float
    rank: int
    evidence_score: float = 0.0
    consistency_score: float = 1.0
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "score": round(self.score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "confidence": round(self.confidence, 4),
            "rank": self.rank,
            "component_scores": {k: round(v, 4) for k, v in self.component_scores.items()},
        }


@dataclass(frozen=True)
class CausalConsistencyResult:
    """Evaluation of causal consistency for a suspected root cause."""

    target_service: str
    temporal_score: float
    topology_score: float
    propagation_score: float
    consistency_score: float
    passed: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_service": self.target_service,
            "temporal_score": round(self.temporal_score, 4),
            "topology_score": round(self.topology_score, 4),
            "propagation_score": round(self.propagation_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "passed": self.passed,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RootCauseDecision:
    """Final decision output from the investigation engine."""

    root_cause_service: str
    ranked_candidates: tuple[RankedHypothesis, ...]
    confidence: float
    budget_used: int
    budget_remaining: int
    supporting_evidence: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    contradicting_evidence: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    causal_consistency: CausalConsistencyResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause_service": self.root_cause_service,
            "confidence": round(self.confidence, 4),
            "budget_used": self.budget_used,
            "budget_remaining": self.budget_remaining,
            "causal_consistency": self.causal_consistency.to_dict() if self.causal_consistency is not None else None,
            "ranked_candidates": [h.to_dict() for h in self.ranked_candidates],
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradicting_evidence": [e.to_dict() for e in self.contradicting_evidence],
        }


@dataclass(frozen=True)
class RemediationAction:
    """Safe, sandboxed remediation proposal."""

    target_service: str
    action_type: str  # "restart_service" | "rollback_deployment" | "scale_service" | "clear_connection_pool"
    preconditions: tuple[str, ...]
    risk_level: str  # "LOW" | "MEDIUM" | "HIGH"
    rollback_action: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemediationExecutionResult:
    """Execution outcome of a simulated remediation action."""

    action: RemediationAction
    success: bool
    execution_log: tuple[str, ...]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "success": self.success,
            "execution_log": list(self.execution_log),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class RecoveryVerificationResult:
    """Outcome of multi-symptom recovery verification comparing pre- and post-remediation states."""

    final_status: str  # "RESOLVED" | "NOT_RESOLVED" | "UNCERTAIN"
    recovered_metrics: tuple[str, ...]
    unresolved_symptoms: tuple[str, ...]
    new_anomalies: tuple[str, ...]
    regression_detected: bool
    comparison_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_status": self.final_status,
            "recovered_metrics": list(self.recovered_metrics),
            "unresolved_symptoms": list(self.unresolved_symptoms),
            "new_anomalies": list(self.new_anomalies),
            "regression_detected": self.regression_detected,
            "comparison_summary": self.comparison_summary,
        }


@dataclass(frozen=True)
class InterventionValidationResult:
    """Post-intervention causal validation comparing target and downstream recovery."""

    target_service: str
    action: str
    pre_intervention_consistency: float
    recovery_verified: bool
    downstream_recovery: bool
    regression_detected: bool
    causal_support: str  # "INTERVENTION_SUPPORTS_HYPOTHESIS" | "INTERVENTION_INCONCLUSIVE" | "INTERVENTION_CONTRADICTS_HYPOTHESIS"
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_service": self.target_service,
            "action": self.action,
            "pre_intervention_consistency": round(self.pre_intervention_consistency, 4),
            "recovery_verified": self.recovery_verified,
            "downstream_recovery": self.downstream_recovery,
            "regression_detected": self.regression_detected,
            "causal_support": self.causal_support,
            "explanation": self.explanation,
        }


@dataclass
class InvestigationState:
    """Mutable container tracking the complete trajectory of an active investigation."""

    incident_id: str
    system: str
    candidate_universe: tuple[str, ...]
    budget: int
    remaining_budget: int
    status: str = "INITIALIZED"  # "INITIALIZED" | "ACTIVE" | "DIAGNOSIS_COMPLETE" | "REMEDIATION_AUTHORIZED" | "BUDGET_EXHAUSTED"
    remediation_authorized: bool = False
    queries_executed: list[ToolQueryRecord] = field(default_factory=list)
    evidence_collected: list[EvidenceItem] = field(default_factory=list)
    current_rankings: tuple[RankedHypothesis, ...] = ()
    decision: RootCauseDecision | None = None
    remediation_action: RemediationAction | None = None
    remediation_result: RemediationExecutionResult | None = None
    verification_result: RecoveryVerificationResult | None = None
    intervention_validation: InterventionValidationResult | None = None

    @property
    def total_query_cost(self) -> int:
        return self.budget - self.remaining_budget

    def add_query_record(self, record: ToolQueryRecord) -> None:
        self.queries_executed.append(record)

    def add_evidence(self, item: EvidenceItem) -> None:
        self.evidence_collected.append(item)

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "system": self.system,
            "status": self.status,
            "remediation_authorized": self.remediation_authorized,
            "budget": self.budget,
            "remaining_budget": self.remaining_budget,
            "total_query_cost": self.total_query_cost,
            "candidate_universe": list(self.candidate_universe),
            "queries_executed": [q.to_dict() for q in self.queries_executed],
            "evidence_count": len(self.evidence_collected),
            "current_rankings": [r.to_dict() for r in self.current_rankings],
            "decision": self.decision.to_dict() if self.decision else None,
            "remediation_action": self.remediation_action.to_dict() if self.remediation_action else None,
            "remediation_result": self.remediation_result.to_dict() if self.remediation_result else None,
            "verification_result": self.verification_result.to_dict() if self.verification_result else None,
            "intervention_validation": self.intervention_validation.to_dict() if self.intervention_validation else None,
        }
