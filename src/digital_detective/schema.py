"""Canonical Incident Schema for Digital Detective.

Standardizes data representation across ingestion, changepoint detection,
feature extraction, RCA ranking, RAG retrieval, agent investigation,
safety policy enforcement, and API/UI serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence
from pydantic import BaseModel, Field, field_validator


class SystemDomain(str, Enum):
    ONLINE_BOUTIQUE = "online-boutique"
    SOCK_SHOP = "sock-shop"
    TRAIN_TICKET = "train-ticket"
    GENERIC = "generic"


class ModalityKind(str, Enum):
    METRICS = "metrics"
    TRACES = "traces"
    LOGS = "logs"
    TOPOLOGY = "topology"


class RemediationActionType(str, Enum):
    SCALE_SERVICE = "scale_service"
    RESTART_SERVICE = "restart_service"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    CLEAR_CONNECTION_POOL = "clear_connection_pool"
    NO_OP = "no_op"


class RemediationStatus(str, Enum):
    RECOMMENDED = "recommended"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    BLOCKED_BY_SAFETY = "blocked_by_safety"


class RecoveryStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"
    REGRESSION_DETECTED = "regression_detected"
    NOT_EVALUATED = "not_evaluated"


class PostInterventionCausalVerdict(str, Enum):
    INTERVENTION_SUPPORTS_HYPOTHESIS = "INTERVENTION_SUPPORTS_HYPOTHESIS"
    INTERVENTION_CONTRADICTS_HYPOTHESIS = "INTERVENTION_CONTRADICTS_HYPOTHESIS"
    INTERVENTION_INCONCLUSIVE = "INTERVENTION_INCONCLUSIVE"
    NOT_EVALUATED = "NOT_EVALUATED"


# ---------------------------------------------------------------------------
# Telemetry and Topology Models
# ---------------------------------------------------------------------------

class ServiceNode(BaseModel):
    service_id: str
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    active_metrics_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TopologyEdge(BaseModel):
    caller: str
    callee: str
    call_count: int = 1
    error_count: int = 0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    elevation_score: float = 0.0


class ServiceTopology(BaseModel):
    nodes: List[ServiceNode] = Field(default_factory=list)
    edges: List[TopologyEdge] = Field(default_factory=list)


class MetricSeriesSummary(BaseModel):
    service: str
    metric_name: str
    baseline_mean: float = 0.0
    baseline_std: float = 1.0
    max_zscore: float = 0.0
    anomaly_detected: bool = False
    first_anomaly_ts: Optional[int] = None
    sample_timestamps: List[int] = Field(default_factory=list)
    sample_values: List[float] = Field(default_factory=list)


class LogEventSummary(BaseModel):
    timestamp: int
    service: str
    level: str = "INFO"
    template_id: Optional[str] = None
    template_text: str
    occurrence_count: int = 1
    is_novel: bool = False


# ---------------------------------------------------------------------------
# Anomaly & Detection Models
# ---------------------------------------------------------------------------

class ChangepointInfo(BaseModel):
    detected: bool = False
    onset_timestamp: Optional[int] = None
    detection_method: str = "bocpd"
    confidence: float = 1.0
    persistence_k: int = 3
    consensus_m: int = 2
    anomalous_entities: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RCA Candidate Ranking Models
# ---------------------------------------------------------------------------

class CandidateRootCause(BaseModel):
    service: str
    rank: int
    score: float
    s_comb_score: Optional[float] = None
    trace_elev_score: Optional[float] = None
    r_strength: Optional[float] = None
    r_early: Optional[float] = None
    r_coverage: Optional[float] = None
    r_prop: Optional[float] = None
    causal_residual_z: Optional[float] = None
    log_anomaly_score: Optional[float] = None
    evidence_summary: str = ""


class RankingResult(BaseModel):
    method: str = "Model B"
    candidates: List[CandidateRootCause] = Field(default_factory=list)
    score_margin: float = 0.0
    uncertainty_detected: bool = False
    disambiguator_invoked: bool = False
    selected_root_cause: str = ""


# ---------------------------------------------------------------------------
# RAG Evidence Models
# ---------------------------------------------------------------------------

class RunbookCitation(BaseModel):
    citation_id: str
    document_id: str
    title: str
    source: str
    section: str
    score: float
    snippet: str
    full_text: Optional[str] = None
    category: str = "operational_guide"
    prerequisites: List[str] = Field(default_factory=list)
    verification_steps: List[str] = Field(default_factory=list)
    rollback_instructions: List[str] = Field(default_factory=list)
    relevance_justification: Optional[str] = None


class RAGContext(BaseModel):
    query: str
    retrieval_method: str = "hybrid_rrf_bm25_bge"
    citations: List[RunbookCitation] = Field(default_factory=list)
    formatted_context_block: str = ""


# ---------------------------------------------------------------------------
# Agent Investigation Models
# ---------------------------------------------------------------------------

class InvestigationStep(BaseModel):
    turn: int
    action_type: str
    tool_name: Optional[str] = None
    target_service: Optional[str] = None
    cost: int = 1
    remaining_budget: int = 19
    observation: str = ""
    reasoning: str = ""


class CausalConsistencyAudit(BaseModel):
    temporal_score: float = 1.0
    topology_score: float = 1.0
    overall_score: float = 1.0
    passed: bool = True
    details: str = ""


class InvestigationFindings(BaseModel):
    status: str = "COMPLETED"
    diagnosed_service: str
    confidence: float
    budget_used: int
    total_budget: int = 20
    turns: int
    reasoning: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    causal_consistency: CausalConsistencyAudit = Field(default_factory=CausalConsistencyAudit)
    trajectory: List[InvestigationStep] = Field(default_factory=list)
    model_name: str = "unknown"
    is_mock: bool = True
    fallback_active: bool = False
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# Safety & Remediation Models
# ---------------------------------------------------------------------------

class SafetyGateCheck(BaseModel):
    supported_action: bool = True
    has_rollback: bool = True
    target_in_universe: bool = True
    target_matches_root_cause: bool = True
    target_shows_degradation: bool = True
    confidence_above_threshold: bool = True
    safety_authorized: bool = False
    rejection_reasons: List[str] = Field(default_factory=list)


class RemediationPlan(BaseModel):
    action: RemediationActionType
    target_service: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    rollback_action: str
    safety_check: SafetyGateCheck
    status: RemediationStatus = RemediationStatus.RECOMMENDED
    justification: str = ""


class RecoveryVerificationResult(BaseModel):
    root_cause_symptoms_resolved: bool = False
    downstream_symptoms_resolved: bool = False
    new_regressions_detected: bool = False
    recovery_status: RecoveryStatus = RecoveryStatus.NOT_EVALUATED
    causal_verdict: PostInterventionCausalVerdict = PostInterventionCausalVerdict.NOT_EVALUATED
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Complete Canonical Incident Model
# ---------------------------------------------------------------------------

class CanonicalIncident(BaseModel):
    incident_id: str
    timestamp: int
    system: str = "online-boutique"
    description: str = ""
    is_demo_fixture: bool = False

    # Telemetry modalities (all optional to prevent crashing on missing data)
    observed_services: List[str] = Field(default_factory=list)
    topology: Optional[ServiceTopology] = None
    metric_summaries: List[MetricSeriesSummary] = Field(default_factory=list)
    recent_logs: List[LogEventSummary] = Field(default_factory=list)

    # Detection & RCA
    changepoint: ChangepointInfo = Field(default_factory=ChangepointInfo)
    ranking: Optional[RankingResult] = None

    # RAG Operational Context
    rag_context: Optional[RAGContext] = None

    # Investigation & Governance
    investigation: Optional[InvestigationFindings] = None
    remediation_plan: Optional[RemediationPlan] = None
    recovery_verification: Optional[RecoveryVerificationResult] = None

    # Metadata & Provenance
    ground_truth_root_cause: Optional[str] = None
    ground_truth_fault_type: Optional[str] = None
    execution_time_seconds: float = 0.0

    @field_validator("incident_id")
    @classmethod
    def validate_incident_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("incident_id cannot be empty")
        return v.strip()
