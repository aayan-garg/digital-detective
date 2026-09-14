"""Digital Detective package."""

from .aggregation import (
    AggregationConfig,
    CaseAlarmSequence,
    STANDARD_CONFIGURATIONS,
    aggregate_metric_anomalies,
)
from .anomaly import MetricAnomalyResult, detect_metric_anomalies
from .evaluation import (
    OUTCOME_CLEAN_DETECTION,
    OUTCOME_COMPLETE_MISS,
    OUTCOME_EARLY_ALARM_ONLY,
    OUTCOME_NOISY_DETECTION,
    VALID_OUTCOMES,
    AggregateEvaluationResult,
    CaseEvaluationResult,
    evaluate_anomaly_detection_case,
    evaluate_anomaly_detection_cases,
)
from .telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from .topology import (
    Dependency,
    EntityAnomalyEvidence,
    EntityGraph,
    EntityNode,
    MetricIdentifier,
    TraceDependencyObservation,
    aggregate_entity_anomaly_evidence,
    build_entity_graph,
    extract_trace_dependencies,
    parse_rcaeval_metric_identifier,
)

__all__ = [
    "AggregateEvaluationResult",
    "AggregationConfig",
    "CaseAlarmSequence",
    "CaseEvaluationResult",
    "CaseMetadata",
    "Dependency",
    "EntityAnomalyEvidence",
    "EntityGraph",
    "EntityNode",
    "GroundTruth",
    "MetricAnomalyResult",
    "MetricIdentifier",
    "ModalityProvenance",
    "OUTCOME_CLEAN_DETECTION",
    "OUTCOME_COMPLETE_MISS",
    "OUTCOME_EARLY_ALARM_ONLY",
    "OUTCOME_NOISY_DETECTION",
    "STANDARD_CONFIGURATIONS",
    "TelemetryCase",
    "TelemetryModality",
    "TraceDependencyObservation",
    "VALID_OUTCOMES",
    "aggregate_entity_anomaly_evidence",
    "aggregate_metric_anomalies",
    "build_entity_graph",
    "detect_metric_anomalies",
    "evaluate_anomaly_detection_case",
    "evaluate_anomaly_detection_cases",
    "extract_trace_dependencies",
    "parse_rcaeval_metric_identifier",
]
