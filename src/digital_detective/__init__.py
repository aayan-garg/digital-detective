"""Digital Detective package."""

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

__all__ = [
    "AggregateEvaluationResult",
    "CaseEvaluationResult",
    "CaseMetadata",
    "GroundTruth",
    "MetricAnomalyResult",
    "ModalityProvenance",
    "OUTCOME_CLEAN_DETECTION",
    "OUTCOME_COMPLETE_MISS",
    "OUTCOME_EARLY_ALARM_ONLY",
    "OUTCOME_NOISY_DETECTION",
    "TelemetryCase",
    "TelemetryModality",
    "VALID_OUTCOMES",
    "detect_metric_anomalies",
    "evaluate_anomaly_detection_case",
    "evaluate_anomaly_detection_cases",
]
