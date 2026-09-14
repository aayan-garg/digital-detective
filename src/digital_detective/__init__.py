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

__all__ = [
    "AggregateEvaluationResult",
    "AggregationConfig",
    "CaseAlarmSequence",
    "CaseEvaluationResult",
    "CaseMetadata",
    "GroundTruth",
    "MetricAnomalyResult",
    "ModalityProvenance",
    "OUTCOME_CLEAN_DETECTION",
    "OUTCOME_COMPLETE_MISS",
    "OUTCOME_EARLY_ALARM_ONLY",
    "OUTCOME_NOISY_DETECTION",
    "STANDARD_CONFIGURATIONS",
    "TelemetryCase",
    "TelemetryModality",
    "VALID_OUTCOMES",
    "aggregate_metric_anomalies",
    "detect_metric_anomalies",
    "evaluate_anomaly_detection_case",
    "evaluate_anomaly_detection_cases",
]
