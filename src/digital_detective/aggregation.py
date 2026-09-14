"""Metric anomaly aggregation strategies for case-level alarm generation.

Operates purely downstream on immutable MetricAnomalyResult evidence.
Causality is strictly preserved: for any observation at index t, persistence
and consensus depend solely on observations at or before index t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .anomaly import MetricAnomalyResult


@dataclass(frozen=True)
class AggregationConfig:
    """Configuration specifying persistence and metric-count consensus parameters."""

    persistence: int = 1
    consensus: int = 1
    name: str | None = None


@dataclass(frozen=True)
class CaseAlarmSequence:
    """Immutable sequence of aggregated case-level alarms."""

    case_id: str
    timestamps: tuple[Any, ...]
    alarms: tuple[bool, ...]
    active_metric_counts: tuple[int, ...]
    persistence: int
    consensus: int


STANDARD_CONFIGURATIONS: Mapping[str, AggregationConfig] = {
    "C0_Baseline": AggregationConfig(persistence=1, consensus=1, name="C0_Baseline"),
    "C1_Persist_3": AggregationConfig(persistence=3, consensus=1, name="C1_Persist_3"),
    "C2_Persist_5": AggregationConfig(persistence=5, consensus=1, name="C2_Persist_5"),
    "C3_Consensus_2": AggregationConfig(persistence=1, consensus=2, name="C3_Consensus_2"),
    "C4_Consensus_3": AggregationConfig(persistence=1, consensus=3, name="C4_Consensus_3"),
    "C5_Persist_3_Consensus_2": AggregationConfig(
        persistence=3, consensus=2, name="C5_Persist_3_Consensus_2"
    ),
}


def aggregate_metric_anomalies(
    anomaly_result: MetricAnomalyResult,
    config: AggregationConfig | None = None,
) -> CaseAlarmSequence:
    """Aggregate metric-level anomaly states into a case-level alarm sequence.

    Parameters
    ----------
    anomaly_result:
        Immutable MetricAnomalyResult containing per-metric evaluation statuses.
    config:
        AggregationConfig specifying persistence and consensus parameters.
        Defaults to persistence=1, consensus=1.
    """
    if config is None:
        config = AggregationConfig()

    eff_persistence = config.persistence
    eff_consensus = config.consensus

    if eff_persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {eff_persistence}")
    if eff_consensus < 1:
        raise ValueError(f"consensus must be >= 1, got {eff_consensus}")

    timestamps = anomaly_result.timestamps
    if not timestamps:
        raise ValueError(f"MetricAnomalyResult for case {anomaly_result.case_id!r} has empty timestamps")

    metric_names = anomaly_result.metric_names
    if not metric_names:
        raise ValueError(f"MetricAnomalyResult for case {anomaly_result.case_id!r} has no metric columns")

    num_steps = len(timestamps)
    active_counts = [0] * num_steps

    # For each metric, track causal consecutive anomaly streak
    for m in metric_names:
        statuses = anomaly_result.evaluation_statuses[m]
        streak = 0
        for idx in range(num_steps):
            if statuses[idx] == "anomaly":
                streak += 1
            else:
                # Normal, missing_observation, insufficient_history, or warmup resets streak
                streak = 0

            if streak >= eff_persistence:
                active_counts[idx] += 1

    case_alarms = tuple(active_counts[idx] >= eff_consensus for idx in range(num_steps))

    return CaseAlarmSequence(
        case_id=anomaly_result.case_id,
        timestamps=timestamps,
        alarms=case_alarms,
        active_metric_counts=tuple(active_counts),
        persistence=eff_persistence,
        consensus=eff_consensus,
    )
