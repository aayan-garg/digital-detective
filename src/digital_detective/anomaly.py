"""Causal rolling window z-score baseline for metric anomaly detection.

Causality is strictly defined over chronological observation order:
For any observation at index k in the chronologically ordered sequence, running
statistics are computed exclusively from observations in the preceding window
[max(0, k - window_size), k - 1]. No observation at or after index k is ever
included in historical parameter estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import pyarrow as pa

from .telemetry import TelemetryCase


@dataclass(frozen=True)
class MetricAnomalyResult:
    """Immutable result of metric anomaly detection for one case."""

    case_id: str
    timestamps: tuple[Any, ...]
    metric_names: tuple[str, ...]
    anomaly_scores: Mapping[str, tuple[float | None, ...]]
    signed_scores: Mapping[str, tuple[float | None, ...]]
    anomalies: Mapping[str, tuple[bool, ...]]
    evaluation_statuses: Mapping[str, tuple[str, ...]]
    valid_history_counts: Mapping[str, tuple[int, ...]]
    summary: Mapping[str, Any]


def detect_metric_anomalies(
    case: TelemetryCase,
    *,
    window_size: int = 60,
    threshold: float = 3.0,
    min_warmup: int | None = None,
    min_valid_history: int | None = None,
    epsilon: float = 1e-6,
) -> MetricAnomalyResult:
    """Detect anomalies across metric time-series using causal rolling z-scores.

    Operates strictly over the preceding ``window_size`` discrete observations
    in chronological order without resampling or interpolation.

    Parameters
    ----------
    case:
        The TelemetryCase containing metrics modality data.
    window_size:
        The number of preceding discrete observations forming the historical window.
    threshold:
        Standard deviation threshold factor for flagging anomalies (|z| >= threshold).
    min_warmup:
        Minimum number of initial observations before evaluation starts. Defaults to window_size.
    min_valid_history:
        Minimum number of valid (non-null, non-NaN) observations required within the
        preceding window_size rows. If fewer valid samples exist, the observation is
        marked as 'insufficient_history' rather than evaluated on an arbitrarily small sample.
        Defaults to max(2, effective_warmup // 2).
    epsilon:
        Numerical stability constant to prevent division by zero in floating-point operations.
    """
    if window_size < 2:
        raise ValueError(f"window_size must be at least 2, got {window_size}")
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    effective_warmup = window_size if min_warmup is None else min_warmup
    if effective_warmup < 2:
        raise ValueError(f"min_warmup must be at least 2, got {effective_warmup}")

    effective_min_valid = (
        max(2, effective_warmup // 2)
        if min_valid_history is None
        else min_valid_history
    )
    if effective_min_valid < 2:
        raise ValueError(f"min_valid_history must be at least 2, got {effective_min_valid}")
    if effective_min_valid > window_size:
        raise ValueError(
            f"min_valid_history ({effective_min_valid}) cannot exceed window_size ({window_size})"
        )

    if case.metrics is None:
        raise ValueError(f"TelemetryCase {case.metadata.case_id!r} has no metrics modality")

    timestamp_field = case.metrics.provenance.timestamp_field
    if not timestamp_field:
        raise ValueError(f"Metrics modality provenance for case {case.metadata.case_id!r} has no timestamp_field")

    timestamps, metric_series = _extract_series(case.metrics.raw_data, timestamp_field)

    num_observations = len(timestamps)
    if num_observations < effective_warmup:
        raise ValueError(
            f"Insufficient observations ({num_observations}) for minimum warmup history ({effective_warmup})"
        )

    # Sort chronologically if timestamps are not monotonically non-decreasing.
    # Python's sorted is a stable sort (Timsort), guaranteeing deterministic
    # tie-breaking for identical timestamps.
    is_sorted = all(timestamps[i] <= timestamps[i + 1] for i in range(num_observations - 1))
    if not is_sorted:
        order = sorted(range(num_observations), key=lambda idx: timestamps[idx])
        timestamps = tuple(timestamps[idx] for idx in order)
        metric_series = {
            col: tuple(vals[idx] for idx in order)
            for col, vals in metric_series.items()
        }

    metric_names = tuple(metric_series.keys())
    anomaly_scores: dict[str, tuple[float | None, ...]] = {}
    signed_scores: dict[str, tuple[float | None, ...]] = {}
    anomalies: dict[str, tuple[bool, ...]] = {}
    evaluation_statuses: dict[str, tuple[str, ...]] = {}
    valid_history_counts: dict[str, tuple[int, ...]] = {}
    metric_summaries: dict[str, dict[str, Any]] = {}

    for metric_name in metric_names:
        s_scores, a_scores, flags, statuses, v_counts = _score_metric_series(
            metric_series[metric_name],
            window_size=window_size,
            min_warmup=effective_warmup,
            min_valid_history=effective_min_valid,
            threshold=threshold,
            epsilon=epsilon,
        )
        signed_scores[metric_name] = s_scores
        anomaly_scores[metric_name] = a_scores
        anomalies[metric_name] = flags
        evaluation_statuses[metric_name] = statuses
        valid_history_counts[metric_name] = v_counts

        anom_count = sum(1 for flag in flags if flag)
        first_idx = next((i for i, flag in enumerate(flags) if flag), None)
        first_ts = timestamps[first_idx] if first_idx is not None else None
        valid_eval_scores = [s for s in a_scores if s is not None]
        peak_score = max(valid_eval_scores) if valid_eval_scores else 0.0

        warmup_count = sum(1 for st in statuses if st == "warmup")
        missing_count = sum(1 for st in statuses if st == "missing_observation")
        insufficient_hist_count = sum(1 for st in statuses if st == "insufficient_history")
        evaluated_count = sum(1 for st in statuses if st in ("normal", "anomaly"))

        metric_summaries[metric_name] = {
            "peak_score": peak_score,
            "anomaly_count": anom_count,
            "first_anomaly_timestamp": first_ts,
            "evaluated_count": evaluated_count,
            "warmup_count": warmup_count,
            "missing_count": missing_count,
            "insufficient_history_count": insufficient_hist_count,
            "min_valid_history": effective_min_valid,
        }

    # Case-level summary statistics
    detected_metrics = tuple(m for m in metric_names if metric_summaries[m]["anomaly_count"] > 0)
    first_timestamps = [
        metric_summaries[m]["first_anomaly_timestamp"]
        for m in detected_metrics
        if metric_summaries[m]["first_anomaly_timestamp"] is not None
    ]
    earliest_detection_ts = min(first_timestamps) if first_timestamps else None

    observation_has_anomaly = [
        any(anomalies[m][t] for m in metric_names)
        for t in range(num_observations)
    ]
    total_anomalous_observations = sum(1 for has_anom in observation_has_anomaly if has_anom)
    total_anomalies = sum(metric_summaries[m]["anomaly_count"] for m in metric_names)

    summary: dict[str, Any] = {
        "metrics": metric_summaries,
        "detected_metrics": detected_metrics,
        "first_anomaly_timestamp": earliest_detection_ts,
        "total_anomalies": total_anomalies,
        "total_anomalous_observations": total_anomalous_observations,
    }

    return MetricAnomalyResult(
        case_id=case.metadata.case_id,
        timestamps=timestamps,
        metric_names=metric_names,
        anomaly_scores=anomaly_scores,
        signed_scores=signed_scores,
        anomalies=anomalies,
        evaluation_statuses=evaluation_statuses,
        valid_history_counts=valid_history_counts,
        summary=summary,
    )


def _score_metric_series(
    values: Sequence[float | None],
    *,
    window_size: int,
    min_warmup: int,
    min_valid_history: int,
    threshold: float,
    epsilon: float,
) -> tuple[
    tuple[float | None, ...],
    tuple[float | None, ...],
    tuple[bool, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    num_points = len(values)
    signed_scores: list[float | None] = [None] * num_points
    anomaly_scores: list[float | None] = [None] * num_points
    anomalies: list[bool] = [False] * num_points
    statuses: list[str] = ["warmup"] * num_points
    valid_counts: list[int] = [0] * num_points

    # Warmup phase: calculate preceding valid counts for transparency
    for i in range(min_warmup):
        win_start = max(0, i - window_size)
        valid_counts[i] = sum(
            1 for v in values[win_start:i]
            if v is not None and not math.isnan(v)
        )
        statuses[i] = "warmup"

    # Evaluation phase: strictly causal over preceding window [i - window_size, i - 1]
    for i in range(min_warmup, num_points):
        win_start = max(0, i - window_size)
        window_slice = values[win_start:i]
        history = [
            v for v in window_slice
            if v is not None and not math.isnan(v)
        ]
        valid_count = len(history)
        valid_counts[i] = valid_count

        cur_val = values[i]

        # 1. Invalid or missing current observation
        if cur_val is None or math.isnan(cur_val):
            statuses[i] = "missing_observation"
            continue

        # 2. Insufficient valid history within the preceding window_size rows
        if valid_count < min_valid_history:
            statuses[i] = "insufficient_history"
            continue

        mean = sum(history) / valid_count
        variance = sum((v - mean) ** 2 for v in history) / valid_count
        std = math.sqrt(variance)

        # 3. Explicit zero-variance semantics
        if math.isclose(std, 0.0, abs_tol=1e-12):
            if math.isclose(cur_val, mean, rel_tol=1e-9, abs_tol=1e-12):
                # Equal to constant history -> evaluated normal
                signed_scores[i] = 0.0
                anomaly_scores[i] = 0.0
                anomalies[i] = False
                statuses[i] = "normal"
            else:
                # Different from constant history -> broken invariant, evaluated anomaly
                deviation = cur_val - mean
                signed_z = deviation / epsilon if epsilon > 0 else (float("inf") if deviation > 0 else float("-inf"))
                signed_scores[i] = signed_z
                anomaly_scores[i] = abs(signed_z)
                anomalies[i] = True
                statuses[i] = "anomaly"
        else:
            # Standard z-score evaluation
            deviation = cur_val - mean
            signed_z = deviation / (std + epsilon)
            score = abs(signed_z)
            signed_scores[i] = signed_z
            anomaly_scores[i] = score
            is_anom = score >= threshold
            anomalies[i] = is_anom
            statuses[i] = "anomaly" if is_anom else "normal"

    return (
        tuple(signed_scores),
        tuple(anomaly_scores),
        tuple(anomalies),
        tuple(statuses),
        tuple(valid_counts),
    )


def _extract_series(
    raw_data: Any,
    timestamp_field: str,
) -> tuple[tuple[Any, ...], dict[str, tuple[float | None, ...]]]:
    # 1. PyArrow Table
    if isinstance(raw_data, pa.Table) or hasattr(raw_data, "column_names"):
        column_names = list(raw_data.column_names)
        if timestamp_field not in column_names:
            raise ValueError(f"Timestamp field {timestamp_field!r} not found in raw data columns")

        timestamps = tuple(raw_data[timestamp_field].to_pylist())
        metrics_data: dict[str, tuple[float | None, ...]] = {}

        for col in column_names:
            if col == timestamp_field:
                continue
            chunked = raw_data[col]
            # Ignore non-numeric columns
            if not (pa.types.is_floating(chunked.type) or pa.types.is_integer(chunked.type)):
                continue
            metrics_data[col] = tuple(
                float(x) if x is not None else None
                for x in chunked.to_pylist()
            )

        return timestamps, metrics_data

    # 2. Sequence of row mappings (list of dicts)
    if isinstance(raw_data, Sequence) and len(raw_data) > 0 and isinstance(raw_data[0], Mapping):
        if timestamp_field not in raw_data[0]:
            raise ValueError(f"Timestamp field {timestamp_field!r} not found in raw data row")

        timestamps = tuple(row[timestamp_field] for row in raw_data)
        metrics_data = {}
        for col in raw_data[0].keys():
            if col == timestamp_field:
                continue
            parsed_col = []
            is_numeric = True
            for row in raw_data:
                val = row.get(col)
                if val is None:
                    parsed_col.append(None)
                elif isinstance(val, (int, float)):
                    parsed_col.append(float(val))
                else:
                    try:
                        parsed_col.append(float(val))
                    except (ValueError, TypeError):
                        is_numeric = False
                        break
            if is_numeric:
                metrics_data[col] = tuple(parsed_col)

        return timestamps, metrics_data

    # 3. Column mapping (dict of lists)
    if isinstance(raw_data, Mapping):
        if timestamp_field not in raw_data:
            raise ValueError(f"Timestamp field {timestamp_field!r} not found in raw data mapping")

        timestamps = tuple(raw_data[timestamp_field])
        metrics_data = {}
        for col, values in raw_data.items():
            if col == timestamp_field:
                continue
            parsed_col = []
            is_numeric = True
            for val in values:
                if val is None:
                    parsed_col.append(None)
                elif isinstance(val, (int, float)):
                    parsed_col.append(float(val))
                else:
                    try:
                        parsed_col.append(float(val))
                    except (ValueError, TypeError):
                        is_numeric = False
                        break
            if is_numeric:
                metrics_data[col] = tuple(parsed_col)

        return timestamps, metrics_data

    raise ValueError(f"Unsupported metrics raw_data type: {type(raw_data).__name__}")
