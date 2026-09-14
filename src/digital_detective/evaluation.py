"""Evaluation methodology for metric anomaly detection on telemetry cases."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from .aggregation import CaseAlarmSequence, aggregate_metric_anomalies
from .anomaly import MetricAnomalyResult
from .telemetry import TelemetryCase


OUTCOME_CLEAN_DETECTION = "Clean_Detection"
OUTCOME_NOISY_DETECTION = "Noisy_Detection"
OUTCOME_EARLY_ALARM_ONLY = "Early_Alarm_Only"
OUTCOME_COMPLETE_MISS = "Complete_Miss"

VALID_OUTCOMES = (
    OUTCOME_CLEAN_DETECTION,
    OUTCOME_NOISY_DETECTION,
    OUTCOME_EARLY_ALARM_ONLY,
    OUTCOME_COMPLETE_MISS,
)


@dataclass(frozen=True)
class CaseEvaluationResult:
    """Exact per-case evaluation result for one telemetry case."""

    case_id: str
    suite: str | None
    system: str | None
    fault_type: str | None
    inject_time: int
    t_eval_end: int
    outcome: str
    first_pre_alarm_ts: int | None
    t_detect: int | None
    detection_latency_sec: int | None
    pre_injection_far: float
    n_eval_pre_steps: int
    n_missing_pre_steps: int
    post_injection_density: float
    top_anomalous_metrics: tuple[str, ...]


@dataclass(frozen=True)
class AggregateEvaluationResult:
    """Exact aggregate evaluation quantities across multiple evaluated cases."""

    total_cases: int
    overall_detection_rate: float
    clean_detection_rate: float
    early_alarm_rate: float
    missed_detection_rate: float
    median_latency_sec: float | None
    p90_latency_sec: float | None
    mean_latency_sec: float | None
    mean_pre_injection_far: float
    by_suite: Mapping[str, AggregateEvaluationResult]
    by_system: Mapping[str, AggregateEvaluationResult]
    by_fault: Mapping[str, AggregateEvaluationResult]


def evaluate_anomaly_detection_case(
    case: TelemetryCase,
    anomaly_result: MetricAnomalyResult,
    case_alarms: CaseAlarmSequence | None = None,
) -> CaseEvaluationResult:
    """Evaluate metric anomaly detection outputs for a single TelemetryCase.

    Uses inject_time strictly as ground truth to evaluate detection fidelity,
    latency, and false alarms without lookahead bias.
    """
    if case.metadata.case_id != anomaly_result.case_id:
        raise ValueError(
            f"Case ID mismatch: TelemetryCase has {case.metadata.case_id!r}, "
            f"MetricAnomalyResult has {anomaly_result.case_id!r}"
        )

    if "inject_time" not in case.ground_truth.values:
        raise ValueError(f"TelemetryCase {case.metadata.case_id!r} has no inject_time in ground_truth")

    try:
        inject_time = int(case.ground_truth.values["inject_time"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Malformed inject_time in case {case.metadata.case_id!r}") from error

    timestamps = anomaly_result.timestamps
    if not timestamps:
        raise ValueError(f"MetricAnomalyResult for case {anomaly_result.case_id!r} has empty timestamps")

    metric_names = anomaly_result.metric_names
    if not metric_names:
        raise ValueError(f"MetricAnomalyResult for case {anomaly_result.case_id!r} has no metric columns")

    if case_alarms is None:
        case_alarms = aggregate_metric_anomalies(anomaly_result)
    else:
        if case_alarms.case_id != case.metadata.case_id:
            raise ValueError(
                f"Case ID mismatch: TelemetryCase has {case.metadata.case_id!r}, "
                f"CaseAlarmSequence has {case_alarms.case_id!r}"
            )
        if len(case_alarms.alarms) != len(timestamps):
            raise ValueError(
                f"Length mismatch: anomaly_result has {len(timestamps)} timestamps, "
                f"CaseAlarmSequence has {len(case_alarms.alarms)} alarms"
            )

    alarms = case_alarms.alarms

    # Terminal observation timestamp of the case telemetry is t_eval_end
    t_eval_end = int(timestamps[-1])

    # Determine warmup cutoff: first observation timestamp where at least one metric is evaluated
    # or leaves "warmup" state
    num_steps = len(timestamps)

    first_non_warmup_idx = None
    for idx in range(num_steps):
        if any(
            anomaly_result.evaluation_statuses[m][idx] != "warmup"
            for m in metric_names
        ):
            first_non_warmup_idx = idx
            break

    if first_non_warmup_idx is None:
        raise ValueError(f"All observations in case {case.metadata.case_id!r} are in warmup state")

    t_warmup = int(timestamps[first_non_warmup_idx])
    if inject_time <= t_warmup:
        raise ValueError(
            f"Case {case.metadata.case_id!r} has invalid_warmup_window: "
            f"inject_time ({inject_time}) <= warmup end ({t_warmup})"
        )

    # 1. Evaluate Pre-Injection Window [t_warmup, inject_time)
    evaluated_pre_steps = 0
    anomalous_pre_steps = 0
    missing_pre_steps = 0
    first_pre_alarm_ts: int | None = None

    for idx in range(first_non_warmup_idx, num_steps):
        ts = int(timestamps[idx])
        if ts >= inject_time:
            break

        # Check if at least one metric was validly evaluated at this step
        step_evaluated = any(
            anomaly_result.evaluation_statuses[m][idx] in ("normal", "anomaly")
            for m in metric_names
        )
        step_has_anomaly = alarms[idx]

        if step_evaluated:
            evaluated_pre_steps += 1
            if step_has_anomaly:
                anomalous_pre_steps += 1
                if first_pre_alarm_ts is None:
                    first_pre_alarm_ts = ts
        else:
            missing_pre_steps += 1

    pre_injection_far = (
        (anomalous_pre_steps / evaluated_pre_steps)
        if evaluated_pre_steps > 0
        else 0.0
    )

    # 2. Evaluate Post-Injection Window [inject_time, t_eval_end]
    evaluated_post_steps = 0
    anomalous_post_steps = 0
    t_detect: int | None = None

    for idx in range(num_steps):
        ts = int(timestamps[idx])
        if ts < inject_time:
            continue
        if ts > t_eval_end:
            break

        step_evaluated = any(
            anomaly_result.evaluation_statuses[m][idx] in ("normal", "anomaly")
            for m in metric_names
        )
        step_has_anomaly = alarms[idx]

        if step_evaluated:
            evaluated_post_steps += 1
            if step_has_anomaly:
                anomalous_post_steps += 1
                if t_detect is None:
                    t_detect = ts

    post_injection_density = (
        (anomalous_post_steps / evaluated_post_steps)
        if evaluated_post_steps > 0
        else 0.0
    )

    # 3. Detection Latency
    # Latency is strictly defined ONLY when detection occurs at or after inject_time
    if t_detect is not None:
        detection_latency_sec = t_detect - inject_time
    else:
        detection_latency_sec = None

    # 4. Mutually Exclusive Case Outcomes
    if t_detect is not None:
        if first_pre_alarm_ts is None:
            outcome = OUTCOME_CLEAN_DETECTION
        else:
            outcome = OUTCOME_NOISY_DETECTION
    else:
        if first_pre_alarm_ts is not None:
            outcome = OUTCOME_EARLY_ALARM_ONLY
        else:
            outcome = OUTCOME_COMPLETE_MISS

    # 5. Top 5 Anomalous Metrics
    metric_metrics = []
    for m in metric_names:
        summary_m = anomaly_result.summary.get("metrics", {}).get(m, {})
        peak = summary_m.get("peak_score", 0.0)
        cnt = summary_m.get("anomaly_count", 0)
        metric_metrics.append((m, peak, cnt))

    metric_metrics.sort(key=lambda item: (-item[1], -item[2], item[0]))
    top_metrics = tuple(m for m, peak, cnt in metric_metrics[:5] if peak > 0.0 or cnt > 0)

    fault_type = case.ground_truth.values.get("fault") or case.ground_truth.values.get("fault_type")

    return CaseEvaluationResult(
        case_id=case.metadata.case_id,
        suite=case.metadata.suite,
        system=case.metadata.system,
        fault_type=fault_type,
        inject_time=inject_time,
        t_eval_end=t_eval_end,
        outcome=outcome,
        first_pre_alarm_ts=first_pre_alarm_ts,
        t_detect=t_detect,
        detection_latency_sec=detection_latency_sec,
        pre_injection_far=pre_injection_far,
        n_eval_pre_steps=evaluated_pre_steps,
        n_missing_pre_steps=missing_pre_steps,
        post_injection_density=post_injection_density,
        top_anomalous_metrics=top_metrics,
    )


def evaluate_anomaly_detection_cases(
    case_results: Sequence[CaseEvaluationResult],
    *,
    compute_breakdowns: bool = True,
) -> AggregateEvaluationResult:
    """Compute aggregate benchmark metrics across multiple CaseEvaluationResults."""
    total_cases = len(case_results)
    if total_cases == 0:
        return AggregateEvaluationResult(
            total_cases=0,
            overall_detection_rate=0.0,
            clean_detection_rate=0.0,
            early_alarm_rate=0.0,
            missed_detection_rate=0.0,
            median_latency_sec=None,
            p90_latency_sec=None,
            mean_latency_sec=None,
            mean_pre_injection_far=0.0,
            by_suite={},
            by_system={},
            by_fault={},
        )

    clean_count = sum(1 for c in case_results if c.outcome == OUTCOME_CLEAN_DETECTION)
    noisy_count = sum(1 for c in case_results if c.outcome == OUTCOME_NOISY_DETECTION)
    early_only_count = sum(1 for c in case_results if c.outcome == OUTCOME_EARLY_ALARM_ONLY)
    miss_count = sum(1 for c in case_results if c.outcome == OUTCOME_COMPLETE_MISS)

    overall_detection_rate = (clean_count + noisy_count) / total_cases
    clean_detection_rate = clean_count / total_cases
    early_alarm_rate = (noisy_count + early_only_count) / total_cases
    missed_detection_rate = miss_count / total_cases

    mean_pre_injection_far = sum(c.pre_injection_far for c in case_results) / total_cases

    # Latency statistics over detected cases (detection_latency_sec is not None)
    detected_latencies = [
        c.detection_latency_sec
        for c in case_results
        if c.detection_latency_sec is not None
    ]

    if detected_latencies:
        mean_latency_sec = float(statistics.fmean(detected_latencies))
        median_latency_sec = float(statistics.median(detected_latencies))
        p90_latency_sec = float(_percentile(detected_latencies, 0.90))
    else:
        mean_latency_sec = None
        median_latency_sec = None
        p90_latency_sec = None

    by_suite: dict[str, AggregateEvaluationResult] = {}
    by_system: dict[str, AggregateEvaluationResult] = {}
    by_fault: dict[str, AggregateEvaluationResult] = {}

    if compute_breakdowns:
        # Group by suite
        suites = set(c.suite or "unknown" for c in case_results)
        for s in sorted(suites):
            sub = [c for c in case_results if (c.suite or "unknown") == s]
            by_suite[s] = evaluate_anomaly_detection_cases(sub, compute_breakdowns=False)

        # Group by system
        systems = set(c.system or "unknown" for c in case_results)
        for sys_name in sorted(systems):
            sub = [c for c in case_results if (c.system or "unknown") == sys_name]
            by_system[sys_name] = evaluate_anomaly_detection_cases(sub, compute_breakdowns=False)

        # Group by fault
        faults = set(c.fault_type or "unknown" for c in case_results)
        for f in sorted(faults):
            sub = [c for c in case_results if (c.fault_type or "unknown") == f]
            by_fault[f] = evaluate_anomaly_detection_cases(sub, compute_breakdowns=False)

    return AggregateEvaluationResult(
        total_cases=total_cases,
        overall_detection_rate=overall_detection_rate,
        clean_detection_rate=clean_detection_rate,
        early_alarm_rate=early_alarm_rate,
        missed_detection_rate=missed_detection_rate,
        median_latency_sec=median_latency_sec,
        p90_latency_sec=p90_latency_sec,
        mean_latency_sec=mean_latency_sec,
        mean_pre_injection_far=mean_pre_injection_far,
        by_suite=by_suite,
        by_system=by_system,
        by_fault=by_fault,
    )


def _percentile(values: Sequence[float | int], p: float) -> float:
    """Calculate percentile using linear interpolation matching numpy.percentile."""
    if not values:
        return 0.0
    sorted_vals = sorted(float(x) for x in values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1
