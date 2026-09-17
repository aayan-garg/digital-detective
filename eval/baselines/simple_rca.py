"""SimpleRCA baseline implementation.

Textual reproduction of SimpleRCA from Fang et al., arXiv:2510.04711 §3.1.2;
no official implementation located.

Specification & Reproducibility Decisions:
------------------------------------------
1. Metrics Anomaly Detection:
   - Evaluated per candidate service using its associated metric series (prefix match).
   - Historical normal baseline: observation timestamps t < onset_ts.
   - Robust 3-sigma threshold: threshold = mean + 3 * max(std, 1e-6).
   - Anomaly condition: any observation in [onset_ts, end_ts] > threshold.
   - Metric alert count for service: number of anomalous metric columns for that service.

2. Traces Anomaly Detection:
   - Evaluated per candidate service appearing in trace spans.
   - Normal period: spans with (startTimeMillis / 1000.0) < onset_ts.
   - Incident period: spans with onset_ts <= (startTimeMillis / 1000.0) <= end_ts.
   - Baseline latency: 95th percentile (P95) duration of normal spans.
   - Incident latency: 95th percentile (P95) duration of incident spans.
   - Threshold condition: incident_P95 >= 3.0 * max(normal_P95, 1e-3) and incident_P95 > 0.
   - Trace alert count for service: 1 alert if threshold exceeded, 0 otherwise.

3. Logs Anomaly Detection:
   - Evaluated per candidate service appearing in log records.
   - Incident period: logs with onset_ts <= timestamp <= end_ts.
   - Keyword match: case-insensitive occurrence of 'error', 'fail', or 'exception'.
   - Log alert count for service: count of matching log records during incident window.

4. Combined Ranking:
   - Total alerts(s) = metric_alerts(s) + trace_alerts(s) + log_alerts(s).
   - Enforces closed candidate universe: candidate services with 0 alerts receive score 0.0.
   - Deterministic tie-breaking: sort descending by (-score, service_name).
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import pyarrow.compute as pc

from ..models import IncidentWindow, RankedEntity
from ..universe import normalize_service_name


def detect_metric_alerts(
    case: Any,
    window: IncidentWindow,
    candidate_universe: Sequence[str],
) -> dict[str, int]:
    """Detect metric threshold alerts using historical 3-sigma baseline.

    Parameters
    ----------
    case:
        TelemetryCase containing metrics series and timestamps.
    window:
        IncidentWindow defining onset_ts and end_ts.
    candidate_universe:
        Closed sequence of candidate services.

    Returns
    -------
    dict[str, int]
        Mapping from candidate service name to number of anomalous metric columns.
    """
    candidate_set = set(candidate_universe)
    alerts: dict[str, int] = {s: 0 for s in candidate_set}

    if not hasattr(case, "metrics") or not getattr(case.metrics, "series", None):
        return alerts

    timestamps = getattr(case.metrics, "timestamps", ())
    if not timestamps:
        return alerts

    if not window.has_detected_window or window.onset_ts is None or window.end_ts is None:
        return alerts

    onset_ts = window.onset_ts
    end_ts = window.end_ts

    # Pre-calculate normal and incident slice indices
    normal_indices = [i for i, t in enumerate(timestamps) if t < onset_ts]
    incident_indices = [i for i, t in enumerate(timestamps) if onset_ts <= t <= end_ts]

    if not normal_indices or not incident_indices:
        return alerts

    for col_name, series in case.metrics.series.items():
        if "_" not in col_name:
            continue
        prefix = col_name.split("_", 1)[0]
        service = normalize_service_name(prefix)
        if service not in candidate_set:
            continue

        normal_vals = [float(series[i]) for i in normal_indices if series[i] is not None]
        if not normal_vals:
            continue

        mean_val = sum(normal_vals) / len(normal_vals)
        variance = sum((x - mean_val) ** 2 for x in normal_vals) / len(normal_vals)
        std_val = math.sqrt(variance)

        threshold = mean_val + 3.0 * max(std_val, 1e-6)

        is_anomalous = False
        for idx in incident_indices:
            val = series[idx]
            if val is not None and float(val) > threshold:
                is_anomalous = True
                break

        if is_anomalous:
            alerts[service] += 1

    return alerts


def detect_trace_alerts(
    case: Any,
    window: IncidentWindow,
    candidate_universe: Sequence[str],
) -> dict[str, int]:
    """Detect trace latency alerts (P95 incident latency >= 3x normal baseline).

    Parameters
    ----------
    case:
        TelemetryCase containing traces.
    window:
        IncidentWindow defining onset_ts and end_ts.
    candidate_universe:
        Closed sequence of candidate services.

    Returns
    -------
    dict[str, int]
        Mapping from candidate service name to trace alert (1 or 0).
    """
    candidate_set = set(candidate_universe)
    alerts: dict[str, int] = {s: 0 for s in candidate_set}

    if not hasattr(case, "traces") or case.traces is None:
        return alerts

    table = getattr(case.traces, "raw_data", None) or getattr(case.traces, "table", None)
    if table is None or table.num_rows == 0:
        return alerts

    if not window.has_detected_window or window.onset_ts is None or window.end_ts is None:
        return alerts

    onset_ts = window.onset_ts
    end_ts = window.end_ts

    # Determine timestamp column and convert to seconds
    if "startTimeMillis" in table.column_names:
        sec_arr = pc.divide(table["startTimeMillis"], 1000.0)
    elif "startTime" in table.column_names:
        sec_arr = pc.divide(table["startTime"], 1_000_000.0)
    elif "time" in table.column_names:
        sec_arr = table["time"]
    else:
        return alerts

    duration_col = table["duration"] if "duration" in table.column_names else None
    service_col = table["serviceName"] if "serviceName" in table.column_names else None

    if duration_col is None or service_col is None:
        return alerts

    norm_mask = pc.less(sec_arr, onset_ts)
    inc_mask = pc.and_(pc.greater_equal(sec_arr, onset_ts), pc.less_equal(sec_arr, end_ts))

    # Evaluate per candidate service
    for service in candidate_set:
        # Check canonical service and alias matches
        srv_match = pc.equal(service_col, service)
        if service == "frontend":
            srv_match = pc.or_(srv_match, pc.equal(service_col, "frontendservice"))

        # Normal P95
        srv_norm_mask = pc.and_(srv_match, norm_mask)
        norm_durations = duration_col.filter(srv_norm_mask)
        if norm_durations.length() == 0:
            normal_p95 = 0.0
        else:
            normal_p95 = float(pc.quantile(norm_durations, q=0.95)[0].as_py() or 0.0)

        # Incident P95
        srv_inc_mask = pc.and_(srv_match, inc_mask)
        inc_durations = duration_col.filter(srv_inc_mask)
        if inc_durations.length() == 0:
            incident_p95 = 0.0
        else:
            incident_p95 = float(pc.quantile(inc_durations, q=0.95)[0].as_py() or 0.0)

        # Alert condition: >= 3x normal baseline threshold
        threshold = 3.0 * max(normal_p95, 1e-3)
        if incident_p95 >= threshold and incident_p95 > 0.0:
            alerts[service] = 1

    return alerts


def detect_log_alerts(
    case: Any,
    window: IncidentWindow,
    candidate_universe: Sequence[str],
) -> dict[str, int]:
    """Detect log error keyword alerts (ERROR / FAIL / EXCEPTION counts).

    Parameters
    ----------
    case:
        TelemetryCase containing logs.
    window:
        IncidentWindow defining onset_ts and end_ts.
    candidate_universe:
        Closed sequence of candidate services.

    Returns
    -------
    dict[str, int]
        Mapping from candidate service name to count of matching error log entries.
    """
    candidate_set = set(candidate_universe)
    alerts: dict[str, int] = {s: 0 for s in candidate_set}

    if not hasattr(case, "logs") or case.logs is None:
        return alerts

    table = getattr(case.logs, "raw_data", None) or getattr(case.logs, "table", None)
    if table is None or table.num_rows == 0:
        return alerts

    if not window.has_detected_window or window.onset_ts is None or window.end_ts is None:
        return alerts

    onset_ts = window.onset_ts
    end_ts = window.end_ts

    ts_col_name = "timestamp" if "timestamp" in table.column_names else "time"
    container_col_name = "container_name" if "container_name" in table.column_names else "serviceName"
    msg_col_name = "message" if "message" in table.column_names else "log"

    if (
        ts_col_name not in table.column_names
        or container_col_name not in table.column_names
        or msg_col_name not in table.column_names
    ):
        return alerts

    ts_arr = table[ts_col_name]
    inc_mask = pc.and_(pc.greater_equal(ts_arr, onset_ts), pc.less_equal(ts_arr, end_ts))
    incident_logs = table.filter(inc_mask)

    if incident_logs.num_rows == 0:
        return alerts

    msg_lower = pc.utf8_lower(incident_logs[msg_col_name])
    has_err = pc.or_(
        pc.match_substring(msg_lower, "error"),
        pc.or_(pc.match_substring(msg_lower, "fail"), pc.match_substring(msg_lower, "exception")),
    )
    err_logs = incident_logs.filter(has_err)

    if err_logs.num_rows == 0:
        return alerts

    counts_struct = pc.value_counts(err_logs[container_col_name]).to_pylist()
    for row in counts_struct:
        raw_name = str(row.get("values") or "")
        norm_name = normalize_service_name(raw_name)
        cnt = int(row.get("counts") or 0)
        if norm_name in candidate_set:
            alerts[norm_name] += cnt

    return alerts


def rank_with_simple_rca(
    case: Any,
    window: IncidentWindow,
    candidate_universe: Sequence[str],
    enabled_modalities: Sequence[str] = ("metrics", "traces", "logs"),
) -> tuple[RankedEntity, ...]:
    """Rank candidate services using SimpleRCA (Fang et al. arXiv:2510.04711 §3.1.2).

    Parameters
    ----------
    case:
        TelemetryCase containing telemetry modalities.
    window:
        IncidentWindow defining temporal boundaries.
    candidate_universe:
        Closed sequence of candidate services to be ranked.
    enabled_modalities:
        Sequence of modalities to include in alert summation ('metrics', 'traces', 'logs').

    Returns
    -------
    tuple[RankedEntity, ...]
        Deterministically ordered ranking of candidate services.
    """
    candidate_list = sorted(set(candidate_universe))
    total_alerts: dict[str, int] = {s: 0 for s in candidate_list}

    if "metrics" in enabled_modalities:
        m_alerts = detect_metric_alerts(case, window, candidate_list)
        for s, cnt in m_alerts.items():
            total_alerts[s] += cnt

    if "traces" in enabled_modalities:
        t_alerts = detect_trace_alerts(case, window, candidate_list)
        for s, cnt in t_alerts.items():
            total_alerts[s] += cnt

    if "logs" in enabled_modalities:
        l_alerts = detect_log_alerts(case, window, candidate_list)
        for s, cnt in l_alerts.items():
            total_alerts[s] += cnt

    # Deterministic tie-breaking: sort descending by score (-alerts), ascending by service_name
    sorted_services = sorted(candidate_list, key=lambda s: (-total_alerts[s], s))

    ranked: list[RankedEntity] = []
    for rank_idx, s in enumerate(sorted_services, start=1):
        score = float(total_alerts[s])
        ranked.append(RankedEntity(entity=s, score=score, rank=rank_idx))

    return tuple(ranked)
