"""Incident window resolution for Digital Detective evaluations.

Distinguishes:
1. Oracle injection-time window (ground-truth inject_time)
2. Estimated/detected window (Digital Detective first detected episode timestamp)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import IncidentWindow


def resolve_incident_window(
    case_or_ground_truth: Any,
    ep_evidence: Mapping[str, Any] | None = None,
    mode: str = "oracle",
    timestamps: Sequence[int] | None = None,
    custom_onset_ts: int | None = None,
    custom_end_ts: int | None = None,
) -> IncidentWindow:
    """Resolve the IncidentWindow for a benchmark case.

    Parameters
    ----------
    case_or_ground_truth:
        Either a TelemetryCase, or a dictionary containing 'inject_time' and optional 'timestamps'.
    ep_evidence:
        Optional mapping of entity name to EntityEpisodeEvidence. Used for mode='detected'.
    mode:
        'oracle' (uses ground-truth injection timestamp) or 'detected' (uses earliest episode start timestamp).
    timestamps:
        Optional explicit sequence of observation timestamps.
    custom_onset_ts:
        Optional explicit override for onset timestamp.
    custom_end_ts:
        Optional explicit override for end timestamp.

    Returns
    -------
    IncidentWindow
        The resolved temporal window.
    """
    if mode not in {"oracle", "detected"}:
        raise ValueError(f"Unsupported incident window mode: {mode!r}. Must be 'oracle' or 'detected'.")

    # Extract inject_time and terminal timestamp
    inject_time: int | None = None
    telemetry_end_ts: int | None = None

    if hasattr(case_or_ground_truth, "ground_truth"):
        gt_vals = getattr(case_or_ground_truth.ground_truth, "values", {})
        if "inject_time" in gt_vals:
            inject_time = int(gt_vals["inject_time"])
    elif isinstance(case_or_ground_truth, dict):
        if "inject_time" in case_or_ground_truth:
            inject_time = int(case_or_ground_truth["inject_time"])
        elif "injection_time" in case_or_ground_truth:
            inject_time = int(case_or_ground_truth["injection_time"])

    if timestamps is not None and len(timestamps) > 0:
        telemetry_end_ts = int(timestamps[-1])
    elif hasattr(case_or_ground_truth, "metrics") and getattr(case_or_ground_truth.metrics, "timestamps", None):
        telemetry_end_ts = int(case_or_ground_truth.metrics.timestamps[-1])
    elif hasattr(case_or_ground_truth, "metrics") and getattr(case_or_ground_truth.metrics, "raw_data", None) is not None:
        raw = case_or_ground_truth.metrics.raw_data
        ts_field = getattr(getattr(case_or_ground_truth.metrics, "provenance", None), "timestamp_field", "time")
        if hasattr(raw, "column_names") and ts_field in raw.column_names:
            col = raw[ts_field]
            if len(col) > 0:
                telemetry_end_ts = int(col[-1].as_py())

    if custom_end_ts is not None:
        end_ts = custom_end_ts
    elif telemetry_end_ts is not None:
        end_ts = telemetry_end_ts
    elif inject_time is not None:
        end_ts = inject_time + 600  # Fallback default 10-minute duration
    else:
        raise ValueError("Cannot determine incident end_ts: no timestamps or inject_time provided.")

    if custom_onset_ts is not None:
        return IncidentWindow(
            onset_ts=custom_onset_ts,
            end_ts=end_ts,
            mode=mode,
            source_description=f"custom_override:onset={custom_onset_ts}",
        )

    if mode == "oracle":
        if inject_time is None:
            raise ValueError("mode='oracle' requested but no ground-truth inject_time is available.")
        return IncidentWindow(
            onset_ts=inject_time,
            end_ts=end_ts,
            mode="oracle",
            source_description=f"oracle:inject_time={inject_time}",
        )

    # mode == "detected"
    earliest_ep_ts: int | None = None
    if ep_evidence is not None:
        ep_starts: list[int] = []
        for ev in ep_evidence.values():
            has_ep = getattr(ev, "has_episode", False)
            first_ts = getattr(ev, "first_episode_start_ts", None)
            if has_ep and first_ts is not None:
                ep_starts.append(int(first_ts))
        if ep_starts:
            earliest_ep_ts = min(ep_starts)

    if earliest_ep_ts is not None:
        return IncidentWindow(
            onset_ts=earliest_ep_ts,
            end_ts=end_ts,
            mode="detected",
            source_description=f"detected:first_episode_ts={earliest_ep_ts}",
        )

    # If no episode was detected, record fallback to injection time or error
    if inject_time is not None:
        return IncidentWindow(
            onset_ts=inject_time,
            end_ts=end_ts,
            mode="detected",
            source_description=f"detected_fallback:no_episode_found(inject_time={inject_time})",
        )

    raise ValueError("mode='detected' requested, but no episodes were found and no inject_time fallback exists.")
