"""Diagnostic trace latency evidence layer.

Computes observed execution relationships, child-covered duration, caller wait
evidence, and downstream latency attribution from distributed trace spans.

Does not claim causality; operates strictly over observed parent-child span
intervals and execution timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .telemetry import TelemetryCase


@dataclass(frozen=True)
class DistributionSummary:
    """Summary statistics for a numeric observation distribution.

    Quantiles use nearest-rank index assignment: `idx = int(round(p * (n - 1)))`.
    For small sample sizes, Python's round-half-to-even rule determines index
    selection (for p=0.10, N <= 6 selects index 0, while N = 7, 8, 9 selects index 1).
    """

    count: int
    min: float
    median: float
    p90: float
    p99: float
    max: float
    mean: float
    p10: float = 0.0

    @classmethod
    def from_values(cls, values: Sequence[float | int]) -> DistributionSummary:
        """Construct distribution summary from a sequence of numeric values."""
        if not values:
            return cls(count=0, min=0.0, median=0.0, p90=0.0, p99=0.0, max=0.0, mean=0.0, p10=0.0)
        sorted_vals = sorted(float(v) for v in values)
        n = len(sorted_vals)

        def quantile(p: float) -> float:
            idx = int(round(p * (n - 1)))
            return sorted_vals[max(0, min(n - 1, idx))]

        return cls(
            count=n,
            min=sorted_vals[0],
            p10=quantile(0.10),
            median=quantile(0.50),
            p90=quantile(0.90),
            p99=quantile(0.99),
            max=sorted_vals[-1],
            mean=sum(sorted_vals) / n,
        )


@dataclass(frozen=True)
class SpanRecord:
    """Immutable representation of an individual trace span."""

    span_id: str
    parent_span_id: str | None
    trace_id: str
    service_name: str
    operation_name: str
    start_time: int
    duration: int
    status_code: int | None = None

    @property
    def end_time(self) -> int:
        """Span end timestamp in microseconds."""
        return self.start_time + self.duration


@dataclass(frozen=True)
class SpanDecomposition:
    """Timing breakdown of a parent span into child-covered time and self time."""

    span_id: str
    service_name: str
    duration: int
    child_covered_duration: int
    self_duration: int
    child_covered_fraction: float
    self_fraction: float
    child_count: int
    has_sibling_overlap: bool


@dataclass(frozen=True)
class CallerCalleeRelationship:
    """An observed cross-service caller -> callee span execution relationship."""

    trace_id: str
    caller_service: str
    callee_service: str
    caller_span_id: str
    callee_span_id: str
    caller_operation: str
    callee_operation: str
    caller_duration: int
    callee_duration: int
    start_lag: int
    return_lag: int
    callee_duration_fraction: float


@dataclass(frozen=True)
class EdgeLatencyEvidence:
    """Aggregate diagnostic latency evidence for an observed caller -> callee edge."""

    caller_service: str
    callee_service: str
    relationship_count: int
    caller_duration_dist: DistributionSummary
    callee_duration_dist: DistributionSummary
    child_covered_fraction_dist: DistributionSummary
    caller_self_fraction_dist: DistributionSummary
    return_lag_dist: DistributionSummary
    fraction_of_caller_time_dist: DistributionSummary
    total_caller_duration: int
    total_callee_duration: int
    representative_examples: tuple[CallerCalleeRelationship, ...] = ()


@dataclass(frozen=True)
class TraceLatencyResult:
    """Container for case-level trace latency analysis and edge-level evidence."""

    case_id: str
    total_spans: int
    total_traces: int
    services: tuple[str, ...]
    edges: tuple[EdgeLatencyEvidence, ...]
    sibling_overlap_count: int
    uninstrumented_services: tuple[str, ...] = ()
    max_span_timestamp: int | None = None
    max_span_end_timestamp: int | None = None
    retained_spans: int = 0
    excluded_spans: int = 0

    def get_edge(self, caller: str, callee: str) -> EdgeLatencyEvidence | None:
        """Lookup evidence for a specific caller -> callee edge, or None."""
        for e in self.edges:
            if e.caller_service == caller and e.callee_service == callee:
                return e
        return None


def compute_interval_union(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Compute minimal disjoint intervals representing the union of input intervals."""
    valid = [(s, e) for s, e in intervals if e > s]
    if not valid:
        return ()
    sorted_intervals = sorted(valid, key=lambda x: (x[0], x[1]))
    merged: list[list[int]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for s, e in sorted_intervals[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return tuple((iv[0], iv[1]) for iv in merged)


def interval_union_duration(intervals: Sequence[tuple[int, int]]) -> int:
    """Compute the total covered duration of the union of intervals."""
    merged = compute_interval_union(intervals)
    return sum(e - s for s, e in merged)


def has_overlapping_intervals(intervals: Sequence[tuple[int, int]]) -> bool:
    """Check if any two intervals in the sequence strictly overlap."""
    valid = [(s, e) for s, e in intervals if e > s]
    if len(valid) <= 1:
        return False
    sorted_ivs = sorted(valid, key=lambda x: (x[0], x[1]))
    max_end = sorted_ivs[0][1]
    for s, e in sorted_ivs[1:]:
        if s < max_end:
            return True
        if e > max_end:
            max_end = e
    return False


def decompose_parent_span(
    parent: SpanRecord,
    children: Sequence[SpanRecord],
) -> SpanDecomposition:
    """Decompose a parent span into child-covered duration and self duration."""
    p_dur = max(0, parent.duration)
    p_start = parent.start_time
    p_end = parent.end_time

    if not children or p_dur == 0:
        return SpanDecomposition(
            span_id=parent.span_id,
            service_name=parent.service_name,
            duration=p_dur,
            child_covered_duration=0,
            self_duration=p_dur,
            child_covered_fraction=0.0,
            self_fraction=1.0,
            child_count=len(children),
            has_sibling_overlap=False,
        )

    child_raw_intervals = [(c.start_time, c.end_time) for c in children]
    has_overlap = has_overlapping_intervals(child_raw_intervals)

    # Clip children to parent interval
    clipped_intervals = [
        (max(p_start, c.start_time), min(p_end, c.end_time))
        for c in children
        if min(p_end, c.end_time) > max(p_start, c.start_time)
    ]
    c_union_dur = interval_union_duration(clipped_intervals)
    c_covered = min(p_dur, max(0, c_union_dur))
    s_dur = max(0, p_dur - c_covered)

    c_fraction = c_covered / p_dur if p_dur > 0 else 0.0
    s_fraction = s_dur / p_dur if p_dur > 0 else 1.0

    return SpanDecomposition(
        span_id=parent.span_id,
        service_name=parent.service_name,
        duration=p_dur,
        child_covered_duration=c_covered,
        self_duration=s_dur,
        child_covered_fraction=c_fraction,
        self_fraction=s_fraction,
        child_count=len(children),
        has_sibling_overlap=has_overlap,
    )


def extract_trace_latency_evidence(
    case: TelemetryCase,
    service_aliases: Mapping[str, str] | None = None,
    expected_services: Sequence[str] | None = None,
    max_examples_per_edge: int = 5,
    max_timestamp: int | float | None = None,
) -> TraceLatencyResult:
    """Extract diagnostic caller -> callee trace latency evidence from a TelemetryCase.

    Parameters
    ----------
    case:
        TelemetryCase containing optional traces modality.
    service_aliases:
        Optional mapping from trace serviceName to canonical entity name
        (e.g. {"frontendservice": "frontend"}).
    expected_services:
        Optional sequence of all known system services to identify uninstrumented entities.
    max_examples_per_edge:
        Maximum number of representative high-duration caller-callee examples to retain.
    max_timestamp:
        Optional causal timestamp cutoff. Spans with start_time > max_timestamp
        are strictly excluded from analysis.

    Returns
    -------
    TraceLatencyResult
        Computed diagnostic evidence across all observed caller -> callee edges.
    """
    case_id = case.metadata.case_id
    if case.traces is None:
        uninst = tuple(sorted(expected_services)) if expected_services else ()
        return TraceLatencyResult(
            case_id=case_id,
            total_spans=0,
            total_traces=0,
            services=(),
            edges=(),
            sibling_overlap_count=0,
            uninstrumented_services=uninst,
            max_span_timestamp=None,
            max_span_end_timestamp=None,
            retained_spans=0,
            excluded_spans=0,
        )

    table = case.traces.raw_data
    if hasattr(table, "column_names"):
        columns = set(table.column_names)
        get_col = lambda name: table.column(name).to_pylist()
    elif isinstance(table, Mapping):
        columns = set(table.keys())
        get_col = lambda name: list(table[name])
    else:
        raise ValueError(f"Unsupported trace raw_data type: {type(table)!r}")

    required_cols = ("spanID", "parentSpanID", "traceID", "serviceName", "operationName", "startTime", "duration")
    missing = [c for c in required_cols if c not in columns]
    if missing:
        raise ValueError(f"Trace data missing required column(s): {missing}")

    span_ids = get_col("spanID")
    parent_ids = get_col("parentSpanID")
    trace_ids = get_col("traceID")
    service_names = get_col("serviceName")
    operation_names = get_col("operationName")
    start_times = get_col("startTime")
    durations = get_col("duration")

    status_codes = get_col("statusCode") if "statusCode" in columns else [None] * len(span_ids)

    # Ingest spans with deduplication for identical duplicate records
    spans_by_id: dict[str, SpanRecord] = {}
    distinct_traces: set[str] = set()
    excluded_spans = 0

    for sid, pid, tid, sname, op, st, dur, sc in zip(
        span_ids, parent_ids, trace_ids, service_names, operation_names, start_times, durations, status_codes
    ):
        if sid is None or sid == "":
            raise ValueError("Trace spanID cannot be null or empty")

        p_id = str(pid) if (pid is not None and str(pid) != "" and str(pid) != "<NA>") else None
        t_id = str(tid) if tid is not None else ""
        raw_svc = str(sname) if sname is not None else ""
        svc = service_aliases.get(raw_svc, raw_svc) if service_aliases else raw_svc
        op_name = str(op) if op is not None else ""
        s_time = int(st) if st is not None else 0
        dur_val = int(dur) if dur is not None else 0
        s_end = s_time + dur_val

        # Causal temporal restriction: exclude spans whose completion time is > max_timestamp
        if max_timestamp is not None:
            if s_time > 1e14 and max_timestamp < 1e11:
                # s_time and s_end in microseconds (~1e15), max_timestamp in seconds (~1e9)
                if (s_end / 1_000_000.0) > max_timestamp:
                    excluded_spans += 1
                    continue
            elif s_time > 1e11 and max_timestamp < 1e11:
                # s_time and s_end in milliseconds (~1e12), max_timestamp in seconds (~1e9)
                if (s_end / 1000.0) > max_timestamp:
                    excluded_spans += 1
                    continue
            else:
                if s_end > max_timestamp:
                    excluded_spans += 1
                    continue
        sc_val = int(sc) if (sc is not None and str(sc) != "<NA>") else None

        record = SpanRecord(
            span_id=str(sid),
            parent_span_id=p_id,
            trace_id=t_id,
            service_name=svc,
            operation_name=op_name,
            start_time=s_time,
            duration=dur_val,
            status_code=sc_val,
        )

        if sid in spans_by_id:
            existing = spans_by_id[sid]
            # Verify record consistency
            if (
                existing.parent_span_id != record.parent_span_id
                or existing.service_name != record.service_name
                or existing.trace_id != record.trace_id
            ):
                raise ValueError(f"Conflicting duplicate spanID detected: {sid!r}")
            continue

        spans_by_id[sid] = record
        if t_id:
            distinct_traces.add(t_id)

    # Build parent -> children map
    children_by_parent: dict[str, list[SpanRecord]] = {}
    for span in spans_by_id.values():
        if span.parent_span_id and span.parent_span_id in spans_by_id:
            children_by_parent.setdefault(span.parent_span_id, []).append(span)

    # Decompose all parent spans and check for sibling overlaps
    sibling_overlap_count = 0
    span_decompositions: dict[str, SpanDecomposition] = {}

    for pid, children in children_by_parent.items():
        parent_span = spans_by_id[pid]
        decomp = decompose_parent_span(parent_span, children)
        span_decompositions[pid] = decomp
        if decomp.has_sibling_overlap:
            sibling_overlap_count += 1

    # Extract cross-service caller -> callee relationships
    # Group by (caller_service, callee_service)
    edge_relationships: dict[tuple[str, str], list[CallerCalleeRelationship]] = {}

    for span in spans_by_id.values():
        if not span.parent_span_id or span.parent_span_id not in spans_by_id:
            continue
        parent_span = spans_by_id[span.parent_span_id]

        caller_svc = parent_span.service_name
        callee_svc = span.service_name

        if not caller_svc or not callee_svc or caller_svc == callee_svc:
            continue

        c_dur = max(0, parent_span.duration)
        e_dur = max(0, span.duration)
        s_lag = span.start_time - parent_span.start_time
        r_lag = parent_span.end_time - span.end_time
        frac = e_dur / c_dur if c_dur > 0 else 0.0

        rel = CallerCalleeRelationship(
            trace_id=span.trace_id,
            caller_service=caller_svc,
            callee_service=callee_svc,
            caller_span_id=parent_span.span_id,
            callee_span_id=span.span_id,
            caller_operation=parent_span.operation_name,
            callee_operation=span.operation_name,
            caller_duration=c_dur,
            callee_duration=e_dur,
            start_lag=s_lag,
            return_lag=r_lag,
            callee_duration_fraction=frac,
        )

        edge_relationships.setdefault((caller_svc, callee_svc), []).append(rel)

    # Build EdgeLatencyEvidence
    edge_evidence_list: list[EdgeLatencyEvidence] = []
    observed_services = set(s.service_name for s in spans_by_id.values())

    for (u, v), rels in sorted(edge_relationships.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        caller_durs = [r.caller_duration for r in rels]
        callee_durs = [r.callee_duration for r in rels]
        return_lags = [r.return_lag for r in rels]
        callee_fracs = [r.callee_duration_fraction for r in rels]

        # Look up parent decompositions for caller spans
        child_covered_fracs = [
            span_decompositions[r.caller_span_id].child_covered_fraction
            for r in rels
            if r.caller_span_id in span_decompositions
        ]
        caller_self_fracs = [
            span_decompositions[r.caller_span_id].self_fraction
            for r in rels
            if r.caller_span_id in span_decompositions
        ]

        # Top representative high-duration examples
        sorted_rels = sorted(rels, key=lambda r: -r.caller_duration)
        reps = tuple(sorted_rels[:max_examples_per_edge])

        edge_ev = EdgeLatencyEvidence(
            caller_service=u,
            callee_service=v,
            relationship_count=len(rels),
            caller_duration_dist=DistributionSummary.from_values(caller_durs),
            callee_duration_dist=DistributionSummary.from_values(callee_durs),
            child_covered_fraction_dist=DistributionSummary.from_values(child_covered_fracs),
            caller_self_fraction_dist=DistributionSummary.from_values(caller_self_fracs),
            return_lag_dist=DistributionSummary.from_values(return_lags),
            fraction_of_caller_time_dist=DistributionSummary.from_values(callee_fracs),
            total_caller_duration=sum(caller_durs),
            total_callee_duration=sum(callee_durs),
            representative_examples=reps,
        )
        edge_evidence_list.append(edge_ev)

    # Determine uninstrumented services
    uninstrumented: tuple[str, ...] = ()
    if expected_services:
        uninst_set = set(expected_services) - observed_services
        uninstrumented = tuple(sorted(uninst_set))

    max_observed_ts = max((s.start_time for s in spans_by_id.values()), default=None)
    max_observed_end_ts = max((s.end_time for s in spans_by_id.values()), default=None)

    return TraceLatencyResult(
        case_id=case_id,
        total_spans=len(spans_by_id),
        total_traces=len(distinct_traces),
        services=tuple(sorted(observed_services)),
        edges=tuple(edge_evidence_list),
        sibling_overlap_count=sibling_overlap_count,
        uninstrumented_services=uninstrumented,
        max_span_timestamp=max_observed_ts,
        max_span_end_timestamp=max_observed_end_ts,
        retained_spans=len(spans_by_id),
        excluded_spans=excluded_spans,
    )
