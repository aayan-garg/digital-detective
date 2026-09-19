"""Sequential Topology-Coherent Episode Confirmation (Sequential TCEC).

Provides a strictly causal, sequential evidence accumulation and detector-horizon
stale-reset mechanism for incident confirmation, replacing the instantaneous
Boolean trigger of Stage 5 TCEC.

Scientific Principles:
----------------------
1. Three-State Sequential Automaton:
   NORMAL -> SUSPECT -> CONFIRMED (with reset from SUSPECT -> NORMAL upon detector horizon elapsed).
2. Directed Topology Evidence:
   Explicitly distinguishes FORWARD (caller -> callee) from REVERSE (callee -> caller)
   structural dependencies instead of collapsing into undirected adjacency.
3. Multi-Step Temporal Corroboration Sequence:
   Requires a strictly time-ordered sequence of corroboration events (Z >= 3: candidate t1,
   first propagation t2 > t1, and second corroboration t3 > t2) while anomalous activity remains active.
4. Physical Detector-Horizon Reset (Zero Arbitrary Parameters):
   If all entities in the suspect set become inactive for longer than the detector's
   rolling window W = 60s, rolling statistics have returned to baseline, resetting
   the suspect hypothesis to NORMAL.
5. Strict Online Causality:
   Evaluated at discrete time steps using only data observable up to step t.
   Trace topology extraction strictly bounded by max_timestamp = curr_ts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .anomaly import MetricAnomalyResult, detect_metric_anomalies
from .bocpd import BOCPDResult
from .episodes import EpisodeConfig, EntityEpisodeEvidence, aggregate_entity_episodes
from .telemetry import TelemetryCase
from .topology import (
    Dependency,
    EntityGraph,
    build_entity_graph,
    extract_trace_dependencies,
)


@dataclass(frozen=True)
class SequentialTCECConfig:
    """Configuration for Sequential Topology-Coherent Episode Confirmation."""

    episode_config: EpisodeConfig = EpisodeConfig(persistence=3, consensus=2)
    memory_window_seconds: int = 60  # Inherited from rolling anomaly detector window W=60
    stopping_threshold: int = 3  # Minimal sequence length for multi-step temporal corroboration

    def __post_init__(self) -> None:
        if self.memory_window_seconds < 1:
            raise ValueError(
                f"memory_window_seconds must be >= 1, got {self.memory_window_seconds}"
            )
        if self.stopping_threshold < 2:
            raise ValueError(
                f"stopping_threshold must be >= 2, got {self.stopping_threshold}"
            )


@dataclass(frozen=True)
class CorroboratingEvidence:
    """Record of a single corroborating entity incorporated into the suspect cascade."""

    entity: str
    connected_to: str
    direction: str  # "FORWARD" (caller -> callee) or "REVERSE" (callee -> caller)
    activation_timestamp: int
    step_index: int


@dataclass(frozen=True)
class SequentialTCECResult:
    """Immutable result of Sequential TCEC incident confirmation for one case."""

    case_id: str
    candidate_onset: int | None
    confirmed_onset: int | None  # Stored identically to confirmation_time (tau)
    status: str  # "confirmed", "unconfirmed", "no_detection"
    candidate_entity: str | None
    corroborating_entities: tuple[str, ...]
    corroborating_details: tuple[CorroboratingEvidence, ...]
    confirmation_latency_sec: int | None
    state_transitions: tuple[dict[str, Any], ...]
    timestamps: tuple[Any, ...]
    audit: Mapping[str, Any]
    confirmation_time: int | None = None
    first_evidence_timestamp: int | None = None


def confirm_sequential_tcec(
    case: TelemetryCase,
    bocpd_result: BOCPDResult,
    *,
    service_aliases: Mapping[str, str] | None = None,
    config: SequentialTCECConfig | None = None,
    anomaly_result: MetricAnomalyResult | None = None,
    graph: EntityGraph | None = None,
) -> SequentialTCECResult:
    """Confirm an incident onset candidate using Sequential TCEC.

    Parameters
    ----------
    case:
        TelemetryCase containing metrics and optional traces.
    bocpd_result:
        BOCPDResult containing candidate changepoints and candidate onset.
    service_aliases:
        Optional mapping from trace service names to metric entity names.
    config:
        Optional SequentialTCECConfig. Defaults to persistence=3, consensus=2,
        memory_window_seconds=60, stopping_threshold=3.
    anomaly_result:
        Optional precomputed MetricAnomalyResult for the case.
    graph:
        Optional pre-built EntityGraph for static topology.

    Returns
    -------
    SequentialTCECResult
        Immutable result containing candidate_onset, confirmed_onset, status,
        corroborating evidence, and state transitions.
    """
    case_id = (
        getattr(getattr(case, "metadata", None), "case_id", None)
        or getattr(case, "case_id", "unknown")
    )
    timestamps = bocpd_result.timestamps
    cfg = config or SequentialTCECConfig()

    # 1. Candidate gate: if BOCPD emitted no candidate, status is no_detection
    if bocpd_result.status != "detected" or bocpd_result.onset_ts is None:
        return SequentialTCECResult(
            case_id=case_id,
            candidate_onset=None,
            confirmed_onset=None,
            status="no_detection",
            candidate_entity=None,
            corroborating_entities=(),
            corroborating_details=(),
            confirmation_latency_sec=None,
            state_transitions=(),
            timestamps=timestamps,
            audit={
                "bocpd_status": bocpd_result.status,
                "reason": "bocpd_emitted_no_candidate",
            },
        )

    t_candidate = int(bocpd_result.onset_ts)

    # 2. Metric anomaly evaluation (causal rolling mean/std baseline)
    det_res = anomaly_result if anomaly_result is not None else detect_metric_anomalies(case)
    ts_list = list(det_res.timestamps)
    ts_to_idx = {int(ts): idx for idx, ts in enumerate(ts_list)}
    num_steps = len(ts_list)

    cand_idx = ts_to_idx.get(t_candidate)
    if cand_idx is None:
        cand_idx = 0
        for idx, ts in enumerate(ts_list):
            if int(ts) >= t_candidate:
                cand_idx = idx
                break

    # 3. Base Entity Topology and Incremental Causal Edge Timeline
    aliases = service_aliases or {"frontendservice": "frontend"}
    has_traces = hasattr(case, "traces") and case.traces is not None

    # Preprocess trace edges once into an immutable causal availability timeline.
    # Mathematically identical to extract_trace_dependencies(case, max_timestamp=t):
    # An edge (parent, child) is observable at time t iff both parent and child spans completed <= t.
    first_edge_ts: dict[tuple[str, str], float] = {}
    if graph is not None:
        for d in graph.dependencies:
            first_edge_ts[(d.source, d.target)] = -float("inf")
    elif has_traces and case.traces is not None:
        table = case.traces.raw_data
        if hasattr(table, "column_names"):
            cols = set(table.column_names)
            get_c = lambda name: table.column(name).to_pylist()
        elif isinstance(table, Mapping):
            cols = set(table.keys())
            get_c = lambda name: list(table[name])
        else:
            cols = set()
            get_c = lambda name: []

        req = ["spanID", "parentSpanID", "serviceName", "startTime"]
        if cols.issuperset(req):
            s_ids = get_c("spanID")
            p_ids = get_c("parentSpanID")
            s_names = get_c("serviceName")
            s_times = get_c("startTime")
            durations = get_c("duration") if "duration" in cols else None

            span_to_svc: dict[str, str] = {}
            span_to_par: dict[str, str | None] = {}
            span_to_end: dict[str, float] = {}
            n_spans = len(s_ids)
            for i in range(n_spans):
                sid = s_ids[i]
                if sid is None or sid in span_to_svc:
                    continue
                st = s_times[i]
                dur = durations[i] if durations is not None else 0
                s_time = int(st) if st is not None else 0
                dur_val = int(dur) if (dur is not None and str(dur) != "<NA>") else 0
                s_end = s_time + dur_val

                if s_time > 1e14:
                    s_end_sec = s_end / 1_000_000.0
                elif s_time > 1e11:
                    s_end_sec = s_end / 1000.0
                else:
                    s_end_sec = float(s_end)

                span_to_svc[sid] = s_names[i]
                span_to_par[sid] = p_ids[i]
                span_to_end[sid] = s_end_sec

            for sid, child_s in span_to_svc.items():
                pid = span_to_par.get(sid)
                if not pid or pid not in span_to_svc:
                    continue
                parent_s = span_to_svc[pid]
                if not parent_s or not child_s:
                    continue
                parent_s = aliases.get(parent_s, parent_s)
                child_s = aliases.get(child_s, child_s)
                if parent_s == child_s:
                    continue
                edge = (parent_s, child_s)
                req_t = max(span_to_end[sid], span_to_end[pid])
                if edge not in first_edge_ts or req_t < first_edge_ts[edge]:
                    first_edge_ts[edge] = req_t

    def get_causal_edges_at(max_ts: int) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Return (forward_edges, reverse_edges) observable at or before max_ts."""
        f_edges: dict[str, set[str]] = {}
        r_edges: dict[str, set[str]] = {}
        for (u, v), req_t in first_edge_ts.items():
            if req_t <= max_ts:
                f_edges.setdefault(u, set()).add(v)  # u calls v (FORWARD)
                r_edges.setdefault(v, set()).add(u)  # v called by u (REVERSE)
        return f_edges, r_edges

    base_graph = graph or build_entity_graph(det_res.metric_names, dependencies=())
    ep_evidence = aggregate_entity_episodes(det_res, base_graph, config=cfg.episode_config)

    # 4. Sequential Evidence State Machine
    state = "NORMAL"
    Z = 0
    candidate_entity: str | None = None
    suspect_set: set[str] = set()
    corroborating_entities: list[str] = []
    corroborating_details: list[CorroboratingEvidence] = []
    last_event_ts = 0
    last_active_ts = 0
    state_transitions: list[dict[str, Any]] = []
    confirmed_onset: int | None = None
    first_evidence_ts: int | None = None

    for step in range(cand_idx, num_steps):
        curr_ts = int(ts_list[step])
        active_ents = {ent for ent, ev in ep_evidence.items() if ev.is_in_episode[step]}

        # A. STATE: NORMAL -> look for candidate trigger
        if state == "NORMAL":
            for ent_name in active_ents:
                ev = ep_evidence[ent_name]
                ep_start = None
                for ep in ev.episodes:
                    if ep.start_idx <= step <= ep.end_idx:
                        ep_start = int(ep.start_timestamp)
                        break

                if ep_start is not None and (ep_start >= t_candidate or ep_start <= t_candidate <= curr_ts):
                    state = "SUSPECT"
                    Z = 1
                    candidate_entity = ent_name
                    suspect_set = {ent_name}
                    corroborating_entities = []
                    corroborating_details = []
                    last_event_ts = curr_ts
                    last_active_ts = curr_ts
                    first_evidence_ts = ep_start
                    state_transitions.append({
                        "timestamp": curr_ts,
                        "step": step,
                        "from_state": "NORMAL",
                        "to_state": "SUSPECT",
                        "trigger_entity": ent_name,
                        "evidence_level": Z,
                    })
                    break

        # B. STATE: SUSPECT -> accumulate sequential corroboration or reset
        elif state == "SUSPECT":
            active_suspect = suspect_set & active_ents
            if active_suspect:
                last_active_ts = curr_ts
            else:
                # Detector memory horizon check: if no suspect entity active for > W seconds, reset
                if (curr_ts - last_active_ts) > cfg.memory_window_seconds:
                    state_transitions.append({
                        "timestamp": curr_ts,
                        "step": step,
                        "from_state": "SUSPECT",
                        "to_state": "NORMAL",
                        "reason": f"detector_memory_horizon_elapsed_{curr_ts - last_active_ts}s",
                        "abandoned_suspect_set": list(suspect_set),
                    })
                    state = "NORMAL"
                    Z = 0
                    candidate_entity = None
                    suspect_set = set()
                    corroborating_entities = []
                    corroborating_details = []
                    last_event_ts = 0
                    last_active_ts = 0
                    first_evidence_ts = None
                    continue

            # Check for sequential corroborating entity in causal topology G_t
            forward_edges, reverse_edges = get_causal_edges_at(curr_ts)

            # Look for a new corroborating entity active at this step
            for ent_name in active_ents:
                if ent_name in suspect_set:
                    continue

                # Must satisfy temporal precedence: step occurred after the previous corroboration event
                if curr_ts <= last_event_ts:
                    continue

                # Must be topologically connected to an entity in suspect_set
                connected_to = None
                direction = None
                for s_ent in suspect_set:
                    if ent_name in forward_edges.get(s_ent, ()):
                        connected_to = s_ent
                        direction = "FORWARD"
                        break
                    elif ent_name in reverse_edges.get(s_ent, ()):
                        connected_to = s_ent
                        direction = "REVERSE"
                        break

                if connected_to is not None and direction is not None:
                    suspect_set.add(ent_name)
                    corroborating_entities.append(ent_name)
                    Z += 1
                    last_event_ts = curr_ts
                    corroborating_details.append(
                        CorroboratingEvidence(
                            entity=ent_name,
                            connected_to=connected_to,
                            direction=direction,
                            activation_timestamp=curr_ts,
                            step_index=step,
                        )
                    )
                    state_transitions.append({
                        "timestamp": curr_ts,
                        "step": step,
                        "event": "CORROBORATION",
                        "entity": ent_name,
                        "connected_to": connected_to,
                        "direction": direction,
                        "evidence_level": Z,
                    })
                    break

            # Stopping Rule: Z >= stopping_threshold AND at least one suspect entity currently active
            if Z >= cfg.stopping_threshold and active_suspect:
                state = "CONFIRMED"
                confirmed_onset = curr_ts
                state_transitions.append({
                    "timestamp": curr_ts,
                    "step": step,
                    "from_state": "SUSPECT",
                    "to_state": "CONFIRMED",
                    "final_evidence_level": Z,
                    "suspect_entities": list(suspect_set),
                    "corroborating_entities": list(corroborating_entities),
                })
                break

    status = "confirmed" if confirmed_onset is not None else "unconfirmed"
    confirmation_time = confirmed_onset  # Exactly tau: zero backdating
    latency = (confirmation_time - t_candidate) if confirmation_time is not None else None

    audit: dict[str, Any] = {
        "candidate_onset": t_candidate,
        "confirmed_onset": confirmed_onset,
        "confirmation_time": confirmation_time,
        "first_evidence_timestamp": first_evidence_ts,
        "status": status,
        "candidate_entity": candidate_entity,
        "corroborating_entities": corroborating_entities,
        "confirmation_latency_sec": latency,
        "total_observations": num_steps,
        "memory_window_seconds": cfg.memory_window_seconds,
        "stopping_threshold": cfg.stopping_threshold,
        "state_transitions_count": len(state_transitions),
    }

    return SequentialTCECResult(
        case_id=case_id,
        candidate_onset=t_candidate,
        confirmed_onset=confirmed_onset,
        status=status,
        candidate_entity=candidate_entity,
        corroborating_entities=tuple(corroborating_entities),
        corroborating_details=tuple(corroborating_details),
        confirmation_latency_sec=latency,
        state_transitions=tuple(state_transitions),
        timestamps=timestamps,
        audit=audit,
        confirmation_time=confirmation_time,
        first_evidence_timestamp=first_evidence_ts,
    )
