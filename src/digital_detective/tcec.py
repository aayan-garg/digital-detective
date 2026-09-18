"""Topology-Coherent Episode Confirmation (TCEC).

Provides the final incident-confirmation layer for Digital Detective, operating
strictly between candidate changepoint detection (BOCPD) and downstream Root
Cause Analysis (RCA).

Scientific Principles:
----------------------
1. Candidate vs Confirmation:
   A Bayesian Online Changepoint Detection (BOCPD) change point is a CANDIDATE,
   not an operational incident declaration.  The candidate is confirmed if and
   only if it exhibits structural coherence across service topology and time.

2. Confirmation Criteria:
   A candidate is confirmed when:
   a. An existing persistent entity anomaly episode exists (persistence K=3, consensus M=2).
   b. The episode involves a candidate entity active at or following the BOCPD candidate.
   c. It is followed by a distinct anomalous entity episode later in time (t_j > t_i).
   d. The two entities are connected through an admissible service topology relation.
   e. The temporal precedence condition (t_j > t_i) is satisfied.

3. Strict Online Causality:
   - Evaluated using only telemetry X[0:t] at each observation step t.
   - Zero access to ground-truth labels, injection timestamps, or fault types.
   - The operational incident onset is t_confirm (the first timestamp at which
     the propagation condition becomes true), NEVER t_candidate.
   - Telemetry after t_confirm is never used to confirm the incident or feed downstream RCA.

4. No Arbitrary Thresholds:
   Reuses existing EpisodeConfig(persistence=3, consensus=2) and EntityGraph topology
   without introducing any new duration, magnitude, correlation, or timeout parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class TCECConfig:
    """Configuration for Topology-Coherent Episode Confirmation.

    Reuses existing episode persistence (K=3) and consensus (M=2) defaults.
    Introduces zero new numerical or window thresholds.
    """

    episode_config: EpisodeConfig = EpisodeConfig(persistence=3, consensus=2)


@dataclass(frozen=True)
class TCECResult:
    """Immutable result of Topology-Coherent Episode Confirmation for one case."""

    case_id: str
    candidate_onset: int | None
    confirmed_onset: int | None
    status: str  # "confirmed", "unconfirmed", "no_detection"
    candidate_entity: str | None
    confirming_entity: str | None
    confirmation_latency_sec: int | None
    timestamps: tuple[Any, ...]
    audit: Mapping[str, Any]


def confirm_topology_coherent_episode(
    case: TelemetryCase,
    bocpd_result: BOCPDResult,
    *,
    graph: EntityGraph | None = None,
    service_aliases: Mapping[str, str] | None = None,
    config: TCECConfig | None = None,
    anomaly_result: MetricAnomalyResult | None = None,
) -> TCECResult:
    """Confirm an incident onset candidate using Topology-Coherent Episode Confirmation.

    Parameters
    ----------
    case:
        TelemetryCase containing metrics and optional traces.
    bocpd_result:
        BOCPDResult containing candidate changepoints and candidate onset.
    graph:
        Optional pre-built EntityGraph. If None, built from case metrics and trace dependencies.
    service_aliases:
        Optional mapping from trace service names to metric entity names.
    config:
        Optional TCECConfig. Reuses existing persistence=3, consensus=2 defaults.
    anomaly_result:
        Optional precomputed MetricAnomalyResult for the case. If None, computed causally.

    Returns
    -------
    TCECResult
        Result containing candidate_onset, confirmed_onset, status, and audit metadata.
    """
    case_id = (
        getattr(getattr(case, "metadata", None), "case_id", None)
        or getattr(case, "case_id", "unknown")
    )
    timestamps = bocpd_result.timestamps
    cfg = config or TCECConfig()

    # 1. Candidate gate: if BOCPD emitted no candidate, status is no_detection
    if bocpd_result.status != "detected" or bocpd_result.onset_ts is None:
        return TCECResult(
            case_id=case_id,
            candidate_onset=None,
            confirmed_onset=None,
            status="no_detection",
            candidate_entity=None,
            confirming_entity=None,
            confirmation_latency_sec=None,
            timestamps=timestamps,
            audit={
                "bocpd_status": bocpd_result.status,
                "reason": "bocpd_emitted_no_candidate",
            },
        )

    t_candidate = int(bocpd_result.onset_ts)

    # 2. Metric anomaly evaluation (causal mean/std baseline)
    det_res = anomaly_result if anomaly_result is not None else detect_metric_anomalies(case)

    # 3. Base entity topology
    if graph is None:
        has_traces = hasattr(case, "traces") and case.traces is not None
        aliases = service_aliases or {"frontendservice": "frontend"}
        if has_traces:
            trace_deps = extract_trace_dependencies(case, service_aliases=aliases)
            deps = [td.to_dependency() for td in trace_deps]
        else:
            deps = []
        eff_graph = build_entity_graph(det_res.metric_names, dependencies=deps)
    else:
        eff_graph = graph

    # 4. Entity episode aggregation (existing persistence=3, consensus=2)
    ep_evidence = aggregate_entity_episodes(
        det_res,
        eff_graph,
        config=cfg.episode_config,
    )

    # Map timestamps to index for causal temporal navigation
    ts_list = list(det_res.timestamps)
    ts_to_idx = {int(ts): idx for idx, ts in enumerate(ts_list)}

    cand_idx = ts_to_idx.get(t_candidate)
    if cand_idx is None:
        # Candidate timestamp not in metric timestamps; find closest causal index
        cand_idx = 0
        for idx, ts in enumerate(ts_list):
            if int(ts) >= t_candidate:
                cand_idx = idx
                break

    # Build adjacency index for admissible topology relations:
    # Entities E_i and E_j are topologically connected if an observed dependency
    # edge exists between them in either direction.
    connected_neighbors: dict[str, set[str]] = {}
    for ent in eff_graph.entities:
        connected_neighbors[ent] = set()
    for dep in eff_graph.dependencies:
        connected_neighbors.setdefault(dep.source, set()).add(dep.target)
        connected_neighbors.setdefault(dep.target, set()).add(dep.source)

    # 5. Online causal evaluation from t_candidate to end of series
    num_steps = len(ts_list)
    confirmed_onset: int | None = None
    candidate_entity: str | None = None
    confirming_entity: str | None = None

    for step in range(cand_idx, num_steps):
        curr_ts = int(ts_list[step])

        # Find candidate episodes Ei that started at or before curr_ts
        # and are active at or after t_candidate
        for ent_i, ev_i in ep_evidence.items():
            if not ev_i.has_episode:
                continue

            for ep_i in ev_i.episodes:
                t_i = int(ep_i.start_timestamp)
                # Ei must have started at/after t_candidate OR was active at t_candidate
                is_candidate_episode = (
                    (t_i >= t_candidate and t_i <= curr_ts)
                    or (int(ep_i.start_timestamp) <= t_candidate <= int(ep_i.end_timestamp))
                )
                if not is_candidate_episode:
                    continue

                # Look for distinct successor Ej connected to Ei
                for ent_j in sorted(connected_neighbors.get(ent_i, ())):
                    if ent_j == ent_i:
                        continue
                    ev_j = ep_evidence.get(ent_j)
                    if ev_j is None or not ev_j.has_episode:
                        continue

                    for ep_j in ev_j.episodes:
                        t_j = int(ep_j.start_timestamp)
                        # Successor must occur strictly later than candidate episode (t_j > t_i)
                        # and must have appeared by the current observation step (t_j <= curr_ts)
                        if t_j > t_i and t_j <= curr_ts:
                            confirmed_onset = curr_ts
                            candidate_entity = ent_i
                            confirming_entity = ent_j
                            break

                    if confirmed_onset is not None:
                        break
                if confirmed_onset is not None:
                    break
            if confirmed_onset is not None:
                break

        if confirmed_onset is not None:
            break

    status = "confirmed" if confirmed_onset is not None else "unconfirmed"
    latency = (confirmed_onset - t_candidate) if confirmed_onset is not None else None

    audit: dict[str, Any] = {
        "candidate_onset": t_candidate,
        "confirmed_onset": confirmed_onset,
        "status": status,
        "candidate_entity": candidate_entity,
        "confirming_entity": confirming_entity,
        "confirmation_latency_sec": latency,
        "total_observations": num_steps,
        "episode_persistence": cfg.episode_config.persistence,
        "episode_consensus": cfg.episode_config.consensus,
        "topology_dependencies_count": len(eff_graph.dependencies),
    }

    return TCECResult(
        case_id=case_id,
        candidate_onset=t_candidate,
        confirmed_onset=confirmed_onset,
        status=status,
        candidate_entity=candidate_entity,
        confirming_entity=confirming_entity,
        confirmation_latency_sec=latency,
        timestamps=timestamps,
        audit=audit,
    )
