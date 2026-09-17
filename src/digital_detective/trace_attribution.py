"""Observational trace latency attribution layer.

Computes benchmark-independent edge-level latency attribution and entity-level
trace delay evidence from observed distributed trace spans.

Does not claim causality; operates strictly as an observational evidence layer
over observed parent-child span durations, return lag, and wait concentration.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Sequence

from .rca import RootCauseScore
from .traces import EdgeLatencyEvidence, TraceLatencyResult


@dataclass(frozen=True)
class EdgeAttribution:
    """Observational attribution evidence for an observed caller -> callee edge u -> v."""

    caller_service: str
    callee_service: str
    elevation: float                 # E_elev = max(0.0, (P90 - P10) / P90) in [0.0, 1.0]
    caller_wait_concentration: float # Phi_wait = TotalWait(u->v) / TotalCallerWait(u) in [0.0, 1.0]
    is_single_child: bool            # True if caller u has exactly 1 observed outbound callee
    sibling_count: int               # Total distinct callees called by caller u
    terminal_factor: float           # T(v) = 1.0 - median(child_covered_fraction(v)) in [0.0, 1.0]
    is_return_lag_dominant: bool     # True if return_lag_p90 >= 0.5 * caller_p90
    attribution_score: float         # A(u, v) = E_elev * Phi_wait * effective_T in [0.0, 1.0]
    call_count: int
    caller_p10: float
    caller_p90: float
    callee_p90: float
    return_lag_p90: float


@dataclass(frozen=True)
class EntityTraceEvidence:
    """Observational entity-level trace delay evidence derived from inbound edges."""

    entity: str
    trace_delay_score: float         # max_{u in callers(v)} A(u, v), bounded in [0.0, 1.0]
    has_trace_evidence: bool         # True if >= 1 inbound trace edge observed; False if uninstrumented / root
    primary_edge: EdgeAttribution | None
    incoming_edges: tuple[EdgeAttribution, ...]


@dataclass(frozen=True)
class TraceAttributionResult:
    """Container for case-level observational trace attribution across entities and edges."""

    case_id: str
    entities: tuple[EntityTraceEvidence, ...]
    edges: tuple[EdgeAttribution, ...]
    uninstrumented_services: tuple[str, ...]

    def get_entity(self, entity: str) -> EntityTraceEvidence | None:
        """Lookup trace attribution evidence for a specific entity."""
        for e in self.entities:
            if e.entity == entity:
                return e
        return None

    def get_edge(self, caller: str, callee: str) -> EdgeAttribution | None:
        """Lookup attribution evidence for a specific caller -> callee edge."""
        for e in self.edges:
            if e.caller_service == caller and e.callee_service == callee:
                return e
        return None


def compute_edge_elevation(p90: float, p10: float) -> float:
    """Compute scale-invariant edge-local tail elevation E_elev in [0.0, 1.0].

    Parameters
    ----------
    p90:
        90th percentile caller span duration for the edge.
    p10:
        10th percentile caller span duration for the edge.

    Returns
    -------
    float
        max(0.0, (P90 - P10) / P90), bounded in [0.0, 1.0].
        Returns 0.0 if P90 <= 0.0 or P10 >= P90.
    """
    if p90 <= 0.0 or p10 >= p90:
        return 0.0
    val = (p90 - p10) / p90
    return max(0.0, min(1.0, float(val)))


def compute_caller_wait_concentration(
    edge_wait: float,
    total_caller_wait: float,
    sibling_count: int,
) -> tuple[float, bool]:
    """Compute caller wait concentration Phi_wait in [0.0, 1.0] and single-child flag.

    Parameters
    ----------
    edge_wait:
        Total observed caller duration for edge u -> v.
    total_caller_wait:
        Total observed caller duration across all outbound edges from caller u.
    sibling_count:
        Number of distinct outbound callees called by caller u.

    Returns
    -------
    tuple[float, bool]
        (phi_wait, is_single_child).
        - phi_wait is bounded in [0.0, 1.0]; returns 0.0 if total_caller_wait <= 0.0.
        - is_single_child is True if sibling_count <= 1, distinguishing trivial
          single-child concentration from multi-sibling concentration.
    """
    is_single_child = (sibling_count <= 1)
    if total_caller_wait <= 0.0 or edge_wait <= 0.0:
        return 0.0, is_single_child
    ratio = edge_wait / total_caller_wait
    return max(0.0, min(1.0, float(ratio))), is_single_child


def compute_terminal_factor(
    median_child_covered_fraction: float | None,
    caller_p90: float,
    return_lag_p90: float,
) -> tuple[float, bool]:
    """Compute terminal non-propagation factor T(v) in [0.0, 1.0] and return-lag dominance flag.

    Parameters
    ----------
    median_child_covered_fraction:
        Pre-computed entity-level median child-covered fraction when callee v acts
        as parent to downstream children, or None if v has no outbound calls (leaf).
    caller_p90:
        Observed P90 caller span duration on edge u -> v.
    return_lag_p90:
        Observed P90 return lag on edge u -> v.

    Returns
    -------
    tuple[float, bool]
        (effective_terminal_factor, is_return_lag_dominant).
        - If return_lag_p90 >= 0.5 * caller_p90 (with caller_p90 > 0):
            Network return lag dominates edge latency, terminating at v's container
            boundary; returns (1.0, True).
        - Else if median_child_covered_fraction is None:
            Callee v is a leaf node with no observed child calls; returns (1.0, False).
        - Else:
            Callee v executes downstream calls; returns
            (max(0.0, min(1.0, 1.0 - median_child_covered_fraction)), False).
    """
    is_return_lag_dominant = bool(caller_p90 > 0.0 and return_lag_p90 >= 0.5 * caller_p90)
    if is_return_lag_dominant:
        return 1.0, True

    if median_child_covered_fraction is None:
        return 1.0, False

    t_val = 1.0 - median_child_covered_fraction
    return max(0.0, min(1.0, float(t_val))), False


def compute_trace_attribution(
    trace_latency: TraceLatencyResult,
    expected_services: Sequence[str] | None = None,
) -> TraceAttributionResult:
    """Compute observational trace attribution across all observed edges and entities.

    Parameters
    ----------
    trace_latency:
        TraceLatencyResult produced by extract_trace_latency_evidence.
    expected_services:
        Optional sequence of all known system services to ensure complete entity
        representation, including uninstrumented nodes.

    Returns
    -------
    TraceAttributionResult
        Case-level attribution results including EdgeAttribution and EntityTraceEvidence.
    """
    # 1. Group outbound edges by caller to determine total caller wait and sibling count
    caller_outbound_edges: dict[str, list[EdgeLatencyEvidence]] = {}
    caller_total_wait: dict[str, float] = {}

    for e in trace_latency.edges:
        caller_outbound_edges.setdefault(e.caller_service, []).append(e)
        caller_total_wait[e.caller_service] = caller_total_wait.get(e.caller_service, 0.0) + float(e.total_caller_duration)

    # 2. Derive entity-level median child-covered fraction for each service when acting as caller
    entity_child_covered_median: dict[str, float] = {}
    for svc, out_edges in caller_outbound_edges.items():
        # Collect medians from caller's outbound edge distributions
        edge_cov_medians = [
            float(e.child_covered_fraction_dist.median)
            for e in out_edges
            if e.child_covered_fraction_dist.count > 0
        ]
        if edge_cov_medians:
            entity_child_covered_median[svc] = float(statistics.median(edge_cov_medians))

    # 3. Compute EdgeAttribution for each observed edge u -> v
    edge_attributions: list[EdgeAttribution] = []
    edges_by_callee: dict[str, list[EdgeAttribution]] = {}

    for e in trace_latency.edges:
        u = e.caller_service
        v = e.callee_service

        c_p10 = float(e.caller_duration_dist.p10)
        c_p90 = float(e.caller_duration_dist.p90)
        ee_p90 = float(e.callee_duration_dist.p90)
        ret_p90 = float(e.return_lag_dist.p90)

        elev = compute_edge_elevation(c_p90, c_p10)

        tot_u_wait = caller_total_wait.get(u, 0.0)
        sib_cnt = len(caller_outbound_edges.get(u, []))
        phi, is_single = compute_caller_wait_concentration(float(e.total_caller_duration), tot_u_wait, sib_cnt)

        # Look up callee's median child-covered fraction (None if callee has no outbound edges)
        v_child_cov = entity_child_covered_median.get(v, None)
        t_factor, is_ret_dom = compute_terminal_factor(v_child_cov, c_p90, ret_p90)

        score = max(0.0, min(1.0, float(elev * phi * t_factor)))

        attr = EdgeAttribution(
            caller_service=u,
            callee_service=v,
            elevation=elev,
            caller_wait_concentration=phi,
            is_single_child=is_single,
            sibling_count=sib_cnt,
            terminal_factor=t_factor,
            is_return_lag_dominant=is_ret_dom,
            attribution_score=score,
            call_count=e.relationship_count,
            caller_p10=c_p10,
            caller_p90=c_p90,
            callee_p90=ee_p90,
            return_lag_p90=ret_p90,
        )
        edge_attributions.append(attr)
        edges_by_callee.setdefault(v, []).append(attr)

    # 4. Determine full set of entities to represent
    all_entities = set(trace_latency.services)
    if expected_services:
        all_entities.update(expected_services)
    for e in trace_latency.edges:
        all_entities.add(e.caller_service)
        all_entities.add(e.callee_service)

    # 5. Build EntityTraceEvidence for each entity
    entity_evidence_list: list[EntityTraceEvidence] = []
    for ent in sorted(all_entities):
        in_edges = edges_by_callee.get(ent, [])
        if not in_edges:
            # Uninstrumented or root caller: absent / neutral evidence
            ev = EntityTraceEvidence(
                entity=ent,
                trace_delay_score=0.0,
                has_trace_evidence=False,
                primary_edge=None,
                incoming_edges=(),
            )
        else:
            sorted_in = sorted(in_edges, key=lambda a: -a.attribution_score)
            best_edge = sorted_in[0]
            ev = EntityTraceEvidence(
                entity=ent,
                trace_delay_score=best_edge.attribution_score,
                has_trace_evidence=True,
                primary_edge=best_edge,
                incoming_edges=tuple(sorted_in),
            )
        entity_evidence_list.append(ev)

    # Determine uninstrumented services
    uninstrumented = trace_latency.uninstrumented_services
    if expected_services and not uninstrumented:
        uninst_set = set(expected_services) - set(trace_latency.services)
        uninstrumented = tuple(sorted(uninst_set))

    return TraceAttributionResult(
        case_id=trace_latency.case_id,
        entities=tuple(entity_evidence_list),
        edges=tuple(edge_attributions),
        uninstrumented_services=uninstrumented,
    )


def rank_with_trace_elevation(
    trace_latency: TraceLatencyResult,
    candidate_universe: Sequence[str] | None = None,
) -> tuple[RootCauseScore, ...]:
    """Rank candidate entities using standalone trace edge elevation (E_elev).

    For each candidate entity v, its score is the maximum elevation across all observed
    incoming caller -> callee edges u -> v:
        E_elev(v) = max_{u in callers(v)} compute_edge_elevation(P90(u->v), P10(u->v))

    Entities with no incoming trace edges (roots, uninstrumented, or non-reporting)
    receive score 0.0 while remaining preserved in the candidate ranking.

    Tie-breaking: (-score, entity_name).

    Parameters
    ----------
    trace_latency:
        TraceLatencyResult produced by extract_trace_latency_evidence.
    candidate_universe:
        Optional explicit sequence of candidate entities to enforce a closed candidate set.
        If None, candidates are derived from trace_latency services and edges.

    Returns
    -------
    tuple[RootCauseScore, ...]
        Candidate entities ranked deterministically descending by E_elev.
    """
    elev_by_callee: dict[str, float] = {}
    for edge in trace_latency.edges:
        p90 = float(edge.caller_duration_dist.p90)
        p10 = float(edge.caller_duration_dist.p10)
        elev = compute_edge_elevation(p90, p10)
        v = edge.callee_service
        if v not in elev_by_callee or elev > elev_by_callee[v]:
            elev_by_callee[v] = elev

    if candidate_universe is not None:
        candidate_entities = sorted(set(candidate_universe))
    else:
        all_ents = set(trace_latency.services)
        for edge in trace_latency.edges:
            all_ents.add(edge.caller_service)
            all_ents.add(edge.callee_service)
        candidate_entities = sorted(all_ents)

    scores: list[RootCauseScore] = []
    for ent in candidate_entities:
        sc = elev_by_callee.get(ent, 0.0)
        scores.append(
            RootCauseScore(
                entity=ent,
                score=sc,
                r_early=0.0,
                r_strength=sc,
                r_coverage=0.0,
                r_prop=0.0,
                first_anomaly_ts=None,
                peak_score=sc,
            )
        )

    # Deterministic tie-breaking: score desc, entity asc
    scores.sort(key=lambda s: (-s.score, s.entity))
    return tuple(scores)

