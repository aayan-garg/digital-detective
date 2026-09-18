"""Parameter-Free Trace Tie-Break for S_comb Ranking.

Applies inbound distributed trace latency elevation (E_elev) as a deterministic,
parameter-free secondary tie-breaker exclusively for candidates with exact S_comb
score ties.

Scientific Principles:
----------------------
1. Non-Tied Invariant:
   If S_comb(A) != S_comb(B), their relative ordering is 100% mathematically preserved.
   Trace evidence NEVER promotes a candidate over another candidate with a higher
   S_comb metric score.

2. Exact Tie Resolution:
   If S_comb(A) == S_comb(B) (exact floating-point equality), inbound trace elevation
   E_elev is used as a secondary tie-breaker descending:
       E_elev(v) = max_{u in callers(v)} (P90(u->v) - P10(u->v)) / P90(u->v)
   This replaces arbitrary alphabetical tie-breaking with observed caller wait elevation.

3. Deterministic Fallback:
   If S_comb(A) == S_comb(B) and E_elev(A) == E_elev(B), the original S_comb tie-break
   order is preserved.

4. Zero Parameters:
   No thresholds, no lambdas, no Top-K windowing, no normalization, and no tuning.
   If trace evidence is absent, the ranking degrades to the original S_comb ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .rca import RootCauseScore
from .trace_attribution import compute_edge_elevation
from .traces import TraceLatencyResult


@dataclass(frozen=True)
class RankedEntity:
    """Standard ranked entity with score and 1-based rank."""

    entity: str
    score: float
    rank: int


@dataclass(frozen=True)
class CandidateTieBreakDetail:
    """Detailed provenance for one candidate entity under trace tie-breaking."""

    entity: str
    s_comb_score: float
    s_comb_rank: int
    e_elev: float
    final_rank: int
    was_tied: bool
    rank_changed: bool


@dataclass(frozen=True)
class TraceTieBreakResult:
    """Complete immutable result of trace tie-breaking for one case."""

    case_id: str
    control_ranking: tuple[RankedEntity, ...]
    treatment_ranking: tuple[RankedEntity, ...]
    candidate_details: tuple[CandidateTieBreakDetail, ...]
    ties_resolved_count: int
    order_changed: bool


def extract_inbound_elevations(
    trace_latency: TraceLatencyResult | None,
    candidate_universe: Sequence[str] | None = None,
) -> dict[str, float]:
    """Extract maximum inbound edge elevation E_elev for each candidate entity.

    Parameters
    ----------
    trace_latency:
        Optional TraceLatencyResult produced by extract_trace_latency_evidence.
    candidate_universe:
        Optional sequence of candidate entities.

    Returns
    -------
    dict[str, float]
        Mapping from entity name to E_elev in [0.0, 1.0].
    """
    elev_by_callee: dict[str, float] = {}
    if trace_latency is not None and trace_latency.edges:
        for edge in trace_latency.edges:
            p90 = float(edge.caller_duration_dist.p90)
            p10 = float(edge.caller_duration_dist.p10)
            elev = compute_edge_elevation(p90, p10)
            v = edge.callee_service
            if v not in elev_by_callee or elev > elev_by_callee[v]:
                elev_by_callee[v] = elev

    if candidate_universe is not None:
        return {ent: elev_by_callee.get(ent, 0.0) for ent in candidate_universe}
    return elev_by_callee


def apply_trace_tiebreak(
    s_comb_ranking: Sequence[RankedEntity | RootCauseScore],
    trace_evidence: TraceLatencyResult | Mapping[str, float] | None = None,
    case_id: str = "UNKNOWN_CASE",
) -> TraceTieBreakResult:
    """Apply parameter-free trace tie-breaking to an S_comb ranking.

    Parameters
    ----------
    s_comb_ranking:
        Original control ranking from rank_with_s_comb.
    trace_evidence:
        Either a TraceLatencyResult or a precomputed mapping of entity -> E_elev.
        If None, ranking degrades to original S_comb ranking.
    case_id:
        Case identifier for provenance tracking.

    Returns
    -------
    TraceTieBreakResult
        Structured output containing control ranking, treatment ranking, and diagnostics.
    """
    # Standardize input control ranking
    control_list: list[RankedEntity] = []
    for idx, item in enumerate(s_comb_ranking, start=1):
        if hasattr(item, "entity") and hasattr(item, "score"):
            control_list.append(RankedEntity(entity=str(item.entity), score=float(item.score), rank=idx))
        elif hasattr(item, "entity"):
            control_list.append(RankedEntity(entity=str(item.entity), score=0.0, rank=idx))
        else:
            control_list.append(RankedEntity(entity=str(item), score=0.0, rank=idx))

    universe = tuple(r.entity for r in control_list)

    # Extract E_elev mapping
    if isinstance(trace_evidence, TraceLatencyResult):
        elev_map = extract_inbound_elevations(trace_evidence, candidate_universe=universe)
    elif isinstance(trace_evidence, Mapping):
        elev_map = {ent: float(trace_evidence.get(ent, 0.0)) for ent in universe}
    else:
        elev_map = {ent: 0.0 for ent in universe}

    # Identify exact ties in S_comb scores
    score_counts: dict[float, int] = {}
    for r in control_list:
        score_counts[r.score] = score_counts.get(r.score, 0) + 1

    # Sorting key:
    # 1. -s_comb_score (primary: strictly preserves non-tied ordering)
    # 2. -e_elev (secondary: breaks exact ties by descending trace elevation)
    # 3. original_rank (tertiary: deterministic fallback preserving original S_comb tie-break)
    def tiebreak_sort_key(r: RankedEntity) -> tuple[float, float, int]:
        return (-r.score, -elev_map.get(r.entity, 0.0), r.rank)

    sorted_treatment = sorted(control_list, key=tiebreak_sort_key)

    treatment_ranking = tuple(
        RankedEntity(entity=r.entity, score=r.score, rank=new_rank)
        for new_rank, r in enumerate(sorted_treatment, start=1)
    )

    # Build provenance details
    details: list[CandidateTieBreakDetail] = []
    ties_resolved = 0
    for r_treat in treatment_ranking:
        orig = next(r for r in control_list if r.entity == r_treat.entity)
        is_tied = score_counts.get(orig.score, 0) > 1
        changed = orig.rank != r_treat.rank
        if is_tied and changed:
            ties_resolved += 1

        details.append(
            CandidateTieBreakDetail(
                entity=r_treat.entity,
                s_comb_score=orig.score,
                s_comb_rank=orig.rank,
                e_elev=elev_map.get(r_treat.entity, 0.0),
                final_rank=r_treat.rank,
                was_tied=is_tied,
                rank_changed=changed,
            )
        )

    order_changed = tuple(r.entity for r in control_list) != tuple(r.entity for r in treatment_ranking)

    return TraceTieBreakResult(
        case_id=case_id,
        control_ranking=tuple(control_list),
        treatment_ranking=treatment_ranking,
        candidate_details=tuple(details),
        ties_resolved_count=ties_resolved,
        order_changed=order_changed,
    )
