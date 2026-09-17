"""Interpretable, dependency-aware Root Cause Analysis (RCA) baselines.

Provides deterministic, parameter-free entity ranking using observational evidence:
- R_strength: Peak active metric count normalized across entities.
- R_early: Earliest episode onset (or anomaly onset) normalized over active entities.
- R_coverage: Proportion of observed direct callees that are anomalous or episode-active.
- R_prop: Temporal propagation consistency along observed direct callee edges (observational only, not causal proof).

Evaluation is strictly isolated from ranking to guarantee zero ground-truth leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .anomaly import MetricAnomalyResult
from .episodes import EntityEpisodeEvidence
from .topology import EntityAnomalyEvidence, EntityGraph


@dataclass(frozen=True)
class RCAConfig:
    """Configuration defining enabled observational scoring components."""

    name: str
    use_earliness: bool = False
    use_strength: bool = False
    use_coverage: bool = False
    use_propagation: bool = False


# Four frozen ablation baselines
RCA_A_ANOMALY_ONLY = RCAConfig(
    name="RCA_A_AnomalyOnly",
    use_strength=True,
    use_earliness=False,
    use_coverage=False,
    use_propagation=False,
)

RCA_B_TEMPORAL = RCAConfig(
    name="RCA_B_Temporal",
    use_strength=True,
    use_earliness=True,
    use_coverage=False,
    use_propagation=False,
)

RCA_C_TOPOLOGY_ONLY = RCAConfig(
    name="RCA_C_TopologyOnly",
    use_strength=True,
    use_earliness=False,
    use_coverage=True,
    use_propagation=False,
)

RCA_D_COMBINED = RCAConfig(
    name="RCA_D_Combined",
    use_strength=True,
    use_earliness=True,
    use_coverage=False,
    use_propagation=True,
)


@dataclass(frozen=True)
class RootCauseScore:
    """Detailed score breakdown and ranking metadata for one candidate entity."""

    entity: str
    score: float
    r_early: float
    r_strength: float
    r_coverage: float
    r_prop: float
    first_anomaly_ts: int | None
    peak_score: float


@dataclass(frozen=True)
class CaseRcaEvaluationResult:
    """Evaluation result for one incident case ranking against ground truth."""

    case_id: str | None
    root_cause: tuple[str, ...]
    top1: bool
    top3: bool
    top5: bool
    reciprocal_rank: float
    first_hit_rank: int | None
    predicted_ranking: tuple[str, ...]
    ac1: float = 0.0
    ac2: float = 0.0
    ac3: float = 0.0
    ac4: float = 0.0
    ac5: float = 0.0
    avg5: float = 0.0


@dataclass(frozen=True)
class AggregateRcaEvaluationResult:
    """Aggregate benchmark evaluation metrics across multiple cases."""

    total_cases: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float
    ac1_accuracy: float = 0.0
    ac2_accuracy: float = 0.0
    ac3_accuracy: float = 0.0
    ac4_accuracy: float = 0.0
    ac5_accuracy: float = 0.0
    avg5_accuracy: float = 0.0


def rank_root_cause_entities(
    evidence: (
        Mapping[str, EntityAnomalyEvidence | EntityEpisodeEvidence]
        | Sequence[EntityAnomalyEvidence | EntityEpisodeEvidence]
    ),
    graph: EntityGraph,
    config: RCAConfig,
) -> tuple[RootCauseScore, ...]:
    """Rank candidate entities using purely observational evidence.

    The ranker receives only EntityEpisodeEvidence (or EntityAnomalyEvidence),
    EntityGraph, and RCAConfig. It has no access to ground truth, injection times,
    or fault types.

    Parameters
    ----------
    evidence:
        Entity-level episode evidence (or raw anomaly evidence).
    graph:
        Observed entity topology with nodes and dependency edges.
    config:
        Ablation configuration specifying which components to enable.

    Returns
    -------
    tuple[RootCauseScore, ...]
        Candidate entities sorted deterministically by plausibility.
    """
    if isinstance(evidence, Mapping):
        ev_map: dict[str, Any] = dict(evidence)
    elif isinstance(evidence, Sequence):
        ev_map = {ev.entity: ev for ev in evidence}
    else:
        raise TypeError(
            f"evidence must be Mapping or Sequence of EntityAnomalyEvidence or EntityEpisodeEvidence, got {type(evidence)!r}"
        )

    # Candidate set is the union of known graph entities and evidence entities
    candidate_entities = sorted(set(graph.entities.keys()) | set(ev_map.keys()))

    # 1. Determine anomalous/episode status for each candidate
    is_anom: dict[str, bool] = {}
    raw_peaks: dict[str, float] = {}
    first_ts_map: dict[str, int | None] = {}

    for e in candidate_entities:
        ev = ev_map.get(e)
        if ev is None:
            is_anom[e] = False
            first_ts_map[e] = None
            raw_peaks[e] = 0.0
        elif hasattr(ev, "has_episode"):  # EntityEpisodeEvidence
            if ev.has_episode and ev.first_episode_start_ts is not None:
                is_anom[e] = True
                first_ts_map[e] = ev.first_episode_start_ts
                raw_peaks[e] = float(ev.peak_active_metrics)
            else:
                is_anom[e] = False
                first_ts_map[e] = None
                raw_peaks[e] = 0.0
        else:  # EntityAnomalyEvidence (backwards compatibility)
            if ev.first_anomaly_ts is not None and any(ev.is_anomalous):
                is_anom[e] = True
                first_ts_map[e] = ev.first_anomaly_ts
                raw_peaks[e] = float(max(ev.active_metric_counts)) if ev.active_metric_counts else 0.0
            else:
                is_anom[e] = False
                first_ts_map[e] = None
                raw_peaks[e] = 0.0


    # 2. Compute R_early:
    # - among anomalous entities only
    # - earliest first anomaly gets 1.0
    # - latest gets 0.0
    # - if all anomalous entities share timestamp, all get 1.0
    # - non-anomalous entity gets 0.0
    anomalous_entities = [e for e in candidate_entities if is_anom[e]]
    r_early_map: dict[str, float] = {}

    if anomalous_entities:
        valid_ts = [first_ts_map[e] for e in anomalous_entities if first_ts_map[e] is not None]
        if valid_ts:
            t_min = min(valid_ts)
            t_max = max(valid_ts)
            if t_min == t_max:
                for e in candidate_entities:
                    r_early_map[e] = 1.0 if is_anom[e] else 0.0
            else:
                span = float(t_max - t_min)
                for e in candidate_entities:
                    if is_anom[e] and first_ts_map[e] is not None:
                        r_early_map[e] = 1.0 - (float(first_ts_map[e] - t_min) / span)
                    else:
                        r_early_map[e] = 0.0
        else:
            for e in candidate_entities:
                r_early_map[e] = 0.0
    else:
        for e in candidate_entities:
            r_early_map[e] = 0.0

    # 3. Compute R_strength:
    # - peak anomaly score normalized by maximum peak across candidate entities
    # - non-anomalous = 0.0
    max_peak = max(raw_peaks.values()) if raw_peaks else 0.0
    r_strength_map: dict[str, float] = {}
    if max_peak > 0.0:
        for e in candidate_entities:
            r_strength_map[e] = (raw_peaks[e] / max_peak) if is_anom[e] else 0.0
    else:
        for e in candidate_entities:
            r_strength_map[e] = 0.0

    # 4. Compute R_coverage:
    # - anomalous observed direct callees / observed direct callees
    # - 0.0 if no observed callees
    # - does not use anomaly timing
    r_coverage_map: dict[str, float] = {}
    for e in candidate_entities:
        if not is_anom[e]:
            r_coverage_map[e] = 0.0
            continue
        callees = graph.callees_of(e) if e in graph.entities else ()
        if not callees:
            r_coverage_map[e] = 0.0
        else:
            anom_callees = sum(1 for c in callees if is_anom.get(c, False))
            r_coverage_map[e] = anom_callees / len(callees)

    # 5. Compute R_prop:
    # - among direct observed callees, count those that:
    #   1. are anomalous
    #   2. have first anomaly timestamp >= candidate first anomaly timestamp
    # - divide by number of observed callees
    # - 0.0 if no observed callees
    # - no future/lookahead beyond completed evidence
    r_prop_map: dict[str, float] = {}
    for e in candidate_entities:
        if not is_anom[e]:
            r_prop_map[e] = 0.0
            continue
        cand_ts = first_ts_map[e]
        if cand_ts is None:
            r_prop_map[e] = 0.0
            continue
        callees = graph.callees_of(e) if e in graph.entities else ()
        if not callees:
            r_prop_map[e] = 0.0
        else:
            propagated = sum(
                1
                for c in callees
                if is_anom.get(c, False)
                and first_ts_map.get(c) is not None
                and first_ts_map[c] >= cand_ts  # type: ignore[operator]
            )
            r_prop_map[e] = propagated / len(callees)

    # 6. Assemble RootCauseScore objects
    scores: list[RootCauseScore] = []
    for e in candidate_entities:
        r_early = r_early_map[e]
        r_strength = r_strength_map[e]
        r_coverage = r_coverage_map[e]
        r_prop = r_prop_map[e]

        # Non-anomalous entities receive score 0.0
        if not is_anom[e]:
            total_score = 0.0
        else:
            total_score = 0.0
            if config.use_strength:
                total_score += r_strength
            if config.use_earliness:
                total_score += r_early
            if config.use_coverage:
                total_score += r_coverage
            if config.use_propagation:
                total_score += r_prop

        scores.append(
            RootCauseScore(
                entity=e,
                score=total_score,
                r_early=r_early,
                r_strength=r_strength,
                r_coverage=r_coverage,
                r_prop=r_prop,
                first_anomaly_ts=first_ts_map[e],
                peak_score=raw_peaks[e],
            )
        )

    # 7. Deterministic tie-breaking:
    # 1. score descending
    # 2. earlier first_anomaly_ts (None goes after any valid timestamp)
    # 3. higher raw peak score
    # 4. lexicographical entity name ascending
    def sort_key(s: RootCauseScore) -> tuple[float, float, float, str]:
        ts_val = float(s.first_anomaly_ts) if s.first_anomaly_ts is not None else math.inf
        return (-s.score, ts_val, -s.peak_score, s.entity)

    scores.sort(key=sort_key)
    return tuple(scores)


def evaluate_root_cause_ranking(
    ranking: Sequence[RootCauseScore | str],
    ground_truth_root_cause: str | Sequence[str],
    case_id: str | None = None,
) -> CaseRcaEvaluationResult:
    """Evaluate a predicted entity ranking against ground-truth root cause(s).

    Parameters
    ----------
    ranking:
        Candidate entity ranking ordered descending by score (RootCauseScore objects or entity names).
    ground_truth_root_cause:
        True root cause service/entity name or sequence of names.
    case_id:
        Optional case identifier for tracking.

    Returns
    -------
    CaseRcaEvaluationResult
        Standard Top-1/3/5, AC@1-5, Avg@5, and reciprocal rank metrics.
    """
    if isinstance(ground_truth_root_cause, str):
        targets = (ground_truth_root_cause,)
    else:
        targets = tuple(ground_truth_root_cause)

    target_set = set(targets)
    predicted_entities = tuple(
        s.entity if hasattr(s, "entity") else str(s)
        for s in ranking
    )

    first_hit_rank: int | None = None
    for rank_idx, entity in enumerate(predicted_entities, start=1):
        if entity in target_set:
            first_hit_rank = rank_idx
            break

    top1 = first_hit_rank is not None and first_hit_rank <= 1
    top3 = first_hit_rank is not None and first_hit_rank <= 3
    top5 = first_hit_rank is not None and first_hit_rank <= 5
    rr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    ac1 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 1) else 0.0
    ac2 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 2) else 0.0
    ac3 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 3) else 0.0
    ac4 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 4) else 0.0
    ac5 = 1.0 if (first_hit_rank is not None and first_hit_rank <= 5) else 0.0
    avg5 = (ac1 + ac2 + ac3 + ac4 + ac5) / 5.0

    return CaseRcaEvaluationResult(
        case_id=case_id,
        root_cause=targets,
        top1=top1,
        top3=top3,
        top5=top5,
        reciprocal_rank=rr,
        first_hit_rank=first_hit_rank,
        predicted_ranking=predicted_entities,
        ac1=ac1,
        ac2=ac2,
        ac3=ac3,
        ac4=ac4,
        ac5=ac5,
        avg5=avg5,
    )


def evaluate_rca_benchmark(
    case_results: Sequence[CaseRcaEvaluationResult],
) -> AggregateRcaEvaluationResult:
    """Aggregate evaluation results across multiple cases.

    Parameters
    ----------
    case_results:
        Sequence of per-case evaluation results.

    Returns
    -------
    AggregateRcaEvaluationResult
        Summary metrics including Top-1, Top-3, Top-5 accuracy, AC@1-5, Avg@5, and MRR.
    """
    total = len(case_results)
    if total == 0:
        return AggregateRcaEvaluationResult(
            total_cases=0,
            top1_accuracy=0.0,
            top3_accuracy=0.0,
            top5_accuracy=0.0,
            mrr=0.0,
            ac1_accuracy=0.0,
            ac2_accuracy=0.0,
            ac3_accuracy=0.0,
            ac4_accuracy=0.0,
            ac5_accuracy=0.0,
            avg5_accuracy=0.0,
        )

    top1_acc = sum(1 for r in case_results if r.top1) / total
    top3_acc = sum(1 for r in case_results if r.top3) / total
    top5_acc = sum(1 for r in case_results if r.top5) / total
    mrr = sum(r.reciprocal_rank for r in case_results) / total

    ac1_acc = sum(r.ac1 for r in case_results) / total
    ac2_acc = sum(r.ac2 for r in case_results) / total
    ac3_acc = sum(r.ac3 for r in case_results) / total
    ac4_acc = sum(r.ac4 for r in case_results) / total
    ac5_acc = sum(r.ac5 for r in case_results) / total
    avg5_acc = sum(r.avg5 for r in case_results) / total

    return AggregateRcaEvaluationResult(
        total_cases=total,
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        top5_accuracy=top5_acc,
        mrr=mrr,
        ac1_accuracy=ac1_acc,
        ac2_accuracy=ac2_acc,
        ac3_accuracy=ac3_acc,
        ac4_accuracy=ac4_acc,
        ac5_accuracy=ac5_acc,
        avg5_accuracy=avg5_acc,
    )


def rank_with_s_comb(
    det_res: MetricAnomalyResult,
    ep_evidence: Mapping[str, EntityEpisodeEvidence],
    graph: EntityGraph | None = None,
    candidate_universe: Sequence[str] | None = None,
) -> tuple[RootCauseScore, ...]:
    """Rank candidate entities using the frozen S_comb metric RCA formulation.

    S_comb combines peak active metric count during confirmed episodes with
    log-transformed anomaly magnitude:
        S_comb(e) = 0.5 * norm_count(e) + 0.5 * norm_log_mag(e)  if has_episode else 0.0

    Tie-breaking: (-score, first_episode_start_ts, -peak_score, entity_name).

    Parameters
    ----------
    det_res:
        MetricAnomalyResult containing raw anomaly scores.
    ep_evidence:
        EntityEpisodeEvidence mapping per entity (K=3, M=2 episode filter).
    graph:
        Optional EntityGraph for candidate entity discovery if candidate_universe is None.
    candidate_universe:
        Optional explicit sequence of candidate entities to enforce a closed candidate set.
        If None, candidates are derived from graph and episode evidence.

    Returns
    -------
    tuple[RootCauseScore, ...]
        Candidate entities ranked deterministically descending by S_comb.
    """
    if candidate_universe is not None:
        candidate_entities = sorted(set(candidate_universe))
    elif graph is not None:
        candidate_entities = sorted(set(graph.entities.keys()) | set(ep_evidence.keys()))
    else:
        candidate_entities = sorted(ep_evidence.keys())

    raw_scores: dict[str, dict[str, float]] = {}
    for e in candidate_entities:
        ev = ep_evidence.get(e)
        if ev is None or not ev.has_episode or not ev.episodes:
            raw_scores[e] = {"count": 0.0, "log_mag_peak": 0.0}
            continue

        count_val = float(ev.peak_active_metrics)
        contrib_metrics = ev.all_contributing_metrics
        episode_indices: set[int] = set()
        for ep in ev.episodes:
            episode_indices.update(range(ep.start_idx, ep.end_idx + 1))

        peak_z_by_metric: dict[str, float] = {}
        for m in contrib_metrics:
            scores = det_res.anomaly_scores.get(m, ())
            m_peak = 0.0
            for idx in episode_indices:
                if idx < len(scores) and scores[idx] is not None:
                    val = float(scores[idx])
                    if val > m_peak:
                        m_peak = val
            peak_z_by_metric[m] = m_peak

        log_mag_peak = sum(math.log1p(z) for z in peak_z_by_metric.values())
        raw_scores[e] = {"count": count_val, "log_mag_peak": log_mag_peak}

    c_raw = {e: raw_scores[e]["count"] for e in candidate_entities}
    m_raw = {e: raw_scores[e]["log_mag_peak"] for e in candidate_entities}
    max_c = max(c_raw.values()) if c_raw else 0.0
    max_m = max(m_raw.values()) if m_raw else 0.0
    norm_c = {e: (c_raw[e] / max_c if max_c > 0.0 else 0.0) for e in candidate_entities}
    norm_m = {e: (m_raw[e] / max_m if max_m > 0.0 else 0.0) for e in candidate_entities}

    scores: list[RootCauseScore] = []
    for e in candidate_entities:
        ev = ep_evidence.get(e)
        has_ep = ev.has_episode if ev else False
        first_ts = ev.first_episode_start_ts if (ev and has_ep) else None
        comb_score = (0.5 * norm_c[e] + 0.5 * norm_m[e]) if has_ep else 0.0
        scores.append(
            RootCauseScore(
                entity=e,
                score=comb_score,
                r_early=0.0,
                r_strength=comb_score,
                r_coverage=0.0,
                r_prop=0.0,
                first_anomaly_ts=first_ts,
                peak_score=comb_score,
            )
        )

    def comb_sort_key(s: RootCauseScore) -> tuple[float, float, float, str]:
        ts_val = float(s.first_anomaly_ts) if s.first_anomaly_ts is not None else math.inf
        return (-s.score, ts_val, -s.peak_score, s.entity)

    scores.sort(key=comb_sort_key)
    return tuple(scores)

