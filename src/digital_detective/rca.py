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


@dataclass(frozen=True)
class AggregateRcaEvaluationResult:
    """Aggregate benchmark evaluation metrics across multiple cases."""

    total_cases: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float


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
    ranking: Sequence[RootCauseScore],
    ground_truth_root_cause: str | Sequence[str],
    case_id: str | None = None,
) -> CaseRcaEvaluationResult:
    """Evaluate a predicted entity ranking against ground-truth root cause(s).

    Parameters
    ----------
    ranking:
        Candidate entity ranking ordered descending by score.
    ground_truth_root_cause:
        True root cause service/entity name or sequence of names.
    case_id:
        Optional case identifier for tracking.

    Returns
    -------
    CaseRcaEvaluationResult
        Standard Top-1/3/5 and reciprocal rank metrics.
    """
    if isinstance(ground_truth_root_cause, str):
        targets = (ground_truth_root_cause,)
    else:
        targets = tuple(ground_truth_root_cause)

    target_set = set(targets)
    predicted_entities = tuple(s.entity for s in ranking)

    first_hit_rank: int | None = None
    for rank_idx, entity in enumerate(predicted_entities, start=1):
        if entity in target_set:
            first_hit_rank = rank_idx
            break

    top1 = first_hit_rank is not None and first_hit_rank <= 1
    top3 = first_hit_rank is not None and first_hit_rank <= 3
    top5 = first_hit_rank is not None and first_hit_rank <= 5
    rr = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0

    return CaseRcaEvaluationResult(
        case_id=case_id,
        root_cause=targets,
        top1=top1,
        top3=top3,
        top5=top5,
        reciprocal_rank=rr,
        first_hit_rank=first_hit_rank,
        predicted_ranking=predicted_entities,
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
        Summary metrics including Top-1, Top-3, Top-5 accuracy and MRR.
    """
    total = len(case_results)
    if total == 0:
        return AggregateRcaEvaluationResult(
            total_cases=0,
            top1_accuracy=0.0,
            top3_accuracy=0.0,
            top5_accuracy=0.0,
            mrr=0.0,
        )

    top1_acc = sum(1 for r in case_results if r.top1) / total
    top3_acc = sum(1 for r in case_results if r.top3) / total
    top5_acc = sum(1 for r in case_results if r.top5) / total
    mrr = sum(r.reciprocal_rank for r in case_results) / total

    return AggregateRcaEvaluationResult(
        total_cases=total,
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        top5_accuracy=top5_acc,
        mrr=mrr,
    )
