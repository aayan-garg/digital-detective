"""Counterfactual Removal Validation (CRV).

A second-stage RCA validator that evaluates whether counterfactually removing
a candidate entity's observed disturbance relieves downstream incident symptoms.

Scientific Principles & Protocol:
---------------------------------
1. Hypothesis:
   Among the top-5 candidates ranked by S_comb, candidate-specific counterfactual
   disturbance removal provides incremental discriminatory information.

2. Structural Causal Model (SCM):
   For each graph node i:
       X_i(t) = f_i(Pa_i(t)) + epsilon_i(t)
   Fitted strictly on reference telemetry preceding the confirmed incident boundary (t < t_confirm).

3. Counterfactual Disturbance Removal:
   For candidate c, set epsilon_c^cf(t) = 0 while keeping all other observed disturbances fixed:
       epsilon_j^cf(t) = epsilon_hat_j(t)  for j != c
   Propagate forward through the directed graph in topological order:
       X_j^cf(t) = f_hat_j(Pa_j^cf(t)) + epsilon_hat_j(t)

4. Target Relief Metric:
   For incident symptom target Y with reference statistics (mu_Y, sigma_Y):
       R_c = mean_t [ (|Y(t) - mu_Y| - |Y_c^cf(t) - mu_Y|) / sigma_Y ]
   If candidate c has no directed path to target Y, R_c = 0.0.

5. Shadow Reranking:
   Reranks the top-5 candidates descending by R_c, tie-breaking by original S_comb rank.
   Candidates 6..N retain their relative positions.
   Zero score fusion (no S_comb + lambda * R_c).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .anomaly import _extract_series
from .telemetry import TelemetryCase
from .topology import EntityGraph


@dataclass(frozen=True)
class RankedEntity:
    """Ranked candidate entity with score and 1-based rank."""

    entity: str
    score: float
    rank: int


@dataclass(frozen=True)
class CRVCandidateValidation:
    """Detailed validation output for a single candidate."""

    candidate: str
    original_rank: int
    relief_score: float
    has_target_path: bool
    counterfactual_rank: int
    factual_target_mean_deviation: float
    cf_target_mean_deviation: float


@dataclass(frozen=True)
class CRVResult:
    """Immutable result of Counterfactual Removal Validation for one execution."""

    case_id: str
    target_entity: str
    original_ranking: tuple[RankedEntity, ...]
    shadow_ranking: tuple[RankedEntity, ...]
    validations: tuple[CRVCandidateValidation, ...]
    order_changed: bool
    top1_changed: bool
    graph_hash: str
    top5_candidates: tuple[str, ...]
    no_path_candidates: tuple[str, ...]


class _LinearStructuralModel:
    """Deterministic linear structural equation for a single DAG node."""

    def __init__(self, node: str, parents: tuple[str, ...]) -> None:
        self.node = node
        self.parents = parents
        self.intercept: float = 0.0
        self.coefs: np.ndarray = np.empty(0, dtype=float)

    def fit(self, data: Mapping[str, np.ndarray]) -> None:
        y = data[self.node]
        if not self.parents:
            self.intercept = float(np.mean(y))
            self.coefs = np.empty(0, dtype=float)
            return

        # Stack parent design matrix: (T, P)
        X_mat = np.column_stack([data[p] for p in self.parents])
        T, P = X_mat.shape

        # Add bias column: (T, P + 1)
        A = np.column_stack([np.ones(T, dtype=float), X_mat])

        # Regularized OLS for numerical stability: (A^T A + lambda I)^{-1} A^T y
        reg = 1e-5 * np.eye(P + 1, dtype=float)
        reg[0, 0] = 0.0  # Do not regularize bias
        try:
            params = np.linalg.solve(A.T @ A + reg, A.T @ y)
        except np.linalg.LinAlgError:
            params, _, _, _ = np.linalg.lstsq(A, y, rcond=1e-7)

        self.intercept = float(params[0])
        self.coefs = np.asarray(params[1:], dtype=float)

    def predict(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        if not self.parents or len(self.coefs) == 0:
            T = len(next(iter(data.values()))) if data else 1
            return np.full(T, self.intercept, dtype=float)

        X_mat = np.column_stack([data[p] for p in self.parents])
        return self.intercept + X_mat @ self.coefs


def extract_standardized_entity_signals(
    case: TelemetryCase,
    graph: EntityGraph,
    *,
    normal_end_ts: int,
    incident_start_ts: int,
    incident_end_ts: int,
    normal_start_ts: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    """Extract standardized entity-level signals for normal and incident windows.

    Strict Causality:
    - Normal window: t < normal_end_ts (strictly preceding confirmed incident onset).
    - Incident window: [incident_start_ts, incident_end_ts].
    - Normalization parameters (mean, std) are computed ONLY from the normal window.
    - Zero access to future telemetry or ground truth.

    Returns:
    (normal_signals, incident_signals, normal_stats_by_entity)
    """
    if case.metrics is None:
        raise ValueError(f"Case {case.metadata.case_id!r} has no metrics modality")

    ts_field = case.metrics.provenance.timestamp_field or "time"
    timestamps, metric_series = _extract_series(case.metrics.raw_data, ts_field)
    ts_arr = np.asarray(timestamps)

    if normal_start_ts is not None:
        norm_mask = (ts_arr >= normal_start_ts) & (ts_arr < normal_end_ts)
    else:
        norm_mask = ts_arr < normal_end_ts

    inc_mask = (ts_arr >= incident_start_ts) & (ts_arr <= incident_end_ts)

    if int(np.sum(norm_mask)) < 2:
        raise ValueError(f"Insufficient normal telemetry samples before {normal_end_ts}")
    if int(np.sum(inc_mask)) < 1:
        raise ValueError(f"Zero incident telemetry samples in [{incident_start_ts}, {incident_end_ts}]")

    all_nodes = sorted(graph.entities.keys())
    norm_signals: dict[str, np.ndarray] = {}
    inc_signals: dict[str, np.ndarray] = {}
    stats: dict[str, tuple[float, float]] = {}

    for node in all_nodes:
        entity_node = graph.entities[node]
        metrics = [m for m in entity_node.metrics if m in metric_series]

        if not metrics:
            norm_signals[node] = np.zeros(int(np.sum(norm_mask)), dtype=float)
            inc_signals[node] = np.zeros(int(np.sum(inc_mask)), dtype=float)
            stats[node] = (0.0, 1.0)
            continue

        norm_cols: list[np.ndarray] = []
        inc_cols: list[np.ndarray] = []

        for m in metrics:
            raw_s = np.asarray(metric_series[m], dtype=float)
            norm_slice = raw_s[norm_mask]
            inc_slice = raw_s[inc_mask]

            valid_norm = norm_slice[~np.isnan(norm_slice)]
            if len(valid_norm) > 0:
                m_mean = float(np.mean(valid_norm))
                m_std = float(np.std(valid_norm))
            else:
                m_mean = 0.0
                m_std = 1.0

            if not math.isfinite(m_mean):
                m_mean = 0.0
            if not math.isfinite(m_std) or m_std < 1e-6:
                m_std = 1.0

            norm_std_slice = np.nan_to_num((norm_slice - m_mean) / m_std, nan=0.0)
            inc_std_slice = np.nan_to_num((inc_slice - m_mean) / m_std, nan=0.0)

            norm_cols.append(norm_std_slice)
            inc_cols.append(inc_std_slice)

        node_norm_sig = np.mean(norm_cols, axis=0)
        node_inc_sig = np.mean(inc_cols, axis=0)

        norm_signals[node] = node_norm_sig
        inc_signals[node] = node_inc_sig
        stats[node] = (float(np.mean(node_norm_sig)), max(1e-6, float(np.std(node_norm_sig))))

    return norm_signals, inc_signals, stats


def compute_directed_reachability(graph: EntityGraph) -> dict[str, set[str]]:
    """Compute reachable descendants for every node in the DAG via transitive closure."""
    nodes = sorted(graph.entities.keys())
    adj_out: dict[str, list[str]] = {n: [] for n in nodes}
    for dep in graph.dependencies:
        if dep.source in adj_out and dep.target in adj_out:
            adj_out[dep.source].append(dep.target)

    descendants: dict[str, set[str]] = {}
    for start_node in nodes:
        visited: set[str] = set()
        queue = list(adj_out.get(start_node, []))
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                queue.extend(adj_out.get(curr, []))
        descendants[start_node] = visited

    return descendants


def get_topological_sort(graph: EntityGraph) -> tuple[str, ...]:
    """Compute a deterministic topological ordering for the entity DAG."""
    nodes = sorted(graph.entities.keys())
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    adj_out: dict[str, list[str]] = {n: [] for n in nodes}

    for dep in graph.dependencies:
        if dep.source in in_degree and dep.target in in_degree:
            adj_out[dep.source].append(dep.target)
            in_degree[dep.target] += 1

    # Deterministic queue sorted alphabetically
    queue = [n for n in nodes if in_degree[n] == 0]
    queue.sort()

    topo_order: list[str] = []
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        for nxt in sorted(adj_out[curr]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    if len(topo_order) < len(nodes):
        # Fallback if cycles exist: append remaining nodes deterministically
        remaining = sorted(set(nodes) - set(topo_order))
        topo_order.extend(remaining)

    return tuple(topo_order)


def run_counterfactual_removal_validation(
    case: TelemetryCase,
    graph: EntityGraph,
    s_comb_ranking: Sequence[RankedEntity],
    *,
    incident_target: str,
    incident_start_ts: int,
    incident_end_ts: int,
    top_k: int = 5,
) -> CRVResult:
    """Execute Counterfactual Removal Validation over top-K S_comb candidates.

    Parameters
    ----------
    case:
        TelemetryCase containing telemetry observations.
    graph:
        Frozen EntityGraph.
    s_comb_ranking:
        Baseline S_comb ranking tuple over canonical candidates.
    incident_target:
        Fixed incident symptom/output target entity.
    incident_start_ts:
        Confirmed incident onset timestamp (analysis boundary).
    incident_end_ts:
        Confirmed incident end timestamp (analysis boundary).
    top_k:
        Number of top candidates to validate (default: 5).

    Returns
    -------
    CRVResult
        Complete evaluation result including shadow ranking and relief scores.
    """
    case_id = getattr(case.metadata, "case_id", "UNKNOWN_CASE")
    nodes = sorted(graph.entities.keys())

    # 1. Extract normal and incident signals
    norm_signals, inc_signals, normal_stats = extract_standardized_entity_signals(
        case,
        graph,
        normal_end_ts=incident_start_ts,
        incident_start_ts=incident_start_ts,
        incident_end_ts=incident_end_ts,
    )

    # 2. Graph parents and topological ordering
    parents_map: dict[str, tuple[str, ...]] = {}
    for n in nodes:
        parents_map[n] = tuple(sorted(graph.callers_of(n)))

    topo_order = get_topological_sort(graph)
    descendants = compute_directed_reachability(graph)

    # Compute graph SHA-256 hash
    edge_str = ";".join(f"{d.source}->{d.target}" for d in sorted(graph.dependencies, key=lambda x: (x.source, x.target)))
    graph_hash = hashlib.sha256(edge_str.encode("utf-8")).hexdigest()[:16]

    # 3. Fit structural models on reference data strictly
    models: dict[str, _LinearStructuralModel] = {}
    for n in nodes:
        m = _LinearStructuralModel(n, parents_map[n])
        m.fit(norm_signals)
        models[n] = m

    # 4. Compute incident disturbances epsilon_hat_i(t)
    incident_disturbances: dict[str, np.ndarray] = {}
    for n in nodes:
        f_hat_inc = models[n].predict(inc_signals)
        incident_disturbances[n] = inc_signals[n] - f_hat_inc

    # 5. Target baseline statistics
    if incident_target in normal_stats:
        mu_Y, sigma_Y = normal_stats[incident_target]
    else:
        mu_Y, sigma_Y = 0.0, 1.0

    if incident_target in inc_signals:
        factual_target = inc_signals[incident_target]
    else:
        factual_target = np.zeros(len(next(iter(inc_signals.values()))), dtype=float)

    factual_target_dev = float(np.mean(np.abs(factual_target - mu_Y)))

    # 6. Evaluate Counterfactual Removal for Top-K candidates
    valid_s_comb = [r for r in s_comb_ranking if r.entity in graph.entities or True]
    top_candidates = valid_s_comb[:top_k]
    top_candidate_entities = tuple(r.entity for r in top_candidates)

    validations: list[CRVCandidateValidation] = []
    no_path_cands: list[str] = []

    for r_entry in top_candidates:
        cand = r_entry.entity
        has_path = (incident_target in descendants.get(cand, set())) or (cand == incident_target)

        if not has_path or cand not in graph.entities:
            # Candidate has no directed path to target
            no_path_cands.append(cand)
            validations.append(
                CRVCandidateValidation(
                    candidate=cand,
                    original_rank=r_entry.rank,
                    relief_score=0.0,
                    has_target_path=False,
                    counterfactual_rank=r_entry.rank,
                    factual_target_mean_deviation=factual_target_dev,
                    cf_target_mean_deviation=factual_target_dev,
                )
            )
            continue

        # Forward propagate intervention: epsilon_c^cf = 0
        cand_desc = descendants.get(cand, set())
        cf_signals: dict[str, np.ndarray] = {}

        for n in topo_order:
            if n == cand:
                # Remove candidate disturbance: X_c^cf = f_hat_c(Pa_c^cf) + 0
                f_pred = models[n].predict(cf_signals)
                cf_signals[n] = f_pred
            elif n in cand_desc:
                # Downstream node: X_j^cf = f_hat_j(Pa_j^cf) + epsilon_hat_j
                f_pred = models[n].predict(cf_signals)
                cf_signals[n] = f_pred + incident_disturbances[n]
            else:
                # Unaffected node: factual trajectory
                cf_signals[n] = inc_signals[n]

        cf_target = cf_signals.get(incident_target, factual_target)
        cf_target_dev = float(np.mean(np.abs(cf_target - mu_Y)))

        # Target relief: mean_t [ (|Y(t) - mu_Y| - |Y_c^cf(t) - mu_Y|) / sigma_Y ]
        relief = float(np.mean((np.abs(factual_target - mu_Y) - np.abs(cf_target - mu_Y)) / sigma_Y))

        validations.append(
            CRVCandidateValidation(
                candidate=cand,
                original_rank=r_entry.rank,
                relief_score=relief,
                has_target_path=True,
                counterfactual_rank=r_entry.rank,
                factual_target_mean_deviation=factual_target_dev,
                cf_target_mean_deviation=cf_target_dev,
            )
        )

    # 7. Shadow Reranking within Top-K
    # Sort top-K by: descending relief score, ascending original rank (tie-breaker)
    sorted_validations = sorted(
        validations,
        key=lambda v: (-v.relief_score, v.original_rank, v.candidate),
    )

    reranked_top: list[RankedEntity] = []
    final_validations: list[CRVCandidateValidation] = []
    for new_rank, v in enumerate(sorted_validations, start=1):
        reranked_top.append(RankedEntity(entity=v.candidate, score=v.relief_score, rank=new_rank))
        final_validations.append(
            CRVCandidateValidation(
                candidate=v.candidate,
                original_rank=v.original_rank,
                relief_score=v.relief_score,
                has_target_path=v.has_target_path,
                counterfactual_rank=new_rank,
                factual_target_mean_deviation=v.factual_target_mean_deviation,
                cf_target_mean_deviation=v.cf_target_mean_deviation,
            )
        )

    # Append remaining candidates (rank > top_k) preserving original relative order
    remaining_ranked: list[RankedEntity] = []
    for new_rank, r_entry in enumerate(valid_s_comb[top_k:], start=len(reranked_top) + 1):
        remaining_ranked.append(RankedEntity(entity=r_entry.entity, score=r_entry.score, rank=new_rank))

    shadow_ranking = tuple(reranked_top + remaining_ranked)

    # Check whether ordering changed
    original_top_entities = [r.entity for r in top_candidates]
    shadow_top_entities = [r.entity for r in reranked_top]
    order_changed = original_top_entities != shadow_top_entities
    top1_changed = bool(original_top_entities and shadow_top_entities and original_top_entities[0] != shadow_top_entities[0])

    return CRVResult(
        case_id=case_id,
        target_entity=incident_target,
        original_ranking=tuple(s_comb_ranking),
        shadow_ranking=shadow_ranking,
        validations=tuple(final_validations),
        order_changed=order_changed,
        top1_changed=top1_changed,
        graph_hash=graph_hash,
        top5_candidates=top_candidate_entities,
        no_path_candidates=tuple(no_path_cands),
    )
