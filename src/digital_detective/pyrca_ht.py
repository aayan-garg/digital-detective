"""PyRCA Hypothesis Testing (HT) Root Cause Analysis Backend.

Provides a minimal, strictly isolated adapter for Salesforce PyRCA's
Hypothesis Testing (HT) RCA algorithm.

Responsibilities:
1. Convert Digital Detective EntityGraph into PyRCA adjacency matrix contract:
   G[i, j] = 1 means directed edge i -> j (node j uses node i as parent/regressor).
2. Extract strictly causal normal and incident telemetry dataframes:
   Normal history strictly precedes the confirmed incident boundary.
   Zero access to inject_time, ground truth labels, or future telemetry.
3. Train HT regressions and compute absolute standardized residual scores:
   aggregator="max", adjustment=False.
4. Project scores onto the canonical candidate universe with deterministic tie-breaking.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from pyrca.analyzers.ht import HT, HTConfig
except ImportError:
    HT = None
    HTConfig = None

@dataclass(frozen=True)
class RankedEntity:
    """Ranked candidate entity with score and 1-based rank."""

    entity: str
    score: float
    rank: int


from .anomaly import _extract_series
from .telemetry import TelemetryCase
from .topology import EntityGraph


def convert_entity_graph_to_pyrca_adjacency(
    graph: EntityGraph,
) -> tuple[pd.DataFrame, str]:
    """Convert an EntityGraph to a PyRCA HT adjacency matrix and compute its SHA-256 hash.

    Matrix Orientation Contract:
        G[i, j] = 1 represents edge i -> j.
        In PyRCA HT, node j uses node i as its regression parent:
        self.graph.predecessors(j) contains i.
        LinearRegression fits normal_df[parents] -> normal_df[j].

    Returns
    -------
    tuple[pd.DataFrame, str]
        (adjacency_df, graph_hash_hex)
    """
    if pd is None:
        raise RuntimeError("pandas is required for pyrca_ht but is not installed in the current environment.")

    nodes = sorted(graph.entities.keys())
    adj = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)

    for dep in graph.dependencies:
        if dep.source in adj.index and dep.target in adj.columns:
            adj.loc[dep.source, dep.target] = 1

    # Deterministic SHA-256 hash of the graph structure
    csv_bytes = adj.to_csv().encode("utf-8")
    graph_hash = hashlib.sha256(csv_bytes).hexdigest()[:16]

    return adj, graph_hash


def build_entity_telemetry_frames(
    case: TelemetryCase,
    graph: EntityGraph,
    *,
    incident_start_ts: int,
    incident_end_ts: int,
    normal_start_ts: int | None = None,
    normal_end_ts: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract causal normal and incident telemetry dataframes for graph entity nodes.

    Strict Causality & Safeguards:
    - Normal data is strictly restricted to timestamps < incident_start_ts
      (or < normal_end_ts if explicitly provided).
    - Incident data is strictly restricted to [incident_start_ts, incident_end_ts].
    - Telemetry strictly after incident_end_ts is NEVER accessed or evaluated.
    - Zero access to ground-truth inject_time or root-cause labels.

    Normalization Policy:
    - Each metric owned by an entity is standardized using the mean and standard
      deviation computed strictly on the normal training window.
    - Zero-variance metrics receive std=1.0 and zero normalized value.
    - The entity signal at discrete time t is the arithmetic mean of its
      standardized metrics at time t.
    """
    if pd is None:
        raise RuntimeError("pandas is required for pyrca_ht but is not installed in the current environment.")

    if case.metrics is None:
        raise ValueError(f"Case {case.metadata.case_id!r} has no metrics modality")

    ts_field = case.metrics.provenance.timestamp_field or "time"
    timestamps, metric_series = _extract_series(case.metrics.raw_data, ts_field)

    num_obs = len(timestamps)
    if num_obs == 0:
        raise ValueError(f"Case {case.metadata.case_id!r} has empty metrics timestamps")

    # Determine window boundaries
    eff_norm_end = incident_start_ts if normal_end_ts is None else normal_end_ts
    if incident_end_ts < incident_start_ts:
        raise ValueError(f"incident_end_ts ({incident_end_ts}) < incident_start_ts ({incident_start_ts})")

    # Boolean masks over timestamps
    ts_arr = np.array(timestamps)
    if normal_start_ts is not None:
        norm_mask = (ts_arr >= normal_start_ts) & (ts_arr < eff_norm_end)
    else:
        norm_mask = ts_arr < eff_norm_end

    inc_mask = (ts_arr >= incident_start_ts) & (ts_arr <= incident_end_ts)

    n_norm = int(np.sum(norm_mask))
    n_inc = int(np.sum(inc_mask))

    if n_norm < 2:
        raise ValueError(
            f"Insufficient normal telemetry samples ({n_norm}) strictly before incident start ({incident_start_ts})"
        )
    if n_inc < 1:
        raise ValueError(
            f"Zero incident telemetry samples in [{incident_start_ts}, {incident_end_ts}]"
        )

    nodes = sorted(graph.entities.keys())
    norm_dict: dict[str, np.ndarray] = {}
    inc_dict: dict[str, np.ndarray] = {}

    for ent in nodes:
        node = graph.entities[ent]
        available_metrics = [m for m in node.metrics if m in metric_series]

        if not available_metrics:
            norm_dict[ent] = np.zeros(n_norm, dtype=float)
            inc_dict[ent] = np.zeros(n_inc, dtype=float)
            continue

        z_norm_list: list[np.ndarray] = []
        z_inc_list: list[np.ndarray] = []

        for m in available_metrics:
            vals = np.array([v if v is not None else np.nan for v in metric_series[m]], dtype=float)
            v_norm = vals[norm_mask]
            v_inc = vals[inc_mask]

            # Standardize strictly using normal statistics
            valid_norm = v_norm[np.isfinite(v_norm)]
            if len(valid_norm) > 0:
                mu = float(np.mean(valid_norm))
                std = float(np.std(valid_norm))
            else:
                mu = 0.0
                std = 1.0

            if math.isnan(std) or std < 1e-6:
                std = 1.0

            z_norm = np.nan_to_num((v_norm - mu) / std, nan=0.0)
            z_inc = np.nan_to_num((v_inc - mu) / std, nan=0.0)

            z_norm_list.append(z_norm)
            z_inc_list.append(z_inc)

        norm_dict[ent] = np.mean(z_norm_list, axis=0)
        inc_dict[ent] = np.mean(z_inc_list, axis=0)

    normal_df = pd.DataFrame(norm_dict)
    incident_df = pd.DataFrame(inc_dict)

    return normal_df, incident_df


def run_pyrca_ht(
    normal_df: pd.DataFrame,
    incident_df: pd.DataFrame,
    graph_adj: pd.DataFrame,
    candidate_universe: Sequence[str],
    *,
    aggregator: str = "max",
    adjustment: bool = False,
) -> tuple[RankedEntity, ...]:
    """Execute PyRCA Hypothesis Testing (HT) and project rankings onto candidate universe.

    Parameters
    ----------
    normal_df:
        DataFrame of pre-incident normal data with columns matching graph_adj.
    incident_df:
        DataFrame of confirmed incident data with columns matching graph_adj.
    graph_adj:
        Adjacency matrix where G[i, j] = 1 represents edge i -> j (j has parent i).
    candidate_universe:
        Canonical candidate universe for root cause projection.
    aggregator:
        Residual aggregation strategy ("max" by specification).
    adjustment:
        Descendant adjustment flag (False by specification).

    Returns
    -------
    tuple[RankedEntity, ...]
        Deterministically ordered ranking of candidates.
    """
    if HT is None or HTConfig is None:
        raise RuntimeError(
            "PyRCA is not installed in the current environment. "
            "Please run using the isolated PyRCA environment (.venv_pyrca)."
        )

    config = HTConfig(
        graph=graph_adj,
        aggregator=aggregator,
        root_cause_top_k=len(graph_adj.columns),
    )
    model = HT(config)
    model.train(normal_df)
    results = model.find_root_causes(incident_df, adjustment=adjustment)

    # Extract raw residual scores from PyRCA results
    raw_scores: dict[str, float] = {}
    for node, score in results.root_cause_nodes:
        raw_scores[node] = float(score)

    # Project onto canonical candidate universe
    # Candidates not present in the graph or unscored receive 0.0
    canonical_candidates = sorted(set(candidate_universe))
    projected: list[tuple[str, float]] = []
    for cand in canonical_candidates:
        score = raw_scores.get(cand, 0.0)
        projected.append((cand, score))

    # Deterministic tie-breaking: descending by score, ascending by candidate name
    projected.sort(key=lambda item: (-item[1], item[0]))

    ranking = tuple(
        RankedEntity(entity=cand, score=score, rank=r_idx)
        for r_idx, (cand, score) in enumerate(projected, start=1)
    )
    return ranking
