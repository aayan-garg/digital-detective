"""Parameter-Free Temporal-Topological Propagation Consistency for RCA.

Implements observational propagation consistency by orienting static topology edges
according to first observed anomaly onset timestamps within the strictly causal prefix.

Formal Definition:
------------------
Let A be anomalous entities observable within X_<= tau_confirm:
For each anomalous entity v in A:
    tau_v = first episode/anomaly onset available within the causal prefix.

Static topology is used strictly as an undirected adjacency relation.
Do not assume static caller -> callee direction is causal.

For each topology-connected anomalous pair {u, v}:
    if tau_u < tau_v:
        add temporal propagation edge u -> v
    elif tau_v < tau_u:
        add temporal propagation edge v -> u
    elif tau_u == tau_v:
        add no propagation edge.

The resulting graph is strictly time-increasing and therefore a DAG (acyclic).

Propagation Coverage P(c):
--------------------------
For candidate c:
    P(c) = number of DISTINCT anomalous entities reachable from c
           through the temporally oriented propagation graph.
    Exclude c itself from the count.
    If c is not anomalous: P(c) = 0.
    If fewer than two anomalous entities exist: P(c) = 0 for all candidates.

Final Ranking:
--------------
Let F(c) = frozen S_fusion(c) = 0.5 * S_comb(c) + 0.5 * E_elev(c).
Lexicographic ordering:
1. higher P(c) first
2. if P(c) ties, higher F(c) first
3. if both tie, preserve existing deterministic frozen tie-break.

Note on Terminology:
--------------------
This is observational propagation consistency, not formal causal inference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .episodes import EntityEpisodeEvidence
from .topology import EntityGraph
from eval.models import RankedEntity


@dataclass(frozen=True)
class PropagationCoverageResult:
    """Provenance and results of temporal-topological propagation coverage."""

    coverage: Mapping[str, int]
    temporal_edges: tuple[tuple[str, str], ...]
    anomalous_entities: tuple[str, ...]
    first_onsets: Mapping[str, int]
    reachable_entities: Mapping[str, tuple[str, ...]]
    reduced_to_baseline: bool


def compute_propagation_coverage(
    ep_evidence: Mapping[str, EntityEpisodeEvidence],
    graph: EntityGraph | None,
    candidate_universe: Sequence[str],
) -> PropagationCoverageResult:
    """Compute parameter-free temporal propagation coverage P(c) for all candidates.

    Parameters
    ----------
    ep_evidence:
        Causal entity episode evidence observable within X_<= tau_confirm.
    graph:
        Causal EntityGraph observable within X_<= tau_confirm.
    candidate_universe:
        Closed set of candidate entities.

    Returns
    -------
    PropagationCoverageResult:
        Immutable container with P(c) per candidate and structured provenance.
    """
    candidates = tuple(sorted(set(candidate_universe)))

    # 1. Identify anomalous entities and their earliest onset tau_v
    anom_entities: list[str] = []
    first_onsets: dict[str, int] = {}

    for c in candidates:
        ev = ep_evidence.get(c)
        if ev is not None and ev.has_episode and ev.first_episode_start_ts is not None:
            anom_entities.append(c)
            first_onsets[c] = int(ev.first_episode_start_ts)

    anom_set = set(anom_entities)

    # If fewer than two anomalous entities exist:
    # P(c) = 0 for all candidates and treatment reduces exactly to frozen baseline.
    if len(anom_entities) < 2:
        return PropagationCoverageResult(
            coverage={c: 0 for c in candidates},
            temporal_edges=(),
            anomalous_entities=tuple(anom_entities),
            first_onsets=first_onsets,
            reachable_entities={c: () for c in candidates},
            reduced_to_baseline=True,
        )

    # 2. Extract undirected topology adjacency between anomalous entities
    adj: dict[str, set[str]] = {e: set() for e in anom_entities}
    if graph is not None:
        for dep in graph.dependencies:
            src, tgt = dep.source, dep.target
            if src in anom_set and tgt in anom_set and src != tgt:
                adj[src].add(tgt)
                adj[tgt].add(src)

    # 3. Orient topology edges temporally: earlier -> later
    # Iterate unique unordered pairs {u, v}
    temporal_edges_list: list[tuple[str, str]] = []
    outgoing: dict[str, set[str]] = {e: set() for e in anom_entities}

    sorted_anom = sorted(anom_entities)
    for i, u in enumerate(sorted_anom):
        for v in sorted_anom[i + 1 :]:
            if v in adj[u]:
                tau_u = first_onsets[u]
                tau_v = first_onsets[v]
                if tau_u < tau_v:
                    temporal_edges_list.append((u, v))
                    outgoing[u].add(v)
                elif tau_v < tau_u:
                    temporal_edges_list.append((v, u))
                    outgoing[v].add(u)
                # If tau_u == tau_v: add no propagation edge

    temporal_edges = tuple(sorted(temporal_edges_list))

    # 4. Compute reachability in the DAG for each candidate
    coverage: dict[str, int] = {}
    reachable_map: dict[str, tuple[str, ...]] = {}

    for c in candidates:
        if c not in anom_set:
            coverage[c] = 0
            reachable_map[c] = ()
            continue

        visited: set[str] = set()
        queue: deque[str] = deque([c])
        while queue:
            curr = queue.popleft()
            for nxt in outgoing[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        # Exclude c itself from the count
        reachable = visited - {c}
        coverage[c] = len(reachable)
        reachable_map[c] = tuple(sorted(reachable))

    return PropagationCoverageResult(
        coverage=coverage,
        temporal_edges=temporal_edges,
        anomalous_entities=tuple(anom_entities),
        first_onsets=first_onsets,
        reachable_entities=reachable_map,
        reduced_to_baseline=False,
    )


def rank_with_propagation_consistency(
    baseline_ranking: Sequence[RankedEntity],
    coverage: Mapping[str, int],
) -> tuple[RankedEntity, ...]:
    """Rank candidate entities primarily by P(c), tie-breaking by baseline S_fusion.

    Ordering is strictly lexicographic:
    1. higher P(c) first
    2. if P(c) ties, higher F(c) first
    3. if both tie, preserve existing deterministic frozen tie-break

    Parameters
    ----------
    baseline_ranking:
        Candidate ranking from frozen S_fusion, ordered descending with deterministic tie-breaks.
    coverage:
        Propagation coverage P(c) mapping per candidate entity.

    Returns
    -------
    tuple[RankedEntity, ...]
        Treatment ranking ordered by (P(c) descending, F(c) descending, baseline_rank ascending).
    """
    baseline_list = list(baseline_ranking)
    baseline_order = {r.entity: idx for idx, r in enumerate(baseline_list)}

    def sort_key(r: RankedEntity) -> tuple[int, float, int]:
        p_val = coverage.get(r.entity, 0)
        return (-p_val, -r.score, baseline_order[r.entity])

    sorted_entities = sorted(baseline_list, key=sort_key)

    return tuple(
        RankedEntity(entity=r.entity, score=r.score, rank=new_rank)
        for new_rank, r in enumerate(sorted_entities, start=1)
    )
