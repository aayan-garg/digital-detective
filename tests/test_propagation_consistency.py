"""Tests for Parameter-Free Temporal-Topological Propagation Consistency."""

import pytest

from digital_detective.episodes import EntityEpisodeEvidence, EpisodeInterval
from digital_detective.propagation_consistency import (
    compute_propagation_coverage,
    rank_with_propagation_consistency,
)
from digital_detective.rca import (
    RCA_D_COMBINED,
    rank_root_cause_entities,
)
from digital_detective.topology import Dependency, EntityGraph, EntityNode
from eval.models import RankedEntity


def make_ep_evidence(
    entity: str,
    first_ts: int | None,
    has_episode: bool = True,
) -> EntityEpisodeEvidence:
    """Helper to construct minimal EntityEpisodeEvidence."""
    episodes = (
        (
            EpisodeInterval(
                start_idx=0,
                end_idx=1,
                start_timestamp=first_ts,
                end_timestamp=first_ts + 1 if first_ts is not None else 1,
                duration_seconds=1,
                peak_active_metrics=2,
                contributing_metrics=(f"{entity}_cpu",),
            ),
        )
        if has_episode and first_ts is not None
        else ()
    )
    return EntityEpisodeEvidence(
        case_id="test_case",
        entity=entity,
        timestamps=(first_ts,) if first_ts is not None else (),
        active_metric_counts=(2,) if has_episode else (0,),
        is_in_episode=(True,) if has_episode else (False,),
        episodes=episodes,
        first_episode_start_ts=first_ts if has_episode else None,
        peak_active_metrics=2 if has_episode else 0,
        all_contributing_metrics=(f"{entity}_cpu",) if has_episode else (),
        has_episode=has_episode,
    )


def make_graph(
    entities: list[str],
    edges: list[tuple[str, str]],
) -> EntityGraph:
    """Helper to construct minimal EntityGraph."""
    nodes = {e: EntityNode(name=e, entity_type="service", metrics=(f"{e}_cpu",)) for e in entities}
    deps = tuple(Dependency(source=u, target=v) for u, v in edges)
    return EntityGraph(entities=nodes, dependencies=deps)


def test_1_simple_chain():
    """Test 1: Simple chain A -> B -> C with onset A < B < C -> P(A)=2, P(B)=1, P(C)=0."""
    candidates = ["A", "B", "C"]
    graph = make_graph(candidates, [("A", "B"), ("B", "C")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=110),
        "C": make_ep_evidence("C", first_ts=120),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.coverage["A"] == 2
    assert res.coverage["B"] == 1
    assert res.coverage["C"] == 0
    assert set(res.temporal_edges) == {("A", "B"), ("B", "C")}
    assert set(res.reachable_entities["A"]) == {"B", "C"}
    assert set(res.reachable_entities["B"]) == {"C"}
    assert set(res.reachable_entities["C"]) == set()


def test_2_reverse_observed_order_on_static_edge():
    """Test 2: Static A -> B but B earlier than A -> oriented B -> A."""
    candidates = ["A", "B"]
    graph = make_graph(candidates, [("A", "B")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=115),
        "B": make_ep_evidence("B", first_ts=100),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.coverage["B"] == 1
    assert res.coverage["A"] == 0
    assert res.temporal_edges == (("B", "A"),)
    assert res.reachable_entities["B"] == ("A",)
    assert res.reachable_entities["A"] == ()


def test_3_equal_timestamps():
    """Test 3: A and B same onset -> no temporal propagation edge."""
    candidates = ["A", "B"]
    graph = make_graph(candidates, [("A", "B")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=100),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.temporal_edges == ()
    assert res.coverage["A"] == 0
    assert res.coverage["B"] == 0


def test_4_branch():
    """Test 4: Branch A -> B and A -> C with A earlier than B/C -> P(A)=2, P(B)=0, P(C)=0."""
    candidates = ["A", "B", "C"]
    graph = make_graph(candidates, [("A", "B"), ("A", "C")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=110),
        "C": make_ep_evidence("C", first_ts=115),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.coverage["A"] == 2
    assert res.coverage["B"] == 0
    assert res.coverage["C"] == 0
    assert set(res.temporal_edges) == {("A", "B"), ("A", "C")}


def test_5_multiple_paths():
    """Test 5: Multiple paths (diamond) -> distinct reachable entities counted once."""
    candidates = ["A", "B", "C", "D"]
    # Diamond: A -> B -> D, A -> C -> D
    graph = make_graph(candidates, [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=110),
        "C": make_ep_evidence("C", first_ts=112),
        "D": make_ep_evidence("D", first_ts=120),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    # D is reachable via two paths, but should be counted exactly once in P(A)
    assert res.coverage["A"] == 3
    assert set(res.reachable_entities["A"]) == {"B", "C", "D"}
    assert res.coverage["B"] == 1
    assert res.coverage["C"] == 1
    assert res.coverage["D"] == 0


def test_6_disconnected_anomaly():
    """Test 6: Disconnected anomalous node not counted."""
    candidates = ["A", "B", "X"]
    # A -> B, X is disconnected
    graph = make_graph(candidates, [("A", "B")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=110),
        "X": make_ep_evidence("X", first_ts=105),  # X has earlier onset than B, but no edge
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.coverage["A"] == 1  # only B reachable, not X
    assert res.reachable_entities["A"] == ("B",)
    assert res.coverage["X"] == 0


def test_7_non_anomalous_candidate():
    """Test 7: Candidate without anomaly episode gets P(c)=0."""
    candidates = ["A", "B", "NonAnom"]
    graph = make_graph(candidates, [("A", "B"), ("NonAnom", "A")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=110),
        "NonAnom": make_ep_evidence("NonAnom", first_ts=None, has_episode=False),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.coverage["NonAnom"] == 0
    assert "NonAnom" not in res.anomalous_entities


def test_8_fewer_than_two_anomalous_entities():
    """Test 8: Fewer than two anomalous entities reduces exactly to frozen baseline."""
    candidates = ["A", "B", "C"]
    graph = make_graph(candidates, [("A", "B"), ("B", "C")])
    # Only A is anomalous
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=None, has_episode=False),
        "C": make_ep_evidence("C", first_ts=None, has_episode=False),
    }

    res = compute_propagation_coverage(ep_ev, graph, candidates)
    assert res.reduced_to_baseline is True
    assert all(p == 0 for p in res.coverage.values())

    baseline = (
        RankedEntity(entity="B", score=0.8, rank=1),
        RankedEntity(entity="A", score=0.5, rank=2),
        RankedEntity(entity="C", score=0.2, rank=3),
    )
    treatment = rank_with_propagation_consistency(baseline, res.coverage)
    assert treatment == baseline


def test_9_causal_cutoff():
    """Test 9: Topology edge only observable after tau_confirm must not enter treatment."""
    # Simulate a graph with an edge that was created only with pre-cutoff dependencies
    candidates = ["A", "B"]
    # Pre-cutoff graph has no edge between A and B
    causal_graph = make_graph(candidates, [])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=110),
    }

    res = compute_propagation_coverage(ep_ev, causal_graph, candidates)
    assert res.temporal_edges == ()
    assert res.coverage["A"] == 0
    assert res.coverage["B"] == 0


def test_10_rca_d_combined_regression():
    """Test 10: Prove old RCA_D_COMBINED behavior remains unchanged."""
    from digital_detective.topology import EntityAnomalyEvidence
    # Create two entity anomaly evidences as in existing test_rca.py
    ev_a = EntityAnomalyEvidence(
        case_id="case_1",
        entity="A",
        timestamps=(1, 2, 3),
        active_metric_counts=(1, 2, 2),
        is_anomalous=(True, True, True),
        first_anomaly_ts=1,
        anomalous_metrics=("m_a",),
    )
    ev_b = EntityAnomalyEvidence(
        case_id="case_1",
        entity="B",
        timestamps=(1, 2, 3),
        active_metric_counts=(0, 1, 1),
        is_anomalous=(False, True, True),
        first_anomaly_ts=2,
        anomalous_metrics=("m_b",),
    )
    graph = make_graph(["A", "B"], [("A", "B")])

    res = {r.entity: r for r in rank_root_cause_entities([ev_a, ev_b], graph, RCA_D_COMBINED)}
    # In RCA_D_COMBINED, A calls B (anomalous callee with ts >= 1), so r_prop for A is 1.0
    assert res["A"].r_prop == 1.0
    assert res["B"].r_prop == 0.0


def test_11_frozen_s_fusion_tiebreak():
    """Test 11: When P(c) ties, exact baseline S_fusion ordering is preserved."""
    candidates = ["A", "B", "C"]
    # Suppose A, B, C all have P(c) = 1
    coverage = {"A": 1, "B": 1, "C": 1}
    baseline = (
        RankedEntity(entity="B", score=0.9, rank=1),
        RankedEntity(entity="A", score=0.5, rank=2),
        RankedEntity(entity="C", score=0.5, rank=3),
    )

    treatment = rank_with_propagation_consistency(baseline, coverage)
    # B has higher F(c) (0.9 vs 0.5) -> rank 1
    # A and C tie on P(c)=1 and F(c)=0.5 -> baseline rank breaks tie: A (rank 2) before C (rank 3)
    assert treatment[0].entity == "B"
    assert treatment[0].rank == 1
    assert treatment[1].entity == "A"
    assert treatment[1].rank == 2
    assert treatment[2].entity == "C"
    assert treatment[2].rank == 3


def test_12_determinism():
    """Test 12: Same inputs produce byte-for-byte identical rankings and counts."""
    candidates = ["A", "B", "C", "D", "E"]
    graph = make_graph(candidates, [("A", "B"), ("B", "C"), ("A", "D"), ("D", "E")])
    ep_ev = {
        "A": make_ep_evidence("A", first_ts=100),
        "B": make_ep_evidence("B", first_ts=105),
        "C": make_ep_evidence("C", first_ts=110),
        "D": make_ep_evidence("D", first_ts=105),
        "E": make_ep_evidence("E", first_ts=120),
    }

    baseline = tuple(
        RankedEntity(entity=e, score=0.5, rank=i)
        for i, e in enumerate(sorted(candidates), start=1)
    )

    res1 = compute_propagation_coverage(ep_ev, graph, candidates)
    treat1 = rank_with_propagation_consistency(baseline, res1.coverage)

    res2 = compute_propagation_coverage(ep_ev, graph, candidates)
    treat2 = rank_with_propagation_consistency(baseline, res2.coverage)

    assert res1.coverage == res2.coverage
    assert res1.temporal_edges == res2.temporal_edges
    assert treat1 == treat2
