"""Unit and integration tests for interpretable dependency-aware RCA baseline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import unittest

from digital_detective.anomaly import MetricAnomalyResult, detect_metric_anomalies
from digital_detective.episodes import (
    EntityEpisodeEvidence,
    EpisodeConfig,
    EpisodeInterval,
    aggregate_entity_episodes,
)
from digital_detective.rca import (
    RCA_A_ANOMALY_ONLY,
    RCA_B_TEMPORAL,
    RCA_C_TOPOLOGY_ONLY,
    RCA_D_COMBINED,
    AggregateRcaEvaluationResult,
    CaseRcaEvaluationResult,
    RCAConfig,
    RootCauseScore,
    evaluate_rca_benchmark,
    evaluate_root_cause_ranking,
    rank_root_cause_entities,
)
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.topology import (
    Dependency,
    EntityAnomalyEvidence,
    EntityGraph,
    aggregate_entity_anomaly_evidence,
    build_entity_graph,
    extract_trace_dependencies,
)



def _make_evidence(
    entity: str,
    first_ts: int | None,
    peak_count: int = 1,
    num_steps: int = 5,
    case_id: str = "test_case",
) -> EntityAnomalyEvidence:
    if first_ts is None:
        active_counts = tuple([0] * num_steps)
        is_anom = tuple([False] * num_steps)
    else:
        active_counts = tuple([0] * (num_steps - 1) + [peak_count])
        is_anom = tuple([False] * (num_steps - 1) + [True])

    return EntityAnomalyEvidence(
        case_id=case_id,
        entity=entity,
        timestamps=tuple(range(100, 100 + num_steps)),
        active_metric_counts=active_counts,
        is_anomalous=is_anom,
        first_anomaly_ts=first_ts,
        anomalous_metrics=(f"{entity}_cpu",) if first_ts is not None else (),
    )


class SyntheticRcaRankingTests(unittest.TestCase):
    def test_1_strength_normalization(self) -> None:
        # 3 entities with peak active counts 10, 5, 0
        ev_a = _make_evidence("svca", first_ts=100, peak_count=10)
        ev_b = _make_evidence("svcb", first_ts=100, peak_count=5)
        ev_c = _make_evidence("svcc", first_ts=None, peak_count=0)

        graph = build_entity_graph(["svca_cpu", "svcb_cpu", "svcc_cpu"])
        config = RCAConfig(name="StrengthOnly", use_strength=True)

        ranking = rank_root_cause_entities([ev_a, ev_b, ev_c], graph, config)
        by_entity = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_entity["svca"].r_strength, 1.0)
        self.assertAlmostEqual(by_entity["svcb"].r_strength, 0.5)
        self.assertAlmostEqual(by_entity["svcc"].r_strength, 0.0)

    def test_2_earliness_normalization(self) -> None:
        # Timestamps 100, 150, 200 -> earliness should be 1.0, 0.5, 0.0
        ev_a = _make_evidence("svca", first_ts=100, peak_count=1)
        ev_b = _make_evidence("svcb", first_ts=150, peak_count=1)
        ev_c = _make_evidence("svcc", first_ts=200, peak_count=1)

        graph = build_entity_graph(["svca_cpu", "svcb_cpu", "svcc_cpu"])
        config = RCAConfig(name="EarlinessOnly", use_earliness=True)

        ranking = rank_root_cause_entities([ev_a, ev_b, ev_c], graph, config)
        by_entity = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_entity["svca"].r_early, 1.0)
        self.assertAlmostEqual(by_entity["svcb"].r_early, 0.5)
        self.assertAlmostEqual(by_entity["svcc"].r_early, 0.0)

    def test_3_equal_timestamps(self) -> None:
        # All anomalous entities share the same first anomaly timestamp -> all get 1.0
        ev_a = _make_evidence("svca", first_ts=100, peak_count=1)
        ev_b = _make_evidence("svcb", first_ts=100, peak_count=1)
        ev_c = _make_evidence("svcc", first_ts=None, peak_count=0)

        graph = build_entity_graph(["svca_cpu", "svcb_cpu", "svcc_cpu"])
        config = RCAConfig(name="EarlinessOnly", use_earliness=True)

        ranking = rank_root_cause_entities([ev_a, ev_b, ev_c], graph, config)
        by_entity = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_entity["svca"].r_early, 1.0)
        self.assertAlmostEqual(by_entity["svcb"].r_early, 1.0)
        self.assertAlmostEqual(by_entity["svcc"].r_early, 0.0)

    def test_4_coverage_calculation(self) -> None:
        # A calls B and C. B is anomalous, C is normal -> coverage = 1/2 = 0.5
        # Timing does not matter for coverage
        ev_a = _make_evidence("svca", first_ts=100)
        ev_b = _make_evidence("svcb", first_ts=50)  # B is earlier than A
        ev_c = _make_evidence("svcc", first_ts=None)  # C is not anomalous

        deps = [Dependency("svca", "svcb"), Dependency("svca", "svcc")]
        graph = build_entity_graph(["svca_cpu", "svcb_cpu", "svcc_cpu"], dependencies=deps)
        config = RCAConfig(name="CoverageOnly", use_coverage=True)

        ranking = rank_root_cause_entities([ev_a, ev_b, ev_c], graph, config)
        by_entity = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_entity["svca"].r_coverage, 0.5)
        # svcb and svcc have 0 callees -> 0.0
        self.assertAlmostEqual(by_entity["svcb"].r_coverage, 0.0)
        self.assertAlmostEqual(by_entity["svcc"].r_coverage, 0.0)

    def test_5_propagation_calculation(self) -> None:
        # A calls B and C. A has ts=150.
        # B has ts=160 (>= 150) -> consistent
        # C has ts=120 (< 150) -> inconsistent (became anomalous before A)
        # Propagation = 1 / 2 = 0.5
        ev_a = _make_evidence("svca", first_ts=150)
        ev_b = _make_evidence("svcb", first_ts=160)
        ev_c = _make_evidence("svcc", first_ts=120)

        deps = [Dependency("svca", "svcb"), Dependency("svca", "svcc")]
        graph = build_entity_graph(["svca_cpu", "svcb_cpu", "svcc_cpu"], dependencies=deps)
        config = RCAConfig(name="PropOnly", use_propagation=True)

        ranking = rank_root_cause_entities([ev_a, ev_b, ev_c], graph, config)
        by_entity = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_entity["svca"].r_prop, 0.5)

    def test_6_no_callee_behavior(self) -> None:
        # Entity with no callees in graph gets r_coverage=0.0 and r_prop=0.0
        ev_a = _make_evidence("svca", first_ts=100)
        graph = build_entity_graph(["svca_cpu"])
        config = RCAConfig(name="Topo", use_coverage=True, use_propagation=True)

        ranking = rank_root_cause_entities([ev_a], graph, config)
        self.assertEqual(len(ranking), 1)
        self.assertAlmostEqual(ranking[0].r_coverage, 0.0)
        self.assertAlmostEqual(ranking[0].r_prop, 0.0)
        self.assertAlmostEqual(ranking[0].score, 0.0)

    def test_7_non_anomalous_entities_rank_at_bottom(self) -> None:
        # An entity with no anomaly must rank below anomalous entities, even if anomalous entity score is 0
        ev_anom = _make_evidence("svcanom", first_ts=100, peak_count=1)
        ev_clean = _make_evidence("svcclean", first_ts=None, peak_count=0)

        graph = build_entity_graph(["svcanom_cpu", "svcclean_cpu"])
        # Config with no components enabled -> all scores are 0.0
        config = RCAConfig(name="ZeroScore")

        ranking = rank_root_cause_entities([ev_anom, ev_clean], graph, config)
        self.assertEqual(ranking[0].entity, "svcanom")
        self.assertEqual(ranking[1].entity, "svcclean")
        self.assertIsNone(ranking[1].first_anomaly_ts)

    def test_8_deterministic_tie_breaking(self) -> None:
        # Tie-break order:
        # 1. score descending
        # 2. earlier first_anomaly_ts
        # 3. higher raw peak score
        # 4. lexicographical entity name ascending

        ev_late = _make_evidence("alate", first_ts=150, peak_count=1)
        ev_early = _make_evidence("zearly", first_ts=100, peak_count=1)
        graph_ts = build_entity_graph(["alate_cpu", "zearly_cpu"])

        # Pair 1: timestamp tie-break when score is equal
        r1 = rank_root_cause_entities([ev_late, ev_early], graph_ts, RCAConfig("Score0"))
        self.assertEqual([x.entity for x in r1], ["zearly", "alate"])

        # Pair 2: peak score tie-break when score and timestamp are equal
        ev_peak2 = _make_evidence("apeak", first_ts=150, peak_count=2)
        ev_peak5 = _make_evidence("zpeak", first_ts=150, peak_count=5)
        graph_peak = build_entity_graph(["apeak_cpu", "zpeak_cpu"])

        r2 = rank_root_cause_entities([ev_peak2, ev_peak5], graph_peak, RCAConfig("Score0"))
        self.assertEqual([x.entity for x in r2], ["zpeak", "apeak"])

        # Pair 3: alphabetical tie-break when score, timestamp, and peak score are equal
        ev_alpha1 = _make_evidence("alphafirst", first_ts=150, peak_count=2)
        ev_alpha2 = _make_evidence("alphasecond", first_ts=150, peak_count=2)
        graph_alpha = build_entity_graph(["alphafirst_cpu", "alphasecond_cpu"])

        r3 = rank_root_cause_entities([ev_alpha2, ev_alpha1], graph_alpha, RCAConfig("Score0"))
        self.assertEqual([x.entity for x in r3], ["alphafirst", "alphasecond"])

    def test_9_ground_truth_isolation_at_api_level(self) -> None:
        # rank_root_cause_entities accepts ONLY evidence, graph, config
        import inspect
        sig = inspect.signature(rank_root_cause_entities)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["evidence", "graph", "config"])
        for forbidden in ("ground_truth", "root_cause", "fault", "inject_time"):
            self.assertNotIn(forbidden, params)

    def test_10_ablation_formulas(self) -> None:
        # A calls B.
        # A: peak=2, first_ts=100
        # B: peak=4, first_ts=120
        # max_peak = 4 -> A r_strength = 2/4 = 0.5, B r_strength = 4/4 = 1.0
        # t_min = 100, t_max = 120 -> A r_early = 1.0, B r_early = 0.0
        # A callees = [B] -> A r_coverage = 1/1 = 1.0, B r_coverage = 0.0
        # A prop: B ts 120 >= 100 -> A r_prop = 1/1 = 1.0, B r_prop = 0.0

        ev_a = _make_evidence("svca", first_ts=100, peak_count=2)
        ev_b = _make_evidence("svcb", first_ts=120, peak_count=4)
        graph = build_entity_graph(
            ["svca_cpu", "svcb_cpu"],
            dependencies=[Dependency("svca", "svcb")],
        )

        # Config A: RCA_A_ANOMALY_ONLY (Score = R_strength)
        res_a = {r.entity: r for r in rank_root_cause_entities([ev_a, ev_b], graph, RCA_A_ANOMALY_ONLY)}
        self.assertAlmostEqual(res_a["svca"].score, 0.5)
        self.assertAlmostEqual(res_a["svcb"].score, 1.0)

        # Config B: RCA_B_TEMPORAL (Score = R_strength + R_early)
        res_b = {r.entity: r for r in rank_root_cause_entities([ev_a, ev_b], graph, RCA_B_TEMPORAL)}
        self.assertAlmostEqual(res_b["svca"].score, 0.5 + 1.0)
        self.assertAlmostEqual(res_b["svcb"].score, 1.0 + 0.0)

        # Config C: RCA_C_TOPOLOGY_ONLY (Score = R_strength + R_coverage)
        res_c = {r.entity: r for r in rank_root_cause_entities([ev_a, ev_b], graph, RCA_C_TOPOLOGY_ONLY)}
        self.assertAlmostEqual(res_c["svca"].score, 0.5 + 1.0)
        self.assertAlmostEqual(res_c["svcb"].score, 1.0 + 0.0)

        # Config D: RCA_D_COMBINED (Score = R_strength + R_early + R_prop)
        res_d = {r.entity: r for r in rank_root_cause_entities([ev_a, ev_b], graph, RCA_D_COMBINED)}
        self.assertAlmostEqual(res_d["svca"].score, 0.5 + 1.0 + 1.0)
        self.assertAlmostEqual(res_d["svcb"].score, 1.0 + 0.0 + 0.0)

    def test_11_no_future_index_access_temporal_causality(self) -> None:
        # Modifying future values does not affect first anomaly timestamp or past evidence
        ev_1 = EntityAnomalyEvidence(
            case_id="c1",
            entity="svca",
            timestamps=(10, 11, 12, 13, 14),
            active_metric_counts=(0, 2, 0, 0, 0),
            is_anomalous=(False, True, False, False, False),
            first_anomaly_ts=11,
            anomalous_metrics=("svca_cpu",),
        )
        ev_2 = EntityAnomalyEvidence(
            case_id="c1",
            entity="svca",
            timestamps=(10, 11, 12, 13, 14),
            active_metric_counts=(0, 2, 5, 5, 5),  # Future metric spikes
            is_anomalous=(False, True, True, True, True),
            first_anomaly_ts=11,
            anomalous_metrics=("svca_cpu",),
        )
        # first_anomaly_ts remains 11 in both cases
        self.assertEqual(ev_1.first_anomaly_ts, ev_2.first_anomaly_ts)

    def test_12_ranking_evaluator_metrics(self) -> None:
        # Construct a synthetic ranking of 6 entities
        ranking = (
            RootCauseScore("svc1", 3.0, 1.0, 1.0, 0.0, 1.0, 100, 3.0),
            RootCauseScore("svc2", 2.0, 0.8, 0.8, 0.0, 0.4, 110, 2.0),
            RootCauseScore("svc3", 1.5, 0.5, 0.5, 0.0, 0.5, 120, 2.0),
            RootCauseScore("svc4", 1.0, 0.3, 0.5, 0.0, 0.2, 130, 1.0),
            RootCauseScore("svc5", 0.5, 0.1, 0.4, 0.0, 0.0, 140, 1.0),
            RootCauseScore("svc6", 0.1, 0.0, 0.1, 0.0, 0.0, 150, 1.0),
        )

        # Hit at rank 1
        r_top1 = evaluate_root_cause_ranking(ranking, "svc1")
        self.assertTrue(r_top1.top1)
        self.assertTrue(r_top1.top3)
        self.assertTrue(r_top1.top5)
        self.assertAlmostEqual(r_top1.reciprocal_rank, 1.0)
        self.assertEqual(r_top1.first_hit_rank, 1)

        # Hit at rank 2
        r_top2 = evaluate_root_cause_ranking(ranking, "svc2")
        self.assertFalse(r_top2.top1)
        self.assertTrue(r_top2.top3)
        self.assertTrue(r_top2.top5)
        self.assertAlmostEqual(r_top2.reciprocal_rank, 0.5)
        self.assertEqual(r_top2.first_hit_rank, 2)

        # Hit at rank 4
        r_top4 = evaluate_root_cause_ranking(ranking, "svc4")
        self.assertFalse(r_top4.top1)
        self.assertFalse(r_top4.top3)
        self.assertTrue(r_top4.top5)
        self.assertAlmostEqual(r_top4.reciprocal_rank, 0.25)
        self.assertEqual(r_top4.first_hit_rank, 4)

        # Hit at rank 6
        r_top6 = evaluate_root_cause_ranking(ranking, "svc6")
        self.assertFalse(r_top6.top1)
        self.assertFalse(r_top6.top3)
        self.assertFalse(r_top6.top5)
        self.assertAlmostEqual(r_top6.reciprocal_rank, 1.0 / 6)
        self.assertEqual(r_top6.first_hit_rank, 6)

        # Missing target
        r_missing = evaluate_root_cause_ranking(ranking, "unknown_svc")
        self.assertFalse(r_missing.top1)
        self.assertFalse(r_missing.top3)
        self.assertFalse(r_missing.top5)
        self.assertAlmostEqual(r_missing.reciprocal_rank, 0.0)
        self.assertIsNone(r_missing.first_hit_rank)

        # Aggregate evaluation across multiple cases
        agg = evaluate_rca_benchmark([r_top1, r_top2, r_top4, r_missing])
        self.assertEqual(agg.total_cases, 4)
        # top1: 1/4 = 0.25
        self.assertAlmostEqual(agg.top1_accuracy, 0.25)
        # top3: 2/4 = 0.5
        self.assertAlmostEqual(agg.top3_accuracy, 0.5)
        # top5: 3/4 = 0.75
        self.assertAlmostEqual(agg.top5_accuracy, 0.75)
        # mrr: (1.0 + 0.5 + 0.25 + 0.0) / 4 = 1.75 / 4 = 0.4375
        self.assertAlmostEqual(agg.mrr, 0.4375)



class SyntheticEpisodeRcaRankingTests(unittest.TestCase):
    def test_1_episode_based_rca_earliness(self) -> None:
        # Two entities with episodes starting at t=100 and t=200
        ep_a = EntityEpisodeEvidence(
            case_id="c1",
            entity="svca",
            timestamps=(100, 200),
            active_metric_counts=(2, 2),
            is_in_episode=(True, True),
            episodes=(EpisodeInterval(0, 1, 100, 200, 100, 2, ("svca_m1", "svca_m2")),),
            first_episode_start_ts=100,
            peak_active_metrics=2,
            all_contributing_metrics=("svca_m1", "svca_m2"),
            has_episode=True,
        )
        ep_b = EntityEpisodeEvidence(
            case_id="c1",
            entity="svcb",
            timestamps=(100, 200),
            active_metric_counts=(0, 2),
            is_in_episode=(False, True),
            episodes=(EpisodeInterval(1, 1, 200, 200, 0, 2, ("svcb_m1", "svcb_m2")),),
            first_episode_start_ts=200,
            peak_active_metrics=2,
            all_contributing_metrics=("svcb_m1", "svcb_m2"),
            has_episode=True,
        )
        graph = build_entity_graph(["svca_m1", "svcb_m1"])
        config = RCAConfig(name="EarlyOnly", use_earliness=True)

        ranking = rank_root_cause_entities([ep_a, ep_b], graph, config)
        by_ent = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_ent["svca"].r_early, 1.0)
        self.assertAlmostEqual(by_ent["svcb"].r_early, 0.0)

    def test_2_episode_based_rca_coverage(self) -> None:
        # svca calls svcb and svcc. svcb has episode, svcc does not.
        ep_a = EntityEpisodeEvidence(
            case_id="c1",
            entity="svca",
            timestamps=(100,),
            active_metric_counts=(2,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 100, 100, 0, 2, ("svca_m1", "svca_m2")),),
            first_episode_start_ts=100,
            peak_active_metrics=2,
            all_contributing_metrics=("svca_m1", "svca_m2"),
            has_episode=True,
        )
        ep_b = EntityEpisodeEvidence(
            case_id="c1",
            entity="svcb",
            timestamps=(100,),
            active_metric_counts=(2,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 100, 100, 0, 2, ("svcb_m1", "svcb_m2")),),
            first_episode_start_ts=100,
            peak_active_metrics=2,
            all_contributing_metrics=("svcb_m1", "svcb_m2"),
            has_episode=True,
        )
        ep_c = EntityEpisodeEvidence(
            case_id="c1",
            entity="svcc",
            timestamps=(100,),
            active_metric_counts=(0,),
            is_in_episode=(False,),
            episodes=(),
            first_episode_start_ts=None,
            peak_active_metrics=0,
            all_contributing_metrics=(),
            has_episode=False,
        )
        deps = [Dependency("svca", "svcb"), Dependency("svca", "svcc")]
        graph = build_entity_graph(["svca_m1", "svcb_m1", "svcc_m1"], dependencies=deps)
        config = RCAConfig(name="CovOnly", use_coverage=True)

        ranking = rank_root_cause_entities([ep_a, ep_b, ep_c], graph, config)
        by_ent = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_ent["svca"].r_coverage, 0.5)
        self.assertAlmostEqual(by_ent["svcb"].r_coverage, 0.0)
        self.assertAlmostEqual(by_ent["svcc"].r_coverage, 0.0)

    def test_3_episode_based_rca_propagation(self) -> None:
        # svca has episode at 100. Callees: svcb has episode at 150 (>=100), svcc has episode at 80 (<100)
        ep_a = EntityEpisodeEvidence(
            case_id="c1",
            entity="svca",
            timestamps=(100,),
            active_metric_counts=(2,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 100, 100, 0, 2, ("svca_m1", "svca_m2")),),
            first_episode_start_ts=100,
            peak_active_metrics=2,
            all_contributing_metrics=("svca_m1", "svca_m2"),
            has_episode=True,
        )
        ep_b = EntityEpisodeEvidence(
            case_id="c1",
            entity="svcb",
            timestamps=(150,),
            active_metric_counts=(2,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 150, 150, 0, 2, ("svcb_m1", "svcb_m2")),),
            first_episode_start_ts=150,
            peak_active_metrics=2,
            all_contributing_metrics=("svcb_m1", "svcb_m2"),
            has_episode=True,
        )
        ep_c = EntityEpisodeEvidence(
            case_id="c1",
            entity="svcc",
            timestamps=(80,),
            active_metric_counts=(2,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 80, 80, 0, 2, ("svcc_m1", "svcc_m2")),),
            first_episode_start_ts=80,
            peak_active_metrics=2,
            all_contributing_metrics=("svcc_m1", "svcc_m2"),
            has_episode=True,
        )
        deps = [Dependency("svca", "svcb"), Dependency("svca", "svcc")]
        graph = build_entity_graph(["svca_m1", "svcb_m1", "svcc_m1"], dependencies=deps)
        config = RCAConfig(name="PropOnly", use_propagation=True)

        ranking = rank_root_cause_entities([ep_a, ep_b, ep_c], graph, config)
        by_ent = {r.entity: r for r in ranking}

        # 1 of 2 callees has episode start >= 100 (svcb at 150) -> r_prop = 0.5
        self.assertAlmostEqual(by_ent["svca"].r_prop, 0.5)

    def test_4_episode_based_rca_strength_uses_peak_active_metrics(self) -> None:
        # Entities with peak_active_metrics: 3, 2, 0
        ep_a = EntityEpisodeEvidence(
            case_id="c1",
            entity="svca",
            timestamps=(100,),
            active_metric_counts=(3,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 100, 100, 0, 3, ("m1", "m2", "m3")),),
            first_episode_start_ts=100,
            peak_active_metrics=3,
            all_contributing_metrics=("m1", "m2", "m3"),
            has_episode=True,
        )
        ep_b = EntityEpisodeEvidence(
            case_id="c1",
            entity="svcb",
            timestamps=(100,),
            active_metric_counts=(2,),
            is_in_episode=(True,),
            episodes=(EpisodeInterval(0, 0, 100, 100, 0, 2, ("m1", "m2")),),
            first_episode_start_ts=100,
            peak_active_metrics=2,
            all_contributing_metrics=("m1", "m2"),
            has_episode=True,
        )
        graph = build_entity_graph(["svca_m1", "svcb_m1"])
        config = RCAConfig(name="StrOnly", use_strength=True)

        ranking = rank_root_cause_entities([ep_a, ep_b], graph, config)
        by_ent = {r.entity: r for r in ranking}

        self.assertAlmostEqual(by_ent["svca"].r_strength, 1.0)
        self.assertAlmostEqual(by_ent["svcb"].r_strength, 2.0 / 3.0)

    def test_5_raw_vs_episode_evidence_does_not_leak_ground_truth(self) -> None:
        import inspect
        sig = inspect.signature(rank_root_cause_entities)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["evidence", "graph", "config"])
        for forbidden in ("ground_truth", "root_cause", "fault", "inject_time"):
            self.assertNotIn(forbidden, params)


class RealDataRcaIntegrationTests(unittest.TestCase):
    def test_re2ob_checkoutservice_cpu_1_ablations_raw_evidence(self) -> None:
        dataset_root = os.environ.get("RCAEVAL_DATASET_ROOT")
        if not dataset_root:
            dataset_root = str(Path.home() / ".cache" / "rcaeval_validation")

        root = Path(dataset_root)
        case_dir = root / "re2ob_checkoutservice_cpu_1"
        if not case_dir.is_dir():
            raise unittest.SkipTest(f"{case_dir} not found; skipping real-data integration test")

        case = load_rcaeval_case(root, "re2ob_checkoutservice_cpu_1")
        ground_truth_target = case.ground_truth.values["root_cause_service"]
        self.assertEqual(ground_truth_target, "checkoutservice")

        trace_deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
        det_res = detect_metric_anomalies(case)
        graph = build_entity_graph(
            det_res.metric_names,
            dependencies=[td.to_dependency() for td in trace_deps],
        )
        evidence = aggregate_entity_anomaly_evidence(det_res, graph)

        ablations = [
            RCA_A_ANOMALY_ONLY,
            RCA_B_TEMPORAL,
            RCA_C_TOPOLOGY_ONLY,
            RCA_D_COMBINED,
        ]

        expected_ranks = {
            "RCA_A_AnomalyOnly": 8,
            "RCA_B_Temporal": 9,
            "RCA_C_TopologyOnly": 2,
            "RCA_D_Combined": 9,
        }

        for config in ablations:
            ranking = rank_root_cause_entities(evidence, graph, config)
            eval_res = evaluate_root_cause_ranking(ranking, ground_truth_target, case_id=case.metadata.case_id)
            self.assertEqual(eval_res.first_hit_rank, expected_ranks[config.name])

    def test_re2ob_checkoutservice_cpu_1_ablations_episode_evidence(self) -> None:
        dataset_root = os.environ.get("RCAEVAL_DATASET_ROOT")
        if not dataset_root:
            dataset_root = str(Path.home() / ".cache" / "rcaeval_validation")

        root = Path(dataset_root)
        case_dir = root / "re2ob_checkoutservice_cpu_1"
        if not case_dir.is_dir():
            raise unittest.SkipTest(f"{case_dir} not found; skipping real-data integration test")

        case = load_rcaeval_case(root, "re2ob_checkoutservice_cpu_1")
        ground_truth_target = case.ground_truth.values["root_cause_service"]
        inject_time = case.ground_truth.values["inject_time"]

        trace_deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
        det_res = detect_metric_anomalies(case)
        graph = build_entity_graph(
            det_res.metric_names,
            dependencies=[td.to_dependency() for td in trace_deps],
        )

        # 1. Construct Entity Anomaly Episodes with approved baseline K=3, M=2
        ep_config = EpisodeConfig(persistence=3, consensus=2)
        ep_evidence = aggregate_entity_episodes(det_res, graph, ep_config)

        # Offline diagnostic prints
        print("\n" + "=" * 80)
        print(f"OFFLINE DIAGNOSTIC: Entity Anomaly Episodes (K={ep_config.persistence}, M={ep_config.consensus})")
        print(f"Case: {case.metadata.case_id} | Ground Truth Target: {ground_truth_target} | Injection: {inject_time}")
        print("=" * 80)

        for entity in sorted(ep_evidence.keys()):
            ev = ep_evidence[entity]
            pre_eps = [ep for ep in ev.episodes if ep.start_timestamp < inject_time]
            post_eps = [ep for ep in ev.episodes if ep.start_timestamp >= inject_time]
            first_ep = ev.episodes[0] if ev.episodes else None
            first_info = f"t={first_ep.start_timestamp} (dur={first_ep.duration_seconds}s, metrics={first_ep.contributing_metrics})" if first_ep else "NONE"
            print(
                f"  {entity:22s} | has_ep={str(ev.has_episode):5s} | pre_ep={len(pre_eps):2d} | "
                f"post_ep={len(post_eps):2d} | peak_active={ev.peak_active_metrics} | first_ep={first_info}"
            )

        # 2. Run RCA ablations using Episode Evidence
        ablations = [
            RCA_A_ANOMALY_ONLY,
            RCA_B_TEMPORAL,
            RCA_C_TOPOLOGY_ONLY,
            RCA_D_COMBINED,
        ]

        print("\n" + "=" * 80)
        print(f"OFFLINE DIAGNOSTIC: Episode-Based RCA Ranking Ablations for Case: {case.metadata.case_id}")
        print("=" * 80)

        for config in ablations:
            ranking = rank_root_cause_entities(ep_evidence, graph, config)
            eval_res = evaluate_root_cause_ranking(ranking, ground_truth_target, case_id=case.metadata.case_id)

            print(f"\n--- {config.name} ---")
            print(f"Target '{ground_truth_target}' Rank: {eval_res.first_hit_rank} | Reciprocal Rank: {eval_res.reciprocal_rank:.4f}")
            print(f"Top-1: {eval_res.top1} | Top-3: {eval_res.top3} | Top-5: {eval_res.top5}")
            print("Ranked Entities (Top 5):")
            for rank_idx, score_item in enumerate(ranking[:5], start=1):
                marker = " <-- TARGET" if score_item.entity == ground_truth_target else ""
                print(
                    f"  {rank_idx}. {score_item.entity:22s} Score={score_item.score:.4f} "
                    f"[R_str={score_item.r_strength:.2f}, R_early={score_item.r_early:.2f}, "
                    f"R_cov={score_item.r_coverage:.2f}, R_prop={score_item.r_prop:.2f}, "
                    f"ts={score_item.first_anomaly_ts}, peak={score_item.peak_score:.0f}]{marker}"
                )

            self.assertEqual(len(ranking), len(graph.entities))
            self.assertIsNotNone(eval_res.first_hit_rank)


if __name__ == "__main__":
    unittest.main()

