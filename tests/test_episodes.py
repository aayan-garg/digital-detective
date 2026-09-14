"""Unit tests for Entity Anomaly Episode construction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import unittest

from digital_detective.anomaly import MetricAnomalyResult
from digital_detective.episodes import (
    EntityEpisodeEvidence,
    EpisodeConfig,
    EpisodeInterval,
    aggregate_entity_episodes,
)
from digital_detective.topology import EntityGraph, build_entity_graph


def _make_anomaly_result(
    statuses: Mapping[str, Sequence[str]],
    case_id: str = "test_case",
    timestamps: Sequence[int] | None = None,
) -> MetricAnomalyResult:
    n = len(next(iter(statuses.values())))
    ts = tuple(range(100, 100 + n)) if timestamps is None else tuple(timestamps)
    return MetricAnomalyResult(
        case_id=case_id,
        timestamps=ts,
        metric_names=tuple(statuses.keys()),
        anomaly_scores={m: tuple(3.5 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        signed_scores={m: tuple(3.5 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        anomalies={m: tuple(s == "anomaly" for s in st) for m, st in statuses.items()},
        evaluation_statuses={m: tuple(st) for m, st in statuses.items()},
        valid_history_counts={m: tuple(60 for _ in st) for m, st in statuses.items()},
        summary={},
    )


class EntityEpisodeTests(unittest.TestCase):
    def test_1_persistence_streak_reset(self) -> None:
        # K=3: anomaly for 2 steps, then normal, then anomaly for 2 steps -> streak never reaches 3
        statuses = {
            "svca_m1": ["anomaly", "anomaly", "normal", "anomaly", "anomaly"],
            "svca_m2": ["anomaly", "anomaly", "normal", "anomaly", "anomaly"],
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=3, consensus=1)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svca"]

        self.assertFalse(ev.has_episode)
        self.assertEqual(len(ev.episodes), 0)
        self.assertEqual(ev.active_metric_counts, (0, 0, 0, 0, 0))
        self.assertIsNone(ev.first_episode_start_ts)

    def test_2_m_of_n_consensus(self) -> None:
        # K=2, M=2. Entity has 3 metrics.
        # At steps 0-1: only m1 is anomalous -> count=1 < M -> no episode
        # At step 2: m2 also reaches streak 2 -> count=2 == M -> episode active
        statuses = {
            "svca_m1": ["anomaly", "anomaly", "anomaly", "normal"],
            "svca_m2": ["normal", "anomaly", "anomaly", "normal"],
            "svca_m3": ["normal", "normal", "normal", "normal"],
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=2, consensus=2)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svca"]

        self.assertTrue(ev.has_episode)
        self.assertEqual(ev.active_metric_counts, (0, 1, 2, 0))
        self.assertEqual(ev.is_in_episode, (False, False, True, False))
        self.assertEqual(len(ev.episodes), 1)
        self.assertEqual(ev.episodes[0].start_idx, 2)
        self.assertEqual(ev.episodes[0].end_idx, 2)

    def test_3_no_silent_consensus_capping(self) -> None:
        # Entity has only 1 metric. M=2.
        # Even if that single metric is anomalous for 10 consecutive steps,
        # it can NEVER reach consensus M=2. Must NOT be silently capped to min(M, 1).
        statuses = {
            "svcsingle_m1": ["anomaly"] * 10,
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=2, consensus=2)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svcsingle"]

        self.assertFalse(ev.has_episode)
        self.assertEqual(len(ev.episodes), 0)
        self.assertIsNone(ev.first_episode_start_ts)
        self.assertEqual(max(ev.active_metric_counts), 1)

    def test_4_interval_start_end(self) -> None:
        # K=1, M=1. Episode from step 2 to step 4.
        statuses = {
            "svca_m1": ["normal", "normal", "anomaly", "anomaly", "anomaly", "normal"],
        }
        det_res = _make_anomaly_result(statuses, timestamps=[100, 101, 102, 103, 104, 105])
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=1, consensus=1)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svca"]

        self.assertEqual(len(ev.episodes), 1)
        ep = ev.episodes[0]
        self.assertEqual(ep.start_idx, 2)
        self.assertEqual(ep.end_idx, 4)
        self.assertEqual(ep.start_timestamp, 102)
        self.assertEqual(ep.end_timestamp, 104)

    def test_5_duration(self) -> None:
        # Duration must equal end_timestamp - start_timestamp in seconds
        statuses = {
            "svca_m1": ["normal", "anomaly", "anomaly", "anomaly", "normal"],
        }
        det_res = _make_anomaly_result(statuses, timestamps=[1000, 1010, 1020, 1030, 1040])
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=1, consensus=1)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ep = evidence["svca"].episodes[0]
        self.assertEqual(ep.duration_seconds, 1030 - 1010)
        self.assertEqual(ep.duration_seconds, 20)

    def test_6_multiple_separated_episodes(self) -> None:
        # K=1, M=1. Anomaly at steps 1-2, then normal at step 3, then anomaly at steps 4-5.
        statuses = {
            "svca_m1": ["normal", "anomaly", "anomaly", "normal", "anomaly", "anomaly", "normal"],
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=1, consensus=1)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svca"]

        self.assertEqual(len(ev.episodes), 2)
        self.assertEqual(ev.episodes[0].start_idx, 1)
        self.assertEqual(ev.episodes[0].end_idx, 2)
        self.assertEqual(ev.episodes[1].start_idx, 4)
        self.assertEqual(ev.episodes[1].end_idx, 5)
        # first_episode_start_ts must be the start of the first episode
        self.assertEqual(ev.first_episode_start_ts, det_res.timestamps[1])

    def test_7_ongoing_episode_ending_at_final_observation(self) -> None:
        # Episode starts at step 2 and stays active through the final step (4)
        statuses = {
            "svca_m1": ["normal", "normal", "anomaly", "anomaly", "anomaly"],
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=1, consensus=1)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svca"]

        self.assertEqual(len(ev.episodes), 1)
        self.assertEqual(ev.episodes[0].start_idx, 2)
        self.assertEqual(ev.episodes[0].end_idx, 4)
        self.assertEqual(ev.episodes[0].end_timestamp, det_res.timestamps[4])

    def test_8_contributing_metrics(self) -> None:
        # svca has 3 metrics: m1 is anomalous during episode, m2 joins midway, m3 is never anomalous
        statuses = {
            "svca_m1": ["normal", "anomaly", "anomaly", "normal"],
            "svca_m2": ["normal", "normal", "anomaly", "normal"],
            "svca_m3": ["normal", "normal", "normal", "normal"],
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=1, consensus=1)

        evidence = aggregate_entity_episodes(det_res, graph, config)
        ev = evidence["svca"]

        self.assertEqual(len(ev.episodes), 1)
        ep = ev.episodes[0]
        self.assertEqual(ep.contributing_metrics, ("svca_m1", "svca_m2"))
        self.assertEqual(ev.all_contributing_metrics, ("svca_m1", "svca_m2"))
        self.assertEqual(ep.peak_active_metrics, 2)

    def test_9_warmup_missing_insufficient_history_reset(self) -> None:
        # Warmup, missing, and insufficient history must all reset the streak
        for non_anom_status in ("warmup", "missing_observation", "insufficient_history", "normal"):
            statuses = {
                "svca_m1": ["anomaly", "anomaly", non_anom_status, "anomaly", "anomaly"],
            }
            det_res = _make_anomaly_result(statuses)
            graph = build_entity_graph(list(statuses.keys()))
            config = EpisodeConfig(persistence=3, consensus=1)

            ev = aggregate_entity_episodes(det_res, graph, config)["svca"]
            self.assertFalse(ev.has_episode, f"Streak failed to reset on status: {non_anom_status}")

    def test_10_no_future_observation_access(self) -> None:
        # Modifying future values does not change episode start index or duration of completed episodes
        statuses_a = {
            "svca_m1": ["normal", "anomaly", "anomaly", "anomaly", "normal", "normal"],
        }
        statuses_b = {
            "svca_m1": ["normal", "anomaly", "anomaly", "anomaly", "normal", "anomaly"],
        }
        det_a = _make_anomaly_result(statuses_a)
        det_b = _make_anomaly_result(statuses_b)
        graph = build_entity_graph(list(statuses_a.keys()))
        config = EpisodeConfig(persistence=2, consensus=1)

        ev_a = aggregate_entity_episodes(det_a, graph, config)["svca"]
        ev_b = aggregate_entity_episodes(det_b, graph, config)["svca"]

        self.assertEqual(ev_a.episodes[0], ev_b.episodes[0])
        self.assertEqual(ev_a.first_episode_start_ts, ev_b.first_episode_start_ts)

    def test_11_no_anomalous_metrics_produces_no_episode(self) -> None:
        statuses = {
            "svca_m1": ["normal", "normal", "normal"],
            "svca_m2": ["normal", "normal", "normal"],
        }
        det_res = _make_anomaly_result(statuses)
        graph = build_entity_graph(list(statuses.keys()))
        config = EpisodeConfig(persistence=1, consensus=1)

        ev = aggregate_entity_episodes(det_res, graph, config)["svca"]
        self.assertFalse(ev.has_episode)
        self.assertEqual(len(ev.episodes), 0)
        self.assertIsNone(ev.first_episode_start_ts)
        self.assertEqual(ev.peak_active_metrics, 0)
        self.assertEqual(ev.all_contributing_metrics, ())


if __name__ == "__main__":
    unittest.main()
