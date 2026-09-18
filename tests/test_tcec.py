"""Unit tests for Topology-Coherent Episode Confirmation (TCEC).

Verifies the 8 mandatory research and causality requirements:
1. Candidate without propagation -> unconfirmed
2. Valid propagation -> t_confirm
3. Wrong topology -> unconfirmed
4. Wrong temporal order -> unconfirmed
5. Causal future invariance
6. No injection-time leakage
7. No arbitrary parameters
8. Existing causality and window resolution compatibility
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from digital_detective.anomaly import MetricAnomalyResult, truncate_metric_anomaly_result
from digital_detective.bocpd import BOCPDResult
from digital_detective.episodes import EpisodeConfig
from digital_detective.tcec import (
    TCECConfig,
    TCECResult,
    confirm_topology_coherent_episode,
)
from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)
from digital_detective.topology import Dependency, EntityGraph, build_entity_graph


def _make_telemetry_case(
    metric_names: Sequence[str],
    timestamps: Sequence[int],
    case_id: str = "case_tcec_test",
    ground_truth_guard: bool = False,
) -> TelemetryCase:
    """Create a minimal TelemetryCase for testing."""
    n = len(timestamps)
    raw_data = [{"time": ts, **{m: 0.0 for m in metric_names}} for ts in timestamps]
    prov = ModalityProvenance(
        source_dataset="RCAEval",
        source_case=case_id,
        source_revision="test",
        original_format="parquet",
        original_field_names=("time", *metric_names),
        timestamp_field="time",
    )

    class GuardedGroundTruth:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"TCEC illegally accessed ground_truth.{name}")

        def __getitem__(self, item: str) -> Any:
            raise AssertionError(f"TCEC illegally accessed ground_truth[{item!r}]")

    gt = GuardedGroundTruth() if ground_truth_guard else GroundTruth(values={"inject_time": 999999})

    return TelemetryCase(
        metadata=CaseMetadata(
            case_id=case_id,
            dataset="RE2-OB",
            suite="RE2",
            system="ob",
            system_name="Online Boutique",
        ),
        metrics=TelemetryModality(raw_data=raw_data, provenance=prov),
        ground_truth=gt,  # type: ignore[arg-type]
    )


def _make_anomaly_result(
    statuses: Mapping[str, Sequence[str]],
    timestamps: Sequence[int],
    case_id: str = "case_tcec_test",
) -> MetricAnomalyResult:
    """Create a synthetic MetricAnomalyResult with predefined evaluation statuses."""
    ts = tuple(timestamps)
    return MetricAnomalyResult(
        case_id=case_id,
        timestamps=ts,
        metric_names=tuple(statuses.keys()),
        anomaly_scores={m: tuple(3.0 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        signed_scores={m: tuple(3.0 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        anomalies={m: tuple(s == "anomaly" for s in st) for m, st in statuses.items()},
        evaluation_statuses={m: tuple(st) for m, st in statuses.items()},
        valid_history_counts={m: tuple(60 for _ in st) for m, st in statuses.items()},
        summary={},
    )


class TestTCEC(unittest.TestCase):
    """Test suite for Topology-Coherent Episode Confirmation."""

    def setUp(self) -> None:
        self.timestamps = tuple(range(100, 120))  # 20 steps: 100 to 119
        self.metric_names = ["svca_m1", "svca_m2", "svcb_m1", "svcb_m2", "svcc_m1", "svcc_m2"]

    def _build_bocpd_result(self, onset_ts: int | None = 100, status: str = "detected") -> BOCPDResult:
        return BOCPDResult(
            case_id="case_tcec_test",
            onset_ts=onset_ts,
            status=status,
            audit={"onset_idx": 0 if onset_ts is not None else None},
            changepoints=(0,) if onset_ts is not None else (),
            timestamps=self.timestamps,
        )

    def test_1_candidate_without_propagation(self) -> None:
        """Candidate occurs, persistent local episode exists, but no successor episode -> unconfirmed."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd_result(onset_ts=100)

        # svca has persistent anomaly (persistence=3, consensus=2) starting at step 0 (ts 100).
        # svcb and svcc remain normal throughout.
        n = len(self.timestamps)
        statuses = {
            "svca_m1": ["anomaly"] * n,
            "svca_m2": ["anomaly"] * n,
            "svcb_m1": ["normal"] * n,
            "svcb_m2": ["normal"] * n,
            "svcc_m1": ["normal"] * n,
            "svcc_m2": ["normal"] * n,
        }
        det_res = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        res = confirm_topology_coherent_episode(
            case,
            bocpd,
            graph=graph,
            anomaly_result=det_res,
        )

        self.assertEqual(res.status, "unconfirmed")
        self.assertEqual(res.candidate_onset, 100)
        self.assertIsNone(res.confirmed_onset)
        self.assertIsNone(res.confirming_entity)

    def test_2_valid_propagation(self) -> None:
        """Candidate episode on svca followed by successor on svcb connected via topology -> confirmed."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd_result(onset_ts=100)

        # svca: anomalous from step 0 (ts 100) -> persistent episode active at step 2 (ts 102), interval starts ts 100
        # svcb: anomalous starting at step 5 (ts 105) -> persistent episode active at step 7 (ts 107), interval starts ts 105
        n = len(self.timestamps)
        statuses = {
            "svca_m1": ["anomaly"] * n,
            "svca_m2": ["anomaly"] * n,
            "svcb_m1": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcb_m2": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcc_m1": ["normal"] * n,
            "svcc_m2": ["normal"] * n,
        }
        det_res = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        res = confirm_topology_coherent_episode(
            case,
            bocpd,
            graph=graph,
            anomaly_result=det_res,
        )

        self.assertEqual(res.status, "confirmed")
        self.assertEqual(res.candidate_onset, 100)
        # First timestamp at which svcb episode is confirmed and visible
        self.assertIsNotNone(res.confirmed_onset)
        self.assertGreater(res.confirmed_onset, 100)
        self.assertEqual(res.candidate_entity, "svca")
        self.assertEqual(res.confirming_entity, "svcb")
        self.assertEqual(res.confirmation_latency_sec, res.confirmed_onset - 100)

    def test_3_wrong_topology(self) -> None:
        """Later anomalous entity svcc exists, but no admissible topology relation with svca -> unconfirmed."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd_result(onset_ts=100)

        # svca anomalous at 100, svcc anomalous at 105.
        # But graph only has edge svcb -> svcc (no edge involving svca).
        n = len(self.timestamps)
        statuses = {
            "svca_m1": ["anomaly"] * n,
            "svca_m2": ["anomaly"] * n,
            "svcb_m1": ["normal"] * n,
            "svcb_m2": ["normal"] * n,
            "svcc_m1": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcc_m2": ["normal"] * 5 + ["anomaly"] * (n - 5),
        }
        det_res = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svcb", "svcc")])

        res = confirm_topology_coherent_episode(
            case,
            bocpd,
            graph=graph,
            anomaly_result=det_res,
        )

        self.assertEqual(res.status, "unconfirmed")
        self.assertIsNone(res.confirmed_onset)
        self.assertIsNone(res.confirming_entity)

    def test_4_wrong_temporal_order(self) -> None:
        """Topology relation exists, but successor evidence occurs BEFORE candidate -> unconfirmed."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd_result(onset_ts=105)  # Candidate at 105

        # svcb had an anomaly earlier at 100..103, but is normal thereafter.
        # svca has anomaly starting at 105.
        # svcb occurred BEFORE svca, violating temporal precedence t_j > t_i.
        n = len(self.timestamps)
        statuses = {
            "svca_m1": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svca_m2": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcb_m1": ["anomaly"] * 4 + ["normal"] * (n - 4),
            "svcb_m2": ["anomaly"] * 4 + ["normal"] * (n - 4),
            "svcc_m1": ["normal"] * n,
            "svcc_m2": ["normal"] * n,
        }
        det_res = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        res = confirm_topology_coherent_episode(
            case,
            bocpd,
            graph=graph,
            anomaly_result=det_res,
        )

        self.assertEqual(res.status, "unconfirmed")
        self.assertIsNone(res.confirmed_onset)

    def test_5_causal_future_invariance(self) -> None:
        """Adding telemetry after an already-confirmed timestamp must not alter earlier confirmation."""
        case_short = _make_telemetry_case(self.metric_names, self.timestamps[:12])
        bocpd_short = BOCPDResult(
            case_id="case_tcec_test",
            onset_ts=100,
            status="detected",
            audit={"onset_idx": 0},
            changepoints=(0,),
            timestamps=self.timestamps[:12],
        )

        statuses_short = {
            "svca_m1": ["anomaly"] * 12,
            "svca_m2": ["anomaly"] * 12,
            "svcb_m1": ["normal"] * 5 + ["anomaly"] * 7,
            "svcb_m2": ["normal"] * 5 + ["anomaly"] * 7,
            "svcc_m1": ["normal"] * 12,
            "svcc_m2": ["normal"] * 12,
        }
        det_short = _make_anomaly_result(statuses_short, self.timestamps[:12])
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        res_short = confirm_topology_coherent_episode(
            case_short,
            bocpd_short,
            graph=graph,
            anomaly_result=det_short,
        )
        self.assertEqual(res_short.status, "confirmed")
        confirmed_ts = res_short.confirmed_onset

        # Extend telemetry with 10 additional observations (with massive spikes on svcc and recovery on svcb)
        extended_ts = tuple(range(100, 130))
        case_long = _make_telemetry_case(self.metric_names, extended_ts)
        bocpd_long = BOCPDResult(
            case_id="case_tcec_test",
            onset_ts=100,
            status="detected",
            audit={"onset_idx": 0},
            changepoints=(0,),
            timestamps=extended_ts,
        )
        statuses_long = {
            "svca_m1": ["anomaly"] * 12 + ["normal"] * 18,
            "svca_m2": ["anomaly"] * 12 + ["normal"] * 18,
            "svcb_m1": ["normal"] * 5 + ["anomaly"] * 7 + ["normal"] * 18,
            "svcb_m2": ["normal"] * 5 + ["anomaly"] * 7 + ["normal"] * 18,
            "svcc_m1": ["normal"] * 15 + ["anomaly"] * 15,
            "svcc_m2": ["normal"] * 15 + ["anomaly"] * 15,
        }
        det_long = _make_anomaly_result(statuses_long, extended_ts)

        res_long = confirm_topology_coherent_episode(
            case_long,
            bocpd_long,
            graph=graph,
            anomaly_result=det_long,
        )

        self.assertEqual(res_long.status, "confirmed")
        self.assertEqual(res_long.confirmed_onset, confirmed_ts)
        self.assertEqual(res_long.candidate_entity, res_short.candidate_entity)
        self.assertEqual(res_long.confirming_entity, res_short.confirming_entity)

    def test_6_no_injection_time_leakage(self) -> None:
        """TCEC must never read inject_time or ground_truth attributes."""
        case_guarded = _make_telemetry_case(self.metric_names, self.timestamps, ground_truth_guard=True)
        bocpd = self._build_bocpd_result(onset_ts=100)

        n = len(self.timestamps)
        statuses = {
            "svca_m1": ["anomaly"] * n,
            "svca_m2": ["anomaly"] * n,
            "svcb_m1": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcb_m2": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcc_m1": ["normal"] * n,
            "svcc_m2": ["normal"] * n,
        }
        det_res = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        # If confirm_topology_coherent_episode touches case.ground_truth, it will raise AssertionError
        res = confirm_topology_coherent_episode(
            case_guarded,
            bocpd,
            graph=graph,
            anomaly_result=det_res,
        )
        self.assertEqual(res.status, "confirmed")

    def test_7_no_arbitrary_parameters(self) -> None:
        """TCECConfig must only reference existing EpisodeConfig and introduce no new thresholds."""
        cfg = TCECConfig()
        self.assertEqual(cfg.episode_config.persistence, 3)
        self.assertEqual(cfg.episode_config.consensus, 2)
        # Ensure no arbitrary duration, magnitude, correlation, or timeout attributes exist
        for forbidden in ("duration_threshold", "magnitude_threshold", "wait_timeout", "correlation_threshold"):
            self.assertFalse(hasattr(cfg, forbidden), f"Forbidden arbitrary parameter found: {forbidden}")

    def test_8_existing_causality_and_window_resolution(self) -> None:
        """Confirmed onset integrates cleanly with downstream truncation and window boundaries."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd_result(onset_ts=100)

        n = len(self.timestamps)
        statuses = {
            "svca_m1": ["anomaly"] * n,
            "svca_m2": ["anomaly"] * n,
            "svcb_m1": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcb_m2": ["normal"] * 5 + ["anomaly"] * (n - 5),
            "svcc_m1": ["normal"] * n,
            "svcc_m2": ["normal"] * n,
        }
        det_res = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        tcec_res = confirm_topology_coherent_episode(case, bocpd, graph=graph, anomaly_result=det_res)
        self.assertEqual(tcec_res.status, "confirmed")
        confirmed_ts = tcec_res.confirmed_onset
        self.assertIsNotNone(confirmed_ts)

        # Truncate anomaly result at confirmed_ts (causal analysis_end)
        truncated = truncate_metric_anomaly_result(det_res, max_timestamp=confirmed_ts)  # type: ignore[arg-type]
        self.assertEqual(truncated.timestamps[-1], confirmed_ts)
        self.assertLessEqual(len(truncated.timestamps), len(det_res.timestamps))


if __name__ == "__main__":
    unittest.main()
