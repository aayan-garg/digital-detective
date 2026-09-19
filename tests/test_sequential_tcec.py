"""Unit tests for Sequential Topology-Coherent Episode Confirmation (Sequential TCEC).

Verifies the mandatory requirements:
1. No corroboration -> remains SUSPECT / unconfirmed.
2. One corroboration -> evidence increases (Z=2), but does not confirm.
3. Later corroboration -> evidence increases again (Z=3) -> triggers confirmation.
4. Confirmation occurs at the first valid stopping timestamp.
5. Future telemetry cannot move confirmation earlier.
6. Future topology edges cannot influence earlier evidence (A -> B at t=200 cannot confirm at t=100).
7. Forward (caller -> callee) and reverse (callee -> caller) topology are distinguished.
8. Stale/reset behavior is deterministic (inactive > W=60s resets to NORMAL).
9. Ground truth is never accessed.
10. Identical input produces identical output (deterministic repeatability).
11. Stage-5 TCEC behavior remains completely unchanged.
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from digital_detective.anomaly import MetricAnomalyResult
from digital_detective.bocpd import BOCPDResult
from digital_detective.episodes import EpisodeConfig
from digital_detective.sequential_tcec import (
    CorroboratingEvidence,
    SequentialTCECConfig,
    SequentialTCECResult,
    confirm_sequential_tcec,
)
from digital_detective.tcec import TCECConfig, confirm_topology_coherent_episode
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
    case_id: str = "case_seq_test",
    ground_truth_guard: bool = False,
    trace_raw_data: Any = None,
) -> TelemetryCase:
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
            raise AssertionError(f"Sequential TCEC illegally accessed ground_truth.{name}")

        def __getitem__(self, item: str) -> Any:
            raise AssertionError(f"Sequential TCEC illegally accessed ground_truth[{item!r}]")

    gt = GuardedGroundTruth() if ground_truth_guard else GroundTruth(values={"inject_time": 999999})

    traces = None
    if trace_raw_data is not None:
        traces = TelemetryModality(raw_data=trace_raw_data, provenance=prov)

    return TelemetryCase(
        metadata=CaseMetadata(
            case_id=case_id,
            dataset="RE2-OB",
            suite="RE2",
            system="ob",
            system_name="Online Boutique",
        ),
        metrics=TelemetryModality(raw_data=raw_data, provenance=prov),
        traces=traces,
        ground_truth=gt,  # type: ignore[arg-type]
    )


def _make_anomaly_result(
    statuses: Mapping[str, Sequence[str]],
    timestamps: Sequence[int],
    case_id: str = "case_seq_test",
) -> MetricAnomalyResult:
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


class TestSequentialTCEC(unittest.TestCase):
    """Test suite for Sequential Topology-Coherent Episode Confirmation."""

    def setUp(self) -> None:
        self.timestamps = tuple(range(100, 250))  # 150 steps: 100 to 249
        self.metric_names = [
            "svca_m1", "svca_m2",
            "svcb_m1", "svcb_m2",
            "svcc_m1", "svcc_m2",
            "svcd_m1", "svcd_m2",
        ]

    def _build_bocpd(self, onset_ts: int | None = 100) -> BOCPDResult:
        return BOCPDResult(
            case_id="case_seq_test",
            onset_ts=onset_ts,
            status="detected" if onset_ts is not None else "no_detection",
            audit={"onset_idx": 0 if onset_ts is not None else None},
            changepoints=(0,) if onset_ts is not None else (),
            timestamps=self.timestamps,
        )

    def test_1_no_corroboration_remains_unconfirmed(self) -> None:
        """Candidate has local anomaly episode on svca, but zero connected corroborating entities -> Z=1 -> unconfirmed."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        self.assertEqual(res.status, "unconfirmed")
        self.assertIsNone(res.confirmed_onset)
        self.assertEqual(len(res.corroborating_entities), 0)

    def test_2_and_3_sequential_corroboration_steps(self) -> None:
        """Candidate on svca (ts=100, Z=1) -> svcb activates at ts=105 (Z=2) -> svcc activates at ts=110 (Z=3) -> CONFIRMED at 110."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        # svcb becomes anomalous at step 5 (ts 105)
        statuses["svcb_m1"] = ["normal"] * 5 + ["anomaly"] * (n - 5)
        statuses["svcb_m2"] = ["normal"] * 5 + ["anomaly"] * (n - 5)
        # svcc becomes anomalous at step 10 (ts 110)
        statuses["svcc_m1"] = ["normal"] * 10 + ["anomaly"] * (n - 10)
        statuses["svcc_m2"] = ["normal"] * 10 + ["anomaly"] * (n - 10)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[Dependency("svca", "svcb"), Dependency("svcb", "svcc")],
        )

        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        self.assertEqual(res.status, "confirmed")
        self.assertEqual(res.candidate_entity, "svca")
        self.assertEqual(res.corroborating_entities, ("svcb", "svcc"))
        # Confirmation occurs at ts=110 (persistence K=3 satisfied for svcc at step 10 + 2 = 12 -> ts 112)
        self.assertEqual(res.confirmed_onset, 112)

    def test_4_confirmation_at_first_stopping_timestamp(self) -> None:
        """Stopping condition triggers at the earliest moment Z >= 3 is satisfied, without further delay."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        statuses["svcb_m1"] = ["normal"] * 3 + ["anomaly"] * (n - 3)
        statuses["svcb_m2"] = ["normal"] * 3 + ["anomaly"] * (n - 3)
        statuses["svcc_m1"] = ["normal"] * 6 + ["anomaly"] * (n - 6)
        statuses["svcc_m2"] = ["normal"] * 6 + ["anomaly"] * (n - 6)
        # svcd also becomes anomalous much later at step 20
        statuses["svcd_m1"] = ["normal"] * 20 + ["anomaly"] * (n - 20)
        statuses["svcd_m2"] = ["normal"] * 20 + ["anomaly"] * (n - 20)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[
                Dependency("svca", "svcb"),
                Dependency("svca", "svcc"),
                Dependency("svca", "svcd"),
            ],
        )
        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        self.assertEqual(res.status, "confirmed")
        self.assertEqual(res.confirmed_onset, 108)  # Step 6 + 2 = 8 -> ts 108
        self.assertNotIn("svcd", res.corroborating_entities)

    def test_5_future_telemetry_cannot_move_confirmation_earlier(self) -> None:
        """Extending telemetry into the future cannot alter an earlier confirmed onset."""
        ts_short = self.timestamps[:30]
        n_short = len(ts_short)
        statuses_short = {m: ["normal"] * n_short for m in self.metric_names}
        statuses_short["svca_m1"] = ["anomaly"] * n_short
        statuses_short["svca_m2"] = ["anomaly"] * n_short
        statuses_short["svcb_m1"] = ["normal"] * 4 + ["anomaly"] * (n_short - 4)
        statuses_short["svcb_m2"] = ["normal"] * 4 + ["anomaly"] * (n_short - 4)
        statuses_short["svcc_m1"] = ["normal"] * 8 + ["anomaly"] * (n_short - 8)
        statuses_short["svcc_m2"] = ["normal"] * 8 + ["anomaly"] * (n_short - 8)

        case_short = _make_telemetry_case(self.metric_names, ts_short)
        bocpd_short = BOCPDResult(
            case_id="case_seq_test",
            onset_ts=100,
            status="detected",
            audit={},
            changepoints=(0,),
            timestamps=ts_short,
        )
        det_short = _make_anomaly_result(statuses_short, ts_short)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[Dependency("svca", "svcb"), Dependency("svca", "svcc")],
        )
        res_short = confirm_sequential_tcec(case_short, bocpd_short, anomaly_result=det_short, graph=graph)
        self.assertEqual(res_short.status, "confirmed")
        t_conf_short = res_short.confirmed_onset

        # Extend with additional chaotic points
        statuses_long = {m: list(statuses_short[m]) + ["anomaly"] * (len(self.timestamps) - n_short) for m in self.metric_names}
        det_long = _make_anomaly_result(statuses_long, self.timestamps)
        case_long = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd_long = self._build_bocpd(onset_ts=100)

        res_long = confirm_sequential_tcec(case_long, bocpd_long, anomaly_result=det_long, graph=graph)
        self.assertEqual(res_long.status, "confirmed")
        self.assertEqual(res_long.confirmed_onset, t_conf_short)

    def test_6_future_topology_edge_cannot_influence_earlier_evidence(self) -> None:
        """Trace dependency edge svca -> svcb only appears at t=200; at t=100 it cannot confirm."""
        trace_data = {
            "spanID": ["s1", "s2"],
            "parentSpanID": [None, "s1"],
            "serviceName": ["svca", "svcb"],
            "startTime": [200, 200],  # Span only occurs at t=200
            "duration": [1, 1],
        }
        case = _make_telemetry_case(self.metric_names, self.timestamps, trace_raw_data=trace_data)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * 20 + ["normal"] * (n - 20)  # active 100 to 120
        statuses["svca_m2"] = ["anomaly"] * 20 + ["normal"] * (n - 20)
        statuses["svcb_m1"] = ["normal"] * 5 + ["anomaly"] * 15 + ["normal"] * (n - 20)
        statuses["svcb_m2"] = ["normal"] * 5 + ["anomaly"] * 15 + ["normal"] * (n - 20)

        det = _make_anomaly_result(statuses, self.timestamps)

        # In the interval 100..120, the trace edge svca -> svcb has NOT occurred yet.
        # When max_timestamp is bounded by curr_ts, no edge exists during 100..120!
        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det)
        self.assertEqual(res.status, "unconfirmed")

    def test_7_forward_and_reverse_topology_distinguished(self) -> None:
        """Verifies that FORWARD (caller -> callee) and REVERSE (callee -> caller) are distinguished in details."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        statuses["svcb_m1"] = ["normal"] * 3 + ["anomaly"] * (n - 3)
        statuses["svcb_m2"] = ["normal"] * 3 + ["anomaly"] * (n - 3)
        statuses["svcc_m1"] = ["normal"] * 6 + ["anomaly"] * (n - 6)
        statuses["svcc_m2"] = ["normal"] * 6 + ["anomaly"] * (n - 6)

        det = _make_anomaly_result(statuses, self.timestamps)
        # svca calls svcb (FORWARD); svcc calls svca (REVERSE)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[Dependency("svca", "svcb"), Dependency("svcc", "svca")],
        )
        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        self.assertEqual(res.status, "confirmed")
        details = {d.entity: d.direction for d in res.corroborating_details}
        self.assertEqual(details["svcb"], "FORWARD")
        self.assertEqual(details["svcc"], "REVERSE")

    def test_8_stale_reset_when_memory_window_elapsed(self) -> None:
        """When an anomalous entity episode ends and > W=60s pass with no active entities, suspect hypothesis resets to NORMAL."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        # svca anomalous for 5 steps (ts 100 to 104), then normal for 70 steps (> W=60s)
        statuses["svca_m1"] = ["anomaly"] * 5 + ["normal"] * 70 + ["anomaly"] * (n - 75)
        statuses["svca_m2"] = ["anomaly"] * 5 + ["normal"] * 70 + ["anomaly"] * (n - 75)
        # svcb anomalous at step 75
        statuses["svcb_m1"] = ["normal"] * 75 + ["anomaly"] * (n - 75)
        statuses["svcb_m2"] = ["normal"] * 75 + ["anomaly"] * (n - 75)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        # Check state transitions: must have reset to NORMAL after 60s of silence
        transitions = [(t.get("from_state"), t.get("to_state")) for t in res.state_transitions]
        self.assertIn(("SUSPECT", "NORMAL"), transitions)

    def test_9_guarded_ground_truth_never_accessed(self) -> None:
        """Ensures ground_truth is strictly unread."""
        case = _make_telemetry_case(self.metric_names, self.timestamps, ground_truth_guard=True)
        bocpd = self._build_bocpd(onset_ts=100)
        det = _make_anomaly_result({m: ["normal"] * len(self.timestamps) for m in self.metric_names}, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[])
        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)
        self.assertEqual(res.status, "unconfirmed")

    def test_10_deterministic_repeatability(self) -> None:
        """Running the same case twice produces byte-identical output."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        statuses["svcb_m1"] = ["normal"] * 3 + ["anomaly"] * (n - 3)
        statuses["svcb_m2"] = ["normal"] * 3 + ["anomaly"] * (n - 3)
        statuses["svcc_m1"] = ["normal"] * 6 + ["anomaly"] * (n - 6)
        statuses["svcc_m2"] = ["normal"] * 6 + ["anomaly"] * (n - 6)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[Dependency("svca", "svcb"), Dependency("svca", "svcc")],
        )

        res1 = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)
        res2 = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        self.assertEqual(res1.status, res2.status)
        self.assertEqual(res1.confirmed_onset, res2.confirmed_onset)
        self.assertEqual(res1.candidate_entity, res2.candidate_entity)
        self.assertEqual(res1.corroborating_entities, res2.corroborating_entities)
        self.assertEqual(res1.state_transitions, res2.state_transitions)

    def test_11_treatment_does_not_modify_existing_tcec_behavior(self) -> None:
        """Verifies that existing Stage 5 confirm_topology_coherent_episode produces identical output."""
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        statuses["svcb_m1"] = ["normal"] * 5 + ["anomaly"] * (n - 5)
        statuses["svcb_m2"] = ["normal"] * 5 + ["anomaly"] * (n - 5)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(self.metric_names, dependencies=[Dependency("svca", "svcb")])

        # Stage 5 TCEC: confirms on single pairwise edge (Z=2)
        res_s5 = confirm_topology_coherent_episode(case, bocpd, graph=graph, anomaly_result=det)
        self.assertEqual(res_s5.status, "confirmed")
        self.assertEqual(res_s5.candidate_entity, "svca")
        self.assertEqual(res_s5.confirming_entity, "svcb")

        # Stage 9 Sequential TCEC: requires Z >= 3, remains unconfirmed on single edge
        res_s9 = confirm_sequential_tcec(case, bocpd, graph=graph, anomaly_result=det)
        self.assertEqual(res_s9.status, "unconfirmed")

    def test_12_non_backdating_stopping_time(self) -> None:
        """Initial evidence at t=100, corroboration at t=110, second corroboration at t=120.

        Stopping condition becomes true at t=120 (or when episode persistence completes at t=122).
        Asserts result.confirmed_onset == 122 (tau), NOT 100 or 110.
        Asserts result.confirmation_time == result.confirmed_onset.
        Asserts result.first_evidence_timestamp == 100.
        """
        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        # svca anomaly starts at t=100 (step 0)
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        # svcb corroboration starts at t=110 (step 10)
        statuses["svcb_m1"] = ["normal"] * 10 + ["anomaly"] * (n - 10)
        statuses["svcb_m2"] = ["normal"] * 10 + ["anomaly"] * (n - 10)
        # svcc second corroboration starts at t=120 (step 20)
        statuses["svcc_m1"] = ["normal"] * 20 + ["anomaly"] * (n - 20)
        statuses["svcc_m2"] = ["normal"] * 20 + ["anomaly"] * (n - 20)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[Dependency("svca", "svcb"), Dependency("svcb", "svcc")],
        )

        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)

        self.assertEqual(res.status, "confirmed")
        # Step 20 + persistence 2 = step 22 -> ts 122
        stopping_time = 122
        self.assertEqual(res.confirmed_onset, stopping_time)
        self.assertEqual(res.confirmation_time, stopping_time)
        self.assertNotEqual(res.confirmed_onset, 100)
        self.assertNotEqual(res.confirmed_onset, 102)
        self.assertNotEqual(res.confirmed_onset, 110)
        # First episode evidence started at ts=102 (persistence K=3 from 100)
        self.assertEqual(res.first_evidence_timestamp, 102)

    def test_13_rca_analysis_end_equals_stopping_time(self) -> None:
        """Verifies that the RCA analysis_end window corresponds exactly to stopping time tau.

        Tests that when earliest evidence onset is much earlier than stopping time,
        the analysis window cutoff matches tau and does NOT leak past tau.
        """
        from eval.windows import resolve_incident_window
        from digital_detective.anomaly import truncate_metric_anomaly_result

        case = _make_telemetry_case(self.metric_names, self.timestamps)
        bocpd = self._build_bocpd(onset_ts=100)
        n = len(self.timestamps)
        statuses = {m: ["normal"] * n for m in self.metric_names}
        statuses["svca_m1"] = ["anomaly"] * n
        statuses["svca_m2"] = ["anomaly"] * n
        statuses["svcb_m1"] = ["normal"] * 10 + ["anomaly"] * (n - 10)
        statuses["svcb_m2"] = ["normal"] * 10 + ["anomaly"] * (n - 10)
        statuses["svcc_m1"] = ["normal"] * 25 + ["anomaly"] * (n - 25)
        statuses["svcc_m2"] = ["normal"] * 25 + ["anomaly"] * (n - 25)

        det = _make_anomaly_result(statuses, self.timestamps)
        graph = build_entity_graph(
            self.metric_names,
            dependencies=[Dependency("svca", "svcb"), Dependency("svcb", "svcc")],
        )

        res = confirm_sequential_tcec(case, bocpd, anomaly_result=det, graph=graph)
        self.assertEqual(res.status, "confirmed")
        tau = res.confirmed_onset
        self.assertIsNotNone(tau)
        self.assertEqual(tau, 127)  # Step 25 + 2 = 27 -> ts 127

        window = resolve_incident_window(
            case,
            mode="detected",
            timestamps=res.timestamps,
            custom_onset_ts=tau,
            custom_end_ts=tau,
        )
        self.assertEqual(window.end_ts, tau)
        self.assertEqual(window.onset_ts, tau)

        truncated_det = truncate_metric_anomaly_result(det, max_timestamp=res.confirmed_onset)
        self.assertEqual(max(truncated_det.timestamps), tau)


if __name__ == "__main__":
    unittest.main()

