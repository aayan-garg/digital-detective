"""Unit tests for metric anomaly aggregation and case alarm generation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import unittest

import pyarrow as pa

from digital_detective.aggregation import (
    AggregationConfig,
    CaseAlarmSequence,
    STANDARD_CONFIGURATIONS,
    aggregate_metric_anomalies,
)
from digital_detective.anomaly import MetricAnomalyResult, detect_metric_anomalies
from digital_detective.evaluation import (
    OUTCOME_CLEAN_DETECTION,
    OUTCOME_NOISY_DETECTION,
    evaluate_anomaly_detection_case,
)
from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)


def _make_anomaly_result(
    statuses: Mapping[str, Sequence[str]],
    case_id: str = "test_case",
    timestamps: Sequence[int] | None = None,
) -> MetricAnomalyResult:
    n = len(next(iter(statuses.values())))
    ts = tuple(range(n)) if timestamps is None else tuple(timestamps)
    return MetricAnomalyResult(
        case_id=case_id,
        timestamps=ts,
        metric_names=tuple(statuses.keys()),
        anomaly_scores={m: tuple(3.5 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        signed_scores={m: tuple(3.5 if s == "anomaly" else 0.0 for s in st) for m, st in statuses.items()},
        anomalies={m: tuple(s == "anomaly" for s in st) for m, st in statuses.items()},
        evaluation_statuses={m: tuple(st) for m, st in statuses.items()},
        valid_history_counts={m: tuple(60 for _ in st) for m, st in statuses.items()},
        summary={
            "metrics": {
                m: {"peak_score": 3.5, "anomaly_count": sum(1 for s in st if s == "anomaly")}
                for m, st in statuses.items()
            }
        },
    )


def _make_case(
    raw_data: Any,
    case_id: str = "test_case",
    inject_time: int = 10,
    timestamp_field: str = "time",
    suite: str = "RE1",
    system: str = "ob",
    fault: str = "cpu",
) -> TelemetryCase:
    field_names = tuple(raw_data.column_names) if hasattr(raw_data, "column_names") else tuple(raw_data.keys())
    return TelemetryCase(
        metadata=CaseMetadata(case_id=case_id, suite=suite, system=system),
        ground_truth=GroundTruth(values={"inject_time": inject_time, "fault": fault}),
        metrics=TelemetryModality(
            raw_data=raw_data,
            provenance=ModalityProvenance(
                source_dataset="test-dataset",
                source_case=case_id,
                original_format="metrics.parquet",
                original_field_names=field_names,
                timestamp_field=timestamp_field,
            ),
        ),
    )


class AggregationTests(unittest.TestCase):
    def test_baseline_compatibility(self) -> None:
        # persistence=1, consensus=1 must be identical to Any-Metric OR
        statuses = {
            "cpu": ["warmup"] * 5 + ["anomaly"] + ["normal"] * 9,
            "mem": ["warmup"] * 5 + ["normal"] * 7 + ["anomaly"] + ["normal"] * 2,
        }
        det_res = _make_anomaly_result(statuses, case_id="case_compat")

        alarms_c0 = aggregate_metric_anomalies(det_res, AggregationConfig(persistence=1, consensus=1))
        self.assertEqual(alarms_c0.persistence, 1)
        self.assertEqual(alarms_c0.consensus, 1)

        expected_or = tuple(
            any(det_res.evaluation_statuses[m][i] == "anomaly" for m in det_res.metric_names)
            for i in range(15)
        )
        self.assertEqual(alarms_c0.alarms, expected_or)

        # Default configuration without config argument must match baseline C0
        alarms_default = aggregate_metric_anomalies(det_res)
        self.assertEqual(alarms_default.alarms, expected_or)
        self.assertEqual(alarms_default.persistence, 1)
        self.assertEqual(alarms_default.consensus, 1)

    def test_persistence_streak_accumulation_and_reset(self) -> None:
        # k=3: Requires 3 consecutive observations with status="anomaly"
        statuses = {
            "cpu": (
                ["warmup"] * 5
                + ["anomaly", "anomaly", "anomaly", "anomaly"]  # indices 5, 6, 7, 8
                + ["normal"]  # index 9
                + ["anomaly", "anomaly"]  # indices 10, 11
                + ["normal"]  # index 12
                + ["normal"] * 7
            )
        }
        det_res = _make_anomaly_result(statuses, case_id="case_persist")

        alarms_k3 = aggregate_metric_anomalies(det_res, AggregationConfig(persistence=3, consensus=1))

        # Index 5 (streak 1), 6 (streak 2): False
        self.assertFalse(alarms_k3.alarms[5])
        self.assertFalse(alarms_k3.alarms[6])
        # Index 7 (streak 3), 8 (streak 4): True
        self.assertTrue(alarms_k3.alarms[7])
        self.assertTrue(alarms_k3.alarms[8])
        # Index 9 (normal): False
        self.assertFalse(alarms_k3.alarms[9])
        # Index 10 (streak 1), 11 (streak 2): False
        self.assertFalse(alarms_k3.alarms[10])
        self.assertFalse(alarms_k3.alarms[11])
        # Index 12 (normal): False
        self.assertFalse(alarms_k3.alarms[12])

    def test_warmup_and_missing_data_resets_streak(self) -> None:
        # Anomaly during warmup cannot contribute to a streak
        # Missing observation (missing_observation/insufficient_history) breaks a streak
        statuses = {
            "cpu": [
                "warmup",
                "warmup",
                "warmup",
                "warmup",
                "warmup",
                "anomaly",  # 5 (streak 1)
                "anomaly",  # 6 (streak 2)
                "missing_observation",  # 7 (streak reset to 0)
                "anomaly",  # 8 (streak 1)
                "insufficient_history",  # 9 (streak reset to 0)
                "anomaly",  # 10 (streak 1)
                "normal",  # 11
            ]
        }
        det_res = _make_anomaly_result(statuses, case_id="case_warmup_reset")

        alarms_k3 = aggregate_metric_anomalies(det_res, AggregationConfig(persistence=3, consensus=1))
        for idx in range(len(statuses["cpu"])):
            self.assertFalse(alarms_k3.alarms[idx])

    def test_metric_count_consensus(self) -> None:
        # 3 metrics: cpu, mem, disk
        # M=2: at least 2 metrics must be anomalous at the same step
        statuses = {
            "cpu": ["warmup"] * 5 + ["normal", "anomaly", "anomaly", "normal"],
            "mem": ["warmup"] * 5 + ["normal", "normal", "anomaly", "normal"],
            "disk": ["warmup"] * 5 + ["normal", "normal", "normal", "normal"],
        }
        det_res = _make_anomaly_result(statuses, case_id="case_consensus")

        # At index 6: only cpu is anomalous (count=1 < 2)
        # At index 7: cpu and mem are anomalous (count=2 >= 2)
        alarms_m2 = aggregate_metric_anomalies(det_res, AggregationConfig(persistence=1, consensus=2))
        self.assertFalse(alarms_m2.alarms[6])
        self.assertTrue(alarms_m2.alarms[7])
        self.assertEqual(alarms_m2.active_metric_counts[6], 1)
        self.assertEqual(alarms_m2.active_metric_counts[7], 2)

        # With consensus=3: neither step 6 nor 7 has 3 metrics
        alarms_m3 = aggregate_metric_anomalies(det_res, AggregationConfig(persistence=1, consensus=3))
        self.assertFalse(alarms_m3.alarms[6])
        self.assertFalse(alarms_m3.alarms[7])

    def test_combined_persistence_and_consensus(self) -> None:
        # k=2, M=2: at least 2 metrics must each have an anomaly streak of >= 2
        statuses = {
            "cpu": ["warmup"] * 5 + ["anomaly", "anomaly", "anomaly", "normal"],  # indices 5, 6, 7
            "mem": ["warmup"] * 5 + ["normal", "anomaly", "anomaly", "normal"],  # indices 6, 7
        }
        det_res = _make_anomaly_result(statuses, case_id="case_combined")

        alarms_k2_m2 = aggregate_metric_anomalies(det_res, AggregationConfig(persistence=2, consensus=2))
        # Index 5: cpu streak=1, mem streak=0 -> active_counts=0
        self.assertEqual(alarms_k2_m2.active_metric_counts[5], 0)
        self.assertFalse(alarms_k2_m2.alarms[5])
        # Index 6: cpu streak=2, mem streak=1 -> active_counts=1 (cpu only)
        self.assertEqual(alarms_k2_m2.active_metric_counts[6], 1)
        self.assertFalse(alarms_k2_m2.alarms[6])
        # Index 7: cpu streak=3, mem streak=2 -> active_counts=2 (both cpu and mem)
        self.assertEqual(alarms_k2_m2.active_metric_counts[7], 2)
        self.assertTrue(alarms_k2_m2.alarms[7])

    def test_causality_strictly_preserved(self) -> None:
        # Changing future observations cannot alter past alarms
        statuses_a = {
            "cpu": ["warmup"] * 5 + ["anomaly", "anomaly", "anomaly"] + ["normal"] * 12,
        }
        statuses_b = {
            "cpu": ["warmup"] * 5 + ["anomaly", "anomaly", "anomaly"] + ["normal"] * 5 + ["anomaly"] * 7,
        }
        det_a = _make_anomaly_result(statuses_a, case_id="case_causal_a")
        det_b = _make_anomaly_result(statuses_b, case_id="case_causal_b")

        alarms_a = aggregate_metric_anomalies(det_a, AggregationConfig(persistence=3, consensus=1))
        alarms_b = aggregate_metric_anomalies(det_b, AggregationConfig(persistence=3, consensus=1))

        # Indices 0..12 are identical in statuses
        self.assertEqual(alarms_a.alarms[:13], alarms_b.alarms[:13])
        self.assertEqual(alarms_a.active_metric_counts[:13], alarms_b.active_metric_counts[:13])

    def test_invalid_parameters_raise_error(self) -> None:
        statuses = {"cpu": ["warmup"] * 5 + ["normal"] * 5}
        det = _make_anomaly_result(statuses)

        with self.assertRaisesRegex(ValueError, "persistence must be >= 1"):
            aggregate_metric_anomalies(det, AggregationConfig(persistence=0))

        with self.assertRaisesRegex(ValueError, "consensus must be >= 1"):
            aggregate_metric_anomalies(det, AggregationConfig(consensus=0))

    def test_case_id_and_length_mismatch_raises_error(self) -> None:
        statuses = {"cpu": ["warmup"] * 5 + ["normal"] * 5}
        det_a = _make_anomaly_result(statuses, case_id="case_a")
        det_b = _make_anomaly_result(statuses, case_id="case_b")
        table = pa.table({"time": list(range(10)), "cpu": [1.0] * 10})
        case_a = _make_case(table, case_id="case_a", inject_time=8)

        alarms_b = aggregate_metric_anomalies(det_b)
        with self.assertRaisesRegex(ValueError, "Case ID mismatch"):
            evaluate_anomaly_detection_case(case_a, det_a, alarms_b)

        # Length mismatch
        short_seq = CaseAlarmSequence(
            case_id="case_a",
            timestamps=(1, 2),
            alarms=(False, False),
            active_metric_counts=(0, 0),
            persistence=1,
            consensus=1,
        )
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            evaluate_anomaly_detection_case(case_a, det_a, short_seq)

    def test_persistence_converts_noisy_to_clean_detection(self) -> None:
        # Pre-injection isolated spike at index 5 (1 observation)
        # Inject time at 10
        # Post-injection sustained failure at index 11, 12, 13 (3 observations)
        n = 15
        statuses = {
            "cpu": (
                ["warmup"] * 5
                + ["anomaly"]  # index 5 (transient pre-injection spike)
                + ["normal"] * 4  # indices 6..9 (inject_time=10)
                + ["normal"]  # index 10
                + ["anomaly", "anomaly", "anomaly", "anomaly"]  # indices 11, 12, 13, 14
            )
        }
        table = pa.table({"time": list(range(n)), "cpu": [1.0] * n})
        case = _make_case(table, case_id="case_clean_convert", inject_time=10)
        det = _make_anomaly_result(statuses, case_id="case_clean_convert")

        # Baseline (k=1, M=1) flags pre-injection alarm at 5 -> Noisy_Detection
        eval_base = evaluate_anomaly_detection_case(case, det)
        self.assertEqual(eval_base.outcome, OUTCOME_NOISY_DETECTION)
        self.assertGreater(eval_base.pre_injection_far, 0.0)

        # With k=3, the 1-observation pre-injection spike is suppressed (FAR=0)
        # Sustained anomaly at 11, 12, 13 reaches streak 3 at index 13 -> Clean_Detection!
        alarms_k3 = aggregate_metric_anomalies(det, AggregationConfig(persistence=3, consensus=1))
        eval_k3 = evaluate_anomaly_detection_case(case, det, alarms_k3)
        self.assertEqual(eval_k3.outcome, OUTCOME_CLEAN_DETECTION)
        self.assertEqual(eval_k3.pre_injection_far, 0.0)
        self.assertIsNone(eval_k3.first_pre_alarm_ts)
        self.assertEqual(eval_k3.t_detect, 13)
        self.assertEqual(eval_k3.detection_latency_sec, 3)

    def test_all_standard_configurations_run(self) -> None:
        statuses = {"cpu": ["warmup"] * 5 + ["normal"] * 10, "mem": ["warmup"] * 5 + ["normal"] * 10}
        table = pa.table({"time": list(range(15)), "cpu": [1.0] * 15, "mem": [2.0] * 15})
        case = _make_case(table, case_id="case_std_configs", inject_time=10)
        det = _make_anomaly_result(statuses, case_id="case_std_configs")

        for name, config in STANDARD_CONFIGURATIONS.items():
            alarms = aggregate_metric_anomalies(det, config)
            self.assertEqual(len(alarms.alarms), 15)
            res = evaluate_anomaly_detection_case(case, det, alarms)
            self.assertEqual(res.case_id, "case_std_configs")


if __name__ == "__main__":
    unittest.main()
