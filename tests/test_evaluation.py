"""Unit tests for anomaly detection evaluator."""

from __future__ import annotations

import math
from typing import Any
import unittest

import pyarrow as pa

from digital_detective.anomaly import MetricAnomalyResult, detect_metric_anomalies
from digital_detective.evaluation import (
    OUTCOME_CLEAN_DETECTION,
    OUTCOME_COMPLETE_MISS,
    OUTCOME_EARLY_ALARM_ONLY,
    OUTCOME_NOISY_DETECTION,
    VALID_OUTCOMES,
    AggregateEvaluationResult,
    CaseEvaluationResult,
    evaluate_anomaly_detection_case,
    evaluate_anomaly_detection_cases,
)
from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
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


class AnomalyEvaluationTests(unittest.TestCase):
    def test_clean_detection(self) -> None:
        # 15 timesteps: 0..4 warmup, 5..9 normal, 10 inject_time
        # Fault at 12
        n = 15
        vals = [1.0] * 12 + [100.0] + [100.0, 100.0]
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_clean", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        self.assertEqual(res.case_id, "case_clean")
        self.assertEqual(res.suite, "RE1")
        self.assertEqual(res.system, "ob")
        self.assertEqual(res.fault_type, "cpu")
        self.assertEqual(res.inject_time, 10)
        self.assertEqual(res.t_eval_end, 14)
        self.assertEqual(res.outcome, OUTCOME_CLEAN_DETECTION)
        self.assertIsNone(res.first_pre_alarm_ts)
        self.assertEqual(res.t_detect, 12)
        self.assertEqual(res.detection_latency_sec, 2)
        self.assertEqual(res.pre_injection_far, 0.0)
        self.assertEqual(res.n_eval_pre_steps, 5)  # 5..9
        self.assertEqual(res.n_missing_pre_steps, 0)
        self.assertGreater(res.post_injection_density, 0.0)
        self.assertIn("cpu", res.top_anomalous_metrics)

    def test_noisy_detection(self) -> None:
        # Pre-injection anomaly at 5, post-injection anomaly at 11
        # Window size = 5. Preceding window for 11 is 6..10 (all 1.0), so 11 is cleanly detected.
        n = 15
        vals = [1.0] * 5 + [100.0] + [1.0] * 5 + [100.0] + [1.0] * 3
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_noisy", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        self.assertEqual(res.outcome, OUTCOME_NOISY_DETECTION)
        self.assertEqual(res.first_pre_alarm_ts, 5)
        self.assertEqual(res.t_detect, 11)
        self.assertEqual(res.detection_latency_sec, 1)
        self.assertGreater(res.pre_injection_far, 0.0)
        self.assertEqual(res.pre_injection_far, 1 / 5)

    def test_early_alarm_only(self) -> None:
        # Pre-injection anomaly at 7, no post-injection anomaly
        n = 15
        vals = [1.0] * 7 + [100.0] + [1.0] * 7
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_early", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        self.assertEqual(res.outcome, OUTCOME_EARLY_ALARM_ONLY)
        self.assertEqual(res.first_pre_alarm_ts, 7)
        self.assertIsNone(res.t_detect)
        self.assertIsNone(res.detection_latency_sec)
        self.assertGreater(res.pre_injection_far, 0.0)
        self.assertEqual(res.post_injection_density, 0.0)

    def test_complete_miss(self) -> None:
        # Flat normal signal throughout: no pre-injection or post-injection alarms
        n = 15
        vals = [1.0] * n
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_miss", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        self.assertEqual(res.outcome, OUTCOME_COMPLETE_MISS)
        self.assertIsNone(res.first_pre_alarm_ts)
        self.assertIsNone(res.t_detect)
        self.assertIsNone(res.detection_latency_sec)
        self.assertEqual(res.pre_injection_far, 0.0)
        self.assertEqual(res.post_injection_density, 0.0)

    def test_detection_exactly_at_injection(self) -> None:
        # Anomaly occurs at exactly index 10 (inject_time=10)
        n = 15
        vals = [1.0] * 10 + [100.0] + [1.0] * 4
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_exact", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        self.assertEqual(res.outcome, OUTCOME_CLEAN_DETECTION)
        self.assertEqual(res.t_detect, 10)
        self.assertEqual(res.detection_latency_sec, 0)

    def test_pre_injection_alarm_does_not_create_negative_latency(self) -> None:
        # Alarm before injection only; verify latency is None, not negative
        n = 15
        vals = [1.0] * 6 + [50.0] + [1.0] * 8
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_early_no_neg_lat", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        self.assertEqual(res.outcome, OUTCOME_EARLY_ALARM_ONLY)
        self.assertEqual(res.first_pre_alarm_ts, 6)
        self.assertIsNone(res.detection_latency_sec)
        self.assertIsNone(res.t_detect)

    def test_warmup_handling_and_exclusion(self) -> None:
        # A spike during the warmup window (0..4) is marked "warmup" and must NOT count as early alarm
        n = 15
        vals = [1.0, 1.0, 999.0, 1.0, 1.0] + [1.0] * 10
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_warmup", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)
        self.assertEqual(anomaly_result.evaluation_statuses["cpu"][2], "warmup")

        res = evaluate_anomaly_detection_case(case, anomaly_result)
        # The spike at index 2 is in warmup, so pre_injection_far must be 0
        self.assertEqual(res.pre_injection_far, 0.0)
        self.assertIsNone(res.first_pre_alarm_ts)
        self.assertEqual(res.outcome, OUTCOME_COMPLETE_MISS)

    def test_invalid_warmup_window_raises_error(self) -> None:
        # inject_time <= t_warmup raises ValueError mentioning invalid_warmup_window
        table = pa.table({"time": list(range(10)), "cpu": [1.0] * 10})
        # min_warmup=5 means t_warmup=5. Setting inject_time=4 violates precondition
        case = _make_case(table, case_id="case_invalid_warmup", inject_time=4)
        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5)

        with self.assertRaisesRegex(ValueError, "invalid_warmup_window"):
            evaluate_anomaly_detection_case(case, anomaly_result)

    def test_missing_and_insufficient_history_denominator_handling(self) -> None:
        # Pre-injection window has 5 steps: 5..9
        # Step 6 has NaN (missing_observation)
        # Step 7 has insufficient history if min_valid_history=5
        # The denominator must exclude steps where all metrics are un-evaluated
        n = 15
        vals = [1.0] * 6 + [float("nan")] + [1.0] * 8
        table = pa.table({"time": list(range(n)), "cpu": vals})
        case = _make_case(table, case_id="case_missing_denom", inject_time=10)

        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, min_valid_history=3)
        res = evaluate_anomaly_detection_case(case, anomaly_result)

        # Pre-injection indices: 5, 6, 7, 8, 9 (total 5)
        # Index 6 is "missing_observation"
        # So evaluated_pre_steps = 4, missing_pre_steps = 1
        self.assertEqual(res.n_eval_pre_steps, 4)
        self.assertEqual(res.n_missing_pre_steps, 1)
        self.assertEqual(res.pre_injection_far, 0.0)

    def test_any_metric_or_gate(self) -> None:
        # Two metrics: cpu stays normal, mem triggers an anomaly at index 11
        n = 15
        table = pa.table({
            "time": list(range(n)),
            "cpu": [1.0] * n,
            "mem": [2.0] * 11 + [200.0] + [2.0] * 3,
        })
        case = _make_case(table, case_id="case_multi_metric", inject_time=10)
        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)

        res = evaluate_anomaly_detection_case(case, anomaly_result)
        self.assertEqual(res.outcome, OUTCOME_CLEAN_DETECTION)
        self.assertEqual(res.t_detect, 11)
        self.assertEqual(res.detection_latency_sec, 1)
        self.assertEqual(res.top_anomalous_metrics[0], "mem")

    def test_case_id_mismatch_and_ground_truth_errors(self) -> None:
        table = pa.table({"time": list(range(10)), "cpu": [1.0] * 10})
        case = _make_case(table, case_id="case_a", inject_time=8)
        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5)

        mismatched_case = _make_case(table, case_id="case_b", inject_time=8)
        with self.assertRaisesRegex(ValueError, "Case ID mismatch"):
            evaluate_anomaly_detection_case(mismatched_case, anomaly_result)

        no_gt_case = TelemetryCase(
            metadata=CaseMetadata(case_id="case_a"),
            ground_truth=GroundTruth(values={}),
            metrics=case.metrics,
        )
        with self.assertRaisesRegex(ValueError, "has no inject_time"):
            evaluate_anomaly_detection_case(no_gt_case, anomaly_result)

    def test_deterministic_repeated_execution(self) -> None:
        n = 20
        vals = [1.0] * 15 + [50.0] * 5
        table = pa.table({"time": list(range(n)), "cpu": vals, "mem": [2.0] * n})
        case = _make_case(table, case_id="case_det", inject_time=14)
        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)

        reference = evaluate_anomaly_detection_case(case, anomaly_result)
        for _ in range(50):
            repeated = evaluate_anomaly_detection_case(case, anomaly_result)
            self.assertEqual(reference, repeated)

    def test_t_eval_end_is_always_final_telemetry_timestamp(self) -> None:
        # Telemetry with irregular or extended timestamps
        timestamps = [100, 110, 125, 140, 160, 180, 210, 250, 300, 350, 420, 500]
        n = len(timestamps)
        vals = [1.0] * 8 + [50.0] * 4
        table = pa.table({"time": timestamps, "cpu": vals})
        case = _make_case(table, case_id="case_horizon_check", inject_time=200)
        anomaly_result = detect_metric_anomalies(case, window_size=5, min_warmup=5, threshold=3.0)

        res = evaluate_anomaly_detection_case(case, anomaly_result)

        # t_eval_end must strictly equal the final observation timestamp
        self.assertEqual(res.t_eval_end, 500)
        self.assertEqual(res.t_eval_end, timestamps[-1])
        self.assertEqual(res.t_eval_end, anomaly_result.timestamps[-1])

        # Verify that caller cannot pass horizon_seconds or any unexpected horizon parameter
        with self.assertRaises(TypeError):
            evaluate_anomaly_detection_case(case, anomaly_result, horizon_seconds=60)  # type: ignore[call-arg]

    def test_aggregate_evaluation_computation(self) -> None:
        # Create 4 synthetic cases representing the 4 mutually exclusive outcomes
        # 1. Clean Detection (latency=2, far=0.0)
        c1 = CaseEvaluationResult(
            case_id="c1",
            suite="RE1",
            system="ob",
            fault_type="cpu",
            inject_time=100,
            t_eval_end=200,
            outcome=OUTCOME_CLEAN_DETECTION,
            first_pre_alarm_ts=None,
            t_detect=102,
            detection_latency_sec=2,
            pre_injection_far=0.0,
            n_eval_pre_steps=50,
            n_missing_pre_steps=0,
            post_injection_density=0.5,
            top_anomalous_metrics=("cpu",),
        )
        # 2. Noisy Detection (latency=8, far=0.1)
        c2 = CaseEvaluationResult(
            case_id="c2",
            suite="RE1",
            system="ob",
            fault_type="cpu",
            inject_time=100,
            t_eval_end=200,
            outcome=OUTCOME_NOISY_DETECTION,
            first_pre_alarm_ts=80,
            t_detect=108,
            detection_latency_sec=8,
            pre_injection_far=0.1,
            n_eval_pre_steps=50,
            n_missing_pre_steps=0,
            post_injection_density=0.4,
            top_anomalous_metrics=("cpu",),
        )
        # 3. Early Alarm Only (latency=None, far=0.2)
        c3 = CaseEvaluationResult(
            case_id="c3",
            suite="RE2",
            system="ss",
            fault_type="mem",
            inject_time=100,
            t_eval_end=200,
            outcome=OUTCOME_EARLY_ALARM_ONLY,
            first_pre_alarm_ts=70,
            t_detect=None,
            detection_latency_sec=None,
            pre_injection_far=0.2,
            n_eval_pre_steps=50,
            n_missing_pre_steps=0,
            post_injection_density=0.0,
            top_anomalous_metrics=("mem",),
        )
        # 4. Complete Miss (latency=None, far=0.0)
        c4 = CaseEvaluationResult(
            case_id="c4",
            suite="RE2",
            system="ss",
            fault_type="mem",
            inject_time=100,
            t_eval_end=200,
            outcome=OUTCOME_COMPLETE_MISS,
            first_pre_alarm_ts=None,
            t_detect=None,
            detection_latency_sec=None,
            pre_injection_far=0.0,
            n_eval_pre_steps=50,
            n_missing_pre_steps=0,
            post_injection_density=0.0,
            top_anomalous_metrics=(),
        )

        agg = evaluate_anomaly_detection_cases([c1, c2, c3, c4])

        self.assertEqual(agg.total_cases, 4)
        self.assertAlmostEqual(agg.overall_detection_rate, 2 / 4)
        self.assertAlmostEqual(agg.clean_detection_rate, 1 / 4)
        self.assertAlmostEqual(agg.early_alarm_rate, 2 / 4)  # noisy + early only
        self.assertAlmostEqual(agg.missed_detection_rate, 1 / 4)
        self.assertAlmostEqual(agg.mean_latency_sec, 5.0)  # (2 + 8) / 2
        self.assertAlmostEqual(agg.median_latency_sec, 5.0)
        self.assertAlmostEqual(agg.mean_pre_injection_far, (0.0 + 0.1 + 0.2 + 0.0) / 4)

        # Verify breakdowns
        self.assertIn("RE1", agg.by_suite)
        self.assertIn("RE2", agg.by_suite)
        self.assertEqual(agg.by_suite["RE1"].total_cases, 2)
        self.assertEqual(agg.by_suite["RE1"].clean_detection_rate, 0.5)

        self.assertIn("ob", agg.by_system)
        self.assertIn("ss", agg.by_system)
        self.assertEqual(agg.by_system["ob"].total_cases, 2)

        self.assertIn("cpu", agg.by_fault)
        self.assertIn("mem", agg.by_fault)
        self.assertEqual(agg.by_fault["cpu"].total_cases, 2)

    def test_aggregate_empty_cases(self) -> None:
        agg = evaluate_anomaly_detection_cases([])
        self.assertEqual(agg.total_cases, 0)
        self.assertEqual(agg.overall_detection_rate, 0.0)
        self.assertIsNone(agg.median_latency_sec)
        self.assertIsNone(agg.mean_latency_sec)


if __name__ == "__main__":
    unittest.main()
