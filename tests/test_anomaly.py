"""Unit tests for basic metric anomaly detection baseline."""

from __future__ import annotations

import math
from typing import Any
import unittest

import pyarrow as pa

from digital_detective.anomaly import (
    detect_metric_anomalies,
    detect_metric_anomalies_median_mad,
    truncate_metric_anomaly_result,
)
from digital_detective.telemetry import (
    CaseMetadata,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)


def _make_case(
    raw_data: Any,
    case_id: str = "test_case",
    timestamp_field: str = "time",
) -> TelemetryCase:
    field_names = tuple(raw_data.column_names) if hasattr(raw_data, "column_names") else tuple(raw_data.keys())
    return TelemetryCase(
        metadata=CaseMetadata(case_id=case_id),
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


class AnomalyDetectionTests(unittest.TestCase):
    def test_missing_metrics_modality_raises_error(self) -> None:
        case = TelemetryCase(metadata=CaseMetadata(case_id="empty_case"), metrics=None)
        with self.assertRaisesRegex(ValueError, "has no metrics modality"):
            detect_metric_anomalies(case)

    def test_insufficient_warmup_history_raises_error(self) -> None:
        table = pa.table({"time": [1, 2, 3], "cpu": [0.1, 0.2, 0.3]})
        case = _make_case(table)
        with self.assertRaisesRegex(ValueError, "Insufficient observations"):
            detect_metric_anomalies(case, window_size=10, min_warmup=10)

    def test_timestamp_column_is_excluded_from_metrics(self) -> None:
        table = pa.table({
            "time": list(range(10)),
            "metric_a": [1.0] * 10,
            "metric_b": [2.0] * 10,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=5, min_warmup=5)

        self.assertNotIn("time", result.metric_names)
        self.assertEqual(result.metric_names, ("metric_a", "metric_b"))
        self.assertEqual(len(result.timestamps), 10)

    def test_output_state_semantics(self) -> None:
        # Verify distinct states: warmup, missing_observation, insufficient_history, normal, anomaly
        n = 15
        # 0..4 warmup
        # 5 normal
        # 6 missing (NaN)
        # 7..10 normal
        # 11 anomaly (100.0)
        vals = [1.0] * 6 + [float("nan")] + [1.0] * 4 + [100.0] + [1.0] * 3
        table = pa.table({
            "time": list(range(n)),
            "val": vals,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=5, min_warmup=5, min_valid_history=3)

        # 1. Warmup (0..4)
        for i in range(5):
            self.assertEqual(result.evaluation_statuses["val"][i], "warmup")
            self.assertIsNone(result.anomaly_scores["val"][i])
            self.assertIsNone(result.signed_scores["val"][i])
            self.assertFalse(result.anomalies["val"][i])

        # 2. Evaluated normal at index 5
        self.assertEqual(result.evaluation_statuses["val"][5], "normal")
        self.assertEqual(result.anomaly_scores["val"][5], 0.0)
        self.assertEqual(result.signed_scores["val"][5], 0.0)
        self.assertFalse(result.anomalies["val"][5])

        # 3. Missing/invalid observation at index 6
        self.assertEqual(result.evaluation_statuses["val"][6], "missing_observation")
        self.assertIsNone(result.anomaly_scores["val"][6])
        self.assertIsNone(result.signed_scores["val"][6])
        self.assertFalse(result.anomalies["val"][6])

        # 4. Evaluated anomaly at index 11
        self.assertEqual(result.evaluation_statuses["val"][11], "anomaly")
        self.assertGreater(result.anomaly_scores["val"][11], 3.0)
        self.assertTrue(result.anomalies["val"][11])

    def test_min_valid_history_rule_and_counts(self) -> None:
        # Window size 6, min_valid_history 4
        # At index 6, preceding window (0..5) has 4 NaNs and 2 valid values
        # Should be marked insufficient_history because 2 < 4
        n = 10
        vals = [1.0, float("nan"), float("nan"), float("nan"), float("nan"), 1.0, 5.0, 5.0, 5.0, 5.0]
        table = pa.table({
            "time": list(range(n)),
            "metric": vals,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=6, min_warmup=6, min_valid_history=4)

        # At index 6, valid history count in preceding 6 rows is 2 (< 4)
        self.assertEqual(result.valid_history_counts["metric"][6], 2)
        self.assertEqual(result.evaluation_statuses["metric"][6], "insufficient_history")
        self.assertIsNone(result.anomaly_scores["metric"][6])
        self.assertFalse(result.anomalies["metric"][6])

    def test_constant_history_equal_is_not_anomalous(self) -> None:
        # History is constant 5.0, current value is 5.0
        n = 20
        table = pa.table({
            "time": list(range(n)),
            "val": [5.0] * n,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10)

        # After warmup (index 10..19), score should be 0.0, status 'normal', no anomaly
        for i in range(10, n):
            self.assertEqual(result.evaluation_statuses["val"][i], "normal")
            self.assertEqual(result.anomaly_scores["val"][i], 0.0)
            self.assertEqual(result.signed_scores["val"][i], 0.0)
            self.assertFalse(result.anomalies["val"][i])
        self.assertEqual(result.summary["total_anomalies"], 0)

    def test_constant_history_different_is_anomalous(self) -> None:
        # History is constant 5.0, at index 10 it steps to 6.0
        n = 20
        values = [5.0] * 10 + [6.0] + [5.0] * 9
        table = pa.table({
            "time": list(range(n)),
            "val": values,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.0)

        # At index 10, invariant was broken -> must be anomalous with status 'anomaly'
        self.assertEqual(result.evaluation_statuses["val"][10], "anomaly")
        self.assertTrue(result.anomalies["val"][10])
        self.assertGreater(result.anomaly_scores["val"][10], 3.0)
        self.assertEqual(result.summary["first_anomaly_timestamp"], 10)

    def test_single_pulse_anomaly(self) -> None:
        # Stationary series around 10.0, with a sharp pulse at step 15
        n = 30
        values = [10.0, 10.1, 9.9, 10.0, 10.2] * 3  # 15 points
        values.append(50.0)  # index 15: extreme spike
        values.extend([10.0, 10.1, 9.9, 10.0, 10.2] * 2 + [10.0, 10.1, 9.9, 10.0])  # 14 points

        table = pa.table({
            "time": list(range(n)),
            "load": values,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.0)

        self.assertEqual(result.evaluation_statuses["load"][15], "anomaly")
        self.assertTrue(result.anomalies["load"][15])
        self.assertGreater(result.anomaly_scores["load"][15], 10.0)
        self.assertGreater(result.signed_scores["load"][15], 0.0)

    def test_step_increase_anomaly(self) -> None:
        # Steps from 0.0 to 10.0 at index 10
        n = 25
        values = [0.0, 0.1, -0.1, 0.05, -0.05] * 2 + [10.0] * 15
        table = pa.table({
            "time": list(range(n)),
            "traffic": values,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.0)

        # Anomaly begins exactly at step 10
        self.assertEqual(result.evaluation_statuses["traffic"][10], "anomaly")
        self.assertTrue(result.anomalies["traffic"][10])
        self.assertEqual(result.summary["metrics"]["traffic"]["first_anomaly_timestamp"], 10)

    def test_nan_and_non_numeric_handling(self) -> None:
        n = 20
        # Include NaNs in numeric column and an extra string tag column
        values = [1.0] * 10 + [float("nan"), 1.0, None, 1.0, 100.0] + [1.0] * 5
        table = pa.table({
            "time": list(range(n)),
            "signal": values,
            "host_name": [f"host-{i}" for i in range(n)],  # string column
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.0)

        # String column host_name must be ignored
        self.assertEqual(result.metric_names, ("signal",))
        # NaN observation at index 10 must be marked missing_observation
        self.assertEqual(result.evaluation_statuses["signal"][10], "missing_observation")
        self.assertFalse(result.anomalies["signal"][10])
        self.assertIsNone(result.anomaly_scores["signal"][10])
        self.assertIsNone(result.signed_scores["signal"][10])
        # None at index 12 must be marked missing_observation
        self.assertEqual(result.evaluation_statuses["signal"][12], "missing_observation")
        self.assertFalse(result.anomalies["signal"][12])
        # 100.0 at index 14 must be flagged anomalous
        self.assertEqual(result.evaluation_statuses["signal"][14], "anomaly")
        self.assertTrue(result.anomalies["signal"][14])

    def test_irregular_timestamps_and_sorting(self) -> None:
        # Unsorted and irregular timestamps
        raw_dict = {
            "time": [30, 10, 5, 50, 20, 15, 40, 25, 35, 45, 0, 55],
            "val": [1.0] * 11 + [100.0],  # 100.0 is at time=55 (the latest)
        }
        case = _make_case(raw_dict)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.0)

        # Output timestamps must be sorted ascending
        self.assertEqual(result.timestamps, (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55))
        # The anomaly occurred at the last timestamp (55)
        self.assertEqual(result.evaluation_statuses["val"][-1], "anomaly")
        self.assertTrue(result.anomalies["val"][-1])
        self.assertEqual(result.summary["first_anomaly_timestamp"], 55)

    def test_deterministic_sorting_with_duplicate_timestamps(self) -> None:
        # Duplicate timestamps: stable sort must preserve relative input order
        raw_dict = {
            "time": [10, 10, 5, 5, 20],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
        case = _make_case(raw_dict)
        result = detect_metric_anomalies(case, window_size=2, min_warmup=2)

        self.assertEqual(result.timestamps, (5, 5, 10, 10, 20))
        # Relative order for t=5 (3.0 before 4.0) and t=10 (1.0 before 2.0) preserved

    def test_metrics_with_different_scales(self) -> None:
        n = 20
        # Metric small on [0, 1], metric large on [1e8, 1e9]
        small_vals = [0.1, 0.12, 0.09, 0.11, 0.1] * 2 + [0.9] + [0.1] * 9
        large_vals = [1e8, 1.01e8, 0.99e8, 1.005e8, 0.995e8] * 2 + [5e8] + [1e8] * 9

        table = pa.table({
            "time": list(range(n)),
            "small_metric": small_vals,
            "large_metric": large_vals,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.0)

        # Both detect anomalies at index 10 regardless of dynamic range differences
        self.assertTrue(result.anomalies["small_metric"][10])
        self.assertTrue(result.anomalies["large_metric"][10])

    def test_no_anomaly_when_series_is_stable(self) -> None:
        n = 30
        table = pa.table({
            "time": list(range(n)),
            "stable_a": [10.0, 10.1, 9.9, 10.0, 10.05] * 6,
            "stable_b": [100.0, 100.2, 99.8, 100.1, 99.9] * 6,
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=10, min_warmup=10, threshold=3.5)

        self.assertEqual(result.summary["total_anomalies"], 0)
        self.assertIsNone(result.summary["first_anomaly_timestamp"])
        self.assertEqual(result.summary["detected_metrics"], ())

    def test_raw_telemetry_not_mutated(self) -> None:
        table = pa.table({
            "time": [10, 20, 30, 40, 50, 60],
            "cpu": [0.1, 0.2, 0.1, 0.2, 0.1, 0.9],
        })
        case = _make_case(table)
        orig_cpu_vals = list(table["cpu"].to_pylist())

        result = detect_metric_anomalies(case, window_size=4, min_warmup=4)
        self.assertIsNotNone(result)
        # Verify table content unchanged
        self.assertEqual(list(case.metrics.raw_data["cpu"].to_pylist()), orig_cpu_vals)

    def test_truncate_metric_anomaly_result(self) -> None:
        table = pa.table({
            "time": [10, 20, 30, 40, 50, 60, 70, 80],
            "cpu": [0.1, 0.2, 0.1, 0.2, 0.1, 0.9, 0.95, 0.99],
        })
        case = _make_case(table)
        result = detect_metric_anomalies(case, window_size=4, min_warmup=4)

        # Truncate at timestamp 50
        truncated = truncate_metric_anomaly_result(result, max_timestamp=50)

        self.assertEqual(truncated.timestamps, (10, 20, 30, 40, 50))
        self.assertEqual(len(truncated.anomaly_scores["cpu"]), 5)
        self.assertEqual(len(truncated.anomalies["cpu"]), 5)
        self.assertEqual(len(truncated.evaluation_statuses["cpu"]), 5)
        # Verify timestamps strictly <= 50
        self.assertTrue(all(t <= 50 for t in truncated.timestamps))


class MedianMadAnomalyDetectionTests(unittest.TestCase):
    """Focused unit tests for the causal rolling median/MAD anomaly detector."""

    def test_causal_history_and_current_sample_excluded_from_baseline(self) -> None:
        """Verify baseline estimation uses strictly preceding observations and excludes current sample."""
        # 10 observations with value 10.0, then an outlier 100.0 at index 10.
        # If index 10 were included in its own baseline, history would be different.
        vals_a = [10.0] * 10 + [100.0] + [10.0] * 5
        table_a = pa.table({"time": list(range(16)), "metric": vals_a})
        case_a = _make_case(table_a)

        res_a = detect_metric_anomalies_median_mad(case_a, window_size=10, min_warmup=10)

        # At index 10: preceding history is 10 samples of 10.0 -> median=10.0, MAD=0.0
        # cur_val=100.0 -> deviation=90.0, robust_sigma=1e-6 -> score=9e7 >> 3.0
        self.assertEqual(res_a.evaluation_statuses["metric"][10], "anomaly")
        self.assertTrue(res_a.anomalies["metric"][10])
        self.assertGreater(res_a.anomaly_scores["metric"][10], 1000.0)

        # Now test causality: future values at indices 11..15 are drastically altered.
        # Observation at index 10 must have identical score and status.
        vals_b = [10.0] * 10 + [100.0] + [999.0] * 5
        table_b = pa.table({"time": list(range(16)), "metric": vals_b})
        case_b = _make_case(table_b)

        res_b = detect_metric_anomalies_median_mad(case_b, window_size=10, min_warmup=10)

        for i in range(11):
            self.assertEqual(res_a.anomaly_scores["metric"][i], res_b.anomaly_scores["metric"][i])
            self.assertEqual(res_a.signed_scores["metric"][i], res_b.signed_scores["metric"][i])
            self.assertEqual(res_a.evaluation_statuses["metric"][i], res_b.evaluation_statuses["metric"][i])
            self.assertEqual(res_a.anomalies["metric"][i], res_b.anomalies["metric"][i])

    def test_normal_stable_history_produces_no_anomalies(self) -> None:
        """Verify normal fluctuations around a stable baseline do not flag false anomalies."""
        n = 40
        # A series fluctuating moderately around 20.0 with nonzero MAD
        vals = [20.0, 20.2, 19.8, 20.1, 19.9, 20.0, 20.1, 19.9, 20.2, 19.8] * 4
        table = pa.table({"time": list(range(n)), "metric": vals})
        case = _make_case(table)

        res = detect_metric_anomalies_median_mad(case, window_size=10, min_warmup=10, threshold=3.0)

        for i in range(10, n):
            self.assertEqual(res.evaluation_statuses["metric"][i], "normal")
            self.assertFalse(res.anomalies["metric"][i])
            self.assertLess(res.anomaly_scores["metric"][i], 3.0)

        self.assertEqual(res.summary["total_anomalies"], 0)
        self.assertIsNone(res.summary["first_anomaly_timestamp"])

    def test_obvious_outlier_detection(self) -> None:
        """Verify obvious outlier is detected with correct signed score and status."""
        # Baseline with non-zero spread
        base = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.4, 9.6, 10.0]
        # At index 10: high spike 50.0. At index 11: low spike -20.0
        vals = base + [50.0, -20.0] + base
        table = pa.table({"time": list(range(len(vals))), "metric": vals})
        case = _make_case(table)

        res = detect_metric_anomalies_median_mad(case, window_size=10, min_warmup=10, threshold=3.0)

        # Index 10: positive spike
        self.assertEqual(res.evaluation_statuses["metric"][10], "anomaly")
        self.assertTrue(res.anomalies["metric"][10])
        self.assertGreater(res.anomaly_scores["metric"][10], 3.0)
        self.assertGreater(res.signed_scores["metric"][10], 0.0)

        # Index 11: negative spike
        self.assertEqual(res.evaluation_statuses["metric"][11], "anomaly")
        self.assertTrue(res.anomalies["metric"][11])
        self.assertGreater(res.anomaly_scores["metric"][11], 3.0)
        self.assertLess(res.signed_scores["metric"][11], 0.0)

    def test_zero_mad_constant_history_behavior(self) -> None:
        """Verify zero-variance / zero-MAD behavior on constant history."""
        # Constant history of 5.0
        # Index 10: 5.0 (equal to constant history -> normal)
        # Index 11: 5.0000001 (practically equal -> normal)
        # Index 12: 6.0 (broken invariant -> anomaly)
        vals = [5.0] * 10 + [5.0, 5.0000000001, 6.0]
        table = pa.table({"time": list(range(len(vals))), "metric": vals})
        case = _make_case(table)

        res = detect_metric_anomalies_median_mad(case, window_size=10, min_warmup=10, threshold=3.0, epsilon=1e-6)

        # Index 10: exact constant -> normal, score=0.0
        self.assertEqual(res.evaluation_statuses["metric"][10], "normal")
        self.assertEqual(res.anomaly_scores["metric"][10], 0.0)
        self.assertFalse(res.anomalies["metric"][10])

        # Index 11: isclose -> normal, score=0.0
        self.assertEqual(res.evaluation_statuses["metric"][11], "normal")
        self.assertEqual(res.anomaly_scores["metric"][11], 0.0)
        self.assertFalse(res.anomalies["metric"][11])

        # Index 12: different from constant history -> broken invariant anomaly
        self.assertEqual(res.evaluation_statuses["metric"][12], "anomaly")
        self.assertTrue(res.anomalies["metric"][12])
        self.assertGreater(res.anomaly_scores["metric"][12], 3.0)

    def test_warmup_and_insufficient_history_behavior(self) -> None:
        """Verify warmup phase and insufficient valid history handling."""
        # Window size 6, min_warmup 6, min_valid_history 4
        # Points 0..5 are warmup
        # Points 6..8 have NaNs in history
        vals = [1.0] * 6 + [float("nan"), float("nan"), float("nan"), 1.0, 1.0]
        table = pa.table({"time": list(range(len(vals))), "metric": vals})
        case = _make_case(table)

        res = detect_metric_anomalies_median_mad(
            case,
            window_size=6,
            min_warmup=6,
            min_valid_history=4,
        )

        # Warmup
        for i in range(6):
            self.assertEqual(res.evaluation_statuses["metric"][i], "warmup")
            self.assertIsNone(res.anomaly_scores["metric"][i])

        # Point 6 is NaN -> missing_observation
        self.assertEqual(res.evaluation_statuses["metric"][6], "missing_observation")
        # Point 7 is NaN -> missing_observation
        self.assertEqual(res.evaluation_statuses["metric"][7], "missing_observation")
        # Point 8 is NaN -> missing_observation
        self.assertEqual(res.evaluation_statuses["metric"][8], "missing_observation")

        # Point 9: cur_val is 1.0. History in [3:9] is [1.0, 1.0, 1.0, NaN, NaN, NaN] -> 3 valid samples < 4
        self.assertEqual(res.evaluation_statuses["metric"][9], "insufficient_history")
        self.assertIsNone(res.anomaly_scores["metric"][9])

    def test_missing_or_invalid_values_handling(self) -> None:
        """Verify None and NaN current observations are marked missing_observation."""
        vals = [1.0] * 8 + [None, float("nan")] + [1.0] * 5
        table = pa.table({"time": list(range(len(vals))), "metric": vals})
        case = _make_case(table)

        res = detect_metric_anomalies_median_mad(case, window_size=5, min_warmup=5, min_valid_history=3)

        self.assertEqual(res.evaluation_statuses["metric"][8], "missing_observation")
        self.assertIsNone(res.anomaly_scores["metric"][8])
        self.assertIsNone(res.signed_scores["metric"][8])
        self.assertFalse(res.anomalies["metric"][8])

        self.assertEqual(res.evaluation_statuses["metric"][9], "missing_observation")
        self.assertIsNone(res.anomaly_scores["metric"][9])
        self.assertIsNone(res.signed_scores["metric"][9])
        self.assertFalse(res.anomalies["metric"][9])

    def test_deterministic_repeated_execution(self) -> None:
        """Verify running the detector twice on the same case yields identical results."""
        vals = [10.0, 12.0, 11.0, 10.5, 9.5, 100.0, 10.0, 11.0, 12.0, 10.0] * 3
        table = pa.table({"time": list(range(len(vals))), "m1": vals, "m2": [v * 2 for v in vals]})
        case = _make_case(table)

        res1 = detect_metric_anomalies_median_mad(case, window_size=8, min_warmup=8)
        res2 = detect_metric_anomalies_median_mad(case, window_size=8, min_warmup=8)

        self.assertEqual(res1.timestamps, res2.timestamps)
        self.assertEqual(res1.metric_names, res2.metric_names)
        self.assertEqual(res1.anomalies, res2.anomalies)
        self.assertEqual(res1.evaluation_statuses, res2.evaluation_statuses)
        self.assertEqual(res1.anomaly_scores, res2.anomaly_scores)
        self.assertEqual(res1.signed_scores, res2.signed_scores)
        self.assertEqual(res1.valid_history_counts, res2.valid_history_counts)
        self.assertEqual(res1.summary, res2.summary)

    def test_wrapper_matches_normalization_argument(self) -> None:
        """Verify detect_metric_anomalies_median_mad matches detect_metric_anomalies(normalization='median_mad')."""
        vals = [10.0] * 10 + [50.0]
        table = pa.table({"time": list(range(len(vals))), "metric": vals})
        case = _make_case(table)

        res_wrapper = detect_metric_anomalies_median_mad(case, window_size=10, min_warmup=10)
        res_direct = detect_metric_anomalies(case, window_size=10, min_warmup=10, normalization="median_mad")

        self.assertEqual(res_wrapper.timestamps, res_direct.timestamps)
        self.assertEqual(res_wrapper.anomaly_scores, res_direct.anomaly_scores)
        self.assertEqual(res_wrapper.signed_scores, res_direct.signed_scores)
        self.assertEqual(res_wrapper.anomalies, res_direct.anomalies)
        self.assertEqual(res_wrapper.evaluation_statuses, res_direct.evaluation_statuses)
        self.assertEqual(res_wrapper.summary, res_direct.summary)


class MeanStdInvarianceTests(unittest.TestCase):
    """Verify that existing mean/std detector behavior remains strictly unchanged."""

    def test_mean_std_detector_outputs_remain_unchanged(self) -> None:
        """Assert calling detect_metric_anomalies without normalization produces identical output to explicit 'mean_std'."""
        vals = [10.0, 12.0, 14.0, 10.0, 12.0, 100.0, 11.0, 13.0, 12.0, 10.0]
        table = pa.table({"time": list(range(len(vals))), "metric": vals})
        case = _make_case(table)

        res_default = detect_metric_anomalies(case, window_size=5, min_warmup=5)
        res_explicit = detect_metric_anomalies(case, window_size=5, min_warmup=5, normalization="mean_std")

        self.assertEqual(res_default.timestamps, res_explicit.timestamps)
        self.assertEqual(res_default.anomaly_scores, res_explicit.anomaly_scores)
        self.assertEqual(res_default.signed_scores, res_explicit.signed_scores)
        self.assertEqual(res_default.anomalies, res_explicit.anomalies)
        self.assertEqual(res_default.evaluation_statuses, res_explicit.evaluation_statuses)
        self.assertEqual(res_default.summary, res_explicit.summary)

        # Verify exact known z-score at index 5:
        # History is [10.0, 12.0, 14.0, 10.0, 12.0] -> mean = 11.6, variance = 2.24, std = 1.4966629547...
        # cur_val = 100.0 -> dev = 88.4, z = 88.4 / (1.4966629547 + 1e-6) ~= 59.0647
        expected_mean = 11.6
        expected_std = math.sqrt(sum((x - expected_mean) ** 2 for x in [10.0, 12.0, 14.0, 10.0, 12.0]) / 5)
        expected_z = (100.0 - expected_mean) / (expected_std + 1e-6)

        self.assertAlmostEqual(res_default.anomaly_scores["metric"][5], expected_z, places=6)
        self.assertTrue(res_default.anomalies["metric"][5])

    def test_invalid_normalization_raises_error(self) -> None:
        """Verify invalid normalization argument raises ValueError."""
        table = pa.table({"time": list(range(10)), "metric": [1.0] * 10})
        case = _make_case(table)
        with self.assertRaisesRegex(ValueError, "Unknown normalization"):
            detect_metric_anomalies(case, normalization="unknown_norm")


if __name__ == "__main__":
    unittest.main()
