"""Unit tests for basic metric anomaly detection baseline."""

from __future__ import annotations

import math
from typing import Any
import unittest

import pyarrow as pa

from digital_detective.anomaly import (
    detect_metric_anomalies,
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


if __name__ == "__main__":
    unittest.main()
