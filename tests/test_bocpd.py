"""Unit tests for causal Bayesian Online Change Point Detection (BOCPD)."""

from __future__ import annotations

from typing import Any
import unittest

import numpy as np
import pyarrow as pa

from digital_detective.bocpd import (
    BOCPDConfig,
    BOCPDResult,
    CausalBOCPD,
    detect_bocpd_onset,
)
from digital_detective.telemetry import (
    CaseMetadata,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)


def _make_telemetry_case(
    raw_data: Any,
    case_id: str = "test_case",
    timestamp_field: str = "time",
) -> TelemetryCase:
    field_names = (
        tuple(raw_data.column_names)
        if hasattr(raw_data, "column_names")
        else tuple(raw_data.keys())
    )
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


class CausalBOCPDTests(unittest.TestCase):
    """Core mathematical and behavioral tests for CausalBOCPD."""

    def test_causal_invariance(self) -> None:
        """Adding future telemetry must not alter an already emitted onset timestamp."""
        rng = np.random.default_rng(42)
        n_warmup = 60
        n_shift = 75
        n_short = 90
        n_long = 200

        # Baseline: N(0, 1)
        base = rng.normal(0.0, 1.0, size=(n_shift, 2))
        # Post-shift: N(5, 1)
        post = rng.normal(5.0, 1.0, size=(n_long - n_shift, 2))
        full_data = np.vstack([base, post])
        timestamps_full = list(range(1000, 1000 + n_long))

        # Run on shorter horizon that encompasses onset
        detector = CausalBOCPD(hazard_lambda=100.0, min_warmup=n_warmup, threshold=0.5)
        onset_short, status_short, audit_short, cps_short = detector.fit_predict(
            timestamps_full[:n_short], full_data[:n_short]
        )

        self.assertEqual(status_short, "detected")
        self.assertIsNotNone(onset_short)

        # Run on longer horizon with 110 additional future steps
        detector_long = CausalBOCPD(hazard_lambda=100.0, min_warmup=n_warmup, threshold=0.5)
        onset_long, status_long, audit_long, cps_long = detector_long.fit_predict(
            timestamps_full, full_data
        )

        self.assertEqual(status_long, "detected")
        self.assertEqual(
            onset_short,
            onset_long,
            f"Causal violation: short onset {onset_short} != long onset {onset_long}",
        )
        self.assertEqual(cps_short[0], cps_long[0])

    def test_stable_constant_sequence(self) -> None:
        """A completely constant stationary sequence produces no false changepoints."""
        n_points = 150
        data = np.ones((n_points, 3), dtype=np.float64) * 42.0
        timestamps = list(range(n_points))

        detector = CausalBOCPD(min_warmup=60)
        onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, data)

        self.assertIsNone(onset_ts)
        self.assertEqual(status, "no_detection")
        self.assertEqual(len(changepoints), 0)
        self.assertEqual(audit["total_observations"], n_points)

    def test_stationary_noise_sequence(self) -> None:
        """Stationary Gaussian noise produces no false changepoints with standard hazard."""
        rng = np.random.default_rng(12345)
        n_points = 200
        data = rng.normal(10.0, 2.0, size=(n_points, 4))
        timestamps = list(range(n_points))

        detector = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, data)

        self.assertIsNone(onset_ts)
        self.assertEqual(status, "no_detection")
        self.assertEqual(len(changepoints), 0)

    def test_regime_shift_detected(self) -> None:
        """A clear synthetic distribution shift after warmup is detected promptly."""
        rng = np.random.default_rng(999)
        t_shift = 75
        n_total = 120

        # Phase 1: mean 0
        p1 = rng.normal(0.0, 0.5, size=(t_shift, 2))
        # Phase 2: mean 6.0
        p2 = rng.normal(6.0, 0.5, size=(n_total - t_shift, 2))
        data = np.vstack([p1, p2])
        timestamps = [1000 + i * 5 for i in range(n_total)]

        detector = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, data)

        self.assertEqual(status, "detected")
        self.assertIsNotNone(onset_ts)
        self.assertGreaterEqual(len(changepoints), 1)

        # Expected onset is at or immediately after the shift at t=75
        first_cp_idx = changepoints[0]
        self.assertGreaterEqual(first_cp_idx, t_shift)
        self.assertLessEqual(first_cp_idx, t_shift + 3)
        self.assertEqual(onset_ts, timestamps[first_cp_idx])

    def test_delayed_shift_requires_evidence(self) -> None:
        """Onset cannot trigger before the actual distribution shift occurs."""
        rng = np.random.default_rng(777)
        t_shift = 80
        n_total = 100

        data = np.vstack([
            rng.normal(0.0, 1.0, size=(t_shift, 2)),
            rng.normal(4.0, 1.0, size=(n_total - t_shift, 2)),
        ])
        timestamps = list(range(n_total))

        detector = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, data)

        self.assertEqual(status, "detected")
        # Cannot declare onset before t_shift
        self.assertGreaterEqual(changepoints[0], t_shift)

    def test_no_detection_on_subtle_variation(self) -> None:
        """Subtle drift within 0.1 std error does not trigger detection."""
        rng = np.random.default_rng(12345)
        n_points = 150
        data = rng.normal(5.0, 1.0, size=(n_points, 2))
        # Add very small perturbation
        data[70:] += 0.02
        timestamps = list(range(n_points))

        detector = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, data)

        self.assertIsNone(onset_ts)
        self.assertEqual(status, "no_detection")
        self.assertEqual(changepoints, ())

    def test_deterministic_behavior(self) -> None:
        """Identical input produces bit-for-bit identical detection outputs."""
        rng = np.random.default_rng(101)
        data = np.vstack([
            rng.normal(0.0, 1.0, size=(70, 3)),
            rng.normal(5.0, 1.0, size=(50, 3)),
        ])
        timestamps = list(range(120))

        d1 = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset1, status1, audit1, cps1 = d1.fit_predict(timestamps, data)

        d2 = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset2, status2, audit2, cps2 = d2.fit_predict(timestamps, data)

        self.assertEqual(onset1, onset2)
        self.assertEqual(status1, status2)
        self.assertEqual(cps1, cps2)
        self.assertEqual(audit1, audit2)

    def test_warmup_boundary_protection(self) -> None:
        """A shift during the warmup window (t < 60) does not trigger early onset."""
        rng = np.random.default_rng(555)
        # Shift at t=20 (well within warmup of 60)
        data = np.vstack([
            rng.normal(0.0, 1.0, size=(20, 2)),
            rng.normal(5.0, 1.0, size=(50, 2)),
        ])
        timestamps = list(range(70))

        detector = CausalBOCPD(hazard_lambda=100.0, min_warmup=60, threshold=0.5)
        onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, data)

        # Even though data shifted at t=20, warmup bounds detection until t >= 60.
        if onset_ts is not None:
            self.assertGreaterEqual(changepoints[0], 60)


class DetectBOCPDOnsetIntegrationTests(unittest.TestCase):
    """Integration tests for detect_bocpd_onset with TelemetryCase."""

    def test_missing_metrics_modality_raises_error(self) -> None:
        case = TelemetryCase(metadata=CaseMetadata(case_id="empty"), metrics=None)
        with self.assertRaisesRegex(ValueError, "has no metrics modality"):
            detect_bocpd_onset(case)

    def test_insufficient_warmup_raises_error(self) -> None:
        table = pa.table({"time": [1, 2, 3], "cpu": [0.1, 0.2, 0.3]})
        case = _make_telemetry_case(table)
        with self.assertRaisesRegex(ValueError, "Insufficient observations"):
            detect_bocpd_onset(case, config=BOCPDConfig(min_warmup=10))

    def test_canonical_metric_universe_uses_all_numeric_metrics(self) -> None:
        """BOCPD evaluates all numeric metrics, matching the canonical control universe."""
        n_obs = 80
        table = pa.table({
            "time": list(range(n_obs)),
            "service_a_cpu": [1.0] * n_obs,
            "service_a_latency": [10.0] * 65 + [100.0] * 15,
            "service_a_error": [0.0] * 65 + [5.0] * 15,
            "service_b_mem": [500.0] * n_obs,
        })
        case = _make_telemetry_case(table)
        result = detect_bocpd_onset(case, config=BOCPDConfig(min_warmup=60))

        self.assertEqual(result.status, "detected")
        self.assertIsNotNone(result.onset_ts)
        self.assertIn("service_a_latency", result.audit["selected_columns"])
        self.assertIn("service_a_error", result.audit["selected_columns"])
        self.assertIn("service_a_cpu", result.audit["selected_columns"])
        self.assertIn("service_b_mem", result.audit["selected_columns"])

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            BOCPDConfig(hazard_lambda=-1.0)
        with self.assertRaises(ValueError):
            BOCPDConfig(min_warmup=1)
        with self.assertRaises(ValueError):
            BOCPDConfig(threshold=0.0)
        with self.assertRaises(ValueError):
            BOCPDConfig(threshold=1.5)


if __name__ == "__main__":
    unittest.main()
