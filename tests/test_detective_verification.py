"""Unit tests for post-remediation recovery verification.
"""

import unittest

from digital_detective.detective.verification import SystemStateSnapshot, verify_recovery


class TestDetectiveVerification(unittest.TestCase):
    def test_full_system_recovery_resolved(self) -> None:
        before = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.85, "frontend": 0.40},
            latencies_p90={"checkoutservice": 250.0, "frontend": 180.0},
            error_rates={"checkoutservice": 12.0, "frontend": 5.0},
            health_statuses={"checkoutservice": "CRITICAL", "frontend": "DEGRADED"},
        )
        after = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.05, "frontend": 0.02},
            latencies_p90={"checkoutservice": 35.0, "frontend": 40.0},
            error_rates={"checkoutservice": 0.0, "frontend": 0.0},
            health_statuses={"checkoutservice": "HEALTHY", "frontend": "HEALTHY"},
        )
        res = verify_recovery(before, after, target_service="checkoutservice")
        self.assertEqual(res.final_status, "RESOLVED")
        self.assertFalse(res.regression_detected)
        self.assertGreater(len(res.recovered_metrics), 0)
        self.assertEqual(len(res.unresolved_symptoms), 0)
        self.assertEqual(len(res.new_anomalies), 0)

    def test_regression_detected_not_resolved(self) -> None:
        before = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.85, "paymentservice": 0.05},
            latencies_p90={"checkoutservice": 250.0, "paymentservice": 20.0},
            error_rates={"checkoutservice": 12.0, "paymentservice": 0.0},
            health_statuses={"checkoutservice": "CRITICAL", "paymentservice": "HEALTHY"},
        )
        # checkoutservice drops, but paymentservice errors spike drastically
        after = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.10, "paymentservice": 0.80},
            latencies_p90={"checkoutservice": 30.0, "paymentservice": 350.0},
            error_rates={"checkoutservice": 0.0, "paymentservice": 45.0},
            health_statuses={"checkoutservice": "HEALTHY", "paymentservice": "CRITICAL"},
        )
        res = verify_recovery(before, after, target_service="checkoutservice")
        self.assertEqual(res.final_status, "NOT_RESOLVED")
        self.assertTrue(res.regression_detected)
        self.assertGreater(len(res.new_anomalies), 0)

    def test_partial_recovery_uncertain(self) -> None:
        before = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.85, "frontend": 0.60},
            latencies_p90={"checkoutservice": 250.0, "frontend": 200.0},
            error_rates={"checkoutservice": 10.0, "frontend": 2.0},
            health_statuses={"checkoutservice": "CRITICAL", "frontend": "CRITICAL"},
        )
        # Target recovered, but frontend latency is still heavily elevated
        after = SystemStateSnapshot(
            anomaly_scores={"checkoutservice": 0.05, "frontend": 0.55},
            latencies_p90={"checkoutservice": 35.0, "frontend": 190.0},
            error_rates={"checkoutservice": 0.0, "frontend": 2.0},
            health_statuses={"checkoutservice": "HEALTHY", "frontend": "DEGRADED"},
        )
        res = verify_recovery(before, after, target_service="checkoutservice")
        self.assertEqual(res.final_status, "UNCERTAIN")
        self.assertFalse(res.regression_detected)
        self.assertGreater(len(res.unresolved_symptoms), 0)


if __name__ == "__main__":
    unittest.main()
