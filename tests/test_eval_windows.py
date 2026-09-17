"""Unit tests for IncidentWindow and resolve_incident_window.
"""

import unittest

from eval.models import IncidentWindow
from eval.windows import resolve_incident_window


class DummyTelemetryCase:
    def __init__(self, inject_time: int, timestamps: list[int]):
        class DummyGroundTruth:
            values = {"inject_time": inject_time}
        class DummyMetrics:
            pass
        self.ground_truth = DummyGroundTruth()
        self.metrics = DummyMetrics()
        self.metrics.timestamps = timestamps


class DummyEvidence:
    def __init__(self, has_episode: bool, first_ts: int | None):
        self.has_episode = has_episode
        self.first_episode_start_ts = first_ts


class TestIncidentWindow(unittest.TestCase):
    def test_window_validation(self) -> None:
        w = IncidentWindow(onset_ts=100, end_ts=200, mode="oracle")
        self.assertEqual(w.onset_ts, 100)
        self.assertEqual(w.end_ts, 200)
        self.assertEqual(w.mode, "oracle")
        self.assertTrue(w.has_detected_window)

        with self.assertRaises(ValueError):
            IncidentWindow(onset_ts=200, end_ts=100, mode="oracle")

        with self.assertRaises(ValueError):
            IncidentWindow(onset_ts=100, end_ts=200, mode="invalid_mode")

        # has_detected_window=False with mode='oracle' is invalid
        with self.assertRaises(ValueError):
            IncidentWindow(onset_ts=None, end_ts=200, mode="oracle", has_detected_window=False)

        # has_detected_window=False with mode='detected' is valid
        w_fail = IncidentWindow(onset_ts=None, end_ts=200, mode="detected", has_detected_window=False)
        self.assertFalse(w_fail.has_detected_window)
        self.assertIsNone(w_fail.onset_ts)

    def test_resolve_oracle_window_uses_inject_time(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        w = resolve_incident_window(case, mode="oracle")
        self.assertEqual(w.mode, "oracle")
        self.assertEqual(w.onset_ts, 1500)
        self.assertEqual(w.end_ts, 2000)
        self.assertTrue(w.has_detected_window)
        self.assertIn("oracle:inject_time=1500", w.source_description)

    def test_resolve_detected_window_uses_episode_timestamp(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        ep_evidence = {
            "s1": DummyEvidence(has_episode=True, first_ts=1510),
            "s2": DummyEvidence(has_episode=True, first_ts=1505),
            "s3": DummyEvidence(has_episode=False, first_ts=None),
        }
        w = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")
        self.assertEqual(w.mode, "detected")
        self.assertEqual(w.onset_ts, 1505)
        self.assertEqual(w.end_ts, 1505)
        self.assertEqual(w.onset_ts, w.end_ts)
        self.assertTrue(w.has_detected_window)
        self.assertIn("detected:first_episode_ts=1505", w.source_description)

    def test_resolve_detected_no_episodes_does_not_use_inject_time(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        ep_evidence = {
            "s1": DummyEvidence(has_episode=False, first_ts=None),
        }
        w = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")
        self.assertEqual(w.mode, "detected")
        self.assertFalse(w.has_detected_window)
        self.assertIsNone(w.onset_ts)
        self.assertNotEqual(w.onset_ts, 1500)
        self.assertEqual(w.end_ts, 2000)
        self.assertIn("no_detected_window", w.source_description)

    def test_custom_overrides(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        w = resolve_incident_window(
            case,
            mode="detected",
            custom_onset_ts=1600,
            custom_end_ts=2100,
        )
        self.assertEqual(w.onset_ts, 1600)
        self.assertEqual(w.end_ts, 2100)
        self.assertTrue(w.has_detected_window)
        self.assertIn("custom_override", w.source_description)

    def test_invalid_modes_fail(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        with self.assertRaises(ValueError):
            resolve_incident_window(case, mode="invalid_mode")


if __name__ == "__main__":
    unittest.main()
