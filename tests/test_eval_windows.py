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

        with self.assertRaises(ValueError):
            IncidentWindow(onset_ts=200, end_ts=100, mode="oracle")

        with self.assertRaises(ValueError):
            IncidentWindow(onset_ts=100, end_ts=200, mode="invalid_mode")

    def test_resolve_oracle_window(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        w = resolve_incident_window(case, mode="oracle")
        self.assertEqual(w.mode, "oracle")
        self.assertEqual(w.onset_ts, 1500)
        self.assertEqual(w.end_ts, 2000)
        self.assertIn("oracle:inject_time=1500", w.source_description)

    def test_resolve_detected_window_with_episodes(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        ep_evidence = {
            "s1": DummyEvidence(has_episode=True, first_ts=1510),
            "s2": DummyEvidence(has_episode=True, first_ts=1505),
            "s3": DummyEvidence(has_episode=False, first_ts=None),
        }
        w = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")
        self.assertEqual(w.mode, "detected")
        self.assertEqual(w.onset_ts, 1505)
        self.assertEqual(w.end_ts, 2000)
        self.assertIn("detected:first_episode_ts=1505", w.source_description)

    def test_resolve_detected_fallback_when_no_episodes(self) -> None:
        case = DummyTelemetryCase(inject_time=1500, timestamps=[1000, 1500, 2000])
        ep_evidence = {
            "s1": DummyEvidence(has_episode=False, first_ts=None),
        }
        w = resolve_incident_window(case, ep_evidence=ep_evidence, mode="detected")
        self.assertEqual(w.mode, "detected")
        self.assertEqual(w.onset_ts, 1500)
        self.assertEqual(w.end_ts, 2000)
        self.assertIn("detected_fallback", w.source_description)


if __name__ == "__main__":
    unittest.main()
