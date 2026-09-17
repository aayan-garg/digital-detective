"""Unit tests for causal-consistency validation.
"""

import unittest

from digital_detective.detective.models import EvidenceItem
from digital_detective.detective.validation import CausalConsistencyValidator


class DummyGraph:
    def __init__(self, callees: dict[str, tuple[str, ...]], callers: dict[str, tuple[str, ...]]) -> None:
        self._callees = callees
        self._callers = callers

    def callees_of(self, service: str) -> tuple[str, ...]:
        return self._callees.get(service, ())

    def callers_of(self, service: str) -> tuple[str, ...]:
        return self._callers.get(service, ())


class TestDetectiveValidation(unittest.TestCase):
    def setUp(self) -> None:
        # Topology: frontend -> checkoutservice -> currencyservice, paymentservice
        callees = {
            "frontend": ("checkoutservice",),
            "checkoutservice": ("currencyservice", "paymentservice"),
        }
        callers = {
            "checkoutservice": ("frontend",),
            "currencyservice": ("checkoutservice",),
            "paymentservice": ("checkoutservice",),
        }
        self.graph = DummyGraph(callees, callers)
        self.validator = CausalConsistencyValidator(graph=self.graph)

    def test_temporal_ordering_cause_before_downstream(self) -> None:
        evs = [
            EvidenceItem(service="checkoutservice", modality="metrics", signal="cpu", magnitude=0.9),
            EvidenceItem(service="checkoutservice", modality="topology", signal="degree", magnitude=0.8),
            EvidenceItem(service="currencyservice", modality="traces", signal="latency", magnitude=0.7),
        ]
        onsets = {
            "checkoutservice": 100,
            "currencyservice": 120,
        }
        res = self.validator.evaluate_consistency("checkoutservice", evs, earliest_onsets=onsets)
        self.assertEqual(res.temporal_score, 1.0)
        self.assertTrue(res.passed)
        self.assertEqual(len(res.warnings), 0)

    def test_temporal_ordering_inverted(self) -> None:
        evs = [
            EvidenceItem(service="checkoutservice", modality="metrics", signal="cpu", magnitude=0.9),
            EvidenceItem(service="currencyservice", modality="traces", signal="latency", magnitude=0.7),
        ]
        onsets = {
            "checkoutservice": 160,
            "currencyservice": 100,  # Downstream started 60s earlier
        }
        res = self.validator.evaluate_consistency("checkoutservice", evs, earliest_onsets=onsets)
        self.assertEqual(res.temporal_score, 0.40)
        self.assertTrue(any("Temporal inversion" in w for w in res.warnings))

    def test_topology_connected_vs_isolated(self) -> None:
        evs = [
            EvidenceItem(service="checkoutservice", modality="metrics", signal="cpu", magnitude=0.9),
            EvidenceItem(service="currencyservice", modality="traces", signal="latency", magnitude=0.7),
        ]
        # Connected
        res_conn = self.validator.evaluate_consistency("checkoutservice", evs)
        self.assertEqual(res_conn.topology_score, 1.0)

        # Disconnected candidate: adservice is not connected to currencyservice
        res_disc = self.validator.evaluate_consistency("adservice", evs)
        self.assertEqual(res_disc.topology_score, 0.50)
        self.assertTrue(any("no direct topology edge" in w for w in res_disc.warnings))

    def test_propagation_consistency(self) -> None:
        # Both internal metric anomaly and trace latency elevation present
        evs_both = [
            EvidenceItem(service="checkoutservice", modality="metrics", signal="cpu", magnitude=0.8),
            EvidenceItem(service="frontend", modality="traces", signal="latency", magnitude=0.6),
        ]
        res_both = self.validator.evaluate_consistency("checkoutservice", evs_both)
        self.assertEqual(res_both.propagation_score, 1.0)

        # Neither internal anomaly nor latency
        evs_none = [
            EvidenceItem(service="checkoutservice", modality="topology", signal="degree", magnitude=0.5),
        ]
        res_none = self.validator.evaluate_consistency("checkoutservice", evs_none)
        self.assertEqual(res_none.propagation_score, 0.50)
        self.assertTrue(any("Weak propagation signal" in w for w in res_none.warnings))


if __name__ == "__main__":
    unittest.main()
