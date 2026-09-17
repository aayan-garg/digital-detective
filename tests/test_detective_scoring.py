"""Unit tests for EvidenceFusion scoring.
"""

import unittest

from digital_detective.detective.models import EvidenceItem
from digital_detective.detective.scoring import EvidenceFusion


class TestDetectiveScoring(unittest.TestCase):
    def test_candidate_universe_preservation(self) -> None:
        fusion = EvidenceFusion()
        universe = ("adservice", "cartservice", "checkoutservice", "redis")
        evs = [
            EvidenceItem(service="cartservice", modality="metrics", signal="cpu", magnitude=0.9),
        ]
        ranked = fusion.rank_hypotheses(candidate_universe=universe, evidence_items=evs)
        self.assertEqual(len(ranked), 4)
        # Check all candidates preserved
        ranked_names = [r.service for r in ranked]
        for c in universe:
            self.assertIn(c, ranked_names)
        self.assertEqual(ranked[0].service, "cartservice")

    def test_multi_modal_weighting(self) -> None:
        fusion = EvidenceFusion()
        universe = ("srv_metric", "srv_trace")
        evs = [
            EvidenceItem(service="srv_metric", modality="metrics", signal="cpu", magnitude=1.0),
            EvidenceItem(service="srv_trace", modality="traces", signal="latency", magnitude=1.0),
        ]
        # Equal metric and trace weights (0.30 each) -> should be tied, tie-broken by name asc
        ranked = fusion.rank_hypotheses(candidate_universe=universe, evidence_items=evs)
        self.assertEqual(len(ranked), 2)
        self.assertAlmostEqual(ranked[0].score, ranked[1].score)
        self.assertEqual(ranked[0].service, "srv_metric")
        self.assertEqual(ranked[1].service, "srv_trace")

    def test_recent_change_boost(self) -> None:
        fusion = EvidenceFusion()
        universe = ("srv_a", "srv_b")
        evs = [
            EvidenceItem(service="srv_a", modality="metrics", signal="cpu", magnitude=0.8),
            EvidenceItem(service="srv_b", modality="metrics", signal="cpu", magnitude=0.8),
            EvidenceItem(service="srv_a", modality="recent_change", signal="deploy", magnitude=1.0),
        ]
        ranked = fusion.rank_hypotheses(candidate_universe=universe, evidence_items=evs)
        self.assertEqual(ranked[0].service, "srv_a")
        self.assertGreater(ranked[0].score, ranked[1].score)


if __name__ == "__main__":
    unittest.main()
