"""Unit tests for closed candidate universe resolution.
"""

import unittest

from eval.universe import (
    CANONICAL_UNIVERSES,
    ONLINE_BOUTIQUE_SERVICES,
    SOCK_SHOP_SERVICES,
    normalize_service_name,
    resolve_candidate_universe,
)


class TestCandidateUniverse(unittest.TestCase):
    def test_canonical_online_boutique_universe(self) -> None:
        univ = resolve_candidate_universe("ob")
        self.assertEqual(len(univ), 11)
        self.assertEqual(univ, ONLINE_BOUTIQUE_SERVICES)
        # All 11 expected services must be present
        for s in (
            "adservice", "cartservice", "checkoutservice", "currencyservice",
            "emailservice", "frontend", "paymentservice", "productcatalogservice",
            "recommendationservice", "redis", "shippingservice"
        ):
            self.assertIn(s, univ)

    def test_exclusion_policy_excludes_frontend_external(self) -> None:
        univ = resolve_candidate_universe("ob")
        self.assertNotIn("frontend-external", univ)
        self.assertNotIn("loadgenerator", univ)

    def test_canonical_sock_shop_universe(self) -> None:
        univ = resolve_candidate_universe("ss")
        self.assertEqual(univ, SOCK_SHOP_SERVICES)
        self.assertIn("front-end", univ)
        self.assertIn("orders", univ)

    def test_alias_normalization(self) -> None:
        self.assertEqual(normalize_service_name("frontendservice"), "frontend")
        self.assertEqual(normalize_service_name("cartservice"), "cartservice")

    def test_custom_override(self) -> None:
        custom = ("service_b", "service_a")
        resolved = resolve_candidate_universe("unknown_system", custom_universe=custom)
        self.assertEqual(resolved, ("service_a", "service_b"))

    def test_zero_anomaly_service_preservation(self) -> None:
        # A candidate service with zero anomaly evidence must not be dropped
        univ = resolve_candidate_universe("ob")
        self.assertIn("redis", univ)
        self.assertIn("shippingservice", univ)


if __name__ == "__main__":
    unittest.main()
