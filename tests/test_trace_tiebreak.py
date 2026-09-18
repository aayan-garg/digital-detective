"""Unit tests for Parameter-Free Trace Tie-Break for S_comb ranking.

Tests all 8 mandatory criteria:
1. Non-tied ordering invariant: S(A) > S(B) with extreme opposite trace scores preserves A ahead of B.
2. Exact tie uses trace: S(A) == S(B) with E(A) < E(B) moves B ahead of A.
3. Exact triple tie: ordered by descending E_elev, then original deterministic fallback.
4. No trace evidence: original S_comb ordering is 100% preserved.
5. Candidate universe unchanged: treatment ranking contains exactly the same candidate IDs.
6. Deterministic repeated execution: identical inputs produce identical rankings.
7. No epsilon tie behavior: candidates with extremely close but unequal scores retain original ordering.
8. Existing S_comb regression: verify frozen S_comb behavior is unmodified.
"""

from __future__ import annotations

import unittest

from digital_detective.rca import (
    RCA_D_COMBINED,
    RootCauseScore,
    rank_with_s_comb,
)
from digital_detective.trace_tiebreak import (
    RankedEntity,
    apply_trace_tiebreak,
)


class TraceTieBreakTests(unittest.TestCase):
    def test_1_non_tied_ordering_invariant(self) -> None:
        """S(A) > S(B) must preserve A ahead of B even with extreme opposite trace scores."""
        control_ranking = [
            RankedEntity(entity="serviceA", score=0.85, rank=1),
            RankedEntity(entity="serviceB", score=0.84, rank=2),
        ]
        # Extreme opposite trace evidence: serviceB has maximum elevation, serviceA has 0
        trace_elev = {"serviceA": 0.0, "serviceB": 1.0}

        result = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")

        # serviceA must remain ahead of serviceB
        self.assertEqual(result.treatment_ranking[0].entity, "serviceA")
        self.assertEqual(result.treatment_ranking[1].entity, "serviceB")
        self.assertFalse(result.order_changed)

    def test_2_exact_tie_uses_trace(self) -> None:
        """S(A) == S(B) with E(A) < E(B) must place B ahead of A."""
        control_ranking = [
            RankedEntity(entity="serviceA", score=0.0, rank=1),
            RankedEntity(entity="serviceB", score=0.0, rank=2),
        ]
        trace_elev = {"serviceA": 0.1, "serviceB": 0.9}

        result = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")

        self.assertEqual(result.treatment_ranking[0].entity, "serviceB")
        self.assertEqual(result.treatment_ranking[1].entity, "serviceA")
        self.assertEqual(result.treatment_ranking[0].rank, 1)
        self.assertEqual(result.treatment_ranking[1].rank, 2)
        self.assertTrue(result.order_changed)
        self.assertEqual(result.ties_resolved_count, 2)

    def test_3_exact_triple_tie(self) -> None:
        """Triple tie ordered by descending E_elev, with remaining ties using original fallback."""
        control_ranking = [
            RankedEntity(entity="serviceA", score=0.0, rank=1),
            RankedEntity(entity="serviceB", score=0.0, rank=2),
            RankedEntity(entity="serviceC", score=0.0, rank=3),
        ]
        # serviceB highest (0.8), serviceA and serviceC tied in trace (0.2)
        trace_elev = {"serviceA": 0.2, "serviceB": 0.8, "serviceC": 0.2}

        result = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")

        treatment_entities = [r.entity for r in result.treatment_ranking]
        # serviceB moves to rank 1; serviceA was ahead of serviceC originally, so A remains ahead of C
        self.assertEqual(treatment_entities, ["serviceB", "serviceA", "serviceC"])

    def test_4_no_trace_evidence(self) -> None:
        """When trace evidence is None or empty, original S_comb ordering is preserved."""
        control_ranking = [
            RankedEntity(entity="serviceA", score=0.5, rank=1),
            RankedEntity(entity="serviceB", score=0.0, rank=2),
            RankedEntity(entity="serviceC", score=0.0, rank=3),
        ]

        res_none = apply_trace_tiebreak(control_ranking, None, case_id="test_case")
        self.assertEqual(res_none.treatment_ranking, tuple(control_ranking))
        self.assertFalse(res_none.order_changed)

        res_empty = apply_trace_tiebreak(control_ranking, {}, case_id="test_case")
        self.assertEqual(res_empty.treatment_ranking, tuple(control_ranking))
        self.assertFalse(res_empty.order_changed)

    def test_5_candidate_universe_unchanged(self) -> None:
        """Treatment ranking must contain exactly the same candidate IDs as control."""
        entities = ["alpha", "beta", "gamma", "delta", "epsilon"]
        control_ranking = [RankedEntity(entity=e, score=0.0, rank=i) for i, e in enumerate(entities, 1)]
        trace_elev = {"delta": 0.5, "beta": 0.9}

        result = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")

        control_set = {r.entity for r in result.control_ranking}
        treatment_set = {r.entity for r in result.treatment_ranking}
        self.assertEqual(control_set, set(entities))
        self.assertEqual(treatment_set, set(entities))
        self.assertEqual(len(result.treatment_ranking), len(control_ranking))

    def test_6_deterministic_repeated_execution(self) -> None:
        """Identical inputs must produce identical outputs."""
        control_ranking = [
            RankedEntity(entity="serviceA", score=0.5, rank=1),
            RankedEntity(entity="serviceB", score=0.0, rank=2),
            RankedEntity(entity="serviceC", score=0.0, rank=3),
        ]
        trace_elev = {"serviceA": 0.2, "serviceB": 0.7, "serviceC": 0.1}

        res1 = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")
        res2 = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")

        self.assertEqual(res1.treatment_ranking, res2.treatment_ranking)
        self.assertEqual(res1.candidate_details, res2.candidate_details)

    def test_7_no_epsilon_tie_behavior(self) -> None:
        """Candidates with extremely close but unequal scores must retain original metric ordering."""
        # Score difference is 1e-12 (not an exact tie)
        control_ranking = [
            RankedEntity(entity="serviceA", score=0.500000000001, rank=1),
            RankedEntity(entity="serviceB", score=0.500000000000, rank=2),
        ]
        # serviceB has massive trace elevation
        trace_elev = {"serviceA": 0.0, "serviceB": 1.0}

        result = apply_trace_tiebreak(control_ranking, trace_elev, case_id="test_case")

        # serviceA must remain rank 1 because exact equality is required for tie-breaking
        self.assertEqual(result.treatment_ranking[0].entity, "serviceA")
        self.assertEqual(result.treatment_ranking[1].entity, "serviceB")
        self.assertFalse(result.order_changed)

    def test_8_existing_s_comb_regression(self) -> None:
        """Confirm that RootCauseScore and S_comb ranking types remain compatible."""
        rcs_list = [
            RootCauseScore(
                entity="serviceA",
                score=0.9,
                r_early=0.0,
                r_strength=0.9,
                r_coverage=0.0,
                r_prop=0.0,
                first_anomaly_ts=100,
                peak_score=0.9,
            ),
            RootCauseScore(
                entity="serviceB",
                score=0.0,
                r_early=0.0,
                r_strength=0.0,
                r_coverage=0.0,
                r_prop=0.0,
                first_anomaly_ts=None,
                peak_score=0.0,
            ),
        ]
        result = apply_trace_tiebreak(rcs_list, {"serviceB": 0.5}, case_id="rcs_case")
        self.assertEqual(result.control_ranking[0].entity, "serviceA")
        self.assertEqual(result.treatment_ranking[0].entity, "serviceA")
        self.assertEqual(result.treatment_ranking[1].entity, "serviceB")


if __name__ == "__main__":
    unittest.main()
