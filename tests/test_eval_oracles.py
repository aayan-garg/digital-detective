"""Unit tests for Stage 1 Oracle Headroom infrastructure.
"""

import unittest

from eval.harness import compute_oracle_headroom
from eval.manifest import ManifestCase
from eval.models import CaseMetrics, MethodRankingResult, RankedEntity


def make_result(case_id: str, method: str, target: str, rank: int) -> MethodRankingResult:
    mrr = 1.0 / rank if rank > 0 else 0.0
    return MethodRankingResult(
        case_id=case_id,
        method_name=method,
        window_mode="oracle",
        ranking=(RankedEntity(target, 1.0, rank),),
        candidate_universe=(target, "other"),
        status="SUCCESS",
        metrics=CaseMetrics(
            top1=(rank == 1),
            top3=(rank <= 3),
            top5=(rank <= 5),
            mrr=mrr,
            ac1=1.0 if rank <= 1 else 0.0,
            ac2=1.0 if rank <= 2 else 0.0,
            ac3=1.0 if rank <= 3 else 0.0,
            ac4=1.0 if rank <= 4 else 0.0,
            ac5=1.0 if rank <= 5 else 0.0,
            avg3=1.0 if rank <= 1 else (0.5 if rank <= 2 else (0.33 if rank <= 3 else 0.0)),
            avg5=1.0 / rank if rank <= 5 else 0.0,
            target_rank=rank,
        ),
        runtime_sec=0.01,
    )


class TestEvalOracles(unittest.TestCase):
    def test_oracle_headroom_computation(self) -> None:
        cases = [
            ManifestCase("c1", "RE2-OB", "ob", "cpu", "service_a", 1, "smoke"),
            ManifestCase("c2", "RE2-OB", "ob", "delay", "service_b", 1, "smoke"),
        ]

        # Case 1 (cpu): metric rank 1, trace rank 3 -> metric wins
        # Case 2 (delay): metric rank 4, trace rank 1 -> trace wins
        case_results = {
            "s_comb": [
                make_result("c1", "s_comb", "service_a", rank=1),
                make_result("c2", "s_comb", "service_b", rank=4),
            ],
            "trace_elevation": [
                make_result("c1", "trace_elevation", "service_a", rank=3),
                make_result("c2", "trace_elevation", "service_b", rank=1),
            ],
        }

        analysis = compute_oracle_headroom(cases, case_results, single_modality_methods=("s_comb", "trace_elevation"))

        # Best fixed modality:
        # s_comb: c1=1.0, c2=0.25 -> avg MRR = 0.625
        # trace_elevation: c1=0.333, c2=1.0 -> avg MRR = 0.6667 (wins best fixed)
        self.assertEqual(analysis.best_fixed_method, "trace_elevation")
        self.assertAlmostEqual(analysis.best_fixed_metrics.mrr, (1.0/3.0 + 1.0)/2.0, places=4)

        # Oracle per fault:
        # fault 'cpu': s_comb (1.0) beats trace (0.333) -> chooses s_comb
        # fault 'delay': trace (1.0) beats s_comb (0.25) -> chooses trace
        # Both c1 and c2 achieve rank 1! MRR = 1.0
        self.assertEqual(analysis.oracle_per_fault_metrics.mrr, 1.0)
        self.assertEqual(analysis.oracle_per_fault_metrics.top1_accuracy, 1.0)

        # Oracle per incident:
        # c1 -> s_comb (rank 1), c2 -> trace (rank 1) -> MRR = 1.0
        self.assertEqual(analysis.oracle_per_incident_metrics.mrr, 1.0)

        # Headroom should be strictly positive
        self.assertGreater(analysis.absolute_headroom_fault["mrr"], 0.0)
        self.assertGreater(analysis.absolute_headroom_incident["mrr"], 0.0)

        # Disclaimer MUST be present
        self.assertIn("theoretical headroom only", analysis.disclaimer)


if __name__ == "__main__":
    unittest.main()
