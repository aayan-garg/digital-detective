"""Unit tests for evaluation metric computation (Top@k, MRR, AC@k, Avg@5).
"""

import unittest

from eval.models import CaseMetrics, RankedEntity, compute_case_metrics, aggregate_case_metrics, MethodRankingResult


class TestEvalMetrics(unittest.TestCase):
    def test_single_target_top1_hit(self) -> None:
        ranking = (
            RankedEntity("cartservice", 10.0, 1),
            RankedEntity("checkoutservice", 5.0, 2),
            RankedEntity("redis", 1.0, 3),
        )
        metrics = compute_case_metrics(ranking, "cartservice")
        self.assertTrue(metrics.top1)
        self.assertTrue(metrics.top3)
        self.assertTrue(metrics.top5)
        self.assertEqual(metrics.target_rank, 1)
        self.assertEqual(metrics.mrr, 1.0)
        self.assertEqual(metrics.ac1, 1.0)
        self.assertEqual(metrics.ac2, 1.0)
        self.assertEqual(metrics.ac3, 1.0)
        self.assertEqual(metrics.ac4, 1.0)
        self.assertEqual(metrics.ac5, 1.0)
        self.assertEqual(metrics.avg3, 1.0)
        self.assertEqual(metrics.avg5, 1.0)

    def test_rank3_hit(self) -> None:
        ranking = (
            RankedEntity("frontend", 10.0, 1),
            RankedEntity("paymentservice", 8.0, 2),
            RankedEntity("cartservice", 5.0, 3),
            RankedEntity("redis", 1.0, 4),
        )
        metrics = compute_case_metrics(ranking, "cartservice")
        self.assertFalse(metrics.top1)
        self.assertTrue(metrics.top3)
        self.assertTrue(metrics.top5)
        self.assertEqual(metrics.target_rank, 3)
        self.assertAlmostEqual(metrics.mrr, 1.0 / 3.0)
        self.assertEqual(metrics.ac1, 0.0)
        self.assertEqual(metrics.ac2, 0.0)
        self.assertEqual(metrics.ac3, 1.0)
        self.assertEqual(metrics.ac4, 1.0)
        self.assertEqual(metrics.ac5, 1.0)
        self.assertAlmostEqual(metrics.avg3, 1.0 / 3.0)
        self.assertAlmostEqual(metrics.avg5, 3.0 / 5.0)

    def test_miss_outside_top5(self) -> None:
        ranking = tuple(RankedEntity(f"s_{i}", 10.0 - i, i) for i in range(1, 10))
        metrics = compute_case_metrics(ranking, "s_7")
        self.assertFalse(metrics.top1)
        self.assertFalse(metrics.top3)
        self.assertFalse(metrics.top5)
        self.assertEqual(metrics.target_rank, 7)
        self.assertAlmostEqual(metrics.mrr, 1.0 / 7.0)
        self.assertEqual(metrics.ac1, 0.0)
        self.assertEqual(metrics.ac5, 0.0)
        self.assertEqual(metrics.avg5, 0.0)

    def test_unranked_target(self) -> None:
        ranking = (RankedEntity("s_1", 1.0, 1),)
        metrics = compute_case_metrics(ranking, "s_missing")
        self.assertIsNone(metrics.target_rank)
        self.assertEqual(metrics.mrr, 0.0)
        self.assertEqual(metrics.avg5, 0.0)

    def test_aggregation(self) -> None:
        r1 = MethodRankingResult(
            case_id="c1",
            method_name="m",
            window_mode="oracle",
            ranking=(RankedEntity("target", 1.0, 1),),
            candidate_universe=("target", "other"),
            status="SUCCESS",
            metrics=CaseMetrics(
                top1=True, top3=True, top5=True, mrr=1.0,
                ac1=1.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                avg3=1.0, avg5=1.0, target_rank=1
            ),
            runtime_sec=0.1,
        )
        r2 = MethodRankingResult(
            case_id="c2",
            method_name="m",
            window_mode="oracle",
            ranking=(RankedEntity("other", 1.0, 1), RankedEntity("target", 0.5, 2)),
            candidate_universe=("target", "other"),
            status="SUCCESS",
            metrics=CaseMetrics(
                top1=False, top3=True, top5=True, mrr=0.5,
                ac1=0.0, ac2=1.0, ac3=1.0, ac4=1.0, ac5=1.0,
                avg3=2.0/3.0, avg5=4.0/5.0, target_rank=2
            ),
            runtime_sec=0.2,
        )
        agg = aggregate_case_metrics("m", [r1, r2])
        self.assertEqual(agg.total_cases, 2)
        self.assertEqual(agg.evaluated_cases, 2)
        self.assertEqual(agg.top1_accuracy, 0.5)
        self.assertEqual(agg.top3_accuracy, 1.0)
        self.assertEqual(agg.mrr, 0.75)
        self.assertEqual(agg.ac1_accuracy, 0.5)
        self.assertEqual(agg.ac2_accuracy, 1.0)
        self.assertAlmostEqual(agg.avg5_accuracy, 0.9)
        self.assertAlmostEqual(agg.mean_runtime_sec, 0.15)


if __name__ == "__main__":
    unittest.main()
