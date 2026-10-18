"""Tests verifying correctness of RAG evaluation metrics against hand-calculated examples."""

import math
import unittest
from typing import Sequence


def compute_precision_at_k(binary_relevances: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(binary_relevances[:k]) / float(k)


def compute_standard_recall_at_k(binary_relevances: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(binary_relevances[:k]) / float(total_relevant)


def compute_hit_at_k(binary_relevances: Sequence[int], k: int) -> float:
    return 1.0 if sum(binary_relevances[:k]) > 0 else 0.0


def compute_mrr(binary_relevances: Sequence[int]) -> float:
    for rank_idx, rel in enumerate(binary_relevances, 1):
        if rel > 0:
            return 1.0 / rank_idx
    return 0.0


def compute_dcg(relevances: Sequence[int], k: int = 5) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def compute_ndcg(relevances: Sequence[int], total_relevant: int, k: int = 5) -> float:
    if total_relevant <= 0:
        return 0.0
    actual_dcg = compute_dcg(relevances, k)
    ideal_relevances = [1] * min(total_relevant, k)
    ideal_dcg = compute_dcg(ideal_relevances, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


class TestRAGMetricsHandCalculated(unittest.TestCase):
    def test_example_1_relevant_at_rank_1(self):
        # Retrieved: [1, 0, 0, 0, 0], total relevant in corpus = 5
        rels = [1, 0, 0, 0, 0]
        total_rel = 5

        self.assertAlmostEqual(compute_precision_at_k(rels, 1), 1.0)
        self.assertAlmostEqual(compute_precision_at_k(rels, 5), 1.0 / 5.0)
        self.assertAlmostEqual(compute_standard_recall_at_k(rels, total_rel, 1), 1.0 / 5.0)
        self.assertAlmostEqual(compute_standard_recall_at_k(rels, total_rel, 5), 1.0 / 5.0)
        self.assertAlmostEqual(compute_hit_at_k(rels, 1), 1.0)
        self.assertAlmostEqual(compute_hit_at_k(rels, 5), 1.0)
        self.assertAlmostEqual(compute_mrr(rels), 1.0)
        # DCG@5 = 1 / log2(2) = 1.0. IDCG@5 = sum_{i=1}^5 1 / log2(i+1) = 1.0 + 0.6309 + 0.5 + 0.4307 + 0.3869 = 2.9485
        # nDCG@5 = 1.0 / 2.9485 = 0.3391
        expected_idcg = sum(1.0 / math.log2(i + 1) for i in range(1, 6))
        self.assertAlmostEqual(compute_ndcg(rels, total_rel, k=5), 1.0 / expected_idcg, places=4)

    def test_example_2_relevant_at_rank_2(self):
        # Retrieved: [0, 1, 0, 0, 0], total relevant in corpus = 2
        rels = [0, 1, 0, 0, 0]
        total_rel = 2

        self.assertAlmostEqual(compute_precision_at_k(rels, 1), 0.0)
        self.assertAlmostEqual(compute_precision_at_k(rels, 2), 0.5)
        self.assertAlmostEqual(compute_hit_at_k(rels, 1), 0.0)
        self.assertAlmostEqual(compute_hit_at_k(rels, 2), 1.0)
        self.assertAlmostEqual(compute_standard_recall_at_k(rels, total_rel, 1), 0.0)
        self.assertAlmostEqual(compute_standard_recall_at_k(rels, total_rel, 2), 0.5)
        self.assertAlmostEqual(compute_mrr(rels), 0.5)

    def test_example_3_no_relevant(self):
        rels = [0, 0, 0, 0, 0]
        total_rel = 10

        self.assertAlmostEqual(compute_precision_at_k(rels, 5), 0.0)
        self.assertAlmostEqual(compute_hit_at_k(rels, 5), 0.0)
        self.assertAlmostEqual(compute_standard_recall_at_k(rels, total_rel, 5), 0.0)
        self.assertAlmostEqual(compute_mrr(rels), 0.0)
        self.assertAlmostEqual(compute_ndcg(rels, total_rel, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
