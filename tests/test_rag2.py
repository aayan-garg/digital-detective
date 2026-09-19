"""Focused tests for the controlled RAG-2 retrieval experiment."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from digital_detective.rag import (
    BM25Retriever,
    CrossEncoderReranker,
    DeterministicRetriever,
    KnowledgeDocument,
    ModelUnavailableError,
    RAG2Experiment,
    RetrievalCondition,
    RetrievalQuery,
    reciprocal_rank_fusion,
)
from digital_detective.rag.models import RetrievalResult
from digital_detective.rag.metrics import paired_bootstrap_mean_ci


class Rag2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            KnowledgeDocument("cpu", "CPU saturation", "cpu utilization worker pool", {"source": "synthetic"}),
            KnowledgeDocument("memory", "Memory pressure", "memory garbage collection", {"source": "synthetic"}),
            KnowledgeDocument("network", "Network latency", "network retries timeout traces", {"source": "synthetic"}),
        )

    def test_bm25_ranking_and_top_k(self) -> None:
        results = BM25Retriever(self.documents).retrieve("CPU worker", k=1)
        self.assertEqual([result.document_id for result in results], ["cpu"])

    def test_rrf_uses_fixed_formula_and_stable_ties(self) -> None:
        first = RetrievalResult("b", 0.1, "B", {}, "b")
        second = RetrievalResult("a", 0.2, "A", {}, "a")
        fused = reciprocal_rank_fusion(((first, second), (second, first)), k=60, candidate_k=2)
        self.assertEqual([result.document_id for result in fused], ["a", "b"])
        self.assertAlmostEqual(fused[0].score, 1 / 61 + 1 / 62)

    def test_dense_model_is_exact_and_fails_explicitly_when_unavailable(self) -> None:
        from digital_detective.rag import DenseRetriever

        try:
            retriever = DenseRetriever(self.documents)
        except ModelUnavailableError as exc:
            self.skipTest(str(exc))
        self.assertEqual(retriever.model_name, "BAAI/bge-small-en-v1.5")
        self.assertEqual(len(retriever.retrieve("CPU saturation", k=1)), 1)

    def test_reranker_orders_candidates_and_preserves_metadata(self) -> None:
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "cross-encoder/ettin-reranker-17m-v1"
        reranker._model = Mock()
        reranker._model.predict.return_value = [0.1, 0.9]
        candidates = (
            RetrievalResult("a", 0.2, "A", {"service": "a"}, "text a"),
            RetrievalResult("b", 0.1, "B", {"service": "b"}, "text b"),
        )
        results = reranker.rerank("query", candidates, k=2)
        self.assertEqual(results[0].document_id, "b")
        self.assertEqual(results[0].metadata["service"], "b")

    def test_same_query_and_corpus_snapshot_across_c0_and_c1(self) -> None:
        experiment = RAG2Experiment(self.documents)
        query = RetrievalQuery("q1", "CPU worker", {"cpu": 1})
        artifact = experiment.run((query,), (RetrievalCondition.C0, RetrievalCondition.C1))
        self.assertEqual(artifact["conditions"], ["C0", "C1"])
        self.assertEqual(len(artifact["records"]), 2)
        self.assertEqual(
            {record["corpus_snapshot"] for record in artifact["records"]},
            {artifact["corpus_snapshot"]},
        )
        self.assertEqual(
            {record["query_hash"] for record in artifact["records"]},
            {artifact["records"][0]["query_hash"]},
        )

    def test_rag2_constants_are_not_tunable_in_experiment(self) -> None:
        from digital_detective.rag.experiment import (
            DENSE_TOP_K,
            FINAL_CONTEXT_K,
            HYBRID_CANDIDATE_K,
            RRF_K,
            SPARSE_TOP_K,
        )

        self.assertEqual((SPARSE_TOP_K, DENSE_TOP_K, RRF_K, HYBRID_CANDIDATE_K, FINAL_CONTEXT_K), (20, 20, 60, 20, 5))

    def test_paired_bootstrap_helper_is_reproducible(self) -> None:
        first = paired_bootstrap_mean_ci((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), samples=100, seed=7)
        second = paired_bootstrap_mean_ci((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), samples=100, seed=7)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
