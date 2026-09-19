"""Focused tests for the isolated retrieval foundation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from digital_detective.rag import (
    DeterministicRetriever,
    KnowledgeDocument,
    build_query,
    format_retrieved_context,
    load_corpus,
)


class RagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            KnowledgeDocument(
                "doc-cpu",
                "CPU saturation",
                "CPU utilization and worker pool saturation checklist.",
                {"service": "checkout", "source": "synthetic"},
            ),
            KnowledgeDocument(
                "doc-memory",
                "Memory pressure",
                "Resident memory and garbage collection checklist.",
                {"service": "cart", "source": "synthetic"},
            ),
            KnowledgeDocument(
                "doc-network",
                "Network latency",
                "Trace timing, retries, and timeout checklist.",
                {"service": "shipping", "source": "synthetic"},
            ),
        )
        self.retriever = DeterministicRetriever(self.documents, snippet_chars=32)

    def test_corpus_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "doc-1",
                            "title": "Local note",
                            "content": "Synthetic content.",
                            "metadata": {"source": "test"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            loaded = load_corpus(path)
        self.assertEqual(loaded[0].document_id, "doc-1")
        self.assertEqual(loaded[0].text, "Synthetic content.")

    def test_deterministic_retrieval_and_score_ordering(self) -> None:
        first = self.retriever.retrieve("checklist CPU", k=3)
        second = self.retriever.retrieve("checklist CPU", k=3)
        self.assertEqual(first, second)
        self.assertEqual(first[0].document_id, "doc-cpu")
        self.assertGreaterEqual(first[0].score, first[1].score)

    def test_top_k_behavior(self) -> None:
        self.assertEqual(len(self.retriever.retrieve("checklist", k=2)), 2)

    def test_empty_and_no_match_behavior(self) -> None:
        self.assertEqual(self.retriever.retrieve("", k=2), ())
        self.assertEqual(self.retriever.retrieve("unseen vocabulary", k=2), ())

    def test_metadata_preservation(self) -> None:
        result = self.retriever.retrieve("memory garbage collection", k=1)[0]
        self.assertEqual(result.metadata["service"], "cart")
        self.assertEqual(result.metadata["source"], "synthetic")

    def test_query_builder_ignores_prohibited_fields(self) -> None:
        query = build_query(
            {
                "candidate_service": "checkout",
                "observed_signals": ["CPU utilization"],
                "modalities": ["metrics"],
                "ground_truth": "secret-root",
                "fault_type": "secret-fault",
                "inject_time": "future-only",
            }
        )
        self.assertEqual(query, "checkout CPU utilization metrics")
        self.assertNotIn("secret-root", query)
        self.assertNotIn("secret-fault", query)
        self.assertNotIn("future-only", query)

    def test_context_formatting_is_bounded_and_preserves_metadata(self) -> None:
        result = self.retriever.retrieve("CPU utilization", k=1)[0]
        context = format_retrieved_context((result,), max_chars=180)
        self.assertLessEqual(len(context), 180)
        self.assertIn("CPU saturation", context)
        self.assertIn("service=checkout", context)
        self.assertIn("Document ID: doc-cpu", context)


if __name__ == "__main__":
    unittest.main()
