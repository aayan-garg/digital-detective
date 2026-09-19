"""Small deterministic BM25 retriever for the frozen RAG-1 corpus."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .models import KnowledgeDocument, RetrievalResult
from .tokenization import tokenize


class BM25Retriever:
    """Rank documents with Okapi BM25 and stable document-ID tie-breaking."""

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        snippet_chars: int = 360,
    ) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("At least one document is required.")
        if k1 < 0 or not 0 <= b <= 1 or snippet_chars <= 0:
            raise ValueError("BM25 parameters or snippet_chars are out of range.")
        identifiers = [document.document_id for document in self.documents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Document IDs must be unique.")
        self._k1 = k1
        self._b = b
        self._snippet_chars = snippet_chars
        self._term_counts = tuple(
            Counter(tokenize(f"{doc.title} {doc.text}")) for doc in self.documents
        )
        self._lengths = tuple(sum(counts.values()) for counts in self._term_counts)
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency = Counter(
            token for counts in self._term_counts for token in counts
        )
        self._idf = {
            token: math.log(
                1.0 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

    def retrieve(self, query: str, k: int = 20) -> tuple[RetrievalResult, ...]:
        if k <= 0:
            raise ValueError("k must be positive.")
        query_terms = tokenize(query)
        query_counts = Counter(query_terms)
        scored: list[tuple[float, KnowledgeDocument]] = []
        for document, counts, length in zip(
            self.documents, self._term_counts, self._lengths
        ):
            score = 0.0
            for term, query_frequency in query_counts.items():
                if term not in counts:
                    continue
                denominator = counts[term] + self._k1 * (
                    1.0 - self._b + self._b * length / self._average_length
                )
                score += self._idf[term] * (
                    counts[term] * (self._k1 + 1.0) / denominator
                ) * query_frequency
            if score > 0.0:
                scored.append((score, document))
        scored.sort(key=lambda item: (-item[0], item[1].document_id))
        return tuple(
            RetrievalResult(
                document_id=document.document_id,
                score=score,
                title=document.title,
                metadata=dict(document.metadata),
                snippet=_snippet(document.text, self._snippet_chars),
            )
            for score, document in scored[:k]
        )


def _snippet(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."
