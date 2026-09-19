"""Deterministic TF-IDF and cosine retrieval without external services."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from .models import KnowledgeDocument, RetrievalResult

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class DeterministicRetriever:
    """Rank local documents by TF-IDF cosine similarity."""

    def __init__(self, documents: Iterable[KnowledgeDocument], snippet_chars: int = 360) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("At least one document is required.")
        if snippet_chars <= 0:
            raise ValueError("snippet_chars must be positive.")
        identifiers = [document.document_id for document in self.documents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Document IDs must be unique.")
        self._snippet_chars = snippet_chars
        token_counts = [Counter(_tokenize(f"{doc.title} {doc.text}")) for doc in self.documents]
        document_frequency = Counter(
            token for counts in token_counts for token in counts
        )
        document_count = len(self.documents)
        self._idf = {
            token: math.log((1 + document_count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._vectors = tuple(
            _normalise({token: count * self._idf[token] for token, count in counts.items()})
            for counts in token_counts
        )

    def retrieve(self, query: str, k: int = 5) -> tuple[RetrievalResult, ...]:
        """Return up to ``k`` positive-scoring documents in stable rank order."""

        if k <= 0:
            raise ValueError("k must be positive.")
        query_vector = _normalise(
            {
                token: count * self._idf[token]
                for token, count in Counter(_tokenize(query)).items()
                if token in self._idf
            }
        )
        if not query_vector:
            return ()

        scored = [
            (sum(query_vector.get(token, 0.0) * value for token, value in vector.items()), document)
            for vector, document in zip(self._vectors, self.documents)
        ]
        scored = [(score, document) for score, document in scored if score > 0.0]
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


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _normalise(vector: dict[str, float]) -> dict[str, float]:
    magnitude = math.sqrt(sum(value * value for value in vector.values()))
    if magnitude == 0.0:
        return {}
    return {token: value / magnitude for token, value in vector.items()}


def _snippet(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."
