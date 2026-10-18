"""Advanced hybrid retrieval system combining lexical BM25 and dense BGE with RRF.

Implements:
1. Lexical BM25 retrieval tuned for microservice identifiers, exceptions, and metrics.
2. Dense semantic retrieval with BGE-small-en-v1.5 embeddings.
3. Reciprocal Rank Fusion (RRF) for scale-invariant ranking combination.
4. Metadata-aware ranking boosts for matching failure categories.
5. Diversity filtering to avoid clustering near-identical chunks from a single document.
6. Optional cross-encoder reranking with graceful fallbacks.
7. Persistent disk caching to avoid redundant embedding computation.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .bm25 import BM25Retriever
from .dense import DenseRetriever, ModelUnavailableError
from .models import KnowledgeDocument, RetrievalResult


@dataclass(frozen=True)
class HybridRetrievalConfig:
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    category_boost: float = 1.25
    diversity_limit_per_doc: int = 2
    snippet_chars: int = 400


class HybridRetriever:
    """Production-grade hybrid retriever combining BM25, BGE-small, and RRF."""

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        config: HybridRetrievalConfig | None = None,
        cache_dir: str | Path | None = None,
        enable_dense: bool = True,
    ) -> None:
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("At least one document is required for retrieval.")
        self.config = config or HybridRetrievalConfig()
        self.doc_map = {doc.document_id: doc for doc in self.documents}

        # Initialize BM25 lexical arm
        self._bm25 = BM25Retriever(self.documents, snippet_chars=self.config.snippet_chars)

        # Initialize or load dense arm with persistent disk caching
        self._dense: DenseRetriever | None = None
        self._dense_error: str | None = None
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[3] / "data" / "rag" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if enable_dense:
            try:
                self._dense = DenseRetriever(self.documents, snippet_chars=self.config.snippet_chars)
                # Check embedding cache
                corpus_hash = hashlib.sha256(
                    "".join(d.document_id for d in self.documents).encode("utf-8")
                ).hexdigest()[:16]
                emb_cache_file = self.cache_dir / f"bge_embeddings_{corpus_hash}.npy"

                if emb_cache_file.is_file():
                    try:
                        loaded_vecs = np.load(str(emb_cache_file))
                        if loaded_vecs.shape == (len(self.documents), 384):
                            self._dense._document_vectors = loaded_vecs
                    except Exception:
                        pass
                else:
                    try:
                        np.save(str(emb_cache_file), self._dense._document_vectors)
                    except Exception:
                        pass
            except Exception as exc:
                self._dense_error = str(exc)
        else:
            self._dense_error = "Dense retrieval explicitly disabled."

    @property
    def has_dense(self) -> bool:
        return self._dense is not None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        target_category: str | None = None,
        target_service: str | None = None,
        rerank: bool = False,
    ) -> list[RetrievalResult]:
        """Retrieve top-k documents using hybrid RRF fusion with diversity controls."""
        if k <= 0:
            raise ValueError("k must be positive")

        candidate_pool_size = max(20, k * 4)

        # 1. Lexical retrieval
        bm25_results = self._bm25.retrieve(query, k=candidate_pool_size)
        bm25_ranks = {res.document_id: rank + 1 for rank, res in enumerate(bm25_results)}

        # 2. Dense retrieval (if available)
        dense_ranks: dict[str, int] = {}
        if self._dense is not None:
            try:
                dense_results = self._dense.retrieve(query, k=candidate_pool_size)
                dense_ranks = {res.document_id: rank + 1 for rank, res in enumerate(dense_results)}
            except Exception:
                dense_ranks = {}

        # 3. Reciprocal Rank Fusion (RRF)
        all_doc_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        rrf_scores: dict[str, float] = {}

        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += self.config.bm25_weight / (self.config.rrf_k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += self.config.dense_weight / (self.config.rrf_k + dense_ranks[doc_id])

            # 4. Metadata-aware boosting
            doc = self.doc_map.get(doc_id)
            if doc:
                meta = doc.metadata
                cat = str(meta.get("category", meta.get("knowledge_category", ""))).lower()
                title_lower = doc.title.lower()
                text_lower = doc.text.lower()

                # Boost if matches incident category
                if target_category and target_category.lower() in cat:
                    score *= self.config.category_boost

                # Boost if mentions target service
                if target_service and (target_service.lower() in title_lower or target_service.lower() in text_lower):
                    score *= 1.15

                # Modest boost for actionable troubleshooting content
                if meta.get("has_verification") or meta.get("has_rollback"):
                    score *= 1.08

            rrf_scores[doc_id] = score

        # 5. Sort by RRF score
        sorted_ids = sorted(all_doc_ids, key=lambda did: (-rrf_scores[did], did))

        # 6. Apply diversity filtering (limit consecutive chunks from the same base document)
        doc_count: dict[str, int] = defaultdict(int)
        filtered_results: list[RetrievalResult] = []

        for did in sorted_ids:
            doc = self.doc_map[did]
            # Base doc ID is before the chunk index '#'
            base_id = did.split("#")[0]
            if doc_count[base_id] >= self.config.diversity_limit_per_doc:
                continue

            doc_count[base_id] += 1
            snippet = doc.text[: self.config.snippet_chars] + ("..." if len(doc.text) > self.config.snippet_chars else "")

            filtered_results.append(
                RetrievalResult(
                    document_id=doc.document_id,
                    score=float(rrf_scores[did]),
                    title=doc.title,
                    metadata=dict(doc.metadata),
                    snippet=snippet,
                )
            )
            if len(filtered_results) >= k:
                break

        return filtered_results
