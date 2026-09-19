"""Unified, fixed-condition RAG-2 experiment runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any, Mapping, Sequence

from .bm25 import BM25Retriever
from .dense import BGE_MODEL_NAME, DenseRetriever
from .fusion import RRF_K, reciprocal_rank_fusion
from .metrics import evaluate_ranking, summarise_metrics
from .models import KnowledgeDocument, RetrievalResult
from .reranker import ETTIN_MODEL_NAME, CrossEncoderReranker
from .retriever import DeterministicRetriever
from .context import format_retrieved_context

SPARSE_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_CANDIDATE_K = 20
FINAL_CONTEXT_K = 5


class RetrievalCondition(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    query: str
    relevance: Mapping[str, float] | None = None


class RAG2Experiment:
    """Run the five fixed retrieval conditions over one corpus/query set."""

    def __init__(self, documents: Sequence[KnowledgeDocument]) -> None:
        self.documents = tuple(documents)
        self.corpus_snapshot = hashlib.sha256(
            json.dumps(
                [
                    {
                        "document_id": doc.document_id,
                        "title": doc.title,
                        "text": doc.text,
                        "metadata": dict(sorted(doc.metadata.items())),
                    }
                    for doc in self.documents
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._tfidf = DeterministicRetriever(self.documents)
        self._bm25 = BM25Retriever(self.documents)
        self._dense: DenseRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None

    def _get_dense(self) -> DenseRetriever:
        if self._dense is None:
            self._dense = DenseRetriever(self.documents, model_name=BGE_MODEL_NAME)
        return self._dense

    def _get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=ETTIN_MODEL_NAME)
        return self._reranker

    def retrieve(self, condition: RetrievalCondition, query: str) -> tuple[RetrievalResult, ...]:
        if condition == RetrievalCondition.C0:
            return self._tfidf.retrieve(query, k=FINAL_CONTEXT_K)
        if condition == RetrievalCondition.C1:
            return self._bm25.retrieve(query, k=FINAL_CONTEXT_K)
        dense = self._get_dense().retrieve(query, k=DENSE_TOP_K)
        if condition == RetrievalCondition.C2:
            return dense[:FINAL_CONTEXT_K]
        sparse = self._bm25.retrieve(query, k=SPARSE_TOP_K)
        hybrid = reciprocal_rank_fusion(
            (sparse, dense), k=RRF_K, candidate_k=HYBRID_CANDIDATE_K
        )
        if condition == RetrievalCondition.C3:
            return hybrid[:FINAL_CONTEXT_K]
        return self._get_reranker().rerank(query, hybrid, k=FINAL_CONTEXT_K)

    def run(
        self,
        queries: Sequence[RetrievalQuery],
        conditions: Sequence[RetrievalCondition] = tuple(RetrievalCondition),
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for condition in conditions:
            condition_metrics: list[Mapping[str, float | None]] = []
            latencies: list[float] = []
            context_sizes: list[int] = []
            for item in queries:
                started = perf_counter()
                results = self.retrieve(condition, item.query)
                latency_ms = (perf_counter() - started) * 1000
                context = format_retrieved_context(results)
                evaluation = (
                    evaluate_ranking(results, item.relevance)
                    if item.relevance is not None
                    else None
                )
                if evaluation is not None:
                    condition_metrics.append(evaluation)
                latencies.append(latency_ms)
                context_sizes.append(len(context.split()))
                records.append(
                    {
                        "query_id": item.query_id,
                        "condition": condition.value,
                        "query": item.query,
                        "query_hash": hashlib.sha256(item.query.encode("utf-8")).hexdigest(),
                        "corpus_snapshot": self.corpus_snapshot,
                        "retrieved": [
                            {
                                "document_id": result.document_id,
                                "score": result.score,
                                "rrf_score": (
                                    result.score
                                    if condition == RetrievalCondition.C3
                                    else None
                                ),
                                "reranker_score": (
                                    result.score
                                    if condition == RetrievalCondition.C4
                                    else None
                                ),
                            }
                            for result in results
                        ],
                        "top_5_document_ids": [result.document_id for result in results[:5]],
                        "latency_ms": latency_ms,
                        "model_name": _model_name(condition),
                        "provenance": [dict(result.metadata) for result in results],
                        "relevance": dict(item.relevance) if item.relevance is not None else None,
                        "context_token_count_est": len(context.split()),
                    }
                )
            summaries.append(
                {
                    "condition": condition.value,
                    **summarise_metrics(condition_metrics, latencies, context_sizes),
                }
            )
        return {
            "schema_version": "rag2-v1",
            "corpus_snapshot": self.corpus_snapshot,
            "conditions": [condition.value for condition in conditions],
            "fixed_parameters": {
                "sparse_top_k": SPARSE_TOP_K,
                "dense_top_k": DENSE_TOP_K,
                "rrf_k": RRF_K,
                "hybrid_candidate_pool": HYBRID_CANDIDATE_K,
                "final_context_k": FINAL_CONTEXT_K,
            },
            "records": records,
            "summary": summaries,
        }


def _model_name(condition: RetrievalCondition) -> str:
    return {
        RetrievalCondition.C0: "rag1-tfidf",
        RetrievalCondition.C1: "bm25",
        RetrievalCondition.C2: BGE_MODEL_NAME,
        RetrievalCondition.C3: f"bm25+{BGE_MODEL_NAME}+rrf-{RRF_K}",
        RetrievalCondition.C4: f"bm25+{BGE_MODEL_NAME}+rrf-{RRF_K}+{ETTIN_MODEL_NAME}",
    }[condition]
