"""Local, deterministic retrieval foundation for operational knowledge."""

from .context import format_retrieved_context
from .corpus import load_corpus
from .bm25 import BM25Retriever
from .dense import BGE_MODEL_NAME, DenseRetriever, ModelUnavailableError
from .experiment import (
    DENSE_TOP_K,
    FINAL_CONTEXT_K,
    HYBRID_CANDIDATE_K,
    RAG2Experiment,
    RetrievalCondition,
    RetrievalQuery,
    SPARSE_TOP_K,
)
from .fusion import RRF_K, reciprocal_rank_fusion
from .models import IncidentQueryContext, KnowledgeDocument, RetrievalResult
from .query import build_query
from .reranker import ETTIN_MODEL_NAME, CrossEncoderReranker
from .retriever import DeterministicRetriever

__all__ = [
    "DeterministicRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "CrossEncoderReranker",
    "IncidentQueryContext",
    "KnowledgeDocument",
    "ModelUnavailableError",
    "RAG2Experiment",
    "RetrievalResult",
    "RetrievalCondition",
    "RetrievalQuery",
    "BGE_MODEL_NAME",
    "ETTIN_MODEL_NAME",
    "RRF_K",
    "SPARSE_TOP_K",
    "DENSE_TOP_K",
    "HYBRID_CANDIDATE_K",
    "FINAL_CONTEXT_K",
    "build_query",
    "format_retrieved_context",
    "load_corpus",
    "reciprocal_rank_fusion",
]
