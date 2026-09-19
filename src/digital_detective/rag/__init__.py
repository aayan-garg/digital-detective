"""Local, deterministic retrieval foundation for operational knowledge."""

from .context import format_retrieved_context
from .corpus import load_corpus
from .models import IncidentQueryContext, KnowledgeDocument, RetrievalResult
from .query import build_query
from .retriever import DeterministicRetriever

__all__ = [
    "DeterministicRetriever",
    "IncidentQueryContext",
    "KnowledgeDocument",
    "RetrievalResult",
    "build_query",
    "format_retrieved_context",
    "load_corpus",
]
