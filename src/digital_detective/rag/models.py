"""Small data models used by the isolated retrieval foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class KnowledgeDocument:
    """A local operational-knowledge document."""

    document_id: str
    title: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked document returned by a deterministic retriever."""

    document_id: str
    score: float
    title: str
    metadata: Mapping[str, str]
    snippet: str


@dataclass(frozen=True)
class IncidentQueryContext:
    """Observed incident fields that are safe to turn into a retrieval query."""

    candidate_service: str = ""
    observed_signals: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
