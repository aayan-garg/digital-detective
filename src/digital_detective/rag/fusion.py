"""Fixed reciprocal-rank fusion for the RAG-2 hybrid arm."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import RetrievalResult

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]],
    *,
    k: int = RRF_K,
    candidate_k: int = 20,
) -> tuple[RetrievalResult, ...]:
    """Fuse rankings using unweighted RRF and stable document-ID ordering."""

    if k <= 0 or candidate_k <= 0:
        raise ValueError("k and candidate_k must be positive.")
    scores: defaultdict[str, float] = defaultdict(float)
    records: dict[str, RetrievalResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking[:candidate_k], 1):
            scores[result.document_id] += 1.0 / (k + rank)
            records.setdefault(result.document_id, result)
    ordered = sorted(records, key=lambda document_id: (-scores[document_id], document_id))
    return tuple(
        RetrievalResult(
            document_id=document_id,
            score=scores[document_id],
            title=records[document_id].title,
            metadata=dict(records[document_id].metadata),
            snippet=records[document_id].snippet,
        )
        for document_id in ordered[:candidate_k]
    )
