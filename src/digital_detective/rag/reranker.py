"""Optional exact Ettin 17M cross-encoder arm."""

from __future__ import annotations

from collections.abc import Sequence

from .dense import ModelUnavailableError
from .models import RetrievalResult

ETTIN_MODEL_NAME = "cross-encoder/ettin-reranker-17m-v1"


class CrossEncoderReranker:
    """Rerank exactly the supplied hybrid candidate pool."""

    def __init__(self, *, model_name: str = ETTIN_MODEL_NAME) -> None:
        if model_name != ETTIN_MODEL_NAME:
            raise ValueError(f"RAG-2 requires model {ETTIN_MODEL_NAME!r}.")
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ModelUnavailableError(
                "Reranking requires sentence-transformers and "
                f"{ETTIN_MODEL_NAME}; neither is installed in the active environment."
            ) from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Unable to load {model_name!r}; no fallback is allowed."
            ) from exc

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], k: int = 5
    ) -> tuple[RetrievalResult, ...]:
        if k <= 0:
            raise ValueError("k must be positive.")
        try:
            scores = self._model.predict(
                [(query, candidate.snippet) for candidate in candidates],
                show_progress_bar=False,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Reranker inference failed for {self.model_name!r}."
            ) from exc
        ranked = sorted(
            zip(list(scores), candidates),
            key=lambda item: (-item[0], item[1].document_id),
        )
        return tuple(
            RetrievalResult(
                document_id=result.document_id,
                score=float(score),
                title=result.title,
                metadata=dict(result.metadata),
                snippet=result.snippet,
            )
            for score, result in ranked[:k]
        )
