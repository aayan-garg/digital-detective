"""Optional BGE-small dense retrieval arm with explicit dependency failures."""

from __future__ import annotations

from typing import Iterable

from .models import KnowledgeDocument, RetrievalResult

BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class ModelUnavailableError(RuntimeError):
    """Raised when an explicitly requested local model cannot be loaded."""


class DenseRetriever:
    """Dense cosine retrieval using the exact RAG-2 BGE model."""

    def __init__(
        self,
        documents: Iterable[KnowledgeDocument],
        *,
        model_name: str = BGE_MODEL_NAME,
        snippet_chars: int = 360,
    ) -> None:
        if model_name != BGE_MODEL_NAME:
            raise ValueError(f"RAG-2 requires model {BGE_MODEL_NAME!r}.")
        if snippet_chars <= 0:
            raise ValueError("snippet_chars must be positive.")
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("At least one document is required.")
        self.model_name = model_name
        self._snippet_chars = snippet_chars
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelUnavailableError(
                "Dense retrieval requires sentence-transformers and "
                f"{BGE_MODEL_NAME}; neither is installed in the active environment."
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
            self._document_vectors = self._model.encode(
                [f"{doc.title}\n{doc.text}" for doc in self.documents],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise ModelUnavailableError(
                f"Unable to load or encode with {model_name!r}; no fallback is allowed."
            ) from exc

    def retrieve(self, query: str, k: int = 20) -> tuple[RetrievalResult, ...]:
        if k <= 0:
            raise ValueError("k must be positive.")
        try:
            query_vector = self._model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        except Exception as exc:
            raise ModelUnavailableError(
                f"Dense inference failed for {self.model_name!r}."
            ) from exc
        scores = self._document_vectors @ query_vector
        ranked = sorted(
            zip(scores.tolist(), self.documents),
            key=lambda item: (-item[0], item[1].document_id),
        )
        return tuple(
            RetrievalResult(
                document_id=document.document_id,
                score=float(score),
                title=document.title,
                metadata=dict(document.metadata),
                snippet=_snippet(document.text, self._snippet_chars),
            )
            for score, document in ranked[:k]
        )


def _snippet(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)].rstrip() + "..."
