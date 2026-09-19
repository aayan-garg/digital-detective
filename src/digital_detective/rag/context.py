"""Formatting of retrieved knowledge for a future downstream context window."""

from __future__ import annotations

from collections.abc import Sequence

from .models import RetrievalResult


def format_retrieved_context(
    results: Sequence[RetrievalResult],
    *,
    max_chars: int = 4000,
) -> str:
    """Format retrieved documents without adding interpretation or conclusions."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    blocks = []
    for rank, result in enumerate(results, 1):
        metadata = ", ".join(
            f"{key}={value}" for key, value in sorted(result.metadata.items())
        )
        metadata_line = f"Metadata: {metadata}\n" if metadata else ""
        blocks.append(
            f"[Document {rank}] {result.title}\n"
            f"Document ID: {result.document_id}\n"
            f"Score: {result.score:.6f}\n"
            f"{metadata_line}"
            f"{result.snippet}"
        )
    return "\n\n".join(blocks)[:max_chars]
