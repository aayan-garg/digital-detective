"""Structure-aware document ingestion and chunking pipeline for Digital Detective RAG.

Preserves document headings, section hierarchies, command blocks, prerequisites,
verification procedures, and rollback instructions from technical runbooks.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from .models import KnowledgeDocument


PREREQ_PATTERNS = re.compile(
    r"\b(prerequisite|before you begin|requirements|prereq|prepare)\b",
    re.IGNORECASE,
)
VERIFY_PATTERNS = re.compile(
    r"\b(verify|verification|check|confirm|validate|inspection)\b",
    re.IGNORECASE,
)
ROLLBACK_PATTERNS = re.compile(
    r"\b(rollback|roll back|undo|revert|restore|recovery)\b",
    re.IGNORECASE,
)
COMMAND_BLOCK_PATTERNS = re.compile(
    r"```(?:bash|sh|shell|console|promql)?\n(.*?)```",
    re.DOTALL,
)


@dataclass(frozen=True)
class StructuredChunk:
    chunk_id: str
    document_id: str
    title: str
    section: str
    text: str
    category: str
    source: str
    version: str
    has_prerequisites: bool
    has_verification: bool
    has_rollback: bool
    command_snippets: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_knowledge_document(self) -> KnowledgeDocument:
        meta = dict(self.metadata)
        meta.update({
            "category": self.category,
            "section": self.section,
            "source": self.source,
            "version": self.version,
            "has_prerequisites": self.has_prerequisites,
            "has_verification": self.has_verification,
            "has_rollback": self.has_rollback,
            "chunk_id": self.chunk_id,
        })
        return KnowledgeDocument(
            document_id=self.chunk_id,
            title=f"{self.title} - {self.section}" if self.section and self.section != self.title else self.title,
            text=self.text,
            metadata=meta,
        )


class RunbookIngestionEngine:
    """Structure-aware ingestion engine parsing and chunking technical runbooks."""

    def __init__(
        self,
        max_chunk_chars: int = 1200,
        chunk_overlap_chars: int = 200,
        min_chunk_chars: int = 80,
        *,
        chunk_size_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        if chunk_size_chars is not None:
            max_chunk_chars = chunk_size_chars
        if overlap_chars is not None:
            chunk_overlap_chars = overlap_chars
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def process_documents(self, documents: Sequence[KnowledgeDocument | Mapping[str, Any]]) -> list[StructuredChunk]:
        """Process a sequence of KnowledgeDocument objects or dictionaries into chunks."""
        all_chunks: list[StructuredChunk] = []
        for doc in documents:
            if isinstance(doc, KnowledgeDocument):
                d_dict = {
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
            else:
                d_dict = doc
            all_chunks.extend(self.ingest_document(d_dict))
        return all_chunks

    def ingest_document(self, doc_dict: Mapping[str, Any]) -> list[StructuredChunk]:
        """Ingest a single document dict and produce structure-aware chunks."""
        doc_id = str(doc_dict.get("document_id", ""))
        title = str(doc_dict.get("title", "Untitled"))
        raw_text = str(doc_dict.get("text", "")).strip()
        metadata = dict(doc_dict.get("metadata", {}))

        category = str(metadata.get("knowledge_category", metadata.get("category", "general")))
        source = str(metadata.get("source", "official"))
        version = str(metadata.get("version", "latest"))
        section = str(metadata.get("section", title))

        if len(raw_text) < self.min_chunk_chars:
            return []

        # Detect commands
        commands = [m.strip() for m in COMMAND_BLOCK_PATTERNS.findall(raw_text) if m.strip()]

        # Split text into structure-aware windows if exceeds max_chunk_chars
        text_windows = self._split_structure_aware(raw_text)
        chunks: list[StructuredChunk] = []

        for idx, win_text in enumerate(text_windows):
            chunk_id = f"{doc_id}#c{idx}"
            has_prereqs = bool(PREREQ_PATTERNS.search(win_text) or PREREQ_PATTERNS.search(section))
            has_verify = bool(VERIFY_PATTERNS.search(win_text) or VERIFY_PATTERNS.search(section))
            has_rollback = bool(ROLLBACK_PATTERNS.search(win_text) or ROLLBACK_PATTERNS.search(section))
            win_commands = [c for c in commands if c in win_text]

            chunk = StructuredChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                title=title,
                section=section,
                text=win_text,
                category=category,
                source=source,
                version=version,
                has_prerequisites=has_prereqs,
                has_verification=has_verify,
                has_rollback=has_rollback,
                command_snippets=tuple(win_commands[:3]),
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def ingest_corpus(self, corpus_path: str | Path) -> list[StructuredChunk]:
        """Ingest an entire jsonl corpus file into structured chunks."""
        path = Path(corpus_path)
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        all_chunks: list[StructuredChunk] = []
        seen_hashes: set[str] = set()

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_chunks = self.ingest_document(doc)
                for chunk in doc_chunks:
                    # Deduplicate near-identical chunks
                    chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                    if chunk_hash in seen_hashes:
                        continue
                    seen_hashes.add(chunk_hash)
                    all_chunks.append(chunk)

        return all_chunks

    def _split_structure_aware(self, text: str) -> list[str]:
        """Split text while preserving sentences and command blocks."""
        if len(text) <= self.max_chunk_chars:
            return [text]

        # Break text into paragraphs or logical sections
        paragraphs = re.split(r"\n\s*\n|(?<=\. )(?=[A-Z])", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            p_len = len(p_clean)

            if current_len + p_len > self.max_chunk_chars and current:
                chunks.append(" ".join(current))
                # Retain overlap from end of previous chunk
                overlap_text = current[-1] if len(current[-1]) <= self.chunk_overlap_chars else ""
                current = [overlap_text, p_clean] if overlap_text else [p_clean]
                current_len = sum(len(x) for x in current)
            else:
                current.append(p_clean)
                current_len += p_len + 1

        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if len(c) >= self.min_chunk_chars]
