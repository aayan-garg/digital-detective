"""Loading of local JSON operational-knowledge documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import KnowledgeDocument


def load_corpus(path: str | Path) -> tuple[KnowledgeDocument, ...]:
    """Load one JSON document file or all JSON files in a directory.

    A file may contain either a list of document objects or an object with a
    ``documents`` list. Directory entries are processed in sorted filename
    order so corpus construction is reproducible.
    """

    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("*.json"))
        if not files:
            raise ValueError(f"No JSON corpus files found in {source}.")
    elif source.is_file():
        files = [source]
    else:
        raise FileNotFoundError(f"Corpus path does not exist: {source}")

    documents: list[KnowledgeDocument] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            with file_path.open(encoding="utf-8") as handle:
                documents.extend(
                    _document_from_entry(json.loads(line), file_path)
                    for line in handle
                    if line.strip()
                )
        else:
            with file_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            entries = payload.get("documents") if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                raise ValueError(f"{file_path} must contain a JSON list or a 'documents' list.")
            documents.extend(_document_from_entry(entry, file_path) for entry in entries)

    identifiers = [document.document_id for document in documents]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Corpus document_id values must be unique.")
    return tuple(documents)


def _document_from_entry(entry: Any, source: Path) -> KnowledgeDocument:
    if not isinstance(entry, dict):
        raise ValueError(f"Each corpus entry in {source} must be an object.")
    document_id = entry.get("document_id", entry.get("id"))
    text = entry.get("text", entry.get("content"))
    title = entry.get("title")
    if not all(isinstance(value, str) and value.strip() for value in (document_id, title, text)):
        raise ValueError(f"Each corpus entry in {source} needs non-empty id, title, and text.")
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError(f"Metadata in {source} must be an object of string values.")
    return KnowledgeDocument(
        document_id=document_id.strip(),
        title=title.strip(),
        text=text.strip(),
        metadata=dict(metadata),
    )
