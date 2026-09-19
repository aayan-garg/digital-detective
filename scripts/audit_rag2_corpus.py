"""Validate the frozen DD-IR-60 operational corpus without reading final queries."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CORPUS = Path("eval/rag2_benchmark/rag2_corpus.jsonl")
ALLOWED_LICENSES = {"Apache-2.0", "CC BY 4.0"}
REQUIRED_METADATA = (
    "source", "source_url", "repository", "version", "commit_tag",
    "retrieved_date", "license", "document_type", "section",
    "admission_status", "provenance",
)
FORBIDDEN_MATERIAL = re.compile(
    r"\b(RE2|RCAEval|postmortem|fault[- ]injection|benchmark label|"
    r"test incident|remediation outcome|root[- ]cause annotation)\b",
    re.IGNORECASE,
)


def audit(path: Path = CORPUS) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("document_id", "") for row in rows]
    texts = [re.sub(r"\s+", " ", row.get("text", "")).strip() for row in rows]
    metadata = [row.get("metadata", {}) for row in rows]
    required_gaps = [
        row["document_id"]
        for row, item in zip(rows, metadata)
        if any(not item.get(field) for field in REQUIRED_METADATA)
    ]
    invalid_licenses = [
        row["document_id"]
        for row, item in zip(rows, metadata)
        if item.get("license") not in ALLOWED_LICENSES
    ]
    forbidden_material = [
        row["document_id"]
        for row, text in zip(rows, texts)
        if FORBIDDEN_MATERIAL.search(text)
    ]
    duplicate_texts = len(texts) - len(set(texts))
    corpus_hash = hashlib.sha256(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")
    ).hexdigest()
    return {
        "retrieval_unit_count": len(rows),
        "unique_id_count": len(set(ids)),
        "duplicate_id_count": len(ids) - len(set(ids)),
        "duplicate_chunk_count": duplicate_texts,
        "required_provenance_gaps": len(required_gaps),
        "invalid_license_count": len(invalid_licenses),
        "temporal_admission_failures": sum(
            item.get("admission_status") != "admitted"
            or not item.get("version")
            or not item.get("commit_tag")
            for item in metadata
        ),
        "prohibited_benchmark_material_count": len(forbidden_material),
        "source_family_count": len({item.get("source") for item in metadata}),
        "source_version_count": len({(item.get("source"), item.get("version")) for item in metadata}),
        "coverage_categories": dict(sorted(Counter(item.get("knowledge_category") for item in metadata).items())),
        "corpus_sha256": corpus_hash,
        "ready_for_annotation": bool(rows)
        and not required_gaps
        and not invalid_licenses
        and not duplicate_texts
        and not forbidden_material,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
