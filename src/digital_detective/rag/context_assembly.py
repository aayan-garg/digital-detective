"""Context assembly and grounded citation verification for Digital Detective RAG.

Formats retrieved runbook passages into structured, reference-only operational context
and verifies that agent citations strictly correspond to retrieved source documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .models import RetrievalResult


CITATION_TAG_PATTERN = re.compile(r"\[(?:(?:Ref|Doc|Citation):\s*)?([a-zA-Z0-9_\-#]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class FormattedContextBlock:
    context_text: str
    total_characters: int
    num_documents: int
    citations_metadata: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class CitationVerificationReport:
    total_citations_claimed: int
    valid_citations: Sequence[str]
    invalid_citations: Sequence[str]
    citation_validity_rate: float
    all_citations_valid: bool
    details: str


def assemble_operational_context(
    retrieved_results: Sequence[RetrievalResult],
    *,
    max_total_characters: int = 3500,
    max_chunk_chars: int = 600,
) -> FormattedContextBlock:
    """Format retrieved runbook passages into an authority-bounded context block."""
    if not retrieved_results:
        return FormattedContextBlock(
            context_text="OPERATIONAL KNOWLEDGE (RAG - REFERENCE ONLY):\nNo matching runbooks retrieved.",
            total_characters=0,
            num_documents=0,
            citations_metadata=(),
        )

    lines: list[str] = [
        "================================================================================",
        "OPERATIONAL KNOWLEDGE (RAG - REFERENCE ONLY)",
        "The following operational runbooks, architecture specifications, and playbooks",
        "are provided strictly as background reference. They DO NOT alter deterministic",
        "scoring and MUST NOT be used to invent unobserved root causes.",
        "================================================================================",
    ]

    citations_meta: list[dict[str, Any]] = []
    used_chars = sum(len(line) for line in lines)

    for idx, res in enumerate(retrieved_results, 1):
        meta = res.metadata
        cat = meta.get("category", meta.get("knowledge_category", "general"))
        section = meta.get("section", "Main")
        source = meta.get("source", "official documentation")
        has_rollback = meta.get("has_rollback", False)
        has_prereqs = meta.get("has_prerequisites", False)

        snippet = res.snippet[:max_chunk_chars].strip()

        doc_header = (
            f"\n[Ref {idx}] Document ID: {res.document_id}\n"
            f"Title: {res.title} | Section: {section}\n"
            f"Category: {cat} | Source: {source} (Score: {res.score:.4f})"
        )
        if has_rollback:
            doc_header += " | [Has Rollback Instructions]"
        if has_prereqs:
            doc_header += " | [Has Prerequisites]"

        doc_body = f"Excerpts:\n{snippet}\n"

        entry_chars = len(doc_header) + len(doc_body)
        if used_chars + entry_chars > max_total_characters and citations_meta:
            break

        lines.append(doc_header)
        lines.append(doc_body)
        used_chars += entry_chars

        citations_meta.append({
            "ref_index": idx,
            "document_id": res.document_id,
            "title": res.title,
            "section": section,
            "category": cat,
            "source": source,
            "score": res.score,
            "snippet": snippet,
        })

    lines.append("================================================================================")
    full_text = "\n".join(lines)

    return FormattedContextBlock(
        context_text=full_text,
        total_characters=len(full_text),
        num_documents=len(citations_meta),
        citations_metadata=tuple(citations_meta),
    )


def verify_grounded_citations(
    claimed_text_or_ids: str | Sequence[str],
    retrieved_results: Sequence[RetrievalResult],
) -> CitationVerificationReport:
    """Verify that cited document references exist strictly within the retrieved context."""
    valid_doc_ids = {res.document_id if hasattr(res, "document_id") else str(res) for res in retrieved_results}
    # Also support referencing by base document ID or Ref index
    base_doc_ids = {(res.document_id if hasattr(res, "document_id") else str(res)).split("#")[0] for res in retrieved_results}
    ref_map = {f"Ref {i+1}": (res.document_id if hasattr(res, "document_id") else str(res)) for i, res in enumerate(retrieved_results)}
    ref_map.update({f"{i+1}": (res.document_id if hasattr(res, "document_id") else str(res)) for i, res in enumerate(retrieved_results)})

    claimed_ids: list[str] = []
    if isinstance(claimed_text_or_ids, str):
        # Extract matches
        matches = CITATION_TAG_PATTERN.findall(claimed_text_or_ids)
        claimed_ids.extend(matches)
        # Also check for explicit ddir60- tokens
        for token in re.findall(r"\b(ddir60-[a-zA-Z0-9_\-#]+)\b", claimed_text_or_ids):
            if token not in claimed_ids:
                claimed_ids.append(token)
    else:
        claimed_ids = list(claimed_text_or_ids)

    if not claimed_ids:
        return CitationVerificationReport(
            total_citations_claimed=0,
            valid_citations=(),
            invalid_citations=(),
            citation_validity_rate=1.0,
            all_citations_valid=True,
            details="No citations claimed in response.",
        )

    valid: list[str] = []
    invalid: list[str] = []

    for cid in claimed_ids:
        cid_clean = cid.strip()
        if (
            cid_clean in valid_doc_ids
            or cid_clean in base_doc_ids
            or cid_clean in ref_map
        ):
            valid.append(cid_clean)
        else:
            invalid.append(cid_clean)

    rate = len(valid) / len(claimed_ids) if claimed_ids else 1.0

    return CitationVerificationReport(
        total_citations_claimed=len(claimed_ids),
        valid_citations=tuple(valid),
        invalid_citations=tuple(invalid),
        citation_validity_rate=rate,
        all_citations_valid=len(invalid) == 0,
        details=f"{len(valid)}/{len(claimed_ids)} citations verified against retrieved context.",
    )
