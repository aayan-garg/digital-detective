"""Construction of retrieval queries from observed incident context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import IncidentQueryContext


def build_query(context: IncidentQueryContext | Mapping[str, object]) -> str:
    """Build a lexical query from explicitly permitted observed fields.

    The mapping form is intentionally allowlisted: fields such as ground truth,
    injection timing, fault labels, and future observations are never read.
    """

    if isinstance(context, IncidentQueryContext):
        values: tuple[object, ...] = (
            context.candidate_service,
            context.observed_signals,
            context.modalities,
        )
    elif isinstance(context, Mapping):
        values = (
            context.get("candidate_service", ""),
            context.get("observed_signals", ()),
            context.get("modalities", ()),
        )
    else:
        raise TypeError("context must be IncidentQueryContext or a mapping.")

    terms: list[str] = []
    for value in values:
        _append_terms(terms, value)
    return " ".join(terms)


def _append_terms(terms: list[str], value: object) -> None:
    if isinstance(value, str):
        if value.strip():
            terms.append(value.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _append_terms(terms, item)
