"""Retrieval metrics and latency summaries for controlled experiments."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from statistics import mean

from .models import RetrievalResult


def evaluate_ranking(
    ranking: Sequence[RetrievalResult],
    relevance: Mapping[str, float],
) -> dict[str, float | None]:
    ranked_ids = [result.document_id for result in ranking]
    relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    rank = next((index + 1 for index, doc_id in enumerate(ranked_ids) if doc_id in relevant), None)
    result: dict[str, float | None] = {
        "recall_at_1": float(bool(rank and rank <= 1)),
        "recall_at_3": float(bool(rank and rank <= 3)),
        "recall_at_5": float(bool(rank and rank <= 5)),
        "recall_at_20": float(bool(rank and rank <= 20)),
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg_at_5": None,
    }
    if any(grade > 1 for grade in relevance.values()):
        ideal = sorted((grade for grade in relevance.values() if grade > 0), reverse=True)[:5]
        dcg = sum(
            relevance.get(doc_id, 0.0) / math.log2(index + 2)
            for index, doc_id in enumerate(ranked_ids[:5])
        )
        idcg = sum(grade / math.log2(index + 2) for index, grade in enumerate(ideal))
        result["ndcg_at_5"] = dcg / idcg if idcg else 0.0
    return result


def percentile(values: Sequence[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile_rank / 100 * len(ordered)) - 1))
    return ordered[index]


def summarise_metrics(
    per_query: Sequence[Mapping[str, float | None]],
    latencies_ms: Sequence[float],
    context_sizes: Sequence[int],
) -> dict[str, float | int | None]:
    fields = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_20", "mrr", "ndcg_at_5")
    summary: dict[str, float | int | None] = {
        "query_count": len(per_query),
        **{
            field: (
                mean(float(item[field]) for item in per_query if item[field] is not None)
                if any(item[field] is not None for item in per_query)
                else None
            )
            for field in fields
        },
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "mean_context_size": mean(context_sizes) if context_sizes else 0.0,
    }
    return summary


def paired_bootstrap_mean_ci(
    first: Sequence[float],
    second: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return mean(second-first) and a paired percentile bootstrap interval."""

    if len(first) != len(second) or not first:
        raise ValueError("Paired samples must be non-empty and have equal length.")
    if samples <= 0 or not 0 < confidence < 1:
        raise ValueError("samples must be positive and confidence must be in (0, 1).")
    differences = [right - left for left, right in zip(first, second)]
    rng = random.Random(seed)
    bootstraps = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) * 50.0
    return (
        mean(differences),
        percentile(bootstraps, alpha),
        percentile(bootstraps, 100.0 - alpha),
    )
