"""Standalone smoke experiment for the local retrieval foundation."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from digital_detective.rag import DeterministicRetriever, build_query, load_corpus


def main() -> None:
    corpus_path = Path(__file__).parents[1] / "data" / "rag" / "operational_knowledge.json"
    retriever = DeterministicRetriever(load_corpus(corpus_path))
    cases = (
        ({"candidate_service": "checkoutservice", "observed_signals": ("CPU utilization",), "modalities": ("metrics",)}, "synthetic-runbook-cpu-saturation"),
        ({"candidate_service": "cartservice", "observed_signals": ("memory garbage collection",), "modalities": ("metrics",)}, "synthetic-runbook-memory-pressure"),
        ({"candidate_service": "shippingservice", "observed_signals": ("network latency retries",), "modalities": ("traces", "logs")}, "synthetic-runbook-network-latency"),
    )
    top_k = 2
    started = perf_counter()
    retrieved_expected = 0
    for context, expected_id in cases:
        results = retriever.retrieve(build_query(context), k=top_k)
        if any(result.document_id == expected_id for result in results):
            retrieved_expected += 1
    elapsed_ms = (perf_counter() - started) * 1000
    print(
        json.dumps(
            {
                "query_count": len(cases),
                "top_k": top_k,
                "retrieval_latency_ms_total": round(elapsed_ms, 3),
                "retrieval_latency_ms_mean": round(elapsed_ms / len(cases), 3),
                "expected_document_retrieved": retrieved_expected,
                "expected_document_retrieval_rate": retrieved_expected / len(cases),
                "note": "Tiny synthetic/local smoke test; not a scientific benchmark.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
