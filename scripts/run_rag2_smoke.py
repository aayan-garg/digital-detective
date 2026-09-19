"""Run the controlled RAG-2 smoke experiment without model fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

from digital_detective.rag import (
    IncidentQueryContext,
    ModelUnavailableError,
    RAG2Experiment,
    RetrievalCondition,
    RetrievalQuery,
    build_query,
    load_corpus,
)


def main() -> None:
    root = Path(__file__).parents[1]
    corpus_path = root / "data" / "rag" / "operational_knowledge.json"
    documents = load_corpus(corpus_path)
    queries = (
        RetrievalQuery(
            "q-cpu",
            build_query(IncidentQueryContext("checkoutservice", ("CPU utilization",), ("metrics",))),
            {"synthetic-runbook-cpu-saturation": 1},
        ),
        RetrievalQuery(
            "q-memory",
            build_query(IncidentQueryContext("cartservice", ("memory garbage collection",), ("metrics",))),
            {"synthetic-runbook-memory-pressure": 1},
        ),
        RetrievalQuery(
            "q-network",
            build_query(IncidentQueryContext("shippingservice", ("network latency retries",), ("traces", "logs"))),
            {"synthetic-runbook-network-latency": 1},
        ),
    )
    experiment = RAG2Experiment(documents)
    all_conditions = tuple(RetrievalCondition)
    artifact = experiment.run(queries, all_conditions)
    blocked: list[dict[str, str]] = []
    successful: list[str] = []
    for condition in all_conditions:
        try:
            experiment.run(queries, (condition,))
            successful.append(condition.value)
        except ModelUnavailableError as exc:
            blocked.append({"condition": condition.value, "status": "blocked", "reason": str(exc)})
    artifact["smoke"] = True
    artifact["evaluation_labels"] = "synthetic local qrels; not a scientific benchmark"
    artifact["blocked_conditions"] = blocked
    artifact["successful_conditions"] = successful
    output_path = root / "eval" / "results" / "rag2_smoke_v1.json"
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(output_path),
        "query_count": len(queries),
        "successful_conditions": successful,
        "blocked_conditions": blocked,
        "note": "Tiny synthetic/local smoke test; no winner or scientific claim is declared.",
    }, indent=2))


if __name__ == "__main__":
    main()
