"""Generate a clearly labeled, stratified candidate query-document pool for human review.

Selects 12 representative queries (2 per fault type: cpu, delay, disk, loss, mem, socket)
and pairs each with the Top-3 hybrid retrieved documents from the frozen corpus.
Outputs an unannotated TSV template ready for expert human grading.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from digital_detective.rag.ingestion import RunbookIngestionEngine
from digital_detective.rag.hybrid import HybridRetriever
from digital_detective.rag.incident_query import build_incident_aware_query

BENCHMARK = ROOT / "eval" / "rag2_benchmark"
CORPUS_PATH = BENCHMARK / "rag2_corpus.jsonl"
QUERIES_PATH = BENCHMARK / "rag2_queries.jsonl"
OUTPUT_TSV = BENCHMARK / "human_annotation_stratified_template.tsv"


def generate_annotation_template():
    # 1. Ingest corpus
    engine = RunbookIngestionEngine()
    chunks = engine.ingest_corpus(CORPUS_PATH)
    docs = [c.to_knowledge_document() for c in chunks]
    doc_map = {d.document_id: d for d in docs}
    retriever = HybridRetriever(docs)

    # 2. Load queries and stratify by fault
    queries = []
    for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(json.loads(line))

    by_fault = {}
    for q in queries:
        fam = q.get("scenario_family", "")
        parts = fam.split("|")
        fault = parts[3] if len(parts) >= 4 else "unknown"
        by_fault.setdefault(fault, []).append(q)

    # Select 2 queries per fault type (repetition 2 and repetition 3)
    stratified_queries = []
    for fault, q_list in sorted(by_fault.items()):
        stratified_queries.extend(q_list[:2])

    print(f"Selected {len(stratified_queries)} stratified queries across {len(by_fault)} fault types.")

    # 3. Retrieve Top-3 candidates for each query
    rows = []
    for q in stratified_queries:
        qid = q["query_id"]
        fam = q["scenario_family"]
        parts = fam.split("|")
        system = parts[1] if len(parts) > 1 else "ob"
        target_svc = parts[2] if len(parts) > 2 else ""
        fault = parts[3] if len(parts) > 3 else ""

        # Build incident-aware query
        pkg = build_incident_aware_query(
            leading_candidates=[target_svc],
            anomalous_metrics=[f"{target_svc} {fault}"],
            system_name=system,
        )

        results = retriever.retrieve(
            pkg.primary_query,
            k=3,
            target_category=pkg.target_category,
            target_service=target_svc,
        )

        for rank, res in enumerate(results, 1):
            doc = doc_map.get(res.document_id)
            title = doc.title if doc else "Unknown"
            category = doc.metadata.get("category", "") if doc else ""
            rows.append({
                "query_id": qid,
                "scenario_family": fam,
                "candidate_rank": rank,
                "document_id": res.document_id,
                "document_title": title,
                "document_category": category,
                "hybrid_score": round(res.score, 4),
                "human_relevance_grade": "",  # Empty for human annotator (0=Irrelevant, 1=Partially Relevant, 2=Highly Relevant)
                "annotator_notes": "",
            })

    # 4. Write TSV template
    fieldnames = [
        "query_id", "scenario_family", "candidate_rank", "document_id",
        "document_title", "document_category", "hybrid_score",
        "human_relevance_grade", "annotator_notes",
    ]
    with open(OUTPUT_TSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidate pairs to: {OUTPUT_TSV}")


if __name__ == "__main__":
    generate_annotation_template()
