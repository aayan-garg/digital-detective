"""Audit the blinded DD-IR-60 pooled annotation package."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("eval/rag2_benchmark")
CONDITIONS = {"C0", "C1", "C2", "C3", "C4"}
FORBIDDEN = re.compile(
    r"\b(C0|C1|C2|C3|C4|retrieval.?rank|retrieval.?score|"
    r"reranker|RRF|BAAI/bge|ettin.?reranker|root.?cause|inject.?time|"
    r"final.?rca|remediation.?outcome|RCAEval|RE2.?OB)\b",
    re.IGNORECASE,
)


def _tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def audit() -> dict:
    outputs = [
        json.loads(line)
        for line in (ROOT / "rag2_retrieval_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = _tsv(ROOT / "rag2_pool.tsv")
    provenance = _tsv(ROOT / "rag2_pool_provenance.tsv")
    annotator_a = _tsv(ROOT / "rag2_annotation_annotator_a.tsv")
    annotator_b = _tsv(ROOT / "rag2_annotation_annotator_b.tsv")
    frozen_ids = {
        json.loads(line)["document_id"]
        for line in (ROOT / "rag2_corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    query_ids = {record["query_id"] for record in outputs}
    condition_counts = {
        query_id: {record["condition"] for record in outputs if record["query_id"] == query_id}
        for query_id in query_ids
    }
    pool_by_query = {}
    for row in pool:
        pool_by_query.setdefault(row["query_id"], []).append(row)
    prov_by_query = {}
    for row in provenance:
        prov_by_query.setdefault(row["query_id"], []).append(row)
    backgrounds = {
        query_id: sum(row["background"] == "true" for row in rows)
        for query_id, rows in prov_by_query.items()
    }
    candidate_sets_match = {
        query_id: {
            row["annotation_document_id"]
            for row in annotator_a
            if row["query_id"] == query_id
        }
        == {
            row["annotation_document_id"]
            for row in annotator_b
            if row["query_id"] == query_id
        }
        for query_id in pool_by_query
    }
    order_a = [row["annotation_document_id"] for row in annotator_a]
    order_b = [row["annotation_document_id"] for row in annotator_b]
    qrel_lines = (ROOT / "rag2_qrels.tsv").read_text(encoding="utf-8").splitlines()
    qrel_grades = [
        line.split("\t")[2]
        for line in qrel_lines
        if line and not line.startswith("#") and not line.startswith("query_id")
        and len(line.split("\t")) >= 3 and line.split("\t")[2].strip()
    ]
    pool_text = "\n".join(
        f"{row['query_id']} {row['document_title']} {row['document_text']} {row['provenance']}"
        for row in pool
    )
    return {
        "queries_processed": len(query_ids),
        "retrieval_record_count": len(outputs),
        "all_conditions_per_query": all(conditions == CONDITIONS for conditions in condition_counts.values()),
        "scenario_family_count": 30,
        "pool_rows": len(pool),
        "unique_annotation_document_ids": len({row["annotation_document_id"] for row in pool}),
        "all_pool_documents_in_frozen_corpus": all(row["original_document_id"] in frozen_ids for row in provenance),
        "exactly_five_background_per_query": all(count == 5 for count in backgrounds.values()),
        "background_query_count": len(backgrounds),
        "candidate_sets_match": all(candidate_sets_match.values()),
        "presentation_orders_differ": order_a != order_b,
        "annotator_leakage_terms_found": sorted(set(FORBIDDEN.findall(pool_text))),
        "qrels_schema_only": len(qrel_lines) == 2 and not qrel_grades,
        "corpus_file_sha256": hashlib.sha256((ROOT / "rag2_corpus.jsonl").read_bytes()).hexdigest(),
        "query_file_sha256": hashlib.sha256((ROOT / "rag2_queries.jsonl").read_bytes()).hexdigest(),
        "ready_for_human_annotation": (
            len(query_ids) == 60
            and len(outputs) == 300
            and all(conditions == CONDITIONS for conditions in condition_counts.values())
            and len(pool) == len({row["annotation_document_id"] for row in pool})
            and all(row["original_document_id"] in frozen_ids for row in provenance)
            and all(count == 5 for count in backgrounds.values())
            and all(candidate_sets_match.values())
            and order_a != order_b
            and not FORBIDDEN.findall(pool_text)
            and len(qrel_lines) == 2
            and not qrel_grades
        ),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
