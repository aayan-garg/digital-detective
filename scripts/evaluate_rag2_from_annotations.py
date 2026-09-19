"""Evaluate RAG-2 retrieval using LLM-assisted preliminary relevance annotations.

Procedure
---------
1. Load pool_provenance.tsv to build:
   - q-NNN -> ddir60-... query ID mapping
   - annotation_document_id -> corpus document_id mapping
2. Load annotation JSONL; map annotation grades to corpus doc IDs.
3. Load retrieval outputs (keyed by ddir60-... query IDs).
4. For each query, evaluate the ranked list against the annotation-derived
   relevance judgments using digital_detective.rag.metrics.
5. Aggregate Recall@5 and MRR.

Caveats
-------
* Labels are preliminary LLM-assisted annotations, NOT human/expert ground truth.
* The benchmark definition (corpus/pool/splits) is NOT modified.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digital_detective.rag.metrics import evaluate_ranking
from digital_detective.rag.models import RetrievalResult

ROOT = Path("eval/rag2_benchmark")
ANNOTATION_FILE = ROOT / "rag2_llm_annotation_v1.jsonl"
RETRIEVAL_FILE = ROOT / "rag2_retrieval_outputs.jsonl"
PROVENANCE_FILE = ROOT / "rag2_pool_provenance.tsv"
LABEL_SOURCE = "LLM-assisted preliminary relevance annotation"


def load_provenance(path):
    """Return two mappings built from pool_provenance.tsv:
      qid_map: q-NNN -> ddir60-... (original_query_id)
      adid_map: annotation_document_id -> original_document_id (corpus doc_id)
    """
    qid_map = {}
    adid_map = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qid_map[row["query_id"]] = row["original_query_id"]
            adid_map[row["annotation_document_id"]] = row["original_document_id"]
    return qid_map, adid_map


def load_annotations(path):
    """Return {q-NNN: {annotation_document_id: grade}} for successful records."""
    by_query = defaultdict(dict)
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not rec.get("success"):
            continue
        grade = rec.get("grade")
        if grade is not None:
            by_query[rec["query_id"]][rec["annotation_document_id"]] = int(grade)
    return dict(by_query)


def load_retrieval_outputs(path):
    """Return {ddir60-...: [retrieved doc dict, ...]} sorted by rank."""
    by_query = defaultdict(list)
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        qid = rec.get("query_id")
        if qid:
            # retrieved is a list of {document_id, score, ...}
            for rank_i, r in enumerate(rec.get("retrieved", []), 1):
                by_query[qid].append({
                    "document_id": r["document_id"],
                    "score": float(r.get("score", 0.0)),
                    "rank": rank_i,
                })
    return dict(by_query)


def main():
    if not PROVENANCE_FILE.is_file():
        print(f"Provenance file not found: {PROVENANCE_FILE}", file=sys.stderr)
        sys.exit(1)

    qid_map, adid_map = load_provenance(PROVENANCE_FILE)
    annotations = load_annotations(ANNOTATION_FILE)
    retrieval = load_retrieval_outputs(RETRIEVAL_FILE)

    if not annotations:
        print(f"No successful annotations in {ANNOTATION_FILE}. Run annotation script first.", file=sys.stderr)
        sys.exit(1)
    if not retrieval:
        print(f"No retrieval outputs in {RETRIEVAL_FILE}.", file=sys.stderr)
        sys.exit(1)

    per_query_metrics = []
    evaluated_queries = 0
    skipped_no_retrieval = 0
    skipped_no_relevant = 0

    for pool_qid, adid_grades in sorted(annotations.items()):
        # Map q-NNN -> ddir60-... retrieval key
        orig_qid = qid_map.get(pool_qid)
        if orig_qid is None or orig_qid not in retrieval:
            skipped_no_retrieval += 1
            continue

        ranked_records = retrieval[orig_qid]
        if not ranked_records:
            skipped_no_retrieval += 1
            continue

        # Build relevance: corpus_doc_id -> grade
        relevance_for_eval: dict[str, float] = {}
        for adid, grade in adid_grades.items():
            corpus_did = adid_map.get(adid)
            if corpus_did:
                # Use max grade if the same corpus doc appears under multiple annotation IDs
                relevance_for_eval[corpus_did] = float(max(grade, relevance_for_eval.get(corpus_did, 0)))

        # Skip queries where no annotated candidate was relevant
        if not any(g > 0 for g in relevance_for_eval.values()):
            skipped_no_relevant += 1
            continue

        ranking = [
            RetrievalResult(
                document_id=r["document_id"],
                score=r["score"],
                title="",
                metadata={},
                snippet="",
            )
            for r in ranked_records
        ]

        metrics = evaluate_ranking(ranking, relevance_for_eval)
        metrics["query_id"] = pool_qid
        per_query_metrics.append(metrics)
        evaluated_queries += 1

    if not per_query_metrics:
        print("No queries could be evaluated.")
        print(f"  annotated={len(annotations)}, retrieval_keys={len(retrieval)}, "
              f"skipped_no_retrieval={skipped_no_retrieval}, skipped_no_relevant={skipped_no_relevant}")
        sys.exit(1)

    recall5 = [float(m["recall_at_5"]) for m in per_query_metrics if m["recall_at_5"] is not None]
    mrr = [float(m["mrr"]) for m in per_query_metrics if m["mrr"] is not None]
    ndcg5 = [float(m["ndcg_at_5"]) for m in per_query_metrics if m.get("ndcg_at_5") is not None]

    summary = {
        "label_source": LABEL_SOURCE,
        "annotation_note": (
            "Preliminary LLM-assisted annotations only. "
            "NOT human/expert ground truth. "
            "Do not treat as final benchmark results."
        ),
        "annotated_queries": len(annotations),
        "evaluated_queries": evaluated_queries,
        "skipped_no_retrieval": skipped_no_retrieval,
        "skipped_no_relevant_in_pool": skipped_no_relevant,
        "recall_at_5": round(mean(recall5), 4) if recall5 else None,
        "mrr": round(mean(mrr), 4) if mrr else None,
        "ndcg_at_5": round(mean(ndcg5), 4) if ndcg5 else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
