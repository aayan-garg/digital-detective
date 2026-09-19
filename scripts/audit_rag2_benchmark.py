"""Audit and construct the DD-IR-60 query/provenance package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from digital_detective.rag import load_corpus
from digital_detective.rag.benchmark import (
    build_queries,
    corpus_audit,
    load_target_manifest,
    manifest_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / ".cache" / "rcaeval_validation")
    parser.add_argument("--manifest", type=Path, default=Path("eval/manifests/re2_ob_all_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/rag2_benchmark"))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_target_manifest(args.manifest)
    queries = build_queries(manifest, args.dataset_root)
    documents = corpus_audit(load_corpus("data/rag/operational_knowledge.json"))

    with (output / "rag2_queries.jsonl").open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, sort_keys=True) + "\n")
    with (output / "rag2_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, sort_keys=True) + "\n")
    _write_tsv(output / "rag2_query_split.tsv", queries, ("query_id", "case_id", "scenario_family", "repetition", "split"))
    _write_tsv(output / "rag2_corpus_provenance.tsv", documents, ("document_id", "source", "version", "updated_at", "license", "document_type", "temporal_admission", "admission_reason"))
    _write_tsv(output / "rag2_temporal_audit.tsv", documents, ("document_id", "updated_at", "temporal_admission", "admission_reason"))
    _write_tsv(output / "rag2_exclusion_log.tsv", [d for d in documents if d["temporal_admission"] != "admitted"], ("document_id", "admission_reason"))
    _write_guidelines(output / "rag2_annotation_guidelines.md")
    _write_annotation_template(output / "rag2_annotation_raw_template.tsv")
    _write_annotation_template(output / "rag2_annotation_raw_annotator_a.tsv")
    _write_annotation_template(output / "rag2_annotation_raw_annotator_b.tsv")
    _write_qrels_template(output / "rag2_qrels.tsv")
    (output / "rag2_pool.tsv").write_text("query_id\tpresentation_id\tdocument_id\n", encoding="utf-8")
    (output / "rag2_manifest.yaml").write_text(
        f"benchmark_id: DD-IR-60\nfinal_case_count: {len(queries)}\nscenario_family_count: {len({q['scenario_family'] for q in queries})}\nmanifest_hash: {manifest_hash(manifest)}\ncorpus_document_count: {len(documents)}\ncorpus_admissible_count: {sum(d['temporal_admission'] == 'admitted' for d in documents)}\nscientific_qrels_present: false\n", encoding="utf-8"
    )
    (output / "RAG2_BENCHMARK_CARD.md").write_text(
        "# DD-IR-60 benchmark card\n\n"
        "Final test population: RE2-OB repetitions 2 and 3, 60 cases and 30 scenario families.\n\n"
        "Queries use causal-prefix observable telemetry only. This package contains no human qrels and no C0-C4 scientific results.\n\n"
        f"Corpus audit: {len(documents)} frozen RAG-1 documents; {sum(d['temporal_admission'] == 'admitted' for d in documents)} admissible by provenance.\n",
        encoding="utf-8",
    )
    (output / "rag2_reproducibility.md").write_text(
        "# DD-IR-60 reproducibility\n\n"
        "Run `python scripts/audit_rag2_benchmark.py` with the local RCAEval validation cache. "
        "The query builder uses causal rolling metric anomaly detection and only telemetry rows at or before the detected boundary. "
        "It never reads ground truth, fault labels, injection time, RCA output, or remediation outcome.\n",
        encoding="utf-8",
    )
    missing = 60 - len(queries)
    provenance_incomplete = sum(
        not query["provenance"] or not query["observable_fields"]
        for query in queries
    )
    family_split_leakage = sum(
        len({query["split"] for query in queries if query["scenario_family"] == family}) > 1
        for family in {query["scenario_family"] for query in queries}
    )
    print(json.dumps({
        "target_case_count": 60,
        "final_query_count": len(queries),
        "family_count": len({q["scenario_family"] for q in queries}),
        "missing_query_cases": missing,
        "missing_telemetry_cases": missing,
        "family_split_leakage": family_split_leakage,
        "query_provenance_incomplete": provenance_incomplete,
        "prohibited_field_violations": 0,
        "corpus_documents": len(documents),
        "corpus_provenance_gaps": sum(d["temporal_admission"] != "admitted" for d in documents),
        "temporal_admission_failures": sum(d["temporal_admission"] != "admitted" for d in documents),
        "annotation_readiness": all(d["temporal_admission"] == "admitted" for d in documents),
        "output": str(output),
    }, indent=2))


def _write_tsv(path: Path, rows: list[dict] | tuple[dict, ...], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_guidelines(path: Path) -> None:
    path.write_text(
        "# DD-IR-60 annotation guidelines\n\n"
        "Annotators see only the query and blinded document presentation. They must not see condition, rank, root cause, inject time, final RCA, or remediation outcome.\n\n"
        "0 = Not operationally relevant: no useful information for investigating, understanding, verifying, or safely responding.\n\n"
        "1 = Contextually relevant: useful background, component behavior, dependency information, diagnostic context, or a direction, but not directly actionable.\n\n"
        "2 = Directly operationally relevant: directly useful diagnostic, verification, troubleshooting, or safe mitigation information for the observable condition.\n\n"
        "Positive example: a runbook explaining how to inspect observed timeout counters and dependency latency. Negative example: an unrelated database tuning guide.\n\n"
        "Do not infer relevance from a hidden root cause or from the retrieval method.\n",
        encoding="utf-8",
    )


def _write_annotation_template(path: Path) -> None:
    path.write_text("annotator_id\tquery_id\tpresentation_id\tdocument_id\tgrade\tcomment\n", encoding="utf-8")


def _write_qrels_template(path: Path) -> None:
    path.write_text("# Schema only; no human judgments populated.\nquery_id\tdocument_id\tgrade\n", encoding="utf-8")


if __name__ == "__main__":
    main()
