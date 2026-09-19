"""Run fixed C0-C4 retrieval and create blinded DD-IR-60 annotation pools."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from datetime import date
from importlib import metadata
from pathlib import Path
from time import perf_counter

from digital_detective.rag.experiment import (
    DENSE_TOP_K,
    FINAL_CONTEXT_K,
    HYBRID_CANDIDATE_K,
    RRF_K,
    SPARSE_TOP_K,
    RAG2Experiment,
    RetrievalCondition,
    RetrievalQuery,
)
from digital_detective.rag.models import KnowledgeDocument

ROOT = Path(".")
BENCHMARK = ROOT / "eval" / "rag2_benchmark"
CORPUS_PATH = BENCHMARK / "rag2_corpus.jsonl"
QUERIES_PATH = BENCHMARK / "rag2_queries.jsonl"
SEED = 60
CORPUS_HASH = "551d72f9087c1495a9e401f5c45d281247968d45f8dd4d11f1048c9ecb4e19dc"
QUERY_HASH = "8B70F7102A80FD00436CE1654A2D8C82E5396C39CCF9387397C6AC8A6E753D26".lower()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs() -> tuple[tuple[KnowledgeDocument, ...], tuple[RetrievalQuery, ...]]:
    documents = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            documents.append(
                KnowledgeDocument(
                    document_id=item["document_id"],
                    title=item["title"],
                    text=item["text"],
                    metadata=item["metadata"],
                )
            )
    queries = tuple(
        RetrievalQuery(query_id=item["query_id"], query=item["query"])
        for item in (
            json.loads(line)
            for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    return tuple(documents), queries


def annotation_id(query_id: str, document_id: str) -> str:
    digest = hashlib.sha256(f"{query_id}:{document_id}".encode("utf-8")).hexdigest()[:16]
    return f"ad-{digest}"


def write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    actual_corpus_hash = sha256(CORPUS_PATH)
    actual_query_hash = sha256(QUERIES_PATH)
    if actual_corpus_hash != CORPUS_HASH or actual_query_hash != QUERY_HASH:
        raise SystemExit(
            "Frozen input hash mismatch: "
            f"corpus={actual_corpus_hash}, query={actual_query_hash}"
        )
    documents, queries = load_inputs()
    if len(documents) != 536 or len(queries) != 60:
        raise SystemExit(f"Unexpected frozen inputs: {len(documents)} documents, {len(queries)} queries")
    families = {item["scenario_family"] for item in (
        json.loads(line)
        for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )}
    if len(families) != 30:
        raise SystemExit(f"Expected 30 scenario families, found {len(families)}")

    experiment = RAG2Experiment(documents)
    started = perf_counter()
    artifact = experiment.run(queries)
    elapsed_ms = (perf_counter() - started) * 1000
    for record in artifact["records"]:
        record["benchmark_corpus_file_sha256"] = actual_corpus_hash
        record["benchmark_query_file_sha256"] = actual_query_hash
        record["corpus_version"] = "ddir60-operational-knowledge-v1"
        record["query_version"] = "DD-IR-60-final-queries-v1"
        record["python_version"] = sys.version.split()[0]
        record["retrieval_code"] = "src/digital_detective/rag/experiment.py"
        record["retrieval_code_revision"] = "working-tree"
    artifact["benchmark_inputs"] = {
        "corpus_path": str(CORPUS_PATH),
        "corpus_file_sha256": actual_corpus_hash,
        "corpus_version": "ddir60-operational-knowledge-v1",
        "query_path": str(QUERIES_PATH),
        "query_file_sha256": actual_query_hash,
        "query_version": "DD-IR-60-final-queries-v1",
        "query_count": len(queries),
        "scenario_family_count": len(families),
    }
    artifact["execution"] = {
        "python_version": sys.version.split()[0],
        "sentence_transformers_version": metadata.version("sentence-transformers"),
        "transformers_version": metadata.version("transformers"),
        "torch_version": metadata.version("torch"),
        "elapsed_ms": elapsed_ms,
        "retrieval_code_revision": "working-tree",
    }
    (BENCHMARK / "rag2_retrieval_outputs.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in artifact["records"]) + "\n",
        encoding="utf-8",
    )

    by_id = {document.document_id: document for document in documents}
    retrieved_by_query: dict[str, list[dict]] = {query.query_id: [] for query in queries}
    for record in artifact["records"]:
        retrieved_by_query[record["query_id"]].append(record)

    pool_rows: list[dict[str, str]] = []
    private_rows: list[dict[str, str]] = []
    blinded_query_ids = {
        query.query_id: f"q-{index:03d}"
        for index, query in enumerate(sorted(queries, key=lambda item: item.query_id), 1)
    }
    for query in queries:
        blinded_query_id = blinded_query_ids[query.query_id]
        query_records = retrieved_by_query[query.query_id]
        retrieved_ids = []
        condition_ranks: dict[str, list[tuple[str, int, float | None]]] = {}
        for record in query_records:
            condition = record["condition"]
            condition_ranks[condition] = []
            for rank, result in enumerate(record["retrieved"][:FINAL_CONTEXT_K], 1):
                document_id = result["document_id"]
                retrieved_ids.append(document_id)
                condition_ranks[condition].append((document_id, rank, result.get("score")))
        unique_retrieved = list(dict.fromkeys(retrieved_ids))
        background_candidates = [
            document.document_id
            for document in documents
            if document.document_id not in unique_retrieved
        ]
        random.Random(f"{SEED}:{query.query_id}").shuffle(background_candidates)
        background_ids = background_candidates[:5]
        pooled_ids = unique_retrieved + background_ids
        presentation_ids = [annotation_id(query.query_id, document_id) for document_id in pooled_ids]
        for document_id, presentation_id in zip(pooled_ids, presentation_ids):
            document = by_id[document_id]
            metadata_for_annotation = {
                "source": document.metadata["source"],
                "version": document.metadata["version"],
                "license": document.metadata["license"],
                "document_type": document.metadata["document_type"],
                "section": document.metadata["section"],
                "knowledge_category": document.metadata["knowledge_category"],
            }
            pool_rows.append({
                "query_id": blinded_query_id,
                "annotation_document_id": presentation_id,
                "document_title": document.title,
                "document_text": document.text,
                "provenance": json.dumps(metadata_for_annotation, sort_keys=True),
            })
            ranks = [
                f"{condition}:{rank}:{score}"
                for condition, values in condition_ranks.items()
                for candidate_id, rank, score in values
                if candidate_id == document_id
            ]
            private_rows.append({
                "query_id": blinded_query_id,
                "original_query_id": query.query_id,
                "annotation_document_id": presentation_id,
                "original_document_id": document_id,
                "source": document.metadata["source"],
                "version": document.metadata["version"],
                "conditions_and_ranks": ";".join(ranks),
                "background": "true" if document_id in background_ids else "false",
            })

    write_tsv(
        BENCHMARK / "rag2_pool.tsv",
        pool_rows,
        ("query_id", "annotation_document_id", "document_title", "document_text", "provenance"),
    )
    write_tsv(
        BENCHMARK / "rag2_pool_provenance.tsv",
        private_rows,
        ("query_id", "original_query_id", "annotation_document_id", "original_document_id", "source", "version", "conditions_and_ranks", "background"),
    )
    for annotator, seed in (("a", SEED + 1), ("b", SEED + 2)):
        rows = list(pool_rows)
        grouped: list[dict[str, str]] = []
        for query_id in sorted({row["query_id"] for row in rows}):
            candidates = [row for row in rows if row["query_id"] == query_id]
            random.Random(f"{seed}:{query_id}").shuffle(candidates)
            grouped.extend(candidates)
        for row in grouped:
            row["grade"] = ""
            row["annotator_note"] = ""
        write_tsv(
            BENCHMARK / f"rag2_annotation_annotator_{annotator}.tsv",
            grouped,
            ("query_id", "annotation_document_id", "document_title", "document_text", "provenance", "grade", "annotator_note"),
        )
    (BENCHMARK / "rag2_qrels.tsv").write_text(
        "# Schema only; human judgments are not populated.\nquery_id\tdocument_id\tgrade\n",
        encoding="utf-8",
    )
    (BENCHMARK / "rag2_pool_manifest.yaml").write_text(
        "benchmark_id: DD-IR-60\n"
        "pool_version: ddir60-annotation-pool-v1\n"
        f"corpus_version: ddir60-operational-knowledge-v1\ncorpus_sha256: {actual_corpus_hash}\n"
        "query_version: DD-IR-60-final-queries-v1\n"
        f"query_sha256: {actual_query_hash}\n"
        "conditions: [C0, C1, C2, C3, C4]\n"
        "models:\n"
        "  C0: rag1-tfidf\n"
        "  C1: bm25\n"
        "  C2: BAAI/bge-small-en-v1.5\n"
        "  C3: bm25+BAAI/bge-small-en-v1.5+rrf-60\n"
        "  C4: bm25+BAAI/bge-small-en-v1.5+rrf-60+cross-encoder/ettin-reranker-17m-v1\n"
        f"python_version: {sys.version.split()[0]}\n"
        f"sentence_transformers_version: {metadata.version('sentence-transformers')}\n"
        f"transformers_version: {metadata.version('transformers')}\n"
        f"torch_version: {metadata.version('torch')}\n"
        "retrieval_code: src/digital_detective/rag/experiment.py\nretrieval_code_revision: working-tree\n"
        "sparse_top_k: 20\ndense_top_k: 20\nrrf_k: 60\nhybrid_candidate_pool: 20\nfinal_top_k: 5\n"
        f"query_count: {len(queries)}\nscenario_family_count: {len(families)}\n"
        f"random_background_count_per_query: 5\nrandom_seed: {SEED}\n"
        f"pooled_candidate_rows: {len(pool_rows)}\n"
        "annotator_identity_blinded: true\nqrels_populated: false\nscientific_comparison_run: false\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "query_count": len(queries),
        "record_count": len(artifact["records"]),
        "pool_rows": len(pool_rows),
        "average_candidates_per_query": len(pool_rows) / len(queries),
        "background_per_query": 5,
        "elapsed_ms": elapsed_ms,
    }, indent=2))


if __name__ == "__main__":
    main()
