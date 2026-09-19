from pathlib import Path

from digital_detective.rag.benchmark import (
    PROHIBITED_FIELDS,
    build_pool,
    corpus_audit,
    load_target_manifest,
    validate_query_leakage,
)
from digital_detective.rag.models import KnowledgeDocument


def test_ddir60_manifest_is_repetitions_two_and_three():
    manifest = load_target_manifest(Path("eval/manifests/re2_ob_all_cases.json"))
    assert len(manifest["cases"]) == 60
    assert {case["repetition"] for case in manifest["cases"]} == {2, 3}
    assert len({case["root_cause_service"] + "|" + case["fault"] for case in manifest["cases"]}) == 30


def test_query_leakage_validator_rejects_prohibited_fields():
    for field in PROHIBITED_FIELDS:
        try:
            validate_query_leakage(f"observed text {field}")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{field} was not rejected")


def test_frozen_corpus_is_explicitly_non_admissible_without_provenance():
    document = KnowledgeDocument(
        document_id="local",
        title="Local",
        text="Synthetic local test knowledge.",
        metadata={"source": "synthetic local runbook"},
    )
    audited = corpus_audit((document,))
    assert audited[0]["temporal_admission"] == "excluded_unknown_provenance"
    assert audited[0]["document_id"] == "local"


def test_pool_deduplicates_and_randomizes_deterministically():
    queries = ({"query_id": "q1"},)
    corpus = tuple(
        {"document_id": f"d{i}", "title": str(i), "text": str(i)} for i in range(8)
    )
    records = (
        {"query_id": "q1", "retrieved": [{"document_id": "d1"}, {"document_id": "d1"}, {"document_id": "d2"}]},
    )
    first = build_pool(queries, records, corpus, background_count=2)
    second = build_pool(queries, records, corpus, background_count=2)
    assert first == second
    assert len(first) == 4
    assert len({row["document_id"] for row in first}) == 4
    assert {row["document_id"] for row in first} >= {"d1", "d2"}
