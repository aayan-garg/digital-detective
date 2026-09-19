import json
import re
from pathlib import Path

from scripts.audit_rag2_corpus import audit


def test_ddir60_corpus_has_stable_unique_provenance_backed_units():
    result = audit()
    assert 250 <= result["retrieval_unit_count"] <= 600
    assert result["retrieval_unit_count"] == result["unique_id_count"]
    assert result["duplicate_id_count"] == 0
    assert result["duplicate_chunk_count"] == 0
    assert result["required_provenance_gaps"] == 0
    assert result["invalid_license_count"] == 0
    assert result["temporal_admission_failures"] == 0
    assert result["prohibited_benchmark_material_count"] == 0


def test_corpus_metadata_matches_provenance_file():
    rows = [
        json.loads(line)
        for line in Path("eval/rag2_benchmark/rag2_corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    provenance = Path("eval/rag2_benchmark/rag2_corpus_provenance.tsv").read_text(encoding="utf-8").splitlines()
    assert len(provenance) == len(rows) + 1
    assert all(row["metadata"]["source_url"].startswith("https://raw.githubusercontent.com/") for row in rows)
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["metadata"]["retrieved_date"]) for row in rows)
