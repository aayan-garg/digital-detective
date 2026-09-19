from scripts.audit_rag2_annotation_pool import audit


def test_blinded_pool_is_ready_without_qrels():
    result = audit()
    assert result["queries_processed"] == 60
    assert result["retrieval_record_count"] == 300
    assert result["all_conditions_per_query"]
    assert result["pool_rows"] == result["unique_annotation_document_ids"]
    assert result["all_pool_documents_in_frozen_corpus"]
    assert result["exactly_five_background_per_query"]
    assert result["candidate_sets_match"]
    assert result["presentation_orders_differ"]
    assert result["annotator_leakage_terms_found"] == []
    assert result["qrels_schema_only"]
    assert result["ready_for_human_annotation"]
