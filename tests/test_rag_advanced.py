"""Unit and integration tests for advanced RAG components."""

from __future__ import annotations

import pytest

from digital_detective.rag.models import KnowledgeDocument, RetrievalResult
from digital_detective.rag.ingestion import RunbookIngestionEngine
from digital_detective.rag.bm25 import BM25Retriever
from digital_detective.rag.hybrid import HybridRetriever, HybridRetrievalConfig
from digital_detective.rag.context_assembly import (
    assemble_operational_context,
    verify_grounded_citations,
)


def _sample_docs() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(
            document_id="doc_k8s_cpu",
            title="Kubernetes CPU Resource Management",
            text="""## Overview
When CPU limit is exceeded, containers are throttled.
### Prerequisites
- kubectl configured
### Remediation Steps
1. Scale deployment replicas using kubectl scale.
2. Adjust CPU requests and limits in the pod spec.
### Verification
Check prometheus container_cpu_usage_seconds_total.
### Rollback
Run kubectl rollout undo deployment.""",
            metadata={"source": "kubernetes/website", "knowledge_category": "cpu"},
        ),
        KnowledgeDocument(
            document_id="doc_k8s_mem",
            title="Kubernetes Memory OOMKilled Troubleshooting",
            text="""## Overview
When memory limit is reached, container is terminated with exit code 137.
### Remediation
Restart service and increase container memory limit.
### Rollback
Restore previous deployment manifest.""",
            metadata={"source": "kubernetes/website", "knowledge_category": "memory"},
        ),
        KnowledgeDocument(
            document_id="doc_grpc_timeout",
            title="gRPC Deadlines and Timeouts",
            text="""## Troubleshooting
gRPC DEADLINE_EXCEEDED indicates downstream service did not reply within timeout.
### Remediation
Inspect connection pool and downstream service latency.""",
            metadata={"source": "grpc/grpc", "knowledge_category": "networking"},
        ),
    ]


class TestRAGAdvanced:
    def test_structure_aware_chunking(self):
        engine = RunbookIngestionEngine(chunk_size_chars=400, overlap_chars=50)
        docs = _sample_docs()
        chunks = engine.process_documents(docs)

        assert len(chunks) >= len(docs)
        # Verify chunk preserves section headings and prerequisites/rollback
        cpu_chunks = [c for c in chunks if "doc_k8s_cpu" in c.document_id]
        assert len(cpu_chunks) > 0
        all_text = " ".join(c.text for c in cpu_chunks)
        assert "Prerequisites" in all_text or "Remediation" in all_text

    def test_bm25_lexical_retrieval(self):
        docs = _sample_docs()
        bm25 = BM25Retriever(docs)

        res = bm25.retrieve("CPU limit throttling kubectl scale", k=2)
        assert len(res) > 0
        assert res[0].document_id == "doc_k8s_cpu"

        res_oom = bm25.retrieve("OOMKilled exit code 137 memory", k=2)
        assert len(res_oom) > 0
        assert res_oom[0].document_id == "doc_k8s_mem"

    def test_hybrid_retriever_rrf(self):
        docs = _sample_docs()
        hybrid = HybridRetriever(docs, config=HybridRetrievalConfig(category_boost=1.5))

        res = hybrid.retrieve("CPU limit exceeded container throttling", k=2, target_category="cpu")
        assert len(res) > 0
        assert res[0].document_id == "doc_k8s_cpu"
        assert res[0].score > 0.0

    def test_context_assembly_and_grounding(self):
        docs = _sample_docs()
        bm25 = BM25Retriever(docs)
        res = bm25.retrieve("CPU limit", k=2)

        context_block = assemble_operational_context(res)
        assert "OPERATIONAL KNOWLEDGE" in context_block.context_text
        assert "REFERENCE ONLY" in context_block.context_text
        assert len(context_block.citations_metadata) == len(res)

    def test_citation_hallucination_detection(self):
        docs = _sample_docs()
        bm25 = BM25Retriever(docs)
        res = bm25.retrieve("CPU limit", k=2)

        # 1. Grounded response
        good_text = f"As explained in [{res[0].document_id}], we should adjust CPU requests."
        audit_good = verify_grounded_citations(good_text, res)
        assert audit_good.all_citations_valid is True
        assert audit_good.citation_validity_rate == 1.0

        # 2. Hallucinated response
        bad_text = "As stated in [fake_phantom_doc_123], restart the node immediately."
        audit_bad = verify_grounded_citations(bad_text, res)
        assert audit_bad.all_citations_valid is False
        assert "fake_phantom_doc_123" in audit_bad.invalid_citations

        # 3. Exists in corpus but NOT retrieved
        unretrieved_doc_id = "doc_grpc_timeout"
        assert unretrieved_doc_id not in [r.document_id for r in res]
        unretrieved_text = f"Per [{unretrieved_doc_id}], check gRPC deadlines."
        audit_unretrieved = verify_grounded_citations(unretrieved_text, res)
        assert audit_unretrieved.all_citations_valid is False
        assert unretrieved_doc_id in audit_unretrieved.invalid_citations

        # 4. Retrieved parent document with multiple chunks resolves correctly
        chunk_results = [
            RetrievalResult("doc_k8s_cpu#chunk_0", 0.9, "CPU Title", {}, "Prerequisites and commands"),
            RetrievalResult("doc_k8s_cpu#chunk_1", 0.8, "CPU Title", {}, "Verification and rollback"),
        ]
        # Base document citation
        base_claim = "See [doc_k8s_cpu] for details."
        audit_base = verify_grounded_citations(base_claim, chunk_results)
        assert audit_base.all_citations_valid is True

        # Specific chunk citation
        chunk_claim = "See [doc_k8s_cpu#chunk_0] for details."
        audit_chunk = verify_grounded_citations(chunk_claim, chunk_results)
        assert audit_chunk.all_citations_valid is True

        # 5. Answer with no supporting runbook / empty retrieval
        empty_audit = verify_grounded_citations("We should run kubectl scale.", [])
        assert empty_audit.total_citations_claimed == 0
        assert empty_audit.all_citations_valid is True

        # Claiming a citation when retrieval is empty
        empty_audit_claim = verify_grounded_citations("According to [doc_k8s_cpu], scale replicas.", [])
        assert empty_audit_claim.total_citations_claimed == 1
        assert empty_audit_claim.all_citations_valid is False

    def test_context_preserves_prerequisites_rollback_and_commands(self):
        docs = _sample_docs()
        cpu_doc = docs[0]
        res = [RetrievalResult(
            document_id=cpu_doc.document_id,
            score=0.95,
            title=cpu_doc.title,
            metadata={"source": "k8s", "knowledge_category": "cpu", "has_rollback": True, "has_prerequisites": True},
            snippet=cpu_doc.text,
        )]

        context_block = assemble_operational_context(res)
        text = context_block.context_text

        # Verify key operational components are preserved in formatted context
        assert "Prerequisites" in text
        assert "Remediation Steps" in text
        assert "Verification" in text
        assert "Rollback" in text
        assert "kubectl scale" in text
        assert "kubectl rollout undo" in text
        assert "[Has Rollback Instructions]" in text
        assert "[Has Prerequisites]" in text
