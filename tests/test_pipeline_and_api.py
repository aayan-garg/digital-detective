"""End-to-end integration tests for InvestigationPipeline and FastAPI service."""

from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from digital_detective.pipeline import InvestigationPipeline
from digital_detective.schema import CanonicalIncident, RemediationStatus
from digital_detective.api.main import app, resolve_allowed_origins, REQUIRED_ALLOWED_ORIGINS

client = TestClient(app)
SAMPLE_CASE = "re2ob_checkoutservice_cpu_1"


class TestPipelineE2E:
    def test_pipeline_real_run(self):
        pipeline = InvestigationPipeline()
        incident = pipeline.run(SAMPLE_CASE, model_name="mock")

        assert isinstance(incident, CanonicalIncident)
        assert incident.incident_id == SAMPLE_CASE
        assert incident.system == "ob"
        assert incident.changepoint.onset_timestamp is not None
        assert incident.ranking is not None
        assert incident.ranking.selected_root_cause == "checkoutservice"
        assert incident.rag_context is not None
        assert len(incident.rag_context.citations) >= 1
        assert incident.remediation_plan is not None
        assert incident.remediation_plan.safety_check is not None
        assert incident.remediation_plan.safety_check.target_matches_root_cause is True
        # Safety gate properly blocks auto-remediation when confidence < 80%
        assert incident.remediation_plan.status in (RemediationStatus.PENDING_APPROVAL, RemediationStatus.BLOCKED_BY_SAFETY)

        # Verify allow_low_confidence authorizes remediation
        incident_auth = pipeline.run(SAMPLE_CASE, model_name="mock", allow_low_confidence=True)
        assert incident_auth.remediation_plan.safety_check.safety_authorized is True

        # Test valid JSON schema serialization
        json_data = incident.model_dump()
        assert json_data["incident_id"] == SAMPLE_CASE
        assert json_data["ranking"]["selected_root_cause"] == "checkoutservice"


class TestFastAPIService:
    def test_deployment_healthz(self):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "digital-detective-api"
        assert "dataset_available" in data

    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["cached_cases_count"] >= 1
        assert data["rag_retriever_ready"] is True

    def test_incidents_list(self):

        resp = client.get("/api/incidents?system=ob&limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["system"] == "ob"

    def test_incident_detail(self):
        resp = client.get(f"/api/incidents/{SAMPLE_CASE}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["case_id"] == SAMPLE_CASE
        assert data["root_cause_service"] == "checkoutservice"

    def test_investigate_endpoint(self):
        resp = client.post(
            "/api/investigate",
            json={"case_id": SAMPLE_CASE, "model_name": "mock"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["incident_id"] == SAMPLE_CASE
        assert data["ranking"]["selected_root_cause"] == "checkoutservice"
        assert len(data["rag_context"]["citations"]) >= 1

    def test_precomputed_demo_investigations_fast(self):
        for cid, expected_rc in [
            ("re2ob_checkoutservice_cpu_1", "checkoutservice"),
            ("re2ob_checkoutservice_mem_1", "checkoutservice"),
            ("re2ob_currencyservice_delay_1", "currencyservice"),
        ]:
            resp = client.post(
                "/api/investigate",
                json={"case_id": cid, "model_name": "mock", "execute_remediation": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["incident_id"] == cid
            assert data["ranking"]["selected_root_cause"] == expected_rc
            assert data["remediation_plan"]["status"] == "executed"


    def test_remediation_approval_workflow(self):
        # 1. Approve
        resp_approve = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "scale_service",
                "target_service": "checkoutservice",
                "user_decision": "approve",
            },
        )
        assert resp_approve.status_code == 200
        data = resp_approve.json()
        assert data["status"] == "executed"
        assert data["execution_success"] is True

        # 2. Reject
        resp_reject = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "scale_service",
                "target_service": "checkoutservice",
                "user_decision": "reject",
            },
        )
        assert resp_reject.status_code == 200
        data_rej = resp_reject.json()
        assert data_rej["status"] == "rejected"
        assert data_rej["execution_success"] is False

    def test_rag_query_endpoint(self):
        resp = client.post(
            "/api/rag/query",
            json={"query": "CPU limit exceeded throttling kubectl scale", "k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["score"] > 0
        assert "document_id" in data[0]

    def test_rate_limiter_unit(self):
        from digital_detective.api.main import InMemoryRateLimiter
        limiter = InMemoryRateLimiter(requests_per_minute=2)
        assert limiter.check("1.2.3.4") is True
        assert limiter.check("1.2.3.4") is True
        assert limiter.check("1.2.3.4") is False
        assert limiter.check("5.6.7.8") is True

    def test_remediation_secret_auth(self, monkeypatch):
        monkeypatch.setenv("REMEDIATION_SECRET", "super-secret-token-123")
        # Missing header -> 401
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "restart_service",
                "target_service": "checkoutservice",
                "user_decision": "dry_run",
            },
        )
        assert resp.status_code == 401

        # Valid header -> 200
        resp_ok = client.post(
            "/api/remediation/action",
            headers={"X-Remediation-Secret": "super-secret-token-123"},
            json={
                "case_id": SAMPLE_CASE,
                "action": "restart_service",
                "target_service": "checkoutservice",
                "user_decision": "dry_run",
            },
        )
        assert resp_ok.status_code == 200

    def test_cors_preflight_allows_production_vercel_origin(self):
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "https://digital-detective-web-nine.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://digital-detective-web-nine.vercel.app"

    def test_cors_resolution_with_stale_environment_variable(self):
        stale_env = '"https://outdated-domain.vercel.app", http://localhost:3000, *'
        resolved = resolve_allowed_origins(stale_env)

        assert "https://outdated-domain.vercel.app" in resolved
        assert "https://digital-detective-web-nine.vercel.app" in resolved
        assert "http://localhost:3000" in resolved
        assert "*" not in resolved
        assert len(resolved) == len(set(resolved))

        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

        @test_app.get("/test")
        def _test():
            return {"ok": True}

        test_client = TestClient(test_app)
        preflight_resp = test_client.options(
            "/test",
            headers={
                "Origin": "https://digital-detective-web-nine.vercel.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert preflight_resp.status_code == 200
        assert (
            preflight_resp.headers.get("access-control-allow-origin")
            == "https://digital-detective-web-nine.vercel.app"
        )
