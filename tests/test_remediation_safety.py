"""Adversarial remediation safety audit tests for Digital Detective API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from digital_detective.api.main import app, _executed_remediations
from digital_detective.schema import RemediationStatus

client = TestClient(app)
SAMPLE_CASE = "re2ob_checkoutservice_cpu_1"


@pytest.fixture(autouse=True)
def reset_executed_remediations():
    """Reset the executed actions registry before each test to guarantee independence."""
    _executed_remediations.clear()
    yield
    _executed_remediations.clear()


class TestRemediationSafetyAdversarialAudit:
    def test_valid_approved_simulated_remediation(self):
        """Valid parameters with explicit approval must succeed as simulated execution."""
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "scale_service",
                "target_service": "checkoutservice",
                "user_decision": "approve",
                "notes": "Scaling checkoutservice replicas to 3 under cpu stress",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "executed"
        assert data["execution_success"] is True
        assert data["is_simulation"] is True
        assert "Real infrastructure operations were NOT executed" in data["message"]
        assert any("rollback_scale_service(checkoutservice)" in a for a in data["audit_log"])

    def test_valid_rejected_remediation(self):
        """Explicit rejection must mark action as rejected with no execution."""
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "restart_service",
                "target_service": "checkoutservice",
                "user_decision": "reject",
                "notes": "Restarting is deemed too risky during peak shopping hours",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert data["execution_success"] is False
        assert data["is_simulation"] is True
        assert "rejected by human operator" in data["message"]

    def test_valid_dry_run_remediation(self):
        """Dry-run must verify prerequisites and mark as approved without execution."""
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "scale_service",
                "target_service": "checkoutservice",
                "user_decision": "dry_run",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["execution_success"] is True
        assert data["is_simulation"] is True

    def test_reject_unsupported_action(self):
        """Server-side policy must block unsupported destructive actions even if approved."""
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "drop_database_and_wipe_pv",
                "target_service": "checkoutservice",
                "user_decision": "approve",
            },
        )
        assert resp.status_code == 400
        assert "Unsupported remediation action" in resp.json()["detail"]

    def test_reject_invalid_target_service_not_in_universe(self):
        """Target service must belong to the incident system's candidate universe."""
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "restart_service",
                "target_service": "malicious_external_host_99",
                "user_decision": "approve",
            },
        )
        assert resp.status_code == 400
        assert "Invalid target service" in resp.json()["detail"]
        assert "candidate universe" in resp.json()["detail"]

    def test_reject_empty_or_whitespace_parameters(self):
        """Missing or whitespace-only parameters must return HTTP 400."""
        # Empty target service
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "scale_service",
                "target_service": "   ",
                "user_decision": "approve",
            },
        )
        assert resp.status_code == 400
        assert "target_service" in resp.json()["detail"]

        # Empty action
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "",
                "target_service": "checkoutservice",
                "user_decision": "approve",
            },
        )
        assert resp.status_code == 400
        assert "action" in resp.json()["detail"]

    def test_reject_invalid_user_decision(self):
        """Decisions other than approve, reject, or dry_run must be rejected."""
        resp = client.post(
            "/api/remediation/action",
            json={
                "case_id": SAMPLE_CASE,
                "action": "scale_service",
                "target_service": "checkoutservice",
                "user_decision": "bypass_safety_gate_force_now",
            },
        )
        assert resp.status_code == 400
        assert "Decision must be 'approve', 'reject', or 'dry_run'" in resp.json()["detail"]

    def test_duplicate_remediation_prevention(self):
        """Duplicate repeated execution of the same action must return HTTP 409 Conflict."""
        payload = {
            "case_id": SAMPLE_CASE,
            "action": "scale_service",
            "target_service": "checkoutservice",
            "user_decision": "approve",
        }
        # First execution succeeds
        resp1 = client.post("/api/remediation/action", json=payload)
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "executed"

        # Repeated execution must be rejected as duplicate
        resp2 = client.post("/api/remediation/action", json=payload)
        assert resp2.status_code == 409
        assert "Duplicate remediation rejected" in resp2.json()["detail"]
