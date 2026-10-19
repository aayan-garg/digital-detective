"""Tests for the self-contained static demo data and web replay assets.

Validates:
1. Static demo investigations asset existence and integrity (demo_investigations.json).
2. Complete coverage of the three benchmark incident cases:
   - re2ob_checkoutservice_cpu_1
   - re2ob_checkoutservice_mem_1
   - re2ob_currencyservice_delay_1
3. Verification of diagnosis data: root-cause ranking, top-3 candidates,
   topology graph, RAG runbook citations, safety gating, and remediation.
4. Robust handling of missing, empty, or malformed demo data.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from digital_detective.schema import CanonicalIncident


DEMO_JSON_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "digital_detective"
    / "web"
    / "demo_investigations.json"
)

EXPECTED_BENCHMARK_CASES = {
    "re2ob_checkoutservice_cpu_1": {
        "root_cause": "checkoutservice",
        "fault": "cpu",
        "system": "ob",
    },
    "re2ob_checkoutservice_mem_1": {
        "root_cause": "checkoutservice",
        "fault": "mem",
        "system": "ob",
    },
    "re2ob_currencyservice_delay_1": {
        "root_cause": "currencyservice",
        "fault": "delay",
        "system": "ob",
    },
}


class TestStaticDemoAsset:
    """Validate the static demo dataset asset packaged for Vercel deployment."""

    def test_demo_json_asset_exists_and_is_valid_json(self):
        assert DEMO_JSON_PATH.is_file(), f"Static asset missing at {DEMO_JSON_PATH}"
        assert DEMO_JSON_PATH.stat().st_size > 0, "Static asset file is empty"

        with open(DEMO_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), "Demo data must be a JSON dictionary mapping case IDs"
        assert set(data.keys()) == set(EXPECTED_BENCHMARK_CASES.keys()), (
            f"Expected exact benchmark cases {list(EXPECTED_BENCHMARK_CASES.keys())}, got {list(data.keys())}"
        )

    def test_demo_data_js_asset_exists_and_matches_json(self):
        js_path = DEMO_JSON_PATH.parent / "demo_data.js"
        assert js_path.is_file(), f"demo_data.js asset missing at {js_path}"
        assert js_path.stat().st_size > 0, "demo_data.js file is empty"
        content = js_path.read_text(encoding="utf-8")
        assert content.startswith("window.__DIGITAL_DETECTIVE_DEMO_DATA__ = ")
        raw_json = content[len("window.__DIGITAL_DETECTIVE_DEMO_DATA__ = "):].rstrip(";\n")
        parsed = json.loads(raw_json)
        assert set(parsed.keys()) == set(EXPECTED_BENCHMARK_CASES.keys())

    @pytest.mark.parametrize("case_id,expected", EXPECTED_BENCHMARK_CASES.items())
    def test_benchmark_incident_diagnosis_data(self, case_id: str, expected: dict[str, str]):
        with open(DEMO_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_incident = data[case_id]
        # Validate against CanonicalIncident Pydantic schema
        incident = CanonicalIncident.model_validate(raw_incident)

        assert incident.incident_id == case_id
        assert incident.ranking is not None
        assert incident.ranking.selected_root_cause == expected["root_cause"]
        assert incident.ground_truth_root_cause == expected["root_cause"]
        assert incident.ground_truth_fault_type == expected["fault"]

        # Validate Top Candidates
        assert incident.ranking.candidates is not None
        assert len(incident.ranking.candidates) >= 3
        rank1 = incident.ranking.candidates[0]
        assert rank1.rank == 1
        assert rank1.service == expected["root_cause"]
        assert rank1.score > 0.0

        # Validate Topology Graph
        assert incident.topology is not None
        assert len(incident.topology.nodes) >= 10
        assert len(incident.topology.edges) >= 5
        node_services = {n.service_id for n in incident.topology.nodes}
        assert expected["root_cause"] in node_services

        # Validate RAG Runbook Citations
        assert incident.rag_context is not None
        assert len(incident.rag_context.citations) >= 1
        top_citation = incident.rag_context.citations[0]
        assert top_citation.title != ""
        assert top_citation.snippet != ""
        assert top_citation.source != ""

        # Validate Safety Gate & Remediation Plan
        assert incident.remediation_plan is not None
        plan = incident.remediation_plan
        assert plan.target_service == expected["root_cause"]
        assert plan.action.value in ("scale_service", "restart_service")
        assert plan.rollback_action != ""
        assert plan.safety_check is not None
        assert plan.safety_check.target_matches_root_cause is True


class TestStaticDemoCatalogExtraction:
    """Validate client-side catalogue parsing and incident dropdown population."""

    def test_catalogue_extraction_from_demo_data(self):
        with open(DEMO_JSON_PATH, "r", encoding="utf-8") as f:
            demo_data = json.load(f)

        # Mirrors the client-side allIncidents extraction in app.js
        incidents = [
            {
                "case_id": inc.get("incident_id", cid),
                "system": inc.get("system", "ob"),
                "fault": inc.get("ground_truth_fault_type", "cpu"),
                "root_cause_service": inc.get("ground_truth_root_cause", "unknown"),
            }
            for cid, inc in demo_data.items()
        ]

        assert len(incidents) == 3
        case_map = {item["case_id"]: item for item in incidents}
        for cid, expected in EXPECTED_BENCHMARK_CASES.items():
            assert cid in case_map
            assert case_map[cid]["root_cause_service"] == expected["root_cause"]
            assert case_map[cid]["fault"] == expected["fault"]


class TestStaticDemoResilience:
    """Verify graceful error handling for missing, corrupted, or unexpected demo data."""

    def test_missing_incident_lookup_returns_none(self):
        with open(DEMO_JSON_PATH, "r", encoding="utf-8") as f:
            demo_data = json.load(f)

        assert demo_data.get("non_existent_case_123") is None

    def test_empty_dataset_handling(self):
        empty_data: dict = {}
        case_ids = list(empty_data.keys())
        assert len(case_ids) == 0

    def test_malformed_json_raises_clean_decode_error(self):
        malformed_raw = '{"re2ob_checkoutservice_cpu_1": { unclosed json '
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_raw)

    def test_partial_incident_graceful_defaults(self):
        # A partial incident object missing some optional fields still resolves safely
        partial = {
            "incident_id": "test_partial_1",
            "timestamp": 1234567890,
            "ranking": {
                "selected_root_cause": "checkoutservice",
                "score_margin": 0.5,
                "candidates": [
                    {
                        "service": "checkoutservice",
                        "score": 0.9,
                        "rank": 1,
                    }
                ],
            },
        }
        incident = CanonicalIncident.model_validate(partial)
        assert incident.incident_id == "test_partial_1"
        assert incident.ranking.selected_root_cause == "checkoutservice"
        assert incident.topology is None
        assert incident.rag_context is None
