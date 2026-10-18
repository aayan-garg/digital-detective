"""Unit tests for margin-triggered uncertainty disambiguation and candidate restriction."""

from __future__ import annotations

import pytest

from digital_detective.agent.disambiguator import (
    CandidateEvidenceProfile,
    Disambiguator,
    DisambiguationDecision,
)


def _make_sample_profiles() -> dict[str, CandidateEvidenceProfile]:
    return {
        "service_a": CandidateEvidenceProfile(
            service_id="service_a",
            model_b_score=0.4501,
            s_comb_score=0.45,
            trace_elev_score=0.40,
            causal_residual_z=1.2,
            log_anomaly_score=0.1,
            anomalous_metrics=("service_a_cpu",),
        ),
        "service_b": CandidateEvidenceProfile(
            service_id="service_b",
            model_b_score=0.4500,  # Delta = 0.0001 < 0.05 -> High Uncertainty!
            s_comb_score=0.44,
            trace_elev_score=0.85, # Strong trace evidence!
            causal_residual_z=5.8, # Significant causal residual!
            log_anomaly_score=0.9, # Novel error bursts!
            anomalous_metrics=("service_b_latency-90",),
            novel_log_templates=("DB timeout on port 5432",),
        ),
        "service_c": CandidateEvidenceProfile(
            service_id="service_c",
            model_b_score=0.0999,
            s_comb_score=0.11,
            trace_elev_score=0.10,
        ),
    }


class TestDisambiguator:
    def test_margin_calculation(self):
        disambiguator = Disambiguator(uncertainty_threshold=0.05)
        profiles = _make_sample_profiles()
        ranked = [("service_a", 0.4501), ("service_b", 0.4500), ("service_c", 0.0999)]

        margin = disambiguator.compute_score_margin(ranked)
        assert round(margin, 5) == 0.0001
        assert disambiguator.is_uncertain(margin) is True

    def test_high_confidence_deterministic_path(self):
        disambiguator = Disambiguator(uncertainty_threshold=0.05)
        # Clear separation between Top-1 and Top-2 (margin = 0.40 > 0.05)
        ranked = [("service_a", 0.80), ("service_b", 0.40), ("service_c", 0.10)]
        profiles = _make_sample_profiles()

        decision = disambiguator.verify_or_disambiguate(
            ranked_tuples=ranked,
            evidence_profiles=profiles,
            agent_model=None,
        )

        assert decision.uncertainty_detected is False
        assert decision.disambiguator_invoked is False
        assert decision.selected_candidate == "service_a"
        assert decision.confidence >= 0.80

    def test_uncertainty_trigger_with_heuristic_evidence(self):
        disambiguator = Disambiguator(uncertainty_threshold=0.05)
        ranked = [("service_a", 0.4501), ("service_b", 0.4500), ("service_c", 0.0999)]
        profiles = _make_sample_profiles()

        decision = disambiguator.verify_or_disambiguate(
            ranked_tuples=ranked,
            evidence_profiles=profiles,
            agent_model=None,  # Fallback to deterministic multi-modal heuristic
        )

        assert decision.uncertainty_detected is True
        assert decision.disambiguator_invoked is True
        # service_b has much higher causal residual (5.8 vs 1.2) and trace elevation (0.85 vs 0.40)
        assert decision.selected_candidate == "service_b"

    def test_candidate_restriction_enforcement(self):
        disambiguator = Disambiguator(uncertainty_threshold=0.05)
        ranked = [("service_a", 0.4501), ("service_b", 0.4500), ("service_c", 0.0999)]
        profiles = _make_sample_profiles()

        # Mock LLM trying to hallucinate an outside service
        class HallucinatingAgent:
            def complete(self, prompt):
                return '{"selected_service": "unrelated_phantom_service", "reasoning": "I think it is phantom"}'

        decision = disambiguator.verify_or_disambiguate(
            ranked_tuples=ranked,
            evidence_profiles=profiles,
            agent_model=HallucinatingAgent(),
        )

        # Candidate restriction MUST reject hallucinated candidate and safely fall back
        assert decision.selected_candidate in ("service_a", "service_b", "service_c")
        assert decision.selected_candidate != "unrelated_phantom_service"
