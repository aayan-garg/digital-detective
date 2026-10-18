"""Margin-triggered candidate verification and disambiguation.

When ranker confidence is high (score margin Delta = s_(1) - s_(2) >= tau),
the deterministic ranking path is preserved.
When uncertainty is detected (Delta < tau), a bounded investigator evaluates
strictly the Top-3 candidates against metric residuals, trace propagation,
onset timing, and logs, falling back safely to deterministic ranking if unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass
class CandidateEvidenceProfile:
    entity: str = ""
    ranker_score: float = 0.0
    s_comb_score: float = 0.0
    trace_elev_score: float = 0.0
    causal_residual_z: float = 0.0
    log_anomaly_score: float = 0.0
    onset_timing: float = 0.0
    propagation_consistency: float = 1.0
    service_id: str = ""
    model_b_score: float = 0.0
    anomalous_metrics: Sequence[str] = ()
    novel_log_templates: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.entity and self.service_id:
            self.entity = self.service_id
        elif not self.service_id and self.entity:
            self.service_id = self.entity
        if self.ranker_score == 0.0 and self.model_b_score != 0.0:
            self.ranker_score = self.model_b_score
        elif self.model_b_score == 0.0 and self.ranker_score != 0.0:
            self.model_b_score = self.ranker_score


@dataclass(frozen=True)
class DisambiguationDecision:
    original_top1: str
    original_top2: str
    score_margin: float
    margin_threshold: float
    uncertainty_triggered: bool
    selected_candidate: str
    was_reordered: bool
    justification: str
    confidence: float
    fallback_used: bool
    top_candidates: Sequence[str] = ()

    @property
    def uncertainty_detected(self) -> bool:
        return self.uncertainty_triggered

    @property
    def disambiguator_invoked(self) -> bool:
        return self.uncertainty_triggered


class Disambiguator:
    """Disambiguate uncertain leading root-cause candidates under strict restrictions."""

    def __init__(self, margin_threshold: float = 0.15, uncertainty_threshold: float | None = None) -> None:
        if uncertainty_threshold is not None:
            margin_threshold = uncertainty_threshold
        if margin_threshold < 0:
            raise ValueError("margin_threshold must be non-negative")
        self.margin_threshold = margin_threshold

    def compute_score_margin(self, ranked_entities: Sequence[tuple[str, float]]) -> float:
        if len(ranked_entities) > 1:
            return float(ranked_entities[0][1] - ranked_entities[1][1])
        return 1.0

    def is_uncertain(self, margin: float) -> bool:
        return margin < self.margin_threshold

    def verify_or_disambiguate(
        self,
        ranked_tuples: Sequence[tuple[str, float]],
        evidence_profiles: Mapping[str, CandidateEvidenceProfile],
        agent_model: Any | None = None,
        rag_context_block: str = "",
    ) -> DisambiguationDecision:
        return self.evaluate(
            ranked_entities=ranked_tuples,
            profiles=evidence_profiles,
            agent_model=agent_model,
            rag_context_block=rag_context_block,
        )

    def evaluate(
        self,
        ranked_entities: Sequence[tuple[str, float]],
        profiles: Mapping[str, CandidateEvidenceProfile],
        agent_model: Any | None = None,
        rag_context_block: str = "",
    ) -> DisambiguationDecision:
        """Evaluate candidate score margin and disambiguate if uncertain."""
        if not ranked_entities:
            raise ValueError("ranked_entities cannot be empty")

        top1 = ranked_entities[0][0]
        s1 = ranked_entities[0][1]

        if len(ranked_entities) > 1:
            top2 = ranked_entities[1][0]
            s2 = ranked_entities[1][1]
            margin = float(s1 - s2)
        else:
            top2 = top1
            s2 = s1
            margin = 1.0

        uncertainty_triggered = margin < self.margin_threshold
        top3_candidates = tuple(ent for ent, _ in ranked_entities[:3])

        if not uncertainty_triggered:
            return DisambiguationDecision(
                original_top1=top1,
                original_top2=top2,
                score_margin=margin,
                margin_threshold=self.margin_threshold,
                uncertainty_triggered=False,
                selected_candidate=top1,
                was_reordered=False,
                justification=f"Deterministic ranking preserved; confidence margin Delta={margin:.3f} >= {self.margin_threshold:.3f}.",
                confidence=float(s1),
                fallback_used=False,
                top_candidates=top3_candidates,
            )

        # Uncertainty triggered: restricted to Top-3 candidates
        allowed_candidates = set(top3_candidates)

        # If agent_model is provided, invoke constrained evaluation
        if agent_model is not None and (hasattr(agent_model, "chat") or hasattr(agent_model, "complete")):
            evidence_summary = []
            for cand in top3_candidates:
                prof = profiles.get(cand)
                if prof:
                    evidence_summary.append({
                        "service": cand,
                        "ranker_score": round(prof.ranker_score, 4),
                        "metric_s_comb": round(prof.s_comb_score, 4),
                        "trace_elevation": round(prof.trace_elev_score, 4),
                        "causal_residual_z": round(prof.causal_residual_z, 4),
                        "log_anomaly_score": round(prof.log_anomaly_score, 4),
                        "onset_timing": round(prof.onset_timing, 4),
                    })

            prompt = (
                f"UNCERTAINTY DISAMBIGUATION REQUIRED:\n"
                f"Leading candidates have low score separation: {top1} (score {s1:.3f}) vs {top2} (score {s2:.3f}) "
                f"with Delta={margin:.3f} < {self.margin_threshold:.3f}.\n\n"
                f"ALLOWED CANDIDATE UNIVERSE: {list(top3_candidates)}\n"
                f"You MUST select exactly one service from this allowed universe. Do NOT invent other services.\n\n"
                f"CANDIDATE TELEMETRY EVIDENCE:\n"
                f"{json.dumps(evidence_summary, indent=2)}\n\n"
                f"RAG OPERATIONAL CONTEXT (REFERENCE ONLY):\n"
                f"{rag_context_block if rag_context_block else 'None'}\n\n"
                f"Respond with strict JSON ONLY:\n"
                f'{{"selected_candidate": "<must be one of {list(top3_candidates)}>", "justification": "<concise reason>", "confidence": <float 0.0 to 1.0>}}'
            )

            try:
                raw_response = agent_model.chat(prompt) if hasattr(agent_model, "chat") else agent_model.complete(prompt)
                # Parse JSON response
                cleaned = raw_response.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0].strip()

                parsed = json.loads(cleaned)
                chosen = str(parsed.get("selected_candidate", "")).strip()

                if chosen in allowed_candidates:
                    was_reordered = (chosen != top1)
                    just = str(parsed.get("justification", "Investigator selected based on evidence profile"))
                    conf = float(parsed.get("confidence", s1))
                    return DisambiguationDecision(
                        original_top1=top1,
                        original_top2=top2,
                        score_margin=margin,
                        margin_threshold=self.margin_threshold,
                        uncertainty_triggered=True,
                        selected_candidate=chosen,
                        was_reordered=was_reordered,
                        justification=just,
                        confidence=conf,
                        fallback_used=False,
                        top_candidates=top3_candidates,
                    )
            except Exception as exc:
                # Fall through to deterministic fallback
                pass

        # Deterministic evidence-weighted fallback across Top-3:
        # Check if a secondary candidate has substantially higher causal residual or trace elevation
        best_cand = top1
        best_score = float(s1)
        fallback_justification = (
            f"Uncertainty margin Delta={margin:.3f} < {self.margin_threshold:.3f}. "
            f"Evaluated multi-modal evidence across Top-3: {list(top3_candidates)}."
        )

        for cand in top3_candidates:
            prof = profiles.get(cand)
            if prof:
                # Combine normalized features: ranker (0.4) + trace elevation (0.3) + causal residual (0.3)
                composite = (
                    0.40 * prof.ranker_score
                    + 0.30 * min(1.0, prof.trace_elev_score)
                    + 0.30 * min(1.0, prof.causal_residual_z / 5.0)
                )
                if composite > best_score * 1.05 and cand in allowed_candidates:
                    best_cand = cand
                    best_score = composite

        was_reordered = (best_cand != top1)
        if was_reordered:
            fallback_justification += f" Reordered to {best_cand} based on composite causal residual and trace elevation evidence."
        else:
            fallback_justification += f" Preserved {best_cand} as leading evidence candidate."

        return DisambiguationDecision(
            original_top1=top1,
            original_top2=top2,
            score_margin=margin,
            margin_threshold=self.margin_threshold,
            uncertainty_triggered=True,
            selected_candidate=best_cand,
            was_reordered=was_reordered,
            justification=fallback_justification,
            confidence=best_score,
            fallback_used=True,
            top_candidates=top3_candidates,
        )
