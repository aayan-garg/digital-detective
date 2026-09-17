"""Autonomous Digital Detective investigation engine.

Coordinates:
1. Closed candidate universe construction.
2. Initial hypothesis screening using frozen S_comb and E_elev evidence.
3. Query-budget-constrained autonomous investigation loop using a deterministic heuristic.
4. Continuous multi-modal evidence fusion.
5. Operational causal consistency audit.
6. Final root cause decision synthesis.
7. Safe remediation proposal, safety gating, and simulated execution.
8. Post-remediation recovery verification.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence

from eval.universe import normalize_service_name, resolve_candidate_universe
from eval.windows import resolve_incident_window

from .models import (
    InvestigationState,
    RootCauseDecision,
)
from .remediation import (
    execute_simulated_remediation,
    propose_remediation,
    simulate_remediation_transformation,
)
from .scoring import EvidenceFusion
from .tools import DEFAULT_TOOL_COSTS, DetectiveTools
from .validation import CausalConsistencyValidator
from .verification import SystemStateSnapshot, validate_intervention, verify_recovery


class InvestigationEngine:
    """Executes deterministic investigation workflows under a strict query budget."""

    def __init__(
        self,
        budget: int = 20,
        confidence_threshold: float = 0.80,
        tool_costs: Mapping[str, int] | None = None,
        fusion_weights: Mapping[str, float] | None = None,
        adaptive_query_selection: bool = True,
    ) -> None:
        self.budget = budget
        self.confidence_threshold = confidence_threshold
        self.tool_costs = dict(DEFAULT_TOOL_COSTS if tool_costs is None else tool_costs)
        self.fusion = EvidenceFusion(weights=fusion_weights)
        self.adaptive_query_selection = adaptive_query_selection

    def investigate(
        self,
        case: Any,
        window_mode: str = "oracle",
        candidate_universe: Sequence[str] | None = None,
        remediation_threshold: float = 0.80,
        allow_low_confidence_remediation: bool = False,
        adaptive_query_selection: bool | None = None,
    ) -> InvestigationState:
        """Run the complete autonomous investigation process for an incident case.

        Parameters
        ----------
        case:
            TelemetryCase containing incident telemetry. Never requires ground truth.
        window_mode:
            'oracle' or 'detected'. Falls back to 'detected' if ground truth is hidden.
        candidate_universe:
            Optional explicit candidate universe override.
        remediation_threshold:
            Minimum diagnosis confidence required to authorize automated remediation.
        allow_low_confidence_remediation:
            If True, explicitly permits remediation execution even when confidence is below threshold.

        Returns
        -------
        InvestigationState
            Complete inspectable investigation trajectory and results.
        """
        cid = str(case.metadata.case_id)
        system = str(case.metadata.system or "ob")

        # 1. Closed Candidate Universe
        universe = resolve_candidate_universe(system, case=case, custom_universe=candidate_universe)

        # 2. Telemetry Precomputations
        has_metrics = hasattr(case, "metrics") and case.metrics is not None
        has_traces = hasattr(case, "traces") and case.traces is not None
        has_logs = hasattr(case, "logs") and case.logs is not None

        det_res = None
        ep_evidence: dict[str, Any] = {}
        graph = None
        trace_lat = None

        if has_metrics:
            det_res = detect_metric_anomalies(case)
            deps = ()
            if has_traces:
                trace_deps = extract_trace_dependencies(case, service_aliases={"frontendservice": "frontend"})
                deps = [td.to_dependency() for td in trace_deps]
            graph = build_entity_graph(det_res.metric_names, dependencies=deps)
            ep_config = EpisodeConfig(persistence=3, consensus=2)
            ep_evidence = dict(aggregate_entity_episodes(det_res, graph, ep_config))

        if has_traces:
            trace_lat = extract_trace_latency_evidence(
                case,
                service_aliases={"frontendservice": "frontend"},
                expected_services=universe,
            )

        # 3. Incident Window (falls back to detected if ground_truth is absent)
        effective_window_mode = window_mode
        if effective_window_mode == "oracle":
            gt = getattr(case, "ground_truth", None)
            if gt is None or not getattr(gt, "values", None):
                effective_window_mode = "detected"
        window = resolve_incident_window(
            case,
            ep_evidence=ep_evidence,
            mode=effective_window_mode,
            timestamps=(det_res.timestamps if det_res else None),
        )

        # 4. Initialize Tools & State
        tools = DetectiveTools(
            case=case,
            candidate_universe=universe,
            graph=graph,
            anomaly_result=det_res,
            ep_evidence=ep_evidence,
            trace_latency=trace_lat,
            costs=self.tool_costs,
        )
        validator = CausalConsistencyValidator(graph=graph)

        state = InvestigationState(
            incident_id=cid,
            system=system,
            candidate_universe=universe,
            budget=self.budget,
            remaining_budget=self.budget,
            status="ACTIVE",
        )

        # 5. Baseline Evidence Seeding (S_comb & E_elev)
        s_comb_scores: dict[str, float] = {}
        if det_res is not None:
            s_comb_list = rank_with_s_comb(det_res, ep_evidence, graph=graph, candidate_universe=universe)
            s_comb_scores = {s.entity: s.score for s in s_comb_list}

        trace_scores: dict[str, float] = {}
        if trace_lat is not None:
            t_list = rank_with_trace_elevation(trace_lat, candidate_universe=universe)
            trace_scores = {s.entity: s.score for s in t_list}

        current_rankings = self.fusion.rank_hypotheses(
            candidate_universe=universe,
            evidence_items=state.evidence_collected,
            initial_metric_scores=s_comb_scores,
            initial_trace_scores=trace_scores,
        )
        state.current_rankings = current_rankings

        # 6. Autonomous Investigation Loop under Query Budget
        earliest_onsets: dict[str, int | None] = {}
        for s, ev in ep_evidence.items():
            valid_eps = [
                ep for ep in getattr(ev, "episodes", ())
                if window.onset_ts is None or getattr(ep, "start_timestamp", 0) >= window.onset_ts
            ]
            if valid_eps:
                earliest_onsets[s] = valid_eps[0].start_timestamp
            else:
                earliest_onsets[s] = None

        queried_pairs: set[tuple[str, str]] = set()

        tool_modality_map = {
            "get_metrics": "metrics",
            "get_service_health": "health",
            "get_recent_change": "recent_change",
            "get_traces": "traces",
            "get_neighbors": "topology",
            "get_logs": "logs",
        }

        effective_adaptive = (
            self.adaptive_query_selection
            if adaptive_query_selection is None
            else adaptive_query_selection
        )

        tool_to_modality_key = {
            "get_metrics": "metric",
            "get_service_health": "metric",
            "get_recent_change": "change",
            "get_traces": "trace",
            "get_neighbors": "topology",
            "get_logs": "log",
        }

        while state.remaining_budget > 0:
            # Deterministic query selection heuristic:
            # In adaptive mode: evaluate modality discrimination across current top-K candidates
            # and normalize by query cost.
            # In legacy mode: use fixed modality priority weights.
            queries_per_service: dict[str, int] = {s: 0 for s in universe}
            observed_modalities_by_service: dict[str, set[str]] = {s: set() for s in universe}
            for q in state.queries_executed:
                if q.status == "SUCCESS":
                    queries_per_service[q.service] = queries_per_service.get(q.service, 0) + 1

            for ev in state.evidence_collected:
                if ev.magnitude > 0:
                    observed_modalities_by_service[ev.service].add(ev.modality)

            # Adaptive modality discrimination across top-K candidates
            top_k = current_rankings[:min(3, len(current_rankings))]
            dispersion_by_modality: dict[str, float] = {}
            for m_key in ("metric", "trace", "change", "log", "topology"):
                vals = [hyp.component_scores.get(m_key, 0.0) for hyp in top_k]
                if len(vals) >= 2:
                    dispersion_by_modality[m_key] = max(vals) - min(vals)
                else:
                    dispersion_by_modality[m_key] = 0.0

            delta = 0.15
            discrimination_by_modality: dict[str, float] = {
                m_key: dispersion_by_modality[m_key] + delta
                for m_key in dispersion_by_modality
            }

            m_disp = dispersion_by_modality.get("metric", 0.0)
            t_disp = dispersion_by_modality.get("trace", 0.0)
            if m_disp >= t_disp:
                policy_note = "metric evidence is currently more discriminative than trace evidence."
            else:
                policy_note = "trace evidence is currently more discriminative than metric evidence."

            candidate_queries: list[tuple[float, str, str, int, str, str, float]] = []

            for hyp in current_rankings:
                s = hyp.service
                base_suspicion = hyp.score
                ev_count = queries_per_service.get(s, 0)
                unexplored_factor = 1.0 / (1.0 + ev_count)

                tools_to_evaluate = [
                    ("get_metrics", self.tool_costs.get("get_metrics", 1), 1.2),
                    ("get_service_health", self.tool_costs.get("get_service_health", 1), 1.1),
                    ("get_recent_change", self.tool_costs.get("get_recent_change", 2), 1.0),
                    ("get_traces", self.tool_costs.get("get_traces", 3), 0.9) if has_traces else None,
                    ("get_neighbors", self.tool_costs.get("get_neighbors", 1), 0.8),
                    ("get_logs", self.tool_costs.get("get_logs", 2), 0.7) if has_logs else None,
                ]

                for item in tools_to_evaluate:
                    if item is None:
                        continue
                    t_name, cost, t_weight = item
                    if (s, t_name) not in queried_pairs and state.remaining_budget >= cost:
                        t_modality = tool_modality_map.get(t_name, "")
                        is_missing = t_modality not in observed_modalities_by_service.get(s, set())
                        missing_factor = 1.6 if is_missing else 0.7

                        if effective_adaptive:
                            m_key = tool_to_modality_key.get(t_name, "metric")
                            disc = discrimination_by_modality.get(m_key, delta)
                            priority = (base_suspicion * unexplored_factor * disc * missing_factor) / cost
                            if is_missing:
                                rationale = (
                                    f"Candidate '{s}' (suspicion={base_suspicion:.2f}) lacks {t_modality} evidence; "
                                    f"querying {t_name} (discrimination={disc:.2f}, cost={cost}) to resolve uncertainty."
                                )
                            else:
                                rationale = (
                                    f"Candidate '{s}' (suspicion={base_suspicion:.2f}) deepening {t_modality} evidence "
                                    f"(discrimination={disc:.2f}, cost={cost})."
                                )
                        else:
                            disc = 0.0
                            priority = base_suspicion * unexplored_factor * t_weight * missing_factor
                            if is_missing:
                                rationale = (
                                    f"Candidate '{s}' (suspicion={base_suspicion:.2f}) lacks {t_modality} evidence; "
                                    f"querying {t_name} to resolve uncertainty."
                                )
                            else:
                                rationale = (
                                    f"Candidate '{s}' (suspicion={base_suspicion:.2f}) deepening {t_modality} evidence."
                                )

                        candidate_queries.append((priority, s, t_name, cost, rationale, policy_note, disc))

            if not candidate_queries:
                # No affordable or remaining queries
                break

            # Pick query with highest deterministic priority; break ties by (-cost, service_name, tool_name)
            candidate_queries.sort(key=lambda x: (-x[0], -x[3], x[1], x[2]))
            _, service_target, tool_name, chosen_cost, chosen_rationale, chosen_policy, chosen_disc = candidate_queries[0]
            queried_pairs.add((service_target, tool_name))

            # Execute tool query (deducts budget & logs record with rationale)
            success, summary, new_evidence = tools.execute_query(
                state=state,
                tool_name=tool_name,
                service=service_target,
                window=window,
                rationale=chosen_rationale,
                leading_hypotheses=[h.service for h in top_k],
                query_policy=chosen_policy,
                discrimination=chosen_disc,
                cost=chosen_cost,
                remaining=(state.remaining_budget - chosen_cost),
                adaptive=effective_adaptive,
            )

            # Re-rank hypotheses with updated evidence and causal consistency
            consistency_scores = {
                s: validator.evaluate_consistency(
                    target_service=s,
                    evidence_items=state.evidence_collected,
                    earliest_onsets=earliest_onsets,
                ).consistency_score
                for s in universe
            }

            current_rankings = self.fusion.rank_hypotheses(
                candidate_universe=universe,
                evidence_items=state.evidence_collected,
                initial_metric_scores=s_comb_scores,
                initial_trace_scores=trace_scores,
                consistency_scores=consistency_scores,
            )
            state.current_rankings = current_rankings

            # Early stopping check:
            # Requires top hypothesis to have high confidence, sufficient separation, and pass causal consistency
            top_hyp = current_rankings[0]
            runner_up_score = current_rankings[1].score if len(current_rankings) > 1 else 0.0
            margin = (top_hyp.score - runner_up_score) / max(top_hyp.score, 1e-6)

            cc_check = validator.evaluate_consistency(
                target_service=top_hyp.service,
                evidence_items=state.evidence_collected,
                earliest_onsets=earliest_onsets,
            )

            # At least 2 queries on top candidate and healthy margin required for early completion
            top_queries = queries_per_service.get(top_hyp.service, 0)
            if top_queries >= 2 and margin >= 0.15 and cc_check.passed and top_hyp.confidence >= self.confidence_threshold:
                break

        # 7. Final Root-Cause Decision
        top_cause = state.current_rankings[0].service
        cc_final = validator.evaluate_consistency(
            target_service=top_cause,
            evidence_items=state.evidence_collected,
            earliest_onsets=earliest_onsets,
        )

        supporting = tuple(
            ev for ev in state.evidence_collected
            if normalize_service_name(ev.service) == top_cause and ev.magnitude > 0
        )
        contradicting = tuple(
            ev for ev in state.evidence_collected
            if normalize_service_name(ev.service) != top_cause and ev.magnitude > state.current_rankings[0].score
        )

        decision = RootCauseDecision(
            root_cause_service=top_cause,
            ranked_candidates=state.current_rankings,
            confidence=state.current_rankings[0].confidence,
            budget_used=state.total_query_cost,
            budget_remaining=state.remaining_budget,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            causal_consistency=cc_final,
        )
        state.decision = decision

        # 8. Safe Remediation Proposal & Safety Gate Evaluation (Zero Ground Truth Access)
        remediation_action = propose_remediation(decision, evidence_items=state.evidence_collected)
        state.remediation_action = remediation_action

        remediation_res = execute_simulated_remediation(
            action=remediation_action,
            state=state,
            min_confidence=remediation_threshold,
            allow_low_confidence=allow_low_confidence_remediation,
        )
        state.remediation_result = remediation_res

        # 9. Post-Remediation Multi-Symptom Recovery Verification
        before_anom = {hyp.service: hyp.score for hyp in state.current_rankings}
        before_lat = {
            s: (trace_scores.get(s, 0.0) * 200.0)
            for s in universe
        }
        before_err = {
            s: sum(ev.magnitude for ev in state.evidence_collected if ev.service == s and ev.modality == "logs")
            for s in universe
        }
        before_health = {
            s: ("CRITICAL" if before_anom[s] > 0.6 else ("DEGRADED" if before_anom[s] > 0.2 else "HEALTHY"))
            for s in universe
        }
        before_snapshot = SystemStateSnapshot(
            anomaly_scores=before_anom,
            latencies_p90=before_lat,
            error_rates=before_err,
            health_statuses=before_health,
        )

        after_snapshot = simulate_remediation_transformation(
            before_state=before_snapshot,
            action=remediation_action,
            remediation_result=remediation_res,
            graph=graph,
        )

        verification = verify_recovery(before_snapshot, after_snapshot, target_service=top_cause)
        state.verification_result = verification

        val_res = validate_intervention(
            verification=verification,
            target_service=top_cause,
            action=(remediation_action.action_type if remediation_action else "none"),
            pre_intervention_consistency=(cc_final.consistency_score if cc_final else 0.0),
            before_state=before_snapshot,
        )
        state.intervention_validation = val_res

        return state
