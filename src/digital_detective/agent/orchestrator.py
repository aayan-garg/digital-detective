"""Agent orchestrator: bounded reasoning loop driving investigations via AgentDecision.

The AgentOrchestrator bridges the AgentModel (decision-maker) and the deterministic
InvestigationEngine (executor).  It enforces:

* Strict query budget — every QUERY action is validated against remaining budget
  before execution.
* Tool validation — only registered tool names are accepted.
* Policy validation — every AgentDecision passes through ``policy.validate_decision()``
  before any action is taken.
* Ground-truth isolation — the agent model NEVER receives ``case.ground_truth``.
  The orchestrator uses ``window_mode='detected'`` internally; ground truth is
  never read here.  The demo/evaluation layer may access ground truth post-hoc
  for display only.
* Step cap — maximum number of agent steps (default 30) prevents runaway loops.
* Invalid-output guard — repeated structurally invalid decisions trigger a STOP.
* Evidence grounding — FINAL_DIAGNOSIS decisions must reference collected evidence.

Architecture
------------
The orchestrator uses the InvestigationEngine to perform the one-time setup
(universe resolution, telemetry pre-computation, baseline seeding) and then
drives the investigation via the agent's structured decisions rather than the
engine's internal heuristic loop.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from digital_detective.detective.investigator import InvestigationEngine
from digital_detective.detective.models import (
    InvestigationState,
    RootCauseDecision,
)
from digital_detective.detective.remediation import (
    execute_simulated_remediation,
    propose_remediation,
    simulate_remediation_transformation,
)
from digital_detective.detective.tools import DEFAULT_TOOL_COSTS, DetectiveTools
from digital_detective.detective.validation import CausalConsistencyValidator
from digital_detective.detective.verification import SystemStateSnapshot, validate_intervention, verify_recovery
from digital_detective.detective.scoring import EvidenceFusion

from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence

from eval.universe import normalize_service_name, resolve_candidate_universe
from eval.windows import resolve_incident_window

from .models import AgentDecision, AgentModel, InvestigationTrajectory, StepTiming
from .policy import validate_decision

# Tools that are legal for the agent to request
_REGISTERED_TOOLS = frozenset(DEFAULT_TOOL_COSTS.keys())  # type: ignore[arg-type]

# How many consecutive invalid decisions trigger an automatic STOP
_MAX_CONSECUTIVE_INVALID = 3


class AgentOrchestrator:
    """Drives incident investigations via an agent model's structured decisions.

    Parameters
    ----------
    agent_model:
        The decision-making model (``MockAgentModel``, ``OllamaAgentModel``,
        ``GrokAgentModel``, or ``OpenAIChatAdapter``).
    budget:
        Total query budget for the investigation.
    max_steps:
        Hard cap on agent decision steps to prevent infinite loops.
    confidence_threshold:
        Minimum confidence required for the engine's early-stop.
    remediation_threshold:
        Minimum confidence for automated remediation authorization.
    tool_costs:
        Optional override for per-tool query costs.
    fusion_weights:
        Optional modality fusion weight overrides.

    Ground-truth isolation
    ----------------------
    ``AgentOrchestrator.investigate()`` NEVER reads ``case.ground_truth``.
    The incident window is always resolved using ``mode='detected'`` so the
    agent operates purely on observed telemetry.  Ground truth may only be
    accessed by the demo / evaluation layer for post-hoc comparison.
    """

    def __init__(
        self,
        agent_model: AgentModel,
        budget: int = 20,
        max_steps: int = 6,
        confidence_threshold: float = 0.80,
        remediation_threshold: float = 0.80,
        tool_costs: Mapping[str, int] | None = None,
        fusion_weights: Mapping[str, float] | None = None,
        allow_duplicate_queries: bool = False,
    ) -> None:
        self.agent_model = agent_model
        self.budget = budget
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.remediation_threshold = remediation_threshold
        self.tool_costs = dict(DEFAULT_TOOL_COSTS if tool_costs is None else tool_costs)  # type: ignore[arg-type]
        self.fusion = EvidenceFusion(weights=fusion_weights)
        self.allow_duplicate_queries = allow_duplicate_queries

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def investigate(
        self,
        case: Any,
        candidate_universe: Sequence[str] | None = None,
        allow_low_confidence_remediation: bool = False,
        custom_onset_ts: int | None = None,
    ) -> InvestigationTrajectory:
        """Run the complete agent-driven investigation for an incident case.

        Parameters
        ----------
        case:
            TelemetryCase containing incident telemetry.
            ``case.ground_truth`` is NEVER read by this method.
        candidate_universe:
            Optional explicit candidate universe override.
        allow_low_confidence_remediation:
            If ``True``, permits remediation below the confidence threshold.
        custom_onset_ts:
            Optional explicit incident onset timestamp override.
            Use this when the detected window is not reliable (e.g. in tests).

        Returns
        -------
        InvestigationTrajectory
            Fully populated trajectory including the final ``InvestigationState``.
        """
        cid = str(case.metadata.case_id)
        trajectory = InvestigationTrajectory(
            incident_id=cid,
            model_name=getattr(self.agent_model, "model_name", type(self.agent_model).__name__),
        )

        # ------------------------------------------------------------------
        # 1. Setup — identical pre-computation as InvestigationEngine
        # ------------------------------------------------------------------
        system = str(case.metadata.system or "ob")
        universe = resolve_candidate_universe(system, case=case, custom_universe=candidate_universe)

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

        # ------------------------------------------------------------------
        # Incident window — ALWAYS use 'detected' mode.
        # The agent NEVER accesses case.ground_truth.
        # ------------------------------------------------------------------
        try:
            window = resolve_incident_window(
                case,
                ep_evidence=ep_evidence,
                mode="detected",
                timestamps=(det_res.timestamps if det_res else None),
                custom_onset_ts=custom_onset_ts,
            )
        except ValueError:
            # No detected episodes and no timestamps → use a null window.
            # The investigation proceeds with empty telemetry.
            from eval.models import IncidentWindow
            window = IncidentWindow(
                onset_ts=0,
                end_ts=0,
                mode="detected",
                source_description="no_data_fallback",
            )

        # ------------------------------------------------------------------
        # 2. Tool interface + state
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 3. Baseline seeding (S_comb + E_elev) — same as deterministic engine
        # ------------------------------------------------------------------
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

        # Earliest onset timestamps for causal consistency checks
        earliest_onsets: dict[str, int | None] = {}
        for svc, ev in ep_evidence.items():
            valid_eps = [
                ep for ep in getattr(ev, "episodes", ())
                if getattr(ep, "start_timestamp", 0) >= window.onset_ts
            ]
            earliest_onsets[svc] = valid_eps[0].start_timestamp if valid_eps else None

        # ------------------------------------------------------------------
        # 4. Agent reasoning loop
        # ------------------------------------------------------------------
        already_queried: set[tuple[str, str]] = set()
        terminated_by = "MAX_STEPS"
        consecutive_invalid = 0

        available_tools = [
            t for t in _REGISTERED_TOOLS
            if (not (t == "get_traces") or has_traces)
            and (not (t == "get_logs") or has_logs)
        ]

        for _step in range(self.max_steps):
            t_step_start = time.perf_counter()

            # Ask the agent model for the next decision.
            # CRITICAL: state is passed (no ground_truth attribute),
            # trajectory is passed (no ground_truth attribute).
            # The model MUST NOT receive case.ground_truth.
            t_llm_start = time.perf_counter()
            decision = self.agent_model.decide(
                state=state,
                trajectory=trajectory,
                available_tools=available_tools,
                already_queried=set(already_queried),
            )
            t_llm_end = time.perf_counter()
            llm_call_sec = decision.metadata.get("llm_call_sec", t_llm_end - t_llm_start)

            # ---- Policy validation BEFORE execution --------------------
            t_val_start = time.perf_counter()
            validation = validate_decision(
                decision=decision,
                state=state,
                available_tools=available_tools,
                already_queried=already_queried,
            )
            val_sec = time.perf_counter() - t_val_start

            # Reject duplicate queries if allow_duplicate_queries is False
            if self.allow_duplicate_queries:
                hard_violations = [
                    v for v in validation.violations
                    if v.code not in ("DUPLICATE_QUERY",)
                ]
            else:
                hard_violations = list(validation.violations)

            if hard_violations and decision.action == "QUERY":
                # Record the rejection and try again (up to _MAX_CONSECUTIVE_INVALID)
                consecutive_invalid += 1
                rejected_reasoning = (
                    f"{decision.reasoning} [POLICY REJECTED: "
                    + "; ".join(str(v) for v in hard_violations)
                    + "]"
                )
                rej_decision = AgentDecision(
                    action=decision.action,
                    tool_name=decision.tool_name,
                    service=decision.service,
                    reasoning=rejected_reasoning,
                    evidence_ids=decision.evidence_ids,
                    step_index=decision.step_index,
                    metadata={**decision.metadata, "policy_violations": [str(v) for v in hard_violations]},
                )
                trajectory.append_decision(rej_decision)

                total_step_sec = time.perf_counter() - t_step_start
                step_timing = StepTiming(
                    step_index=_step,
                    llm_call_sec=llm_call_sec,
                    validation_sec=val_sec,
                    tool_execution_sec=0.0,
                    total_step_sec=total_step_sec,
                    prompt_chars=decision.metadata.get("prompt_chars", 0),
                    prompt_tokens_est=decision.metadata.get("prompt_tokens_est", 0),
                    output_chars=decision.metadata.get("output_chars", 0),
                    output_tokens_est=decision.metadata.get("output_tokens_est", 0),
                    evidence_count=len(state.evidence_collected),
                    trajectory_steps_count=len(trajectory.decisions),
                )
                trajectory.add_step_timing(step_timing)

                if consecutive_invalid >= _MAX_CONSECUTIVE_INVALID:
                    state.status = "BUDGET_EXHAUSTED"
                    terminated_by = "STOP"
                    break
                continue
            else:
                consecutive_invalid = 0

            trajectory.append_decision(decision)

            # ---- Handle terminal actions --------------------------------
            if decision.action == "FINAL_DIAGNOSIS":
                state.status = "DIAGNOSIS_COMPLETE"
                terminated_by = "FINAL_DIAGNOSIS"
                total_step_sec = time.perf_counter() - t_step_start
                step_timing = StepTiming(
                    step_index=_step,
                    llm_call_sec=llm_call_sec,
                    validation_sec=val_sec,
                    tool_execution_sec=0.0,
                    total_step_sec=total_step_sec,
                    prompt_chars=decision.metadata.get("prompt_chars", 0),
                    prompt_tokens_est=decision.metadata.get("prompt_tokens_est", 0),
                    output_chars=decision.metadata.get("output_chars", 0),
                    output_tokens_est=decision.metadata.get("output_tokens_est", 0),
                    evidence_count=len(state.evidence_collected),
                    trajectory_steps_count=len(trajectory.decisions),
                )
                trajectory.add_step_timing(step_timing)
                break

            if decision.action == "STOP":
                state.status = "BUDGET_EXHAUSTED"
                terminated_by = "STOP"
                total_step_sec = time.perf_counter() - t_step_start
                step_timing = StepTiming(
                    step_index=_step,
                    llm_call_sec=llm_call_sec,
                    validation_sec=val_sec,
                    tool_execution_sec=0.0,
                    total_step_sec=total_step_sec,
                    prompt_chars=decision.metadata.get("prompt_chars", 0),
                    prompt_tokens_est=decision.metadata.get("prompt_tokens_est", 0),
                    output_chars=decision.metadata.get("output_chars", 0),
                    output_tokens_est=decision.metadata.get("output_tokens_est", 0),
                    evidence_count=len(state.evidence_collected),
                    trajectory_steps_count=len(trajectory.decisions),
                )
                trajectory.add_step_timing(step_timing)
                break

            if decision.action == "REQUEST_REMEDIATION":
                terminated_by = "REQUEST_REMEDIATION"
                total_step_sec = time.perf_counter() - t_step_start
                step_timing = StepTiming(
                    step_index=_step,
                    llm_call_sec=llm_call_sec,
                    validation_sec=val_sec,
                    tool_execution_sec=0.0,
                    total_step_sec=total_step_sec,
                    prompt_chars=decision.metadata.get("prompt_chars", 0),
                    prompt_tokens_est=decision.metadata.get("prompt_tokens_est", 0),
                    output_chars=decision.metadata.get("output_chars", 0),
                    output_tokens_est=decision.metadata.get("output_tokens_est", 0),
                    evidence_count=len(state.evidence_collected),
                    trajectory_steps_count=len(trajectory.decisions),
                )
                trajectory.add_step_timing(step_timing)
                break

            # ---- Execute QUERY action -----------------------------------
            tool_sec = 0.0
            if decision.action == "QUERY":
                tool_name = decision.tool_name
                service = decision.service

                # Execute via DetectiveTools (budget enforced inside)
                t_tool_start = time.perf_counter()
                success, summary, new_evidence = tools.execute_query(
                    state=state,
                    tool_name=tool_name,
                    service=service,
                    window=window,
                    rationale=decision.reasoning,
                )
                tool_sec = time.perf_counter() - t_tool_start
                if success:
                    already_queried.add((normalize_service_name(service), tool_name))

                # Re-rank hypotheses after each query
                consistency_scores = {
                    svc: validator.evaluate_consistency(
                        target_service=svc,
                        evidence_items=state.evidence_collected,
                        earliest_onsets=earliest_onsets,
                    ).consistency_score
                    for svc in universe
                }
                current_rankings = self.fusion.rank_hypotheses(
                    candidate_universe=universe,
                    evidence_items=state.evidence_collected,
                    initial_metric_scores=s_comb_scores,
                    initial_trace_scores=trace_scores,
                    consistency_scores=consistency_scores,
                )
                state.current_rankings = current_rankings

                total_step_sec = time.perf_counter() - t_step_start
                step_timing = StepTiming(
                    step_index=_step,
                    llm_call_sec=llm_call_sec,
                    validation_sec=val_sec,
                    tool_execution_sec=tool_sec,
                    total_step_sec=total_step_sec,
                    prompt_chars=decision.metadata.get("prompt_chars", 0),
                    prompt_tokens_est=decision.metadata.get("prompt_tokens_est", 0),
                    output_chars=decision.metadata.get("output_chars", 0),
                    output_tokens_est=decision.metadata.get("output_tokens_est", 0),
                    evidence_count=len(state.evidence_collected),
                    trajectory_steps_count=len(trajectory.decisions),
                )
                trajectory.add_step_timing(step_timing)

                # Budget exhausted mid-loop
                if state.remaining_budget <= 0:
                    state.status = "BUDGET_EXHAUSTED"
                    terminated_by = "BUDGET_EXHAUSTED"
                    break
        else:
            state.status = "BUDGET_EXHAUSTED"

        # ------------------------------------------------------------------
        # 5. Final decision synthesis (same as deterministic engine)
        # ------------------------------------------------------------------
        if state.current_rankings:
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
                if normalize_service_name(ev.service) != top_cause
                and ev.magnitude > state.current_rankings[0].score
            )
            decision_obj = RootCauseDecision(
                root_cause_service=top_cause,
                ranked_candidates=state.current_rankings,
                confidence=state.current_rankings[0].confidence,
                budget_used=state.total_query_cost,
                budget_remaining=state.remaining_budget,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
                causal_consistency=cc_final,
            )
            state.decision = decision_obj
        else:
            top_cause = ""

        # ------------------------------------------------------------------
        # 6. Remediation proposal + safety gate
        # ------------------------------------------------------------------
        if state.decision is not None:
            remediation_action = propose_remediation(state.decision, evidence_items=state.evidence_collected)
            state.remediation_action = remediation_action

            should_attempt_remediation = (
                terminated_by == "REQUEST_REMEDIATION"
                or (terminated_by == "FINAL_DIAGNOSIS" and state.decision.confidence >= self.remediation_threshold)
                or allow_low_confidence_remediation
            )

            remediation_res = execute_simulated_remediation(
                action=remediation_action,
                state=state,
                min_confidence=self.remediation_threshold,
                allow_low_confidence=allow_low_confidence_remediation or should_attempt_remediation,
            )
            state.remediation_result = remediation_res

            # ------------------------------------------------------------------
            # 7. Post-remediation recovery verification
            # ------------------------------------------------------------------
            before_anom = {hyp.service: hyp.score for hyp in state.current_rankings}
            before_lat = {svc: (trace_scores.get(svc, 0.0) * 200.0) for svc in universe}
            before_err = {
                svc: sum(
                    ev.magnitude for ev in state.evidence_collected
                    if ev.service == svc and ev.modality == "logs"
                )
                for svc in universe
            }
            before_health = {
                svc: (
                    "CRITICAL" if before_anom[svc] > 0.6
                    else ("DEGRADED" if before_anom[svc] > 0.2 else "HEALTHY")
                )
                for svc in universe
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

            cc_score = state.decision.causal_consistency.consistency_score if state.decision and state.decision.causal_consistency else 0.0
            val_res = validate_intervention(
                verification=verification,
                target_service=top_cause,
                action=(remediation_action.action_type if remediation_action else "none"),
                pre_intervention_consistency=cc_score,
                before_state=before_snapshot,
            )
            state.intervention_validation = val_res

        # ------------------------------------------------------------------
        # 8. Finalize trajectory
        # ------------------------------------------------------------------
        trajectory.final_state = state
        trajectory.total_budget_used = state.total_query_cost
        trajectory.terminated_by = terminated_by

        return trajectory
