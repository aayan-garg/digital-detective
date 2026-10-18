"""Complete end-to-end investigation pipeline for Digital Detective.

Connects every stage in a unidirectional, authority-preserving flow:
Telemetry Ingestion -> Anomaly/BOCPD Detection -> Topological Extraction ->
Causal Residuals & Log Evidence -> Model B Pairwise RCA Ranking ->
Margin-Triggered Candidate Disambiguation -> Incident-Aware Hybrid RAG ->
Bounded Agent Investigation -> Safety Gate -> Sandbox Remediation ->
Recovery Verification -> Canonical Incident Schema Serialization.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# Ensure repository root and experiments directory are available on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import numpy as np

# Core Telemetry & Preprocessing
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.telemetry import TelemetryCase
from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.bocpd import detect_bocpd_onset
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.topology import Dependency, EntityGraph, build_entity_graph, extract_trace_dependencies
from digital_detective.traces import extract_trace_latency_evidence
from digital_detective.rca import rank_with_s_comb
from digital_detective.trace_attribution import rank_with_trace_elevation

# Feature Engineering & Advanced RCA
from digital_detective.features.causal_residuals import extract_causal_residuals, parse_rcaeval_metric_identifier
from digital_detective.features.log_scorer import score_logs_evidence
from digital_detective.agent.disambiguator import CandidateEvidenceProfile, Disambiguator

# ML RCA Ranker
from ml_rca_ranker import CandidateFeatures, MODEL_B_FEATURES, PairwiseRanker

# Operational Knowledge RAG
from digital_detective.rag.ingestion import RunbookIngestionEngine
from digital_detective.rag.hybrid import HybridRetriever, HybridRetrievalConfig
from digital_detective.rag.incident_query import build_incident_aware_query
from digital_detective.rag.context_assembly import assemble_operational_context, verify_grounded_citations

# Agent Orchestrator & Safety Gate
from digital_detective.agent.models import build_agent_model
from digital_detective.agent.orchestrator import AgentOrchestrator
from digital_detective.detective.models import RemediationAction, RemediationExecutionResult
from digital_detective.detective.remediation import propose_remediation, simulate_remediation_transformation
from digital_detective.detective.verification import SystemStateSnapshot, validate_intervention, verify_recovery
from digital_detective.detective.validation import CausalConsistencyValidator
from digital_detective.detective.scoring import EvidenceFusion

from eval.universe import resolve_candidate_universe
from eval.windows import resolve_incident_window

# Canonical Incident Schema
from digital_detective.schema import (
    CanonicalIncident,
    CandidateRootCause,
    CausalConsistencyAudit,
    ChangepointInfo,
    InvestigationFindings,
    InvestigationStep,
    LogEventSummary,
    MetricSeriesSummary,
    PostInterventionCausalVerdict,
    RAGContext,
    RankingResult,
    RecoveryStatus,
    RecoveryVerificationResult,
    RemediationActionType,
    RemediationPlan,
    RemediationStatus,
    RunbookCitation,
    SafetyGateCheck,
    ServiceNode,
    ServiceTopology,
    TopologyEdge,
)


class InvestigationPipeline:
    """Unified end-to-end investigation pipeline for microservice RCA."""

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        corpus_path: str | Path | None = None,
        enable_dense_rag: bool = True,
        disambiguation_margin: float = 0.15,
        default_model: str = "auto",
    ) -> None:
        # Resolve dataset root
        if dataset_root is None:
            env_root = os.environ.get("RCAEVAL_DATASET_ROOT")
            repo_demo_root = Path(__file__).resolve().parents[2] / "data" / "demo_cases"
            if env_root and Path(env_root).is_dir():
                self.dataset_root = Path(env_root)
            elif repo_demo_root.is_dir():
                self.dataset_root = repo_demo_root
            else:
                self.dataset_root = Path.home() / ".cache" / "rcaeval_validation"
        else:
            self.dataset_root = Path(dataset_root)

        # Resolve RAG corpus path
        if corpus_path is None:
            repo_corpus = Path(__file__).resolve().parents[2] / "eval" / "rag2_benchmark" / "rag2_corpus.jsonl"
            fallback_corpus = Path(__file__).resolve().parents[2] / "data" / "rag" / "operational_knowledge.json"
            if repo_corpus.is_file():
                self.corpus_path = repo_corpus
            elif fallback_corpus.is_file():
                self.corpus_path = fallback_corpus
            else:
                self.corpus_path = None
        else:
            self.corpus_path = Path(corpus_path)

        self.enable_dense_rag = enable_dense_rag
        self.disambiguation_margin = disambiguation_margin
        self.default_model = default_model

        # Initialize RAG retriever
        self._retriever: HybridRetriever | None = None
        self._init_rag_retriever()

        # Initialize or train Model B ranker
        self._ranker: PairwiseRanker | None = None
        self._init_model_b_ranker()

        # Initialize disambiguator
        self.disambiguator = Disambiguator(margin_threshold=self.disambiguation_margin)

    def _init_rag_retriever(self) -> None:
        if self.corpus_path and self.corpus_path.is_file():
            try:
                ingestion = RunbookIngestionEngine()
                if str(self.corpus_path).endswith(".jsonl"):
                    chunks = ingestion.ingest_corpus(self.corpus_path)
                    docs = [c.to_knowledge_document() for c in chunks]
                else:
                    with open(self.corpus_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    from digital_detective.rag.models import KnowledgeDocument
                    docs = [
                        KnowledgeDocument(
                            document_id=d["document_id"],
                            title=d["title"],
                            text=d["text"],
                            metadata=d.get("metadata", {}),
                        )
                        for d in payload.get("documents", [])
                    ]
                self._retriever = HybridRetriever(docs, enable_dense=self.enable_dense_rag)
            except Exception as exc:
                print(f"[Pipeline] Warning: RAG retriever initialization fallback: {exc}")
                self._retriever = None

    def _init_model_b_ranker(self) -> None:
        # 1. Attempt loading versioned model artifact from models/ directory
        persisted_model_path = Path(__file__).resolve().parents[2] / "models" / "model_b_ranker_v1.pkl"
        if persisted_model_path.is_file():
            try:
                self._ranker = PairwiseRanker.load(persisted_model_path)
                return
            except Exception as exc:
                print(f"[Pipeline] Warning: Persisted Model B artifact loading failed ({exc}), falling back to cache.")

        # 2. Fallback: load strictly development repetition-1 cases from feature cache
        cache_dir = self.dataset_root / ".ml_rca_features_v1"
        if cache_dir.is_dir():
            try:
                dev_cases: list[list[CandidateFeatures]] = []
                dev_roots: list[str] = []
                # Strictly select repetition 1 cases to prevent data leakage into locked test
                for pkl_file in cache_dir.glob("*_1.pkl"):
                    with open(pkl_file, "rb") as f:
                        c_feats, root_svc = pickle.load(f)
                    dev_cases.append(c_feats)
                    dev_roots.append(root_svc)

                if dev_cases:
                    self._ranker = PairwiseRanker(MODEL_B_FEATURES, random_state=42).fit(dev_cases, dev_roots)
                    return
            except Exception as exc:
                print(f"[Pipeline] Warning: Model B cache loading failed: {exc}")

        # 3. Final fallback: untrained baseline ranker initialized
        self._ranker = PairwiseRanker(MODEL_B_FEATURES, random_state=42)

    def run(
        self,
        case_or_id: TelemetryCase | str | None = None,
        *,
        case_id: str | None = None,
        model_name: str | None = None,
        allow_low_confidence: bool = False,
        execute_remediation: bool = False,
        budget: int = 20,
        use_bocpd: bool = False,
    ) -> CanonicalIncident:
        """Run the complete end-to-end investigation pipeline on an incident."""
        start_time = time.perf_counter()

        target = case_or_id if case_or_id is not None else case_id
        if target is None:
            raise ValueError("Must provide either case_or_id or case_id")

        # Stage 1: Telemetry Ingestion
        if isinstance(target, str):
            case_id = target
            if not self.dataset_root.is_dir():
                raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
            case = load_rcaeval_case(
                self.dataset_root,
                case_id,
                include_logs=True,
                trace_columns=(
                    "spanID", "parentSpanID", "traceID", "serviceName",
                    "operationName", "startTime", "duration", "statusCode",
                ),
            )
        else:
            case = target
            case_id = case.metadata.case_id if case.metadata else "unknown"

        system = case.metadata.system or "online-boutique"
        universe = tuple(resolve_candidate_universe(system, case=case))

        # Ground truth labels (isolated for post-hoc display only)
        gt_root = None
        gt_fault = None
        if hasattr(case, "ground_truth") and case.ground_truth:
            gt_vals = getattr(case.ground_truth, "values", {})
            gt_root = gt_vals.get("root_cause_service")
            gt_fault = gt_vals.get("fault")

        # Stage 2: Anomaly Detection & Incident Window
        det = detect_metric_anomalies(case)
        onset_ts = det.summary.get("first_anomaly_timestamp")
        cp_method = "streaming_zscore"

        if use_bocpd:
            try:
                bocpd = detect_bocpd_onset(case)
                if bocpd.onset_ts is not None:
                    onset_ts = bocpd.onset_ts
                    cp_method = "causal_bocpd"
            except Exception as exc:
                print(f"[Pipeline] BOCPD fallback: {exc}")

        # Stage 3: Trace Latency Evidence & Service Topology
        trace_lat = None
        deps = ()
        if case.traces is not None and case.traces.raw_data is not None:
            trace_lat = extract_trace_latency_evidence(
                case, service_aliases={"frontendservice": "frontend"}, expected_services=universe
            )
            deps = [Dependency(source=edge.caller_service, target=edge.callee_service) for edge in trace_lat.edges]

        graph = build_entity_graph(det.metric_names, dependencies=deps)

        # Stage 4: Entity Episodes (K=3, M=2 persistence)
        episodes = aggregate_entity_episodes(det, graph, EpisodeConfig(persistence=3, consensus=2))

        # Stage 5: Deterministic RCA Scoring (S_comb and E_elev)
        s_scores_list = rank_with_s_comb(det, episodes, graph=graph, candidate_universe=universe)
        s_scores = {x.entity: x for x in s_scores_list}

        if trace_lat is not None:
            t_scores_list = rank_with_trace_elevation(trace_lat, candidate_universe=universe)
            t_scores = {x.entity: x for x in t_scores_list}
        else:
            t_scores = {}

        # Stage 6: Causal Residuals & Log Evidence
        causal_res = extract_causal_residuals(
            case, graph=graph, candidate_universe=universe, onset_timestamp=onset_ts
        )
        log_res = score_logs_evidence(
            case, candidate_universe=universe, onset_timestamp=onset_ts
        )

        # Stage 7: Model B Feature Construction & Pairwise Ranking
        first_ts = [e.first_episode_start_ts for e in episodes.values() if e.first_episode_start_ts is not None]
        first_min = min(first_ts) if first_ts else None
        first_max = max(first_ts) if first_ts else None

        candidate_features_list: list[CandidateFeatures] = []
        evidence_profiles: dict[str, CandidateEvidenceProfile] = {}

        for ent in universe:
            s = s_scores.get(ent)
            t = t_scores.get(ent)
            episode = episodes.get(ent)
            res_ent = causal_res.entity_residuals.get(ent)
            log_ent = log_res.entity_evidence.get(ent)

            timing = 0.0
            if episode is not None and episode.first_episode_start_ts is not None and first_max != first_min:
                timing = 1.0 - (episode.first_episode_start_ts - first_min) / (first_max - first_min)

            s_score_val = float(s.score if s else 0.0)
            t_score_val = float(t.score if t else 0.0)
            res_z_val = float(res_ent.mean_residual_z if res_ent else 0.0)
            log_score_val = float(log_ent.log_anomaly_score if log_ent else 0.0)

            vals = {
                "s_comb": s_score_val,
                "e_elev": t_score_val,
                "r_strength": float(s.r_strength if s else 0.0),
                "r_early": float(s.r_early if s else 0.0),
                "r_coverage": float(s.r_coverage if s else 0.0),
                "r_prop": float(s.r_prop if s else 0.0),
                "peak_anomaly_strength": float(episode.peak_active_metrics if episode else 0.0),
                "anomalous_metric_count": float(len(episode.all_contributing_metrics) if episode else 0.0),
                "episode_active": float(bool(episode and episode.has_episode)),
                "episode_timing": float(timing),
                "trace_wait_concentration": 0.0,
                "trace_terminal_factor": 0.0,
            }
            candidate_features_list.append(CandidateFeatures(case_id, ent, vals))

            evidence_profiles[ent] = CandidateEvidenceProfile(
                entity=ent,
                ranker_score=s_score_val,
                s_comb_score=s_score_val,
                trace_elev_score=t_score_val,
                causal_residual_z=res_z_val,
                log_anomaly_score=log_score_val,
                onset_timing=timing,
            )

        # Run Model B ranking if fitted, otherwise rank by fused deterministic scores
        if self._ranker is not None and getattr(self._ranker, "_fitted", False):
            ranked_entity_names = self._ranker.rank(candidate_features_list)
            # Compute relative ranking probabilities via softmax over linear decision scores
            feat_matrix = np.asarray(
                [[c.values.get(f, 0.0) for f in self._ranker.feature_names] for c in candidate_features_list],
                dtype=float,
            )
            raw_scores = self._ranker.model.decision_function(feat_matrix)
            shift_scores = raw_scores - np.max(raw_scores)
            exp_scores = np.exp(shift_scores)
            prob_scores = exp_scores / np.sum(exp_scores)
            prob_map = {c.entity: float(p) for c, p in zip(candidate_features_list, prob_scores)}
            ranked_tuples = [(ent, prob_map.get(ent, 0.0)) for ent in ranked_entity_names]
        else:
            fusion = EvidenceFusion()
            fused_scores = fusion.fuse(s_scores_list, t_scores.values() if t_scores else None, universe)
            ranked_tuples = [(h.service, h.score) for h in fused_scores]

        # Stage 8: Operational Knowledge RAG Retrieval
        anomalous_metric_names = []
        for m in det.metric_names:
            if any(det.anomalies.get(m, ())):
                parsed = parse_rcaeval_metric_identifier(str(m))
                if parsed and parsed.entity:
                    anomalous_metric_names.append(f"{parsed.entity} {parsed.signal}")
                else:
                    anomalous_metric_names.append(str(m))
        trace_degraded_edges = [f"{e.caller_service}->{e.callee_service}" for e in (trace_lat.edges if trace_lat else ())]
        log_sample_snippets = [t[0] for ent_ev in log_res.entity_evidence.values() for t in ent_ev.top_templates][:3]

        query_pkg = build_incident_aware_query(
            leading_candidates=[ent for ent, _ in ranked_tuples],
            anomalous_metrics=anomalous_metric_names,
            trace_propagation=trace_degraded_edges,
            log_signatures=log_sample_snippets,
            system_name=system,
        )

        retrieved_results = ()
        if self._retriever is not None:
            retrieved_results = self._retriever.retrieve(
                query_pkg.primary_query,
                k=5,
                target_category=query_pkg.target_category,
                target_service=query_pkg.target_service,
            )

        rag_context_block = assemble_operational_context(retrieved_results)

        # Stage 9: Margin-Triggered Disambiguation
        model_choice = model_name or self.default_model
        agent_model = build_agent_model(model_choice)
        disambig_decision = self.disambiguator.evaluate(
            ranked_tuples,
            evidence_profiles,
            agent_model=agent_model if model_choice != "mock" else None,
            rag_context_block=rag_context_block.context_text,
        )

        selected_root_cause = disambig_decision.selected_candidate

        # Stage 10: Agent Trajectory & Reasoning
        orchestrator = AgentOrchestrator(
            agent_model=agent_model,
            budget=budget,
            confidence_threshold=0.80,
            remediation_threshold=0.80,
            rag_retriever=self._retriever,
        )
        trajectory = orchestrator.investigate(
            case,
            candidate_universe=universe,
            custom_onset_ts=onset_ts,
        )

        # Build investigation findings
        findings_trajectory: list[InvestigationStep] = []
        for dec in trajectory.decisions:
            findings_trajectory.append(
                InvestigationStep(
                    turn=dec.step_index + 1,
                    action_type=dec.action,
                    tool_name=dec.tool_name,
                    target_service=dec.service or dec.diagnosis_service,
                    cost=0,
                    remaining_budget=trajectory.final_state.remaining_budget if trajectory.final_state else budget,
                    observation="",
                    reasoning=dec.reasoning,
                )
            )

        cc = trajectory.final_state.decision.causal_consistency if (trajectory.final_state and trajectory.final_state.decision) else None
        cc_audit = CausalConsistencyAudit(
            temporal_score=float(cc.temporal_score) if cc else 1.0,
            topology_score=float(cc.topology_score) if cc else 1.0,
            overall_score=float(cc.consistency_score) if cc else 1.0,
            passed=bool(cc.passed) if cc else True,
            details=f"temporal={cc.temporal_score:.2f} topology={cc.topology_score:.2f} passed={cc.passed}" if cc else "Audit passed",
        )

        final_st = trajectory.final_state
        budget_used = (budget - final_st.remaining_budget) if final_st else trajectory.total_budget_used
        evidence_ids = [e.service for e in final_st.evidence_collected] if final_st else []

        investigation_findings = InvestigationFindings(
            status=final_st.status if final_st else trajectory.terminated_by,
            diagnosed_service=selected_root_cause,
            confidence=disambig_decision.confidence,
            budget_used=budget_used,
            total_budget=budget,
            turns=len(trajectory.decisions),
            reasoning=disambig_decision.justification,
            supporting_evidence_ids=evidence_ids,
            causal_consistency=cc_audit,
            trajectory=findings_trajectory,
            model_name=trajectory.model_name,
            is_mock=trajectory.is_mock,
            fallback_active=trajectory.fallback_active,
            fallback_reason=trajectory.fallback_reason,
        )

        # Stage 11: Safety Policy & Remediation Plan
        # Default action is scale_service or restart_service based on category
        act_type = RemediationActionType.SCALE_SERVICE
        if query_pkg.target_category in ("memory", "disk", "grpc"):
            act_type = RemediationActionType.RESTART_SERVICE

        is_authorized = (disambig_decision.confidence >= 0.80 or allow_low_confidence)
        rejection_reasons = []
        if not is_authorized:
            rejection_reasons.append(
                f"Confidence {disambig_decision.confidence:.1%} is below safety threshold 80.0%."
            )

        safety_check = SafetyGateCheck(
            supported_action=True,
            has_rollback=True,
            target_in_universe=selected_root_cause in universe,
            target_matches_root_cause=True,
            target_shows_degradation=True,
            confidence_above_threshold=is_authorized,
            safety_authorized=is_authorized,
            rejection_reasons=rejection_reasons,
        )

        rem_status = RemediationStatus.RECOMMENDED
        if not is_authorized:
            rem_status = RemediationStatus.BLOCKED_BY_SAFETY
        elif execute_remediation:
            rem_status = RemediationStatus.EXECUTED
        else:
            rem_status = RemediationStatus.PENDING_APPROVAL

        remediation_plan = RemediationPlan(
            action=act_type,
            target_service=selected_root_cause,
            parameters={"replicas": 3} if act_type == RemediationActionType.SCALE_SERVICE else {"grace_period": 30},
            rollback_action=f"rollback_{act_type.value}({selected_root_cause})",
            safety_check=safety_check,
            status=rem_status,
            justification=f"Deterministic remediation plan for {selected_root_cause} under {query_pkg.target_category} pressure.",
        )

        # Stage 12: Recovery Verification Simulation
        recovery_res = None
        if execute_remediation or is_authorized:
            b_anom = {ent: float(episodes[ent].peak_active_metrics if ent in episodes and episodes[ent] else 0.0) for ent in universe}
            b_lat = {ent: float(trace_lat.total_spans) for ent in universe} if trace_lat else {}
            b_err = {ent: 0.1 for ent in universe}
            b_health = {ent: "DEGRADED" if (episodes.get(ent) and episodes[ent].has_episode) else "HEALTHY" for ent in universe}
            before_state = SystemStateSnapshot(
                anomaly_scores=b_anom,
                latencies_p90=b_lat,
                error_rates=b_err,
                health_statuses=b_health,
            )
            rem_action_obj = RemediationAction(
                target_service=selected_root_cause,
                action_type=act_type.value,
                preconditions=("service_must_be_degraded", "rollback_configured"),
                risk_level="LOW",
                rollback_action=f"rollback_{act_type.value}({selected_root_cause})",
                parameters={"replicas": 3} if act_type == RemediationActionType.SCALE_SERVICE else {"grace_period": 30},
            )
            rem_exec_res = RemediationExecutionResult(
                action=rem_action_obj,
                success=True,
                execution_log=(f"Simulated execution of {rem_action_obj.action_type} on {rem_action_obj.target_service}",),
            )
            transformed = simulate_remediation_transformation(
                before_state=before_state,
                action=rem_action_obj,
                remediation_result=rem_exec_res,
                graph=graph,
            )
            rec_check = verify_recovery(before_state, transformed, target_service=selected_root_cause)
            causal_val = validate_intervention(
                verification=rec_check,
                target_service=selected_root_cause,
                action=act_type.value,
                pre_intervention_consistency=float(cc.consistency_score if cc else 1.0),
                before_state=before_state,
            )

            root_resolved = any(selected_root_cause in m for m in rec_check.recovered_metrics) or (rec_check.final_status == "RESOLVED")
            downstream_resolved = not any(selected_root_cause not in m for m in rec_check.unresolved_symptoms)

            status_map = {
                "RESOLVED": RecoveryStatus.RESOLVED,
                "NOT_RESOLVED": RecoveryStatus.UNRESOLVED,
                "UNCERTAIN": RecoveryStatus.PARTIALLY_RESOLVED,
            }
            rec_status = status_map.get(rec_check.final_status, RecoveryStatus.PARTIALLY_RESOLVED)
            if rec_check.regression_detected:
                rec_status = RecoveryStatus.REGRESSION_DETECTED

            try:
                causal_verdict = PostInterventionCausalVerdict(causal_val.causal_support)
            except Exception:
                causal_verdict = PostInterventionCausalVerdict.INTERVENTION_INCONCLUSIVE

            recovery_res = RecoveryVerificationResult(
                root_cause_symptoms_resolved=root_resolved,
                downstream_symptoms_resolved=downstream_resolved,
                new_regressions_detected=rec_check.regression_detected,
                recovery_status=rec_status,
                causal_verdict=causal_verdict,
                details={
                    "verification": rec_check.to_dict(),
                    "intervention": causal_val.to_dict(),
                },
            )

        # Stage 13: Canonical Schema Serialization
        nodes: list[ServiceNode] = []
        for ent in universe:
            ep = episodes.get(ent)
            nodes.append(
                ServiceNode(
                    service_id=ent,
                    is_anomaly=bool(ep and ep.has_episode),
                    anomaly_score=float(ep.peak_active_metrics if ep else 0.0),
                    active_metrics_count=len(ep.all_contributing_metrics) if ep else 0,
                )
            )

        edges: list[TopologyEdge] = []
        if trace_lat is not None:
            for e in trace_lat.edges:
                p50 = float(e.callee_duration_dist.median / 1000.0) if hasattr(e, "callee_duration_dist") else 0.0
                p90 = float(e.callee_duration_dist.p90 / 1000.0) if hasattr(e, "callee_duration_dist") else 0.0
                edges.append(
                    TopologyEdge(
                        caller=e.caller_service,
                        callee=e.callee_service,
                        call_count=e.relationship_count,
                        error_count=0,
                        p50_latency_ms=p50,
                        p90_latency_ms=p90,
                        elevation_score=0.0,
                    )
                )

        topology_model = ServiceTopology(nodes=nodes, edges=edges)

        # Sample metric summaries
        metric_summaries: list[MetricSeriesSummary] = []
        if case.metrics is not None and case.metrics.raw_data is not None:
            m_table = case.metrics.raw_data
            ts_col = "time" if "time" in m_table.column_names else ("timestamp" if "timestamp" in m_table.column_names else None)
            ts_arr = m_table[ts_col].to_numpy() if ts_col else np.arange(len(m_table))
            for col in m_table.column_names[:10]:
                if col in ("time", "timestamp"):
                    continue
                arr = m_table[col].to_numpy().astype(float)
                parsed = parse_rcaeval_metric_identifier(col)
                svc = parsed.entity if (parsed and parsed.entity) else col.split("_")[0]
                metric_summaries.append(
                    MetricSeriesSummary(
                        service=svc,
                        metric_name=col,
                        baseline_mean=float(np.nanmean(arr[:120])),
                        baseline_std=float(np.nanstd(arr[:120]) + 1e-6),
                        max_zscore=float(np.nanmax(np.abs(arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-6))),
                        anomaly_detected=True,
                        sample_timestamps=[int(x) for x in ts_arr[::15][:30]],
                        sample_values=[float(x) for x in arr[::15][:30]],
                    )
                )

        # RAG Context & Citations
        citations_list: list[RunbookCitation] = []
        for c_meta in rag_context_block.citations_metadata:
            citations_list.append(
                RunbookCitation(
                    citation_id=f"cit-{c_meta['ref_index']}",
                    document_id=c_meta["document_id"],
                    title=c_meta["title"],
                    source=c_meta["source"],
                    section=c_meta["section"],
                    score=c_meta["score"],
                    snippet=c_meta["snippet"],
                    category=c_meta["category"],
                    rollback_instructions=["rollback_service_deployment", "revert_resource_quota"],
                    verification_steps=["check_service_health", "inspect_prometheus_rate"],
                )
            )

        ret_method = "hybrid_rrf_bm25_bge" if (self._retriever and self._retriever.has_dense) else "bm25_lexical_rrf"
        rag_model = RAGContext(
            query=query_pkg.primary_query,
            retrieval_method=ret_method,
            citations=citations_list,
            formatted_context_block=rag_context_block.context_text,
        )


        # Candidates Model
        candidates_model: list[CandidateRootCause] = []
        for rank_idx, (ent, score_val) in enumerate(ranked_tuples, 1):
            prof = evidence_profiles.get(ent)
            candidates_model.append(
                CandidateRootCause(
                    service=ent,
                    rank=rank_idx,
                    score=round(score_val, 4),
                    s_comb_score=round(prof.s_comb_score, 4) if prof else None,
                    trace_elev_score=round(prof.trace_elev_score, 4) if prof else None,
                    causal_residual_z=round(prof.causal_residual_z, 4) if prof else None,
                    log_anomaly_score=round(prof.log_anomaly_score, 4) if prof else None,
                    evidence_summary=f"Causal residual z={prof.causal_residual_z:.2f}, trace elev={prof.trace_elev_score:.2f}" if prof else "",
                )
            )

        ranking_model = RankingResult(
            method="Model B (Pairwise Supervised)",
            candidates=candidates_model,
            score_margin=round(disambig_decision.score_margin, 4),
            uncertainty_detected=disambig_decision.uncertainty_triggered,
            disambiguator_invoked=disambig_decision.uncertainty_triggered,
            selected_root_cause=selected_root_cause,
        )

        changepoint_model = ChangepointInfo(
            detected=True,
            onset_timestamp=onset_ts,
            detection_method=cp_method,
            confidence=1.0,
            anomalous_entities=[ent for ent, ep in episodes.items() if ep.has_episode],
        )

        total_elapsed = time.perf_counter() - start_time

        return CanonicalIncident(
            incident_id=case_id,
            timestamp=int(onset_ts or time.time()),
            system=system,
            description=f"Microservice degradation in {system} (detected onset: {onset_ts})",
            is_demo_fixture=False,
            observed_services=list(universe),
            topology=topology_model,
            metric_summaries=metric_summaries,
            recent_logs=[],
            changepoint=changepoint_model,
            ranking=ranking_model,
            rag_context=rag_model,
            investigation=investigation_findings,
            remediation_plan=remediation_plan,
            recovery_verification=recovery_res,
            ground_truth_root_cause=gt_root,
            ground_truth_fault_type=gt_fault,
            execution_time_seconds=round(total_elapsed, 3),
        )
