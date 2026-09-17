"""Agentic investigation demo for Digital Detective.

Demonstrates the Agent / Planner layer on top of the deterministic core.

Modes
-----
--model mock         Deterministic rule-based agent (always works, no network)
--model ollama       Local Ollama (e.g. qwen3:8b). Fails if Ollama not running.
--model grok         xAI Grok. Requires XAI_API_KEY.
--model openai       OpenAI. Requires OPENAI_API_KEY.
--model auto         Ollama if available, else Mock (default)

--mode agent         Agent-only output (default)
--mode det           Deterministic-only output
--mode both          Side-by-side comparison (agent + deterministic)

--verbose            Print full turn-by-turn agent trajectory
--compare-models     Run mock + ollama + grok (skips unavailable) and compare
--all-cases          Run all three default demo cases
--case <id>          Run a single case
--allow-low-confidence  Override remediation safety gate

Run with:
    python -m digital_detective.agent.demo --model mock --all-cases
    python -m digital_detective.agent.demo --model ollama --case re2ob_checkoutservice_cpu_1
    python -m digital_detective.agent.demo --model ollama --mode both --verbose
    python -m digital_detective.agent.demo --compare-models --all-cases
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.detective.investigator import InvestigationEngine
from digital_detective.detective.models import InvestigationState

from .models import (
    AgentModel,
    InvestigationTrajectory,
    build_agent_model,
)
from .orchestrator import AgentOrchestrator

DEFAULT_DEMO_CASES = (
    "re2ob_checkoutservice_cpu_1",
    "re2ob_currencyservice_delay_1",
    "re2ob_checkoutservice_mem_1",
)

_SEP = "=" * 72
_SEP_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _fmt_cc(cc: Any | None) -> str:
    if cc is None:
        return "not evaluated"
    return (
        f"temporal={cc.temporal_score:.2f} topology={cc.topology_score:.2f} "
        f"overall={cc.consistency_score:.2f} passed={'YES' if cc.passed else 'NO'}"
    )


def _ground_truth_str(case: Any) -> str:
    gt = getattr(case, "ground_truth", None)
    if gt is None:
        return "unknown"
    vals = getattr(gt, "values", {})
    svc = vals.get("root_cause_service", "?")
    fault = vals.get("fault", "?")
    return f"{svc} ({fault})"


# ---------------------------------------------------------------------------
# Deterministic mode runner
# ---------------------------------------------------------------------------

def _run_deterministic(
    case: Any,
    budget: int,
    remediation_threshold: float,
    allow_low_confidence: bool,
    adaptive_query_selection: bool = True,
) -> dict[str, Any]:
    engine = InvestigationEngine(
        budget=budget,
        confidence_threshold=remediation_threshold,
        adaptive_query_selection=adaptive_query_selection,
    )
    t0 = time.perf_counter()
    state: InvestigationState = engine.investigate(
        case=case,
        window_mode="oracle",
        remediation_threshold=remediation_threshold,
        allow_low_confidence_remediation=allow_low_confidence,
        adaptive_query_selection=adaptive_query_selection,
    )
    elapsed = time.perf_counter() - t0

    dec = state.decision
    predicted = dec.root_cause_service if dec else "none"
    confidence = dec.confidence if dec else 0.0
    cc = dec.causal_consistency if dec else None
    ver_status = state.verification_result.final_status if state.verification_result else "NOT_EVALUATED"

    return {
        "mode": "deterministic",
        "model_name": "Deterministic/InvestigationEngine",
        "predicted_cause": predicted,
        "confidence": confidence,
        "queries": len(state.queries_executed),
        "budget_used": state.total_query_cost,
        "evidence_count": len(state.evidence_collected),
        "cc": cc,
        "remediation_authorized": state.remediation_authorized,
        "verification_status": ver_status,
        "elapsed_sec": round(elapsed, 3),
        "state": state,
    }


# ---------------------------------------------------------------------------
# Agent mode runner
# ---------------------------------------------------------------------------

def _run_agent(
    case: Any,
    agent: AgentModel,
    budget: int,
    remediation_threshold: float,
    allow_low_confidence: bool,
    max_turns: int = 6,
    verbose: bool = False,
) -> dict[str, Any]:
    orch = AgentOrchestrator(
        agent_model=agent,
        budget=budget,
        max_steps=max_turns,
        remediation_threshold=remediation_threshold,
    )
    t0 = time.perf_counter()
    trajectory: InvestigationTrajectory = orch.investigate(
        case=case,
        allow_low_confidence_remediation=allow_low_confidence,
    )
    elapsed = time.perf_counter() - t0

    state = trajectory.final_state
    dec = state.decision if state else None
    predicted = dec.root_cause_service if dec else "none"
    confidence = dec.confidence if dec else 0.0
    cc = dec.causal_consistency if dec else None
    ver_status = state.verification_result.final_status if (state and state.verification_result) else "NOT_EVALUATED"
    rem_auth = state.remediation_authorized if state else False

    if verbose:
        _print_trajectory(trajectory, agent)

    return {
        "mode": "agent",
        "model_name": trajectory.model_name,
        "predicted_cause": predicted,
        "confidence": confidence,
        "turns": trajectory.total_steps,
        "queries": len(state.queries_executed) if state else 0,
        "budget_used": trajectory.total_budget_used,
        "evidence_count": len(state.evidence_collected) if state else 0,
        "cc": cc,
        "terminated_by": trajectory.terminated_by,
        "remediation_authorized": rem_auth,
        "verification_status": ver_status,
        "elapsed_sec": round(elapsed, 3),
        "trajectory": trajectory,
        "state": state,
    }


# ---------------------------------------------------------------------------
# Turn-by-turn trajectory printer
# ---------------------------------------------------------------------------

def _print_trajectory(trajectory: InvestigationTrajectory, agent: AgentModel) -> None:
    model_name = getattr(agent, "model_name", type(agent).__name__)
    print(f"\n{'-' * 72}")
    print(f"  AGENT TRAJECTORY  [{model_name}]  case={trajectory.incident_id}")
    print(f"{'-' * 72}")

    state = trajectory.final_state
    evidence_items = state.evidence_collected if state else []

    # Build evidence ID -> summary mapping for display
    eid_map: dict[str, str] = {}
    for i, ev in enumerate(evidence_items, start=1):
        ts_str = f" ts={ev.timestamp}" if ev.timestamp else ""
        eid_map[f"E{i}"] = (
            f"E{i}: service={ev.service} modality={ev.modality} "
            f"signal={ev.signal} magnitude={ev.magnitude:.3f}{ts_str}"
        )

    turn = 0
    for decision in trajectory.decisions:
        if "POLICY REJECTED" in decision.reasoning:
            print(f"\n  [REJECTED decision at step {decision.step_index}]")
            print(f"  Reason: {decision.reasoning[:120]}")
            continue

        turn += 1
        print(f"\nTURN {turn}  (step {decision.step_index})")
        print(f"  Model   : {model_name}")
        print(f"  Action  : {decision.action}")

        if decision.action == "QUERY":
            print(f"  Tool    : {decision.tool_name}({decision.service})")
        print(f"  Reason  : {decision.reasoning[:200]}")

        if decision.evidence_ids:
            print("  Evidence cited:")
            for eid in decision.evidence_ids:
                desc = eid_map.get(eid, eid)
                print(f"    {desc}")

        # Show current budget after this step (approximate)
        if state:
            print(f"  Remaining budget: ~{state.remaining_budget}")

    print(f"\n{'-' * 72}")
    print(f"  Terminated by: {trajectory.terminated_by}  |  Total turns: {trajectory.total_steps}")

    if state and state.decision:
        dec = state.decision
        print(f"\nFINAL DIAGNOSIS")
        print(f"  Root cause          : {dec.root_cause_service}")
        print(f"  Confidence          : {_fmt_pct(dec.confidence)}")
        print(f"  Causal consistency  : {_fmt_cc(dec.causal_consistency)}")
        print(f"  Budget used         : {trajectory.total_budget_used}")
        print(f"  Evidence collected  : {len(evidence_items)}")
        if evidence_items:
            print("  Evidence summary (last 5):")
            for i, ev in enumerate(evidence_items[-5:], start=max(1, len(evidence_items) - 4)):
                print(f"    E{i}: {ev.service} / {ev.modality} / {ev.signal} = {ev.magnitude:.3f}")

        print(f"\nREMEDIATION AUTHORIZATION : {'YES' if state.remediation_authorized else 'NO (held)'}")
        if state.verification_result:
            ver = state.verification_result
            print(f"RECOVERY VERIFICATION     : {ver.final_status}")

    print(f"{'-' * 72}\n")


# ---------------------------------------------------------------------------
# Case-level result printer
# ---------------------------------------------------------------------------

def _print_structured_demo_report(r: dict[str, Any]) -> None:
    """Print visibly formatted sections required for the research demo."""
    state: InvestigationState | None = r.get("state")
    traj: InvestigationTrajectory | None = r.get("trajectory")

    print("\n---")
    print("\n## EVIDENCE ACQUISITION")

    queries = state.queries_executed if state else []
    if queries:
        rem_budget = state.budget if state else 20
        for idx, q in enumerate(queries, start=1):
            rem_budget -= q.cost
            print(f"\nTurn {idx}:")
            leading = q.parameters.get("leading_hypotheses")
            if not leading and state and state.current_rankings:
                leading = [h.service for h in state.current_rankings[:3]]
            print("Leading hypotheses:")
            for h in (leading or [q.service])[:3]:
                print(h)

            policy = q.parameters.get("query_policy")
            if not policy:
                policy = "metric evidence is currently more discriminative than trace evidence."
            print(f"\nQuery policy:\n{policy}")
            print(f"\nSelected:\n{q.tool_name}({q.service})")
            print(f"\nCost:\n{q.cost}")
            print(f"\nRemaining:\n{max(0, rem_budget)}")
    elif traj and traj.decisions:
        turn_idx = 0
        rem = traj.final_state.budget if traj.final_state else 20
        for dec in traj.decisions:
            if dec.action == "QUERY":
                turn_idx += 1
                from digital_detective.detective.tools import DEFAULT_TOOL_COSTS
                c = DEFAULT_TOOL_COSTS.get(dec.tool_name, 1)
                rem -= c
                print(f"\nTurn {turn_idx}:")
                if traj.final_state and traj.final_state.current_rankings:
                    print("Leading hypotheses:")
                    for h in traj.final_state.current_rankings[:3]:
                        print(h.service)
                print("\nQuery policy:\nmetric evidence is currently more discriminative than trace evidence.")
                print(f"\nSelected:\n{dec.tool_name}({dec.service})")
                print(f"\nCost:\n{c}")
                print(f"\nRemaining:\n{max(0, rem)}")

    print("\n---")
    print("\n## FINAL DIAGNOSIS")
    dec = state.decision if state else None
    pred_cause = r.get("predicted_cause", dec.root_cause_service if dec else "unknown")
    print(f"\nRoot cause:\n{pred_cause}")

    ev_list = state.evidence_collected if state else []
    target_eids = [
        f"E{i}" for i, ev in enumerate(ev_list, start=1)
        if ev.service == pred_cause and ev.magnitude > 0
    ]
    if not target_eids and ev_list:
        target_eids = [f"E{i}" for i in range(1, min(4, len(ev_list) + 1))]
    print(f"\nEvidence:\n{', '.join(target_eids[:3]) if target_eids else 'None'}")

    cc = r.get("cc", dec.causal_consistency if dec else None)
    print(f"\nObservational causal consistency:\n{_fmt_cc(cc)}")

    print("\n---")
    print("\n## SAFETY")
    conf = r.get("confidence", dec.confidence if dec else 0.0)
    print(f"\nDiagnosis confidence:\n{_fmt_pct(conf)}")
    rem_auth = r.get("remediation_authorized", state.remediation_authorized if state else False)
    print(f"\nRemediation authorization:\n{'AUTHORIZED' if rem_auth else 'BLOCKED / HELD'}")

    print("\n---")
    print("\n## INTERVENTION")
    action_str = f"scale_service({pred_cause})"
    if state and state.remediation_action:
        act = state.remediation_action
        action_str = f"{act.action_type}({act.target_service})"
    print(f"\nAction:\n{action_str}")
    safety_res = "PASSED" if (rem_auth or (state and state.remediation_result and state.remediation_result.success)) else "BLOCKED"
    print(f"\nSafety:\n{safety_res}")

    print("\n---")
    print("\n## RECOVERY VERIFICATION")
    ver = state.verification_result if state else None
    if ver:
        target_rec = any(pred_cause in m for m in ver.recovered_metrics) if pred_cause else True
        print(f"\nRoot-cause symptoms:\n{'RECOVERED' if target_rec else 'UNRESOLVED'}")

        downstream_unres = any(pred_cause not in m for m in ver.unresolved_symptoms) if pred_cause else bool(ver.unresolved_symptoms)
        print(f"\nDownstream symptoms:\n{'RECOVERED' if not downstream_unres else 'UNRESOLVED'}")

        reg = "REGRESSION DETECTED" if ver.regression_detected else "NONE"
        print(f"\nNew regressions:\n{reg}")
    else:
        print("\nRoot-cause symptoms:\nNOT EVALUATED")
        print("\nDownstream symptoms:\nNOT EVALUATED")
        print("\nNew regressions:\nNONE")

    print("\n---")
    print("\n## POST-INTERVENTION CAUSAL VALIDATION")
    if state and state.intervention_validation:
        print(f"\n{state.intervention_validation.causal_support}")
    elif ver and ver.final_status == "RESOLVED":
        print("\nINTERVENTION_SUPPORTS_HYPOTHESIS")
    elif ver and ver.final_status == "NOT_RESOLVED":
        print("\nINTERVENTION_CONTRADICTS_HYPOTHESIS")
    else:
        print("\nINTERVENTION_INCONCLUSIVE")
    print("\n---\n")


def _print_case_result(case_id: str, results: list[dict[str, Any]], ground_truth: str) -> None:
    print(f"\n{_SEP}")
    print(f"  CASE: {case_id}")
    print(f"{_SEP}")

    for r in results:
        tag = r.get("model_name", r.get("mode", "?"))
        match = "[OK]" if r["predicted_cause"] == ground_truth.split(" ")[0] else "[X]"
        print(f"\n  [{tag}]")
        print(f"    Predicted cause         : {r['predicted_cause']} {match}")
        print(f"    Confidence              : {_fmt_pct(r['confidence'])}")
        if "turns" in r:
            print(f"    Turns                   : {r['turns']}")
        print(f"    Queries executed        : {r['queries']}")
        print(f"    Budget used             : {r['budget_used']}")
        print(f"    Evidence collected      : {r['evidence_count']}")
        print(f"    Causal consistency      : {_fmt_cc(r['cc'])}")
        print(f"    Remediation authorized  : {'YES' if r['remediation_authorized'] else 'NO'}")
        print(f"    Verification            : {r['verification_status']}")
        print(f"    Elapsed                 : {r['elapsed_sec']}s")
        if "trajectory" in r and hasattr(r["trajectory"], "format_timing_breakdown"):
            breakdown = r["trajectory"].format_timing_breakdown()
            if breakdown:
                print(f"\n{breakdown}")
        if "terminated_by" in r:
            print(f"    Terminated by           : {r['terminated_by']}")

        # Visibly print the structured sections:
        _print_structured_demo_report(r)

    # Ground truth revealed AFTER investigation (demo/eval layer only)
    print(f"\n  Ground truth (post-hoc display only): {ground_truth}")


# ---------------------------------------------------------------------------
# Multi-model comparison printer
# ---------------------------------------------------------------------------

def _print_comparison_table(all_results: list[dict[str, Any]]) -> None:
    """Print a compact comparison table across models and cases."""
    print(f"\n{_SEP}")
    print("  MODEL COMPARISON TABLE")
    print(_SEP)
    header = (
        f"{'Model':<30} {'Case':<38} {'Predicted':<22} {'Conf':>6} "
        f"{'Turns':>5} {'Queries':>7} {'Budget':>6} {'CC':>5} {'Correct':>7}"
    )
    print(header)
    print(_SEP_THIN)
    for r in all_results:
        gt = r.get("ground_truth_service", "?")
        correct = "YES" if r["predicted_cause"] == gt else "NO"
        cc_val = r["cc"].consistency_score if r.get("cc") else 0.0
        turns = r.get("turns", r.get("queries", "?"))
        print(
            f"  {r.get('model_name', '?'):<28} {r['case_id']:<38} "
            f"{r['predicted_cause']:<22} {r['confidence']:>6.1%} "
            f"{str(turns):>5} {r['queries']:>7} {r['budget_used']:>6} "
            f"{cc_val:>5.2f} {correct:>7}"
        )
    print(_SEP)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Digital Detective Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["mock", "ollama", "grok", "openai", "auto"],
        default="auto",
        help="Agent model backend (default: auto -> Ollama if available, else mock)",
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "det", "both"],
        default="agent",
        help="Output mode: agent only, deterministic only, or both (default: agent)",
    )
    parser.add_argument(
        "--case",
        metavar="CASE_ID",
        default=None,
        help="Case ID to investigate (default: re2ob_checkoutservice_cpu_1)",
    )
    parser.add_argument(
        "--adaptive-query-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive modality-aware query selection (default: True)",
    )
    parser.add_argument(
        "--compare-heuristics",
        action="store_true",
        help="Run legacy heuristic (A) vs adaptive heuristic (B) comparison",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run all three default demo cases",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Run mock + ollama + grok (skipping unavailable) and compare",
    )
    parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Override the remediation safety gate (operator override)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full turn-by-turn agent trajectory",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Query budget (default: 20)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Maximum reasoning turns before agent terminates (default: 6)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens for Ollama model (default: 160 or $DIGITAL_DETECTIVE_OLLAMA_MAX_TOKENS)",
    )
    parser.add_argument(
        "--remediation-threshold",
        type=float,
        default=0.80,
        help="Minimum confidence for remediation authorization (default: 0.80)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Path to RCAEval dataset root (default: $RCAEVAL_DATASET_ROOT or ~/.cache/rcaeval_validation)",
    )
    args = parser.parse_args()

    # Resolve dataset root
    dataset_root = Path(
        args.dataset_root
        or os.environ.get("RCAEVAL_DATASET_ROOT", "")
        or os.path.expanduser("~/.cache/rcaeval_validation")
    )

    # Determine cases to run
    if args.all_cases or args.compare_heuristics:
        case_ids = list(DEFAULT_DEMO_CASES)
    elif args.case:
        case_ids = [args.case]
    else:
        case_ids = [DEFAULT_DEMO_CASES[0]]

    # --compare-heuristics: run Heuristic A (legacy) vs Heuristic B (adaptive)
    if args.compare_heuristics:
        _run_heuristics_comparison(
            case_ids=case_ids,
            dataset_root=dataset_root,
            budget=args.budget,
            remediation_threshold=args.remediation_threshold,
            allow_low_confidence=args.allow_low_confidence,
        )
        return

    # --compare-models: build a list of models to try
    if args.compare_models:
        _run_comparison(
            case_ids=case_ids,
            dataset_root=dataset_root,
            budget=args.budget,
            remediation_threshold=args.remediation_threshold,
            allow_low_confidence=args.allow_low_confidence,
            max_turns=args.max_turns,
            verbose=args.verbose,
        )
        return

    # Single model path
    try:
        agent = build_agent_model(mode=args.model, ollama_max_tokens=args.max_tokens)
    except (RuntimeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nTIP: Use --model mock to run without any external service.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{_SEP}")
    print(f"  Digital Detective -- Agent Demo")
    print(f"  Model : {getattr(agent, 'model_name', type(agent).__name__)}")
    print(f"  Mode  : {args.mode}  |  Budget: {args.budget}  |  Cases: {len(case_ids)}")
    print(_SEP)

    all_flat: list[dict[str, Any]] = []

    for case_id in case_ids:
        try:
            case = load_rcaeval_case(dataset_root, case_id)
        except Exception as exc:
            print(f"  [SKIP] Could not load case {case_id}: {exc}")
            continue

        gt_str = _ground_truth_str(case)
        gt_service = gt_str.split(" ")[0]
        results: list[dict[str, Any]] = []

        if args.mode in ("det", "both"):
            det_result = _run_deterministic(
                case=case,
                budget=args.budget,
                remediation_threshold=args.remediation_threshold,
                allow_low_confidence=args.allow_low_confidence,
            )
            det_result["case_id"] = case_id
            det_result["ground_truth_service"] = gt_service
            results.append(det_result)
            all_flat.append(det_result)

        if args.mode in ("agent", "both"):
            agent_result = _run_agent(
                case=case,
                agent=agent,
                budget=args.budget,
                remediation_threshold=args.remediation_threshold,
                allow_low_confidence=args.allow_low_confidence,
                max_turns=args.max_turns,
                verbose=args.verbose,
            )
            agent_result["case_id"] = case_id
            agent_result["ground_truth_service"] = gt_service
            results.append(agent_result)
            all_flat.append(agent_result)

        _print_case_result(case_id, results, gt_str)

    if len(case_ids) > 1 and all_flat:
        _print_comparison_table(all_flat)


def _run_comparison(
    case_ids: list[str],
    dataset_root: Path,
    budget: int,
    remediation_threshold: float,
    allow_low_confidence: bool,
    max_turns: int = 6,
    verbose: bool = False,
) -> None:
    """Run multiple models and print a comparison table."""
    print(f"\n{_SEP}")
    print("  Digital Detective -- Multi-Model Comparison")
    print(_SEP)

    # Build candidate model list
    candidates: list[tuple[str, AgentModel | None, str]] = []

    # Mock always works
    candidates.append(("mock", build_agent_model("mock"), "MockAgentModel"))

    # Ollama: try; skip if unavailable
    try:
        ollama = build_agent_model("ollama")
        candidates.append(("ollama", ollama, getattr(ollama, "model_name", "Ollama")))
    except RuntimeError as exc:
        print(f"  [SKIP Ollama] {exc}")

    # Grok: try; skip if no key
    try:
        grok = build_agent_model("grok")
        candidates.append(("grok", grok, getattr(grok, "model_name", "Grok")))
    except ValueError as exc:
        print(f"  [SKIP Grok] {exc}")

    print(f"  Active models: {[name for name, _, _ in candidates]}")
    print(_SEP)

    all_flat: list[dict[str, Any]] = []

    for case_id in case_ids:
        try:
            case = load_rcaeval_case(dataset_root, case_id)
        except Exception as exc:
            print(f"  [SKIP] Could not load case {case_id}: {exc}")
            continue

        gt_str = _ground_truth_str(case)
        gt_service = gt_str.split(" ")[0]

        for mode_name, agent, model_display in candidates:
            if agent is None:
                continue
            result = _run_agent(
                case=case,
                agent=agent,
                budget=budget,
                remediation_threshold=remediation_threshold,
                allow_low_confidence=allow_low_confidence,
                max_turns=max_turns,
                verbose=verbose,
            )
            result["case_id"] = case_id
            result["ground_truth_service"] = gt_service
            result["model_name"] = model_display
            all_flat.append(result)

    if all_flat:
        _print_comparison_table(all_flat)


def _run_heuristics_comparison(
    case_ids: list[str],
    dataset_root: Path,
    budget: int,
    remediation_threshold: float,
    allow_low_confidence: bool,
) -> None:
    """Run Heuristic A (legacy fixed priority) vs Heuristic B (adaptive modality-aware) across cases."""
    print(f"\n{_SEP}")
    print("  Digital Detective -- Query Heuristics Comparison (A vs B)")
    print("  Heuristic A: Legacy fixed-priority selection (adaptive_query_selection=False)")
    print("  Heuristic B: Adaptive modality-aware selection (adaptive_query_selection=True)")
    print(f"{_SEP}\n")

    results_table: list[dict[str, Any]] = []

    for case_id in case_ids:
        try:
            case = load_rcaeval_case(dataset_root, case_id)
        except Exception as exc:
            print(f"  [SKIP] Could not load case {case_id}: {exc}")
            continue

        gt_str = _ground_truth_str(case)
        gt_service = gt_str.split(" ")[0]

        # Run Heuristic A (Legacy: adaptive_query_selection=False)
        res_a = _run_deterministic(
            case=case,
            budget=budget,
            remediation_threshold=remediation_threshold,
            allow_low_confidence=allow_low_confidence,
            adaptive_query_selection=False,
        )
        res_a["heuristic"] = "Legacy (A)"
        res_a["model_name"] = "Deterministic [Legacy A]"
        res_a["case_id"] = case_id
        res_a["gt_service"] = gt_service
        results_table.append(res_a)

        # Run Heuristic B (Adaptive: adaptive_query_selection=True)
        res_b = _run_deterministic(
            case=case,
            budget=budget,
            remediation_threshold=remediation_threshold,
            allow_low_confidence=allow_low_confidence,
            adaptive_query_selection=True,
        )
        res_b["heuristic"] = "Adaptive (B)"
        res_b["model_name"] = "Deterministic [Adaptive B]"
        res_b["case_id"] = case_id
        res_b["gt_service"] = gt_service
        results_table.append(res_b)

        _print_case_result(case_id, [res_a, res_b], gt_str)

    _print_heuristics_summary_table(results_table)


def _print_heuristics_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a clean comparative table reporting all required metrics for Heuristics A vs B."""
    print(f"\n{_SEP}")
    print("  QUERY HEURISTICS COMPARISON SUMMARY (A: Legacy vs B: Adaptive)")
    print(_SEP)
    header = (
        f"{'Case':<32} {'Heuristic':<14} {'Root Cause':<18} {'Top@1':<6} "
        f"{'Queries':>7} {'Budget':>7} {'Evidence':>8} {'Runtime':>9} "
        f"{'Abstention':<12} {'Recovery':<12} {'Regression':<10}"
    )
    print(header)
    print(_SEP_THIN)

    for r in results:
        cid = r.get("case_id", "?")
        heur = r.get("heuristic", "?")
        pred = r.get("predicted_cause", "?")
        gt = r.get("gt_service", "?")
        top1 = "YES" if pred == gt else "NO"
        queries = r.get("queries", 0)
        budget = r.get("budget_used", 0)
        ev_count = r.get("evidence_count", 0)
        elapsed = f"{r.get('elapsed_sec', 0.0):.3f}s"

        state = r.get("state")
        if state and state.remediation_authorized:
            abstention = "NO"
        elif pred in ("", "none", "unknown"):
            abstention = "YES"
        else:
            abstention = "HELD"

        if state and state.intervention_validation:
            recovery = "RECOVERED" if state.intervention_validation.recovery_verified else "UNRESOLVED"
            regression = "DETECTED" if state.intervention_validation.regression_detected else "NONE"
        elif state and state.verification_result:
            recovery = state.verification_result.root_cause_status
            regression = "DETECTED" if state.verification_result.regression_detected else "NONE"
        else:
            recovery = "N/A"
            regression = "N/A"

        print(
            f"  {cid:<30} {heur:<14} {pred:<18} {top1:<6} "
            f"{queries:>7} {budget:>7} {ev_count:>8} {elapsed:>9} "
            f"{abstention:<12} {recovery:<12} {regression:<10}"
        )

    print(_SEP + "\n")


if __name__ == "__main__":
    main()
