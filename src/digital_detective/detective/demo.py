"""Runnable demonstration scenario for Digital Detective autonomous investigation engine.

Demonstrates the full end-to-end incident lifecycle:
Incident intake
-> Closed candidate universe
-> Initial hypothesis ranking
-> Query-budget-constrained autonomous tool queries
-> Evidence fusion & confidence convergence
-> Causal-consistency validation
-> Root-cause decision
-> Safe remediation proposal & safety gating
-> Simulated execution
-> Multi-symptom recovery verification & final incident status

Run with:
    python -m digital_detective.detective.demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

from digital_detective.rcaeval import load_rcaeval_case
from .investigator import InvestigationEngine


DEFAULT_DEMO_CASES = (
    "re2ob_checkoutservice_cpu_1",
    "re2ob_currencyservice_delay_1",
    "re2ob_checkoutservice_mem_1",
)


def run_single_case_demo(
    case_id: str,
    root_path: Path,
    budget: int = 20,
    remediation_threshold: float = 0.80,
    allow_low_confidence: bool = False,
) -> dict[str, Any]:
    print("=" * 72)
    print(f"DIGITAL DETECTIVE DEMO: {case_id}")
    print("=" * 72)

    # 1. Intake
    t_load_start = time.perf_counter()
    case = load_rcaeval_case(root_path, case_id)
    t_load = time.perf_counter() - t_load_start

    # Read ground truth ONLY for post-run demonstration comparison
    gt_vals = getattr(case.ground_truth, "values", {})
    ground_truth_target = str(gt_vals.get("root_cause_service") or gt_vals.get("target") or "unknown")
    fault_type = str(gt_vals.get("fault") or gt_vals.get("fault_type") or "unknown")
    inject_time = gt_vals.get("inject_time")

    print(f"\n[1] INCIDENT INTAKE")
    print(f"  System:               {case.metadata.system} ({case.metadata.system_name or 'Online Boutique'})")
    print(f"  Fault Type:           {fault_type}")
    print(f"  Ground Truth Target:  {ground_truth_target} (DEMO DISPLAY ONLY — NOT provided to engine)")
    print(f"  Injection Time (t):   {inject_time}")
    print(f"  Telemetry Modalities: Metrics: {case.metrics is not None}, Traces: {case.traces is not None}, Logs: {case.logs is not None}")
    print(f"  Load Time:            {t_load:.3f}s")

    # 2. Engine Invocation
    engine = InvestigationEngine(budget=budget, confidence_threshold=remediation_threshold)

    print(f"\n[2] INVESTIGATION CONFIGURATION")
    print(f"  Query Budget:              {budget} units")
    print(f"  Remediation Threshold:     {remediation_threshold:.0%}")
    print(f"  Allow Low-Confidence Flag: {allow_low_confidence}")

    t_inv_start = time.perf_counter()
    state = engine.investigate(
        case=case,
        window_mode="oracle",
        remediation_threshold=remediation_threshold,
        allow_low_confidence_remediation=allow_low_confidence,
    )
    t_inv = time.perf_counter() - t_inv_start

    print(f"  Candidate Universe:        ({len(state.candidate_universe)} services)")
    print(f"    {', '.join(state.candidate_universe)}")

    # 3. Autonomous Exploration Log
    print(f"\n[3] AUTONOMOUS EXPLORATION LOG (Under Query Budget)")
    print(f"  Total queries executed: {len(state.queries_executed)} | Budget used: {state.total_query_cost}/{state.budget}")
    print("  " + "-" * 68)
    for idx, q in enumerate(state.queries_executed[:6], start=1):
        print(f"  Step {idx:02d} | Tool: {q.tool_name:<18} | Target: {q.service:<18} | Cost: {q.cost} | Status: {q.status}")
        if q.rationale:
            print(f"          Rationale: {q.rationale}")
        print(f"          Summary:   {q.result_summary}")
    if len(state.queries_executed) > 6:
        print(f"  ... [{len(state.queries_executed) - 6} additional queries executed within budget] ...")

    # 4. Evidence Fusion & Decision
    print(f"\n[4] EVIDENCE FUSION & ROOT-CAUSE DECISION")
    predicted_cause = "none"
    confidence = 0.0
    if state.decision is not None:
        dec = state.decision
        predicted_cause = dec.root_cause_service
        confidence = dec.confidence
        is_hit = (predicted_cause == ground_truth_target)
        print(f"  Predicted Root Cause:  '{predicted_cause}' {'[CORRECT MATCH]' if is_hit else '[INCORRECT]'}")
        print(f"  Decision Confidence:   {confidence:.1%}")
        print(f"  Remaining Budget:      {dec.budget_remaining} units")
        print("\n  Top 3 Ranked Hypotheses:")
        for h in dec.ranked_candidates[:3]:
            mark = " <-- PREDICTED CAUSE" if h.service == dec.root_cause_service else ""
            print(
                f"    {h.rank}. {h.service:<22} Score: {h.score:.3f} (Conf: {h.confidence:.1%}, Ev: {h.evidence_score:.3f}, CC: {h.consistency_score:.2f}){mark}"
            )

    # 5. Causal-Consistency Audit
    print(f"\n[5] CAUSAL-CONSISTENCY AUDIT")
    cc_passed = False
    if state.decision is not None:
        cc = state.decision.causal_consistency
        cc_passed = cc.passed if cc else False
        if cc:
            print(f"  Target Service:        {cc.target_service}")
            print(f"  Temporal Ordering:     {cc.temporal_score:.2f}/1.00")
            print(f"  Topology Consistency:  {cc.topology_score:.2f}/1.00")
            print(f"  Propagation Matching:  {cc.propagation_score:.2f}/1.00")
            print(f"  Overall Consistency:   {cc.consistency_score:.2f}/1.00 -> {'PASSED' if cc.passed else 'FAILED'}")
            if cc.warnings:
                print("  Warnings:")
                for w in cc.warnings:
                    print(f"    ! {w}")

    # 6. Remediation Safety Layer
    print(f"\n[6] SAFE REMEDIATION LAYER")
    print(f"  Investigation Status:  {state.status}")
    print(f"  Remediation Auth:      {'AUTHORIZED' if state.remediation_authorized else 'BLOCKED / HELD'}")
    if state.remediation_action is not None and state.remediation_result is not None:
        act = state.remediation_action
        res = state.remediation_result
        print(f"  Proposed Action:       {act.action_type} on '{act.target_service}'")
        print(f"  Safety Gate Verdict:   {'PASSED' if res.success else 'BLOCKED'}")
        for line in res.execution_log[:2]:
            print(f"    {line}")

    # 7. Recovery Verification
    print(f"\n[7] POST-REMEDIATION RECOVERY VERIFICATION")
    ver_status = "NOT_EVALUATED"
    if state.verification_result is not None:
        ver = state.verification_result
        ver_status = ver.final_status
        print(f"  Final Status:          {ver_status}")
        print(f"  Regression Detected:   {ver.regression_detected}")
        print(f"  Recovered Metrics:     {len(ver.recovered_metrics)} metrics")
        print(f"  Unresolved Symptoms:   {len(ver.unresolved_symptoms)} symptoms")

    print(f"\n  Case Elapsed Time:     {t_inv:.3f}s\n")
    return {
        "case_id": case_id,
        "fault_type": fault_type,
        "ground_truth": ground_truth_target,
        "predicted_cause": predicted_cause,
        "hit": (predicted_cause == ground_truth_target),
        "confidence": confidence,
        "queries": len(state.queries_executed),
        "budget_used": state.total_query_cost,
        "cc_passed": cc_passed,
        "remediation_status": state.status,
        "remediation_authorized": state.remediation_authorized,
        "verification_status": ver_status,
        "elapsed_sec": round(t_inv, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Detective Autonomous Investigation Demo")
    parser.add_argument("--case", type=str, default="", help="Specific case ID to run")
    parser.add_argument("--all-cases", action="store_true", help="Run all 3 demo cases")
    parser.add_argument("--dataset-root", type=str, default="", help="Path to RCAEval validation cache")
    parser.add_argument("--budget", type=int, default=20, help="Investigation query budget")
    parser.add_argument("--threshold", type=float, default=0.80, help="Remediation authorization threshold")
    parser.add_argument("--allow-low-confidence", action="store_true", help="Explicitly authorize remediation below threshold")
    args = parser.parse_args()

    root_path = Path(args.dataset_root or os.environ.get("RCAEVAL_DATASET_ROOT") or os.path.expanduser("~/.cache/rcaeval_validation"))

    cases_to_run = [args.case] if args.case else list(DEFAULT_DEMO_CASES)

    results = []
    t_start = time.perf_counter()
    for cid in cases_to_run:
        res = run_single_case_demo(
            case_id=cid,
            root_path=root_path,
            budget=args.budget,
            remediation_threshold=args.threshold,
            allow_low_confidence=args.allow_low_confidence,
        )
        results.append(res)
    total_time = time.perf_counter() - t_start

    print("=" * 72)
    print("DEMO SUMMARY TABLE")
    print("=" * 72)
    print(f"{'Case ID':<32} {'Fault':<7} {'GroundTruth':<16} {'Predicted':<16} {'Conf':<7} {'Queries':<8} {'Auth':<12} {'Verified'}")
    print("-" * 115)
    for r in results:
        auth_str = "YES" if r["remediation_authorized"] else "NO (HELD)"
        print(
            f"{r['case_id']:<32} {r['fault_type']:<7} {r['ground_truth']:<16} {r['predicted_cause']:<16} "
            f"{r['confidence']:<6.1%} {r['queries']}/{r['budget_used']}     {auth_str:<12} {r['verification_status']}"
        )
    print("-" * 115)
    print(f"Total Demo Runtime: {total_time:.2f}s\n")


if __name__ == "__main__":
    main()
