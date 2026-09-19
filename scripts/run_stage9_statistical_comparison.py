#!/usr/bin/env python3
"""Execute the formal paired statistical comparison for Stage 9: Sequential TCEC vs Stage 5 Control TCEC.

Methodology:
- Population: RE2-OB Repetitions 2 and 3 (N=60, 30 scenario families, 2 executions per family).
- Scenario family as inferential cluster (N=30).
- Comparison: Sequential TCEC (treatment) vs Stage 5 TCEC (control).
- Primary endpoint: False-Early Rate.
- Secondary endpoints: Detection Delay, S_comb MRR, Fixed Fusion MRR, S_comb Top@1, Top@3, Top@5.
- Paired cluster-level randomization test (10,000 permutations, seed=42).
- Clustered bootstrap confidence intervals (10,000 resamples, seed=42, Type-7 percentile CI).
- Holm-Bonferroni correction across evaluated endpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from eval.manifest import BenchmarkManifest
from eval.stats import (
    PairedCaseObservation,
    apply_holm_correction,
    cluster_bootstrap_ci,
    paired_cluster_randomization_test,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    treatment_path = repo_root / "eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json"
    control_path = repo_root / "eval/results/stage5_tcec_treatment_v1.json"
    manifest_path = repo_root / "eval/manifests/re2_ob_all_cases.json"

    if not treatment_path.exists():
        print(f"Error: Treatment artifact {treatment_path} does not exist. Run treatment script first.")
        sys.exit(1)
    if not control_path.exists():
        print(f"Error: Control artifact {control_path} does not exist.")
        sys.exit(1)

    treatment_hash = compute_file_sha256(treatment_path)
    control_hash = compute_file_sha256(control_path)
    manifest = BenchmarkManifest.load(manifest_path)
    manifest_hash = manifest.compute_hash()

    with open(treatment_path, "r", encoding="utf-8") as f:
        treatment_data = json.load(f)
    with open(control_path, "r", encoding="utf-8") as f:
        control_data = json.load(f)

    treat_cases = {c["case_id"]: c for c in treatment_data["per_case_results"]}
    ctrl_cases = {c["case_id"]: c for c in control_data["per_case_results"]}

    assert len(treat_cases) == 60, f"Expected 60 treatment cases, got {len(treat_cases)}"
    assert len(ctrl_cases) == 60, f"Expected 60 control cases, got {len(ctrl_cases)}"

    print("=" * 90)
    print("STAGE 9 STATISTICAL COMPARISON: SEQUENTIAL TCEC vs STAGE 5 CONTROL TCEC")
    print("=" * 90)
    print(f"Manifest Hash:        {manifest_hash}")
    print(f"Control Hash:         {control_hash}")
    print(f"Treatment Hash:       {treatment_hash}")
    print(f"Total Cases:          {len(treat_cases)}")

    # 1. Prepare paired observations
    fe_obs: list[PairedCaseObservation] = []
    delay_obs: list[PairedCaseObservation] = []
    s_mrr_obs: list[PairedCaseObservation] = []
    s_top1_obs: list[PairedCaseObservation] = []
    s_top3_obs: list[PairedCaseObservation] = []
    s_top5_obs: list[PairedCaseObservation] = []
    f_mrr_obs: list[PairedCaseObservation] = []

    for mc in manifest.cases:
        cid = mc.case_id
        fam = tuple(mc.scenario_family)
        c_treat = treat_cases[cid]
        c_ctrl = ctrl_cases[cid]

        # False-early (1.0 if confirmed and confirmed_onset < inject_time, else 0.0)
        fe_t = 1.0 if c_treat.get("is_false_early") else 0.0
        fe_c = 1.0 if c_ctrl.get("is_false_early") else 0.0
        fe_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=fe_t, value_b=fe_c))

        # Confirmed delay (confirmed_onset - inject_time, in seconds)
        # For unconfirmed cases, use 0 or leave out? In Stage 5, delay was computed for confirmed cases.
        d_t = float(c_treat["confirmed_delay_sec"]) if c_treat.get("confirmed_delay_sec") is not None else 0.0
        d_c = float(c_ctrl["confirmed_delay_sec"]) if c_ctrl.get("confirmed_delay_sec") is not None else 0.0
        delay_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=d_t, value_b=d_c))

        # S_comb RCA metrics
        s_mrr_t = float(c_treat["methods"]["s_comb"]["mrr"])
        s_mrr_c = float(c_ctrl["methods"]["s_comb"]["mrr"])
        s_mrr_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=s_mrr_t, value_b=s_mrr_c))

        s_t1_t = 1.0 if c_treat["methods"]["s_comb"]["top1"] else 0.0
        s_t1_c = 1.0 if c_ctrl["methods"]["s_comb"]["top1"] else 0.0
        s_top1_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=s_t1_t, value_b=s_t1_c))

        s_t3_t = 1.0 if c_treat["methods"]["s_comb"]["top3"] else 0.0
        s_t3_c = 1.0 if c_ctrl["methods"]["s_comb"]["top3"] else 0.0
        s_top3_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=s_t3_t, value_b=s_t3_c))

        s_t5_t = 1.0 if c_treat["methods"]["s_comb"]["top5"] else 0.0
        s_t5_c = 1.0 if c_ctrl["methods"]["s_comb"]["top5"] else 0.0
        s_top5_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=s_t5_t, value_b=s_t5_c))

        # Fixed fusion MRR
        f_mrr_t = float(c_treat["methods"]["fixed_equal_weight_fusion"]["mrr"])
        f_mrr_c = float(c_ctrl["methods"]["fixed_equal_weight_fusion"]["mrr"])
        f_mrr_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=f_mrr_t, value_b=f_mrr_c))

    endpoints = [
        ("False-Early Rate", fe_obs, True, "PRIMARY"),
        ("Confirmed Delay (s)", delay_obs, False, "SECONDARY"),
        ("S_comb MRR", s_mrr_obs, False, "SECONDARY"),
        ("S_comb Top@1", s_top1_obs, True, "SECONDARY"),
        ("S_comb Top@3", s_top3_obs, True, "SECONDARY"),
        ("S_comb Top@5", s_top5_obs, True, "SECONDARY"),
        ("Fixed Fusion MRR", f_mrr_obs, False, "SECONDARY"),
    ]

    results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for name, obs, is_pct, role in endpoints:
        diff_obs, p_val = paired_cluster_randomization_test(obs, randomization_replicates=10000, seed=42)
        _, ci_lower, ci_upper = cluster_bootstrap_ci(obs, bootstrap_replicates=10000, seed=42)
        mean_treat = sum(o.value_a for o in obs) / len(obs)
        mean_ctrl = sum(o.value_b for o in obs) / len(obs)
        raw_p_values.append(p_val)

        results.append({
            "metric": name,
            "role": role,
            "mean_treatment": mean_treat,
            "mean_control": mean_ctrl,
            "difference": diff_obs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "raw_p_value": p_val,
            "is_percentage": is_pct,
        })

    adjusted_p_values = apply_holm_correction(raw_p_values)

    print("\nStatistical Analysis Results (Cluster-Level Permutation & Bootstrap N=10,000):")
    print("-" * 90)
    print(f"{'Metric':<22} | {'Treatment':<11} | {'Control':<11} | {'Delta':<10} | {'95% Clustered CI':<20} | {'Raw p':<8} | {'Adj p':<8}")
    print("-" * 90)

    for res, adj_p in zip(results, adjusted_p_values):
        res["adjusted_p_value"] = adj_p
        res["significant_raw"] = res["raw_p_value"] < 0.05
        res["significant_adjusted"] = adj_p < 0.05

        fmt = "{:.1%}" if res["is_percentage"] else "{:.2f}"
        diff_fmt = "{:+.1%}" if res["is_percentage"] else "{:+.2f}"

        val_treat_str = fmt.format(res["mean_treatment"])
        val_ctrl_str = fmt.format(res["mean_control"])
        diff_str = diff_fmt.format(res["difference"])
        ci_str = f"[{diff_fmt.format(res['ci_lower'])}, {diff_fmt.format(res['ci_upper'])}]"

        print(f"{res['metric']:<22} | {val_treat_str:<11} | {val_ctrl_str:<11} | {diff_str:<10} | {ci_str:<20} | {res['raw_p_value']:<8.4f} | {adj_p:<8.4f}")

    print("-" * 90)

    fe_res = next(r for r in results if r["metric"] == "False-Early Rate")
    # For False-Early, a reduction (delta < 0) is an improvement
    if fe_res["ci_upper"] < 0:
        decision = "SUPPORT_FALSE_EARLY_REDUCTION"
        conclusion = "95% clustered bootstrap CI for Delta False-Early is entirely below zero -> Statistically significant reduction in premature confirmations."
    elif fe_res["ci_lower"] > 0:
        decision = "FALSIFY_HYPOTHESIS"
        conclusion = "95% clustered bootstrap CI for Delta False-Early is entirely above zero -> False-early rate worsened."
    else:
        decision = "INCONCLUSIVE"
        conclusion = "95% clustered bootstrap CI for Delta False-Early crosses zero -> Inconclusive."

    print(f"\nPrimary Hypothesis Decision: {decision}")
    print(f"Scientific Conclusion:        {conclusion}")

    out_artifact = {
        "schema_version": "stage9_sequential_tcec_statistical_comparison_v1",
        "experiment_id": "stage9_re2ob_sequential_tcec_statistical_comparison_v1",
        "treatment_artifact": "eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json",
        "control_artifact": "eval/results/stage5_tcec_treatment_v1.json",
        "treatment_hash": treatment_hash,
        "control_hash": control_hash,
        "manifest_hash": manifest_hash,
        "population": {
            "dataset": "RE2-OB",
            "total_cases": 60,
            "scenario_families": 30,
            "repetitions": [2, 3],
        },
        "hypothesis_decision": decision,
        "scientific_conclusion": conclusion,
        "comparison_results": results,
    }

    out_file = repo_root / "eval/results/stage9_sequential_tcec_statistical_comparison_v1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)

    stat_hash = compute_file_sha256(out_file)
    print(f"Artifact Saved:               {out_file}")
    print(f"Statistical Artifact SHA-256: {stat_hash}")
    print("=" * 90)


if __name__ == "__main__":
    main()
