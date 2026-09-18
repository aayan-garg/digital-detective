#!/usr/bin/env python3
"""Execute the formal paired statistical comparison between CRV Shadow Ranking and S_comb.

Scientific Protocol:
- Population: RE2-OB Repetitions 2 and 3 (N=60, 30 scenario families, 2 executions per family).
- Scenario family as inferential cluster (N=30).
- Comparison: CRV Shadow (treatment) vs S_comb (control).
- Paired cluster-level randomization test (10,000 permutations, seed=42).
- Clustered bootstrap confidence intervals (10,000 resamples, seed=42, Type-7 percentile CI).
- Holm-Bonferroni correction across evaluated metrics (Top@1, Top@3, Top@5, MRR).
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
    paired_cluster_randomization_test,
    cluster_bootstrap_ci,
    apply_holm_correction,
)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> None:
    treatment_path = repo_root / "eval/results/stage7_re2ob_crv_treatment_v1.json"
    manifest_path = repo_root / "eval/manifests/re2_ob_all_cases.json"

    if not treatment_path.exists():
        print(f"Error: Treatment artifact {treatment_path} does not exist. Run treatment script first.")
        sys.exit(1)

    treatment_hash = compute_file_sha256(treatment_path)
    manifest = BenchmarkManifest.load(manifest_path)
    manifest_hash = manifest.compute_hash()

    with open(treatment_path, "r", encoding="utf-8") as f:
        treatment_data = json.load(f)

    per_case = treatment_data["per_case_results"]
    assert len(per_case) == 60, f"Expected 60 cases, got {len(per_case)}"

    print("=" * 85)
    print("STAGE 7 STATISTICAL COMPARISON: CRV SHADOW vs DIGITAL DETECTIVE S_COMB")
    print("=" * 85)
    print(f"Manifest Hash:        {manifest_hash}")
    print(f"Treatment Hash:       {treatment_hash}")
    print(f"Total Cases:          {len(per_case)}")

    top1_obs = []
    top3_obs = []
    top5_obs = []
    mrr_obs = []

    for c in per_case:
        cid = c["case_id"]
        fam = tuple(c["scenario_family"])
        m_crv = c["methods"]["crv_shadow"]
        m_sc = c["methods"]["s_comb"]

        top1_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=1.0 if m_crv["top1"] else 0.0, value_b=1.0 if m_sc["top1"] else 0.0))
        top3_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=1.0 if m_crv["top3"] else 0.0, value_b=1.0 if m_sc["top3"] else 0.0))
        top5_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=1.0 if m_crv["top5"] else 0.0, value_b=1.0 if m_sc["top5"] else 0.0))
        mrr_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=float(m_crv["mrr"]), value_b=float(m_sc["mrr"])))

    metrics_to_test = [
        ("Top@1 Accuracy", top1_obs, True),
        ("Top@3 Accuracy", top3_obs, True),
        ("Top@5 Accuracy", top5_obs, True),
        ("MRR", mrr_obs, False),
    ]

    results = []
    raw_p_values = []

    for name, obs, is_pct in metrics_to_test:
        diff_obs, p_val = paired_cluster_randomization_test(obs, randomization_replicates=10000, seed=42)
        _, ci_lower, ci_upper = cluster_bootstrap_ci(obs, bootstrap_replicates=10000, seed=42)
        mean_crv = sum(o.value_a for o in obs) / len(obs)
        mean_sc = sum(o.value_b for o in obs) / len(obs)
        raw_p_values.append(p_val)

        results.append({
            "metric": name,
            "mean_crv_shadow": mean_crv,
            "mean_s_comb": mean_sc,
            "difference": diff_obs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "raw_p_value": p_val,
            "is_percentage": is_pct,
        })

    # Holm-Bonferroni correction
    adjusted_p_values = apply_holm_correction(raw_p_values)

    print("\nStatistical Analysis Results (Cluster-Level Permutation & Bootstrap N=10,000):")
    print("-" * 85)
    print(f"{'Metric':<18} | {'CRV Shadow':<11} | {'S_comb':<11} | {'Delta':<10} | {'95% Clustered CI':<20} | {'Raw p':<8} | {'Adj p':<8}")
    print("-" * 85)

    for res, adj_p in zip(results, adjusted_p_values):
        res["adjusted_p_value"] = adj_p
        res["significant_raw"] = res["raw_p_value"] < 0.05
        res["significant_adjusted"] = adj_p < 0.05

        fmt = "{:.1%}" if res["is_percentage"] else "{:.4f}"
        diff_fmt = "{:+.1%}" if res["is_percentage"] else "{:+.4f}"

        val_crv_str = fmt.format(res["mean_crv_shadow"])
        val_sc_str = fmt.format(res["mean_s_comb"])
        diff_str = diff_fmt.format(res["difference"])
        ci_str = f"[{diff_fmt.format(res['ci_lower'])}, {diff_fmt.format(res['ci_upper'])}]"

        print(f"{res['metric']:<18} | {val_crv_str:<11} | {val_sc_str:<11} | {diff_str:<10} | {ci_str:<20} | {res['raw_p_value']:<8.4f} | {adj_p:<8.4f}")

    print("-" * 85)

    # Scientific hypothesis test evaluation
    mrr_res = next(r for r in results if r["metric"] == "MRR")
    mrr_ci_low = mrr_res["ci_lower"]
    mrr_ci_high = mrr_res["ci_upper"]

    if mrr_ci_low > 0:
        decision = "SUPPORT_H1"
        conclusion = "Clustered 95% CI for Delta MRR is entirely above zero -> Supports H1 (CRV provides incremental discriminatory information)."
    elif mrr_ci_high < 0:
        decision = "FALSIFY_H1"
        conclusion = "Clustered 95% CI for Delta MRR is entirely below zero -> Falsifies H1 (CRV significantly degrades ranking quality)."
    else:
        decision = "INCONCLUSIVE"
        conclusion = "Clustered 95% CI for Delta MRR crosses zero -> Inconclusive (no statistically significant incremental discriminatory information)."

    print(f"\nPrimary Hypothesis Decision: {decision}")
    print(f"Scientific Conclusion:        {conclusion}")

    out_artifact = {
        "schema_version": "stage7_statistical_comparison_v1",
        "experiment_id": "stage7_re2ob_crv_shadow_statistical_comparison_v1",
        "treatment_artifact": "eval/results/stage7_re2ob_crv_treatment_v1.json",
        "treatment_hash": treatment_hash,
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
        "crv_summary": treatment_data.get("crv_summary", {}),
    }

    out_file = repo_root / "eval/results/stage7_crv_statistical_comparison_v1.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_artifact, f, indent=2)

    stat_hash = compute_file_sha256(out_file)
    print(f"Artifact Saved:               {out_file}")
    print(f"Statistical Artifact SHA-256: {stat_hash}")
    print("=" * 85)


if __name__ == "__main__":
    main()
