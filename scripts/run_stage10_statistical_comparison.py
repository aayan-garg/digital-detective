#!/usr/bin/env python3
"""Execute the formal paired statistical comparison for Stage 10: Propagation Consistency vs Frozen Stage 9 Control.

Methodology:
- Population: RE2-OB Repetitions 2 and 3 (N=60, 30 scenario families, 2 executions per family).
- Scenario family as inferential cluster (N=30).
- Comparison: Temporal-Topological Propagation Consistency (treatment) vs Frozen Stage-9 S_fusion (control).
- Primary endpoints: Top@1, Top@3, Top@5, MRR.
- Secondary endpoints: Avg@5, True Root Promoted, Downstream Symptom Demoted, Unrelated Promoted.
- Paired cluster-level randomization test (10,000 permutations, seed=42).
- Clustered bootstrap confidence intervals (10,000 resamples, seed=42, Type-7 percentile CI).
- Holm-Bonferroni correction across primary endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
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
    treatment_path = repo_root / "eval/results/stage10_re2ob_propagation_consistency_treatment_v1.json"
    manifest_path = repo_root / "eval/manifests/re2_ob_all_cases.json"

    if not treatment_path.exists():
        print(f"Error: Treatment artifact {treatment_path} does not exist.")
        sys.exit(1)

    treatment_hash = compute_file_sha256(treatment_path)
    manifest = BenchmarkManifest.load(manifest_path)
    manifest_hash = manifest.compute_hash()

    with open(treatment_path, "r", encoding="utf-8") as f:
        treatment_data = json.load(f)

    treat_cases = {c["case_id"]: c for c in treatment_data["per_case_results"]}
    assert len(treat_cases) == 60, f"Expected 60 cases, got {len(treat_cases)}"

    print("=" * 90)
    print("STAGE 10 STATISTICAL COMPARISON: PROPAGATION CONSISTENCY vs STAGE 9 FROZEN CONTROL")
    print("=" * 90)
    print(f"Manifest Hash:        {manifest_hash}")
    print(f"Treatment Hash:       {treatment_hash}")
    print(f"Total Cases:          {len(treat_cases)}")

    # 1. Prepare paired observations for each endpoint
    top1_obs: list[PairedCaseObservation] = []
    top3_obs: list[PairedCaseObservation] = []
    top5_obs: list[PairedCaseObservation] = []
    mrr_obs: list[PairedCaseObservation] = []
    avg5_obs: list[PairedCaseObservation] = []

    family_consistency: dict[tuple[str, ...], dict[str, float]] = {}

    for mc in manifest.cases:
        cid = mc.case_id
        fam = tuple(mc.scenario_family)
        c = treat_cases[cid]
        m = c["metrics"]

        ctrl_m = m["control"]
        treat_m = m["treatment"]

        top1_obs.append(
            PairedCaseObservation(
                case_id=cid,
                scenario_family=fam,
                value_a=1.0 if treat_m["top1"] else 0.0,
                value_b=1.0 if ctrl_m["top1"] else 0.0,
            )
        )
        top3_obs.append(
            PairedCaseObservation(
                case_id=cid,
                scenario_family=fam,
                value_a=1.0 if treat_m["top3"] else 0.0,
                value_b=1.0 if ctrl_m["top3"] else 0.0,
            )
        )
        top5_obs.append(
            PairedCaseObservation(
                case_id=cid,
                scenario_family=fam,
                value_a=1.0 if treat_m["top5"] else 0.0,
                value_b=1.0 if ctrl_m["top5"] else 0.0,
            )
        )
        mrr_obs.append(
            PairedCaseObservation(
                case_id=cid,
                scenario_family=fam,
                value_a=float(treat_m["mrr"]),
                value_b=float(ctrl_m["mrr"]),
            )
        )
        avg5_obs.append(
            PairedCaseObservation(
                case_id=cid,
                scenario_family=fam,
                value_a=float(treat_m["avg5"]),
                value_b=float(ctrl_m["avg5"]),
            )
        )

        if fam not in family_consistency:
            family_consistency[fam] = {"diff_mrr_sum": 0.0, "count": 0}
        family_consistency[fam]["diff_mrr_sum"] += float(treat_m["mrr"]) - float(ctrl_m["mrr"])
        family_consistency[fam]["count"] += 1

    # 2. Run statistical tests
    primary_endpoints = [
        ("Top@1", top1_obs),
        ("Top@3", top3_obs),
        ("Top@5", top5_obs),
        ("MRR", mrr_obs),
    ]

    primary_results = []
    p_values_to_correct = []

    for name, obs in primary_endpoints:
        diff_obs, p_val = paired_cluster_randomization_test(obs, randomization_replicates=10000, seed=42)
        _, ci_lower, ci_upper = cluster_bootstrap_ci(obs, bootstrap_replicates=10000, seed=42)
        mean_treat = sum(o.value_a for o in obs) / len(obs)
        mean_ctrl = sum(o.value_b for o in obs) / len(obs)
        p_values_to_correct.append(p_val)
        primary_results.append(
            {
                "name": name,
                "obs": obs,
                "diff": diff_obs,
                "p_val": p_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mean_treat": mean_treat,
                "mean_ctrl": mean_ctrl,
            }
        )

    corrected_p_values = apply_holm_correction(p_values_to_correct)

    # 3. Print primary comparison table
    print("\n" + "=" * 90)
    print("PRIMARY ENDPOINTS (Treatment vs Control)")
    print("=" * 90)
    print(
        f"{'Endpoint':<10} | {'Treatment':<10} | {'Control':<10} | {'Diff':<9} | "
        f"{'95% Clustered CI':<22} | {'Raw p':<9} | {'Holm p':<9}"
    )
    print("-" * 90)

    endpoint_records = []
    for res, p_corr in zip(primary_results, corrected_p_values):
        name = res["name"]
        ci_str = f"[{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]"
        print(
            f"{name:<10} | {res['mean_treat']:<10.4f} | {res['mean_ctrl']:<10.4f} | {res['diff']:+.4f}   | "
            f"{ci_str:<22} | {res['p_val']:<9.4f} | {p_corr:<9.4f}"
        )
        endpoint_records.append(
            {
                "endpoint": name,
                "treatment_mean": round(res["mean_treat"], 4),
                "control_mean": round(res["mean_ctrl"], 4),
                "diff": round(res["diff"], 4),
                "ci_lower": round(res["ci_lower"], 4),
                "ci_upper": round(res["ci_upper"], 4),
                "raw_p_value": round(res["p_val"], 4),
                "holm_p_value": round(p_corr, 4),
                "significant": p_corr < 0.05,
            }
        )

    # Secondary: Avg@5
    diff_avg5, p_avg5 = paired_cluster_randomization_test(avg5_obs, randomization_replicates=10000, seed=42)
    _, ci_lower_avg5, ci_upper_avg5 = cluster_bootstrap_ci(avg5_obs, bootstrap_replicates=10000, seed=42)
    mean_treat_avg5 = sum(o.value_a for o in avg5_obs) / len(avg5_obs)
    mean_ctrl_avg5 = sum(o.value_b for o in avg5_obs) / len(avg5_obs)
    print("\nSECONDARY ENDPOINT:")
    print(
        f"Avg@5: Treat={mean_treat_avg5:.4f}, Ctrl={mean_ctrl_avg5:.4f}, Diff={diff_avg5:+.4f}, "
        f"95% CI=[{ci_lower_avg5:+.4f}, {ci_upper_avg5:+.4f}], p={p_avg5:.4f}"
    )

    # 4. Family-level consistency analysis
    fam_improved = sum(1 for v in family_consistency.values() if v["diff_mrr_sum"] > 0)
    fam_degraded = sum(1 for v in family_consistency.values() if v["diff_mrr_sum"] < 0)
    fam_tied = sum(1 for v in family_consistency.values() if v["diff_mrr_sum"] == 0)

    print("\n" + "=" * 90)
    print("FAMILY-LEVEL CONSISTENCY (N=30 families)")
    print("=" * 90)
    print(f"  Families with Treatment MRR > Control MRR: {fam_improved:2d} / 30 ({fam_improved/30*100:.1f}%)")
    print(f"  Families with Treatment MRR < Control MRR: {fam_degraded:2d} / 30 ({fam_degraded/30*100:.1f}%)")
    print(f"  Families with Treatment MRR == Control MRR:{fam_tied:2d} / 30 ({fam_tied/30*100:.1f}%)")

    # 5. Ranking shift analysis
    shifts = treatment_data["shifts_summary"]
    print("\n" + "=" * 90)
    print("RANKING SHIFTS SUMMARY")
    print("=" * 90)
    print(f"  True root promoted:       {shifts['true_root_promoted']} / 60 ({shifts['true_root_promoted']/60*100:.1f}%)")
    print(f"  True root demoted:        {shifts['true_root_demoted']} / 60 ({shifts['true_root_demoted']/60*100:.1f}%)")
    print(f"  True root unchanged:      {shifts['true_root_unchanged']} / 60 ({shifts['true_root_unchanged']/60*100:.1f}%)")
    print(f"  Winner changed:           {shifts['winner_changed']} / 60 ({shifts['winner_changed']/60*100:.1f}%)")
    print(f"  Downstream demoted:       {shifts['downstream_symptom_demoted']} / 60 ({shifts['downstream_symptom_demoted']/60*100:.1f}%)")
    print(f"  Unrelated promoted:       {shifts['unrelated_candidate_promoted']} / 60 ({shifts['unrelated_candidate_promoted']/60*100:.1f}%)")

    # 6. Evaluation Verdict
    # Success requires:
    # 1. At least one primary RCA metric improves under the existing corrected paired clustered analysis.
    # 2. No primary metric significantly degrades.
    # 3. Ranking-change analysis shows improvement is primarily true-root promotion and/or downstream-symptom suppression.
    # 4. No causal leakage.
    # 5. No evaluation-derived tuning.
    has_significant_improvement = any(
        r["significant"] and r["diff"] > 0 for r in endpoint_records
    )
    has_significant_degradation = any(
        r["significant"] and r["diff"] < 0 for r in endpoint_records
    )

    if has_significant_improvement and not has_significant_degradation:
        decision = "SUCCESS"
    elif has_significant_degradation or all(r["diff"] <= 0 for r in endpoint_records):
        decision = "FAILURE"
    else:
        decision = "NEUTRAL / INCONCLUSIVE"

    print("\n" + "=" * 90)
    print(f"EXPERIMENTAL VERDICT: {decision}")
    print("=" * 90)

    # Save output comparison artifact
    comp_artifact = {
        "schema_version": "1.0",
        "comparison_id": "stage10_propagation_consistency_statistical_comparison_v1",
        "treatment_artifact": str(treatment_path.name),
        "treatment_hash": treatment_hash,
        "manifest_hash": manifest_hash,
        "methodology": "paired_cluster_randomization_and_bootstrap_ci_10k_seed42",
        "primary_endpoints": endpoint_records,
        "secondary_endpoints": {
            "avg5": {
                "treatment_mean": round(mean_treat_avg5, 4),
                "control_mean": round(mean_ctrl_avg5, 4),
                "diff": round(diff_avg5, 4),
                "ci_lower": round(ci_lower_avg5, 4),
                "ci_upper": round(ci_upper_avg5, 4),
                "raw_p_value": round(p_avg5, 4),
            },
        },
        "family_consistency": {
            "improved": fam_improved,
            "degraded": fam_degraded,
            "tied": fam_tied,
        },
        "shifts_summary": shifts,
        "verdict": decision,
    }

    comp_path = repo_root / "eval/results/stage10_propagation_consistency_statistical_comparison_v1.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_artifact, f, indent=2)

    comp_hash = compute_file_sha256(comp_path)
    print(f"Saved statistical comparison artifact to: {comp_path}")
    print(f"Artifact SHA-256:                         {comp_hash}")


if __name__ == "__main__":
    main()
