#!/usr/bin/env python3
"""Execute the formal paired statistical comparison between PyRCA HT and S_comb.

Methodology:
- Population: RE2-OB Repetitions 2 and 3 (N=60, 30 scenario families, 2 executions per family).
- Scenario family as inferential cluster (N=30).
- Comparison: PyRCA HT (treatment) vs S_comb (control).
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
    treatment_path = repo_root / "eval/results/stage6_re2ob_pyrca_ht_treatment_v1.json"
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
    print("STAGE 6 STATISTICAL COMPARISON: PyRCA HT vs DIGITAL DETECTIVE S_COMB")
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
        m_ht = c["methods"]["pyrca_ht"]
        m_sc = c["methods"]["s_comb"]

        top1_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=1.0 if m_ht["top1"] else 0.0, value_b=1.0 if m_sc["top1"] else 0.0))
        top3_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=1.0 if m_ht["top3"] else 0.0, value_b=1.0 if m_sc["top3"] else 0.0))
        top5_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=1.0 if m_ht["top5"] else 0.0, value_b=1.0 if m_sc["top5"] else 0.0))
        mrr_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=float(m_ht["mrr"]), value_b=float(m_sc["mrr"])))

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
        mean_ht = sum(o.value_a for o in obs) / len(obs)
        mean_sc = sum(o.value_b for o in obs) / len(obs)
        raw_p_values.append(p_val)

        results.append({
            "metric": name,
            "mean_pyrca_ht": mean_ht,
            "mean_s_comb": mean_sc,
            "diff": diff_obs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "raw_p": p_val,
            "is_pct": is_pct,
        })

    adj_p_values = apply_holm_correction(raw_p_values)
    for r, adj_p in zip(results, adj_p_values):
        r["adj_p"] = adj_p

    print(f"\n{'Metric':<18} | {'PyRCA HT':<10} | {'S_comb':<10} | {'Diff':<10} | {'95% CI':<20} | {'Raw p':<9} | {'Holm p':<9}")
    print("-" * 95)
    for r in results:
        if r["is_pct"]:
            ht_s = f"{r['mean_pyrca_ht']*100:6.2f}%"
            sc_s = f"{r['mean_s_comb']*100:6.2f}%"
            d_s = f"{r['diff']*100:+6.2f}%"
            ci_s = f"[{r['ci_lower']*100:+5.2f}%, {r['ci_upper']*100:+5.2f}%]"
        else:
            ht_s = f"{r['mean_pyrca_ht']:6.4f}"
            sc_s = f"{r['mean_s_comb']:6.4f}"
            d_s = f"{r['diff']:+6.4f}"
            ci_s = f"[{r['ci_lower']:+6.4f}, {r['ci_upper']:+6.4f}]"

        print(f"{r['metric']:<18} | {ht_s:<10} | {sc_s:<10} | {d_s:<10} | {ci_s:<20} | {r['raw_p']:<9.4f} | {r['adj_p']:<9.4f}")

    # Save statistical comparison artifact
    out_stat_path = repo_root / "eval/results/stage6_pyrca_ht_statistical_comparison_v1.json"
    stat_artifact = {
        "schema_version": "1.0.0",
        "comparison_id": "stage6_pyrca_ht_vs_s_comb_statistical_comparison",
        "manifest_hash": manifest_hash,
        "treatment_hash": treatment_hash,
        "population": {"suite": "RE2-OB", "repetitions": [2, 3], "total_cases": 60, "scenario_families": 30},
        "results": results,
    }
    with open(out_stat_path, "w", encoding="utf-8") as f:
        json.dump(stat_artifact, f, indent=2)
    print(f"\nStatistical artifact saved to: {out_stat_path}")


if __name__ == "__main__":
    main()
