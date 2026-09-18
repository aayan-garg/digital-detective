#!/usr/bin/env python3
"""Execute the formal paired statistical comparison between Stage 5 TCEC treatment and Stage 4 BOCPD treatment.

Methodological requirements:
- Population: RE2-OB Repetitions 2 and 3 (N=60, 30 scenario families, 2 executions per family).
- Scenario family as inferential cluster.
- Paired cluster-level randomization/sign-swapping test (10,000 randomizations, seed=42).
- Cluster bootstrap (10,000 resamples, seed=42, Type-7 percentile CI).
- Primary comparison: Stage 5 TCEC treatment vs Stage 4 BOCPD treatment.
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
    ctrl_path = repo_root / "eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json"
    bocpd_path = repo_root / "eval/results/stage4_re2ob_detected_bocpd_rep2_rep3_treatment_v1.json"
    tcec_path = repo_root / "eval/results/stage5_tcec_treatment_v1.json"
    manifest_path = repo_root / "eval/manifests/re2_ob_all_cases.json"

    # Integrity check on baseline artifacts
    ctrl_hash = compute_file_sha256(ctrl_path)
    bocpd_hash = compute_file_sha256(bocpd_path)
    tcec_hash = compute_file_sha256(tcec_path)

    expected_ctrl_hash = "3C01E2CA28F89AA805A8109152E0B8D992F32330C3E70CDADC19A645C1110157"
    expected_bocpd_hash = "60FEC0F98C0A1486A1575CF8F1B653026BB60ACE1F02149EFA58DDDE2E4D0F9A"

    assert ctrl_hash == expected_ctrl_hash, f"Control hash mismatch: {ctrl_hash}"
    assert bocpd_hash == expected_bocpd_hash, f"BOCPD hash mismatch: {bocpd_hash}"

    with open(ctrl_path, "r", encoding="utf-8") as f:
        ctrl_data = json.load(f)
    with open(bocpd_path, "r", encoding="utf-8") as f:
        bocpd_data = json.load(f)
    with open(tcec_path, "r", encoding="utf-8") as f:
        tcec_data = json.load(f)

    manifest = BenchmarkManifest.load(manifest_path)
    manifest_hash = manifest.compute_hash()

    print("=" * 85)
    print("STAGE 5 STATISTICAL COMPARISON: TCEC TREATMENT vs STAGE 4 BOCPD TREATMENT")
    print("=" * 85)
    print(f"Manifest Hash:        {manifest_hash}")
    print(f"Control Hash:         {ctrl_hash}")
    print(f"BOCPD Baseline Hash:  {bocpd_hash}")
    print(f"TCEC Treatment Hash:  {tcec_hash}")

    bocpd_by_id = {r["case_id"]: r for r in bocpd_data["per_case_results"]}
    tcec_by_id = {r["case_id"]: r for r in tcec_data["per_case_results"]}

    # Verify alignment
    assert set(bocpd_by_id.keys()) == set(tcec_by_id.keys())
    assert len(tcec_by_id) == 60

    # 1. Detection Delay Comparison (Confirmed Delay vs BOCPD Delay)
    delay_obs = []
    fe_obs = []
    scomb_obs = []
    fusion_obs = []

    for cid, c_case in tcec_by_id.items():
        b_case = bocpd_by_id[cid]
        fam = tuple(c_case["scenario_family"])

        # Delay
        c_delay = c_case["confirmed_delay_sec"] if c_case["confirmed_delay_sec"] is not None else -1000.0
        b_delay = b_case["detection_delay_sec"] if b_case["detection_delay_sec"] is not None else -1000.0
        delay_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=c_delay, value_b=b_delay))

        # False early (1.0 if false early, 0.0 otherwise)
        c_fe = 1.0 if c_case.get("is_false_early") else 0.0
        b_fe = 1.0 if b_case.get("is_false_early") else 0.0
        fe_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=c_fe, value_b=b_fe))

        # S_comb MRR
        c_scomb = c_case["methods"]["s_comb"]["mrr"]
        b_scomb = b_case["methods"]["s_comb"]["mrr"]
        scomb_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=c_scomb, value_b=b_scomb))

        # Fusion MRR
        c_fusion = c_case["methods"]["fixed_equal_weight_fusion"]["mrr"]
        b_fusion = b_case["methods"]["fixed_equal_weight_fusion"]["mrr"]
        fusion_obs.append(PairedCaseObservation(case_id=cid, scenario_family=fam, value_a=c_fusion, value_b=b_fusion))

    metrics_to_test = [
        ("Detection Delay (s)", delay_obs),
        ("False-Early Rate", fe_obs),
        ("S_comb MRR", scomb_obs),
        ("Fixed Fusion MRR", fusion_obs),
    ]

    stats_results = []
    raw_p_values = []

    for name, obs_list in metrics_to_test:
        mean_diff, p_val = paired_cluster_randomization_test(obs_list, randomization_replicates=10000, seed=42)
        _, ci_lower, ci_upper = cluster_bootstrap_ci(obs_list, confidence_level=0.95, bootstrap_replicates=10000, seed=42)
        mean_tcec = sum(o.value_a for o in obs_list) / len(obs_list)
        mean_bocpd = sum(o.value_b for o in obs_list) / len(obs_list)
        stats_results.append({
            "metric": name,
            "mean_tcec": mean_tcec,
            "mean_bocpd": mean_bocpd,
            "diff": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val,
        })
        raw_p_values.append(p_val)

    adj_p = apply_holm_correction(raw_p_values)
    for s, ap in zip(stats_results, adj_p):
        s["p_value_holm"] = ap

    print("\nSTATISTICAL COMPARISON TABLE:")
    print(f"{'Metric':<24} | {'TCEC Mean':<10} | {'BOCPD Mean':<10} | {'Diff':<10} | {'95% Cluster CI':<22} | {'p-val':<8} | {'Holm p':<8}")
    print("-" * 105)
    for s in stats_results:
        ci_str = f"[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}]"
        print(f"{s['metric']:<24} | {s['mean_tcec']:<10.4f} | {s['mean_bocpd']:<10.4f} | {s['diff']:<+10.4f} | {ci_str:<22} | {s['p_value']:<8.4f} | {s['p_value_holm']:<8.4f}")

    output_comparison_path = repo_root / "eval/results/stage5_tcec_statistical_comparison_v1.json"
    with open(output_comparison_path, "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nSaved statistical comparison to: {output_comparison_path}")


if __name__ == "__main__":
    main()
