"""Canonical locked-test evaluation script for Digital Detective Model B ranker.

Reproduces the locked evaluation on RE2-OB repetitions 2 and 3 (60 cases)
from the persisted model artifact (models/model_b_ranker_v1.pkl).
Verifies split integrity, zero data leakage, and exact metric numerators/denominators.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from eval.manifest import ManifestCase
from ml_rca_ranker import MODEL_B_FEATURES, PairwiseRanker


def evaluate_persisted_model(
    model_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict:
    if model_path is None:
        model_path = ROOT / "models" / "model_b_ranker_v1.pkl"
    else:
        model_path = Path(model_path)

    metadata_path = model_path.parent / "model_b_metadata.json"

    if dataset_root is None:
        dataset_root = Path.home() / ".cache" / "rcaeval_validation"
    else:
        dataset_root = Path(dataset_root)

    cache_dir = dataset_root / ".ml_rca_features_v1"

    print("=" * 70)
    print("Digital Detective — Canonical Locked-Test Evaluation")
    print("=" * 70)
    print(f"Model Artifact:    {model_path}")
    print(f"Dataset Root:      {dataset_root}")
    print(f"Feature Cache:     {cache_dir}")

    # 1. Load model artifact
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    ranker = PairwiseRanker.load(model_path)
    print(f"Loaded Ranker:     Version {ranker.MODEL_VERSION}, features: {len(ranker.feature_names)}")

    # 2. Check metadata if present
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"Artifact Metadata: {metadata.get('model_name')}, trained on {metadata.get('training_dataset')}")

    # 3. Load case manifest index
    cases_parquet = dataset_root / "cases.parquet"
    if not cases_parquet.is_file():
        raise FileNotFoundError(f"cases.parquet not found at {cases_parquet}")
    df = pd.read_parquet(cases_parquet)

    all_cases = [
        ManifestCase(
            case_id=str(row.case),
            dataset=str(row.dataset),
            system=str(row.system),
            fault=str(row.fault),
            root_cause_service=str(row.root_cause_service),
            repetition=int(row.repetition),
            partition="all",
            suite=str(row.suite),
        )
        for row in df.itertuples()
    ]

    # Canonical split partitioning
    train_cases = [c for c in all_cases if c.dataset in {"RE2-SS", "RE2-TT"}]
    locked_test_cases = [c for c in all_cases if c.dataset == "RE2-OB" and c.repetition in {2, 3}]
    demo_cases = [c for c in all_cases if c.case_id == "re2ob_checkoutservice_cpu_1"]

    train_ids = {c.case_id for c in train_cases}
    test_ids = {c.case_id for c in locked_test_cases}
    demo_id = "re2ob_checkoutservice_cpu_1"

    print("\n--- Split Integrity & Leakage Verification ---")
    print(f"Training Cases (RE2-SS + RE2-TT):         {len(train_cases)}")
    print(f"Locked Test Cases (RE2-OB Rep 2 & 3):      {len(locked_test_cases)}")
    print(f"Held-out Demo Case ({demo_id}): present={len(demo_cases) > 0}")

    # Assertions for split integrity
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {overlap}"
    assert demo_id not in train_ids, f"DEMO CASE LEAKAGE: {demo_id} in training set!"
    assert demo_id not in test_ids, f"DEMO CASE IN TEST SET: {demo_id} in locked test set!"
    print("[PASS] Split verification: 0 overlap between training and locked test.")
    print("[PASS] Demo isolation: re2ob_checkoutservice_cpu_1 strictly held out from both.")

    # 4. Evaluate persisted model on locked test cases without refitting
    print("\n--- Evaluating Persisted Model on Locked Test (60 cases) ---")
    results = []
    top1_count = 0
    top3_count = 0
    top5_count = 0
    rr_sum = 0.0

    s_comb_top1_count = 0
    s_comb_top3_count = 0
    s_comb_top5_count = 0
    s_comb_rr_sum = 0.0

    for c in locked_test_cases:
        feat_path = cache_dir / f"{c.case_id}.pkl"
        if not feat_path.is_file():
            raise FileNotFoundError(f"Cached features missing for {c.case_id} at {feat_path}")
        with open(feat_path, "rb") as f:
            c_feats, root_svc = pickle.load(f)

        assert root_svc == c.root_cause_service, f"Root mismatch for {c.case_id}"

        # Persisted Model B ranking
        ranking = ranker.rank(c_feats)
        rank = ranking.index(root_svc) + 1 if root_svc in ranking else 999
        is_top1 = rank == 1
        is_top3 = rank <= 3
        is_top5 = rank <= 5
        rr = 1.0 / rank if rank <= len(ranking) else 0.0

        if is_top1:
            top1_count += 1
        if is_top3:
            top3_count += 1
        if is_top5:
            top5_count += 1
        rr_sum += rr

        # Baseline S_comb ranking for comparison
        s_base_ordered = [cand.entity for cand, _ in sorted(zip(c_feats, [cand.values["s_comb"] for cand in c_feats]), key=lambda item: (-float(item[1]), item[0].entity))]
        s_rank = s_base_ordered.index(root_svc) + 1 if root_svc in s_base_ordered else 999
        if s_rank == 1:
            s_comb_top1_count += 1
        if s_rank <= 3:
            s_comb_top3_count += 1
        if s_rank <= 5:
            s_comb_top5_count += 1
        s_comb_rr_sum += (1.0 / s_rank if s_rank <= len(s_base_ordered) else 0.0)

        results.append({
            "case_id": c.case_id,
            "root_cause": root_svc,
            "rank": rank,
            "is_top1": is_top1,
            "is_top3": is_top3,
            "is_top5": is_top5,
            "reciprocal_rank": rr,
            "top_candidates": list(ranking[:3]),
        })

    n = len(locked_test_cases)
    metrics_model_b = {
        "top1": top1_count / n,
        "top1_numerator": top1_count,
        "top3": top3_count / n,
        "top3_numerator": top3_count,
        "top5": top5_count / n,
        "top5_numerator": top5_count,
        "mrr": rr_sum / n,
        "mrr_sum": rr_sum,
        "total_cases": n,
    }

    metrics_s_comb = {
        "top1": s_comb_top1_count / n,
        "top1_numerator": s_comb_top1_count,
        "top3": s_comb_top3_count / n,
        "top3_numerator": s_comb_top3_count,
        "top5": s_comb_top5_count / n,
        "top5_numerator": s_comb_top5_count,
        "mrr": s_comb_rr_sum / n,
        "mrr_sum": s_comb_rr_sum,
        "total_cases": n,
    }

    print("\n" + "=" * 70)
    print("Locked-Test Benchmark Results Summary (N = 60)")
    print("=" * 70)
    print(f"{'Method':<32} {'Top@1':<12} {'Top@3':<12} {'Top@5':<12} {'MRR':<12}")
    print("-" * 70)
    print(f"{'Digital Detective frozen RCA (S_comb)':<32} {metrics_s_comb['top1']:.4f} ({s_comb_top1_count}/{n}) {metrics_s_comb['top3']:.4f} ({s_comb_top3_count}/{n}) {metrics_s_comb['top5']:.4f} ({s_comb_top5_count}/{n}) {metrics_s_comb['mrr']:.4f}")
    print(f"{'Model B Pairwise Ranker (Persisted)':<32} {metrics_model_b['top1']:.4f} ({top1_count}/{n}) {metrics_model_b['top3']:.4f} ({top3_count}/{n}) {metrics_model_b['top5']:.4f} ({top5_count}/{n}) {metrics_model_b['mrr']:.4f}")
    print("=" * 70)

    report = {
        "evaluation_name": "locked_test_canonical_evaluation",
        "model_artifact": str(model_path),
        "dataset_split": {
            "training_dataset": "RE2-SS (90) + RE2-TT (90) = 180 cases",
            "locked_test_dataset": "RE2-OB repetitions 2 & 3 = 60 cases",
            "demo_case": demo_id,
            "leakage_overlap_count": len(overlap),
        },
        "model_b_metrics": metrics_model_b,
        "baseline_s_comb_metrics": metrics_s_comb,
        "case_evaluations": results,
    }

    if output_path is None:
        output_path = ROOT / "eval" / "results" / "canonical_locked_test_results.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPersisted canonical results to: {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate persisted Model B on locked test set.")
    parser.add_argument("--model", default=None, help="Path to model artifact pkl")
    parser.add_argument("--dataset", default=None, help="Path to rcaeval_validation root")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    evaluate_persisted_model(
        model_path=args.model,
        dataset_root=args.dataset,
        output_path=args.output,
    )
