"""Cross-application zero-shot generalization experiment.

Protocol:
- Train Model B pairwise ranker strictly on Online Boutique (RE2-OB, 90 cases).
- Evaluate zero-shot transfer performance on:
  1. Sock Shop (RE2-SS, 90 cases)
  2. Train Ticket (RE2-TT, 90 cases)
- Feature definitions and extractors are identical across training and transfer.
- Evaluates Top@1, Top@3, Top@5, and MRR.
- Outputs results to eval/results/cross_application_generalization.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from eval.manifest import ManifestCase
from experiments.ml_rca_ranker import MODEL_B_FEATURES, CandidateFeatures, PairwiseRanker
from experiments.run_ml_rca_experiment import _cached_extract, _metrics, _aggregate

DATASET = Path.home() / ".cache" / "rcaeval_validation"


def run_cross_application_study() -> dict[str, Any]:
    print("=" * 70)
    print("Cross-Application RCA Generalization Study")
    print("=" * 70)

    index = pd.read_parquet(DATASET / "cases.parquet")
    all_cases = tuple(
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
        for row in index.itertuples()
    )

    ob_train_cases = [c for c in all_cases if c.dataset == "RE2-OB"]
    ss_test_cases = [c for c in all_cases if c.dataset == "RE2-SS"]
    tt_test_cases = [c for c in all_cases if c.dataset == "RE2-TT"]

    print(f"Source Training Set: Online Boutique (RE2-OB): {len(ob_train_cases)} cases")
    print(f"Target Zero-Shot Set 1: Sock Shop (RE2-SS):     {len(ss_test_cases)} cases")
    print(f"Target Zero-Shot Set 2: Train Ticket (RE2-TT):  {len(tt_test_cases)} cases")

    # Extract features using existing caching
    print("\nExtracting/loading features...")
    ob_features: list[tuple[list[CandidateFeatures], str]] = []
    for c in ob_train_cases:
        ob_features.append(_cached_extract(c))

    ss_features: list[tuple[list[CandidateFeatures], str]] = []
    for c in ss_test_cases:
        ss_features.append(_cached_extract(c))

    tt_features: list[tuple[list[CandidateFeatures], str]] = []
    for c in tt_test_cases:
        tt_features.append(_cached_extract(c))

    # Train Model B strictly on RE2-OB
    print("\nTraining Pairwise Model B on RE2-OB...")
    ranker = PairwiseRanker(MODEL_B_FEATURES)
    t0 = time.perf_counter()
    train_cands = [cands for cands, _ in ob_features]
    train_roots = [root for _, root in ob_features]
    ranker.fit(train_cands, train_roots)
    train_time = time.perf_counter() - t0
    print(f"Model B trained in {train_time:.3f}s")

    # Evaluate on Sock Shop (RE2-SS)
    print("\nEvaluating Zero-Shot Transfer on Sock Shop (RE2-SS)...")
    ss_results = []
    for (candidates, root), c in zip(ss_features, ss_test_cases):
        ranking = ranker.rank(candidates)
        m = _metrics(ranking, root)
        m["case_id"] = c.case_id
        m["root_cause"] = root
        m["top_pred"] = ranking[0] if ranking else ""
        ss_results.append(m)

    ss_agg = _aggregate(ss_results)
    print(f"  Sock Shop (RE2-SS) Zero-Shot Results (N={len(ss_results)}):")
    print(f"    Top@1: {ss_agg['top1']:.4f}")
    print(f"    Top@3: {ss_agg['top3']:.4f}")
    print(f"    Top@5: {ss_agg['top5']:.4f}")
    print(f"    MRR:   {ss_agg['mrr']:.4f}")

    # Evaluate on Train Ticket (RE2-TT)
    print("\nEvaluating Zero-Shot Transfer on Train Ticket (RE2-TT)...")
    tt_results = []
    for (candidates, root), c in zip(tt_features, tt_test_cases):
        ranking = ranker.rank(candidates)
        m = _metrics(ranking, root)
        m["case_id"] = c.case_id
        m["root_cause"] = root
        m["top_pred"] = ranking[0] if ranking else ""
        tt_results.append(m)

    tt_agg = _aggregate(tt_results)
    print(f"  Train Ticket (RE2-TT) Zero-Shot Results (N={len(tt_results)}):")
    print(f"    Top@1: {tt_agg['top1']:.4f}")
    print(f"    Top@3: {tt_agg['top3']:.4f}")
    print(f"    Top@5: {tt_agg['top5']:.4f}")
    print(f"    MRR:   {tt_agg['mrr']:.4f}")

    # Combine overall zero-shot metrics
    all_zero_shot = ss_results + tt_results
    all_agg = _aggregate(all_zero_shot)
    print(f"\n  Combined Cross-Application Zero-Shot (N={len(all_zero_shot)}):")
    print(f"    Top@1: {all_agg['top1']:.4f}")
    print(f"    Top@3: {all_agg['top3']:.4f}")
    print(f"    Top@5: {all_agg['top5']:.4f}")
    print(f"    MRR:   {all_agg['mrr']:.4f}")

    output_payload = {
        "experiment": "cross_application_generalization_v1",
        "training_source": {
            "dataset": "RE2-OB",
            "case_count": len(ob_train_cases),
            "model": "Model B (Pairwise Linear SVM)",
            "features": list(MODEL_B_FEATURES),
            "train_duration_sec": round(train_time, 4),
        },
        "zero_shot_transfer": {
            "RE2-SS": {
                "system": "sock-shop",
                "case_count": len(ss_results),
                "aggregate": ss_agg,
                "cases_sample": ss_results[:10],
            },
            "RE2-TT": {
                "system": "train-ticket",
                "case_count": len(tt_results),
                "aggregate": tt_agg,
                "cases_sample": tt_results[:10],
            },
            "combined": {
                "case_count": len(all_zero_shot),
                "aggregate": all_agg,
            },
        },
    }

    out_file = ROOT / "eval" / "results" / "cross_application_generalization.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_file}")
    return output_payload


if __name__ == "__main__":
    run_cross_application_study()
