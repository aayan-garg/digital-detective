"""Regression tests for Model B evaluation, split integrity, and artifact schema safety."""

from __future__ import annotations

import copy
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from digital_detective.schema import CanonicalIncident
from ml_rca_ranker import CandidateFeatures, MODEL_B_FEATURES, PairwiseRanker
from scripts.evaluate_locked_test import evaluate_persisted_model


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path.home() / ".cache" / "rcaeval_validation"
MODEL_PATH = ROOT / "models" / "model_b_ranker_v1.pkl"


class TestModelBEvaluationRegression:
    def test_split_integrity_and_zero_leakage(self):
        """Verify train, test, and demo sets are strictly disjoint."""
        cases_parquet = DATASET_ROOT / "cases.parquet"
        if not cases_parquet.is_file():
            pytest.skip("cases.parquet not found")

        df = pd.read_parquet(cases_parquet)
        train_cases = set(df[(df["dataset"].isin(["RE2-SS", "RE2-TT"]))]["case"])
        test_cases = set(df[(df["dataset"] == "RE2-OB") & (df["repetition"].isin([2, 3]))]["case"])
        demo_case = "re2ob_checkoutservice_cpu_1"

        assert len(train_cases) == 180
        assert len(test_cases) == 60
        assert len(train_cases & test_cases) == 0, "Train and locked test sets must not overlap"
        assert demo_case not in train_cases, "Demo case must not be in training set"
        assert demo_case not in test_cases, "Demo case must not be in locked test set"

    def test_persisted_artifact_schema_validation(self):
        """Verify model loads correctly and rejects inputs with missing features."""
        if not MODEL_PATH.is_file():
            pytest.skip("Model artifact not found")

        ranker = PairwiseRanker.load(MODEL_PATH)
        assert ranker._fitted is True
        assert tuple(ranker.feature_names) == MODEL_B_FEATURES

        # Valid candidate features
        valid_cand = CandidateFeatures(
            case_id="test_case",
            entity="service_a",
            values={f: 0.5 for f in MODEL_B_FEATURES},
        )
        ranker.validate_candidate_schema(valid_cand)

        # Invalid candidate missing 's_comb'
        invalid_values = {f: 0.5 for f in MODEL_B_FEATURES if f != "s_comb"}
        invalid_cand = CandidateFeatures(
            case_id="test_case",
            entity="service_a",
            values=invalid_values,
        )
        with pytest.raises(ValueError, match="Feature schema mismatch.*missing \\['s_comb'\\]"):
            ranker.validate_candidate_schema(invalid_cand)

    def test_deterministic_evaluation_and_no_accidental_refitting(self):
        """Verify ranking is deterministic and model parameters are unchanged during inference."""
        if not MODEL_PATH.is_file():
            pytest.skip("Model artifact not found")

        ranker = PairwiseRanker.load(MODEL_PATH)
        initial_coef = ranker.model.coef_.copy()
        initial_intercept = ranker.model.intercept_.copy()

        cands = [
            CandidateFeatures("case_1", "svc_a", {f: 0.8 if f == "s_comb" else 0.2 for f in MODEL_B_FEATURES}),
            CandidateFeatures("case_1", "svc_b", {f: 0.3 if f == "s_comb" else 0.7 for f in MODEL_B_FEATURES}),
            CandidateFeatures("case_1", "svc_c", {f: 0.5 for f in MODEL_B_FEATURES}),
        ]

        ranking_1 = ranker.rank(cands)
        ranking_2 = ranker.rank(cands)

        assert ranking_1 == ranking_2
        assert len(ranking_1) == 3
        assert set(ranking_1) == {"svc_a", "svc_b", "svc_c"}
        # Verify model weights are strictly unmodified after inference
        np.testing.assert_array_equal(ranker.model.coef_, initial_coef)
        np.testing.assert_array_equal(ranker.model.intercept_, initial_intercept)

    def test_canonical_evaluation_reproducibility(self):
        """Run canonical evaluation and verify exact Top@1 and MRR metrics."""
        if not MODEL_PATH.is_file() or not (DATASET_ROOT / "cases.parquet").is_file():
            pytest.skip("Required evaluation inputs missing")

        report = evaluate_persisted_model(model_path=MODEL_PATH, dataset_root=DATASET_ROOT)
        metrics = report["model_b_metrics"]

        assert metrics["total_cases"] == 60
        assert metrics["top1_numerator"] == 49
        assert pytest.approx(metrics["top1"], abs=1e-4) == 0.8167
        assert metrics["top3_numerator"] == 58
        assert pytest.approx(metrics["top3"], abs=1e-4) == 0.9667
        assert metrics["top5_numerator"] == 60
        assert pytest.approx(metrics["top5"], abs=1e-4) == 1.0000
        assert pytest.approx(metrics["mrr"], abs=1e-4) == 0.8917
