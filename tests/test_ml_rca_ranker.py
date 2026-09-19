from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from ml_rca_ranker import CandidateFeatures, MODEL_A_FEATURES, PairwiseRanker, build_pairwise_examples, validate_feature_names


def _case(root: str) -> list[CandidateFeatures]:
    return [CandidateFeatures("c", "root", {"s_comb": 1.0}), CandidateFeatures("c", root if root != "root" else "other", {"s_comb": 0.0})]


def test_pair_construction_uses_root_minus_negative() -> None:
    x, y = build_pairwise_examples([_case("other")], ["root"], ("s_comb",))
    assert x.tolist() == [[1.0]]
    assert y.tolist() == [1]


def test_prohibited_feature_check() -> None:
    with pytest.raises(ValueError):
        validate_feature_names(("fault_type",))


def test_pairwise_inference_is_deterministic() -> None:
    cases = [_case("other"), [CandidateFeatures("d", "root", {"s_comb": 0.8}), CandidateFeatures("d", "other", {"s_comb": 0.2})]]
    model = PairwiseRanker(MODEL_A_FEATURES).fit(cases, ["root", "root"])
    candidates = cases[0]
    assert model.rank(candidates) == model.rank(candidates)


def test_ranking_output_contains_all_candidates() -> None:
    model = PairwiseRanker(("s_comb",)).fit([_case("other")], ["root"])
    assert model.rank(_case("other")) == ("root", "other")
