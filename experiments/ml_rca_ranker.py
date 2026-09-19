"""Compact pairwise ML ranker over existing Digital Detective evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


MODEL_A_FEATURES = (
    "s_comb",
    "e_elev",
    "r_strength",
    "r_early",
    "r_coverage",
    "r_prop",
)
MODEL_B_FEATURES = MODEL_A_FEATURES + (
    "peak_anomaly_strength",
    "anomalous_metric_count",
    "episode_active",
    "episode_timing",
    "trace_wait_concentration",
    "trace_terminal_factor",
)
PROHIBITED_FEATURE_TOKENS = (
    "service_id",
    "fault",
    "family",
    "repetition",
    "inject",
    "root",
    "label",
    "timestamp",
)


@dataclass(frozen=True)
class CandidateFeatures:
    case_id: str
    entity: str
    values: Mapping[str, float]


def validate_feature_names(names: Iterable[str]) -> None:
    names = tuple(names)
    for name in names:
        if any(token in name.lower() for token in PROHIBITED_FEATURE_TOKENS):
            raise ValueError(f"Prohibited feature name: {name}")


def build_pairwise_examples(
    cases: Sequence[Sequence[CandidateFeatures]],
    roots: Sequence[str],
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Create positive root-vs-negative difference examples."""
    validate_feature_names(feature_names)
    if len(cases) != len(roots):
        raise ValueError("cases and roots must have equal length")
    rows: list[np.ndarray] = []
    for candidates, root in zip(cases, roots):
        by_entity = {c.entity: c for c in candidates}
        if root not in by_entity:
            raise ValueError(f"Root {root!r} is absent from candidate set")
        root_values = np.asarray([by_entity[root].values.get(f, 0.0) for f in feature_names], dtype=float)
        for candidate in sorted(candidates, key=lambda c: c.entity):
            if candidate.entity == root:
                continue
            negative = np.asarray([candidate.values.get(f, 0.0) for f in feature_names], dtype=float)
            rows.append(root_values - negative)
    if not rows:
        raise ValueError("No pairwise examples were generated")
    return np.vstack(rows), np.ones(len(rows), dtype=int)


class PairwiseRanker:
    """Binary logistic model trained on root-minus-negative feature differences."""

    def __init__(self, feature_names: Sequence[str], *, random_state: int = 0) -> None:
        validate_feature_names(feature_names)
        self.feature_names = tuple(feature_names)
        self.model = LogisticRegression(C=1.0, solver="liblinear", random_state=random_state)
        self._fitted = False

    def fit(self, cases: Sequence[Sequence[CandidateFeatures]], roots: Sequence[str]) -> PairwiseRanker:
        x, _ = build_pairwise_examples(cases, roots, self.feature_names)
        # A root-minus-negative example is always the positive class. Mirroring
        # it supplies the negative class without changing the ranking objective.
        self.model.fit(np.vstack((x, -x)), np.r_[np.ones(len(x)), np.zeros(len(x))])
        self._fitted = True
        return self

    def rank(self, candidates: Sequence[CandidateFeatures]) -> tuple[str, ...]:
        if not self._fitted:
            raise RuntimeError("PairwiseRanker must be fitted before inference")
        matrix = np.asarray(
            [[candidate.values.get(f, 0.0) for f in self.feature_names] for candidate in candidates],
            dtype=float,
        )
        scores = self.model.predict_proba(matrix)[:, 1]
        ordered = sorted(
            zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0].entity),
        )
        return tuple(candidate.entity for candidate, _ in ordered)
