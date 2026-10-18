"""Compact pairwise ML ranker over existing Digital Detective evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    MODEL_VERSION = "1.0.0"

    def __init__(self, feature_names: Sequence[str], *, random_state: int = 0) -> None:
        validate_feature_names(feature_names)
        self.feature_names = tuple(feature_names)
        self.model = LogisticRegression(C=1.0, solver="liblinear", random_state=random_state)
        self._fitted = False
        self.metadata: dict = {
            "model_version": self.MODEL_VERSION,
            "feature_names": list(self.feature_names),
            "random_state": random_state,
        }

    def fit(self, cases: Sequence[Sequence[CandidateFeatures]], roots: Sequence[str]) -> PairwiseRanker:
        x, _ = build_pairwise_examples(cases, roots, self.feature_names)
        # A root-minus-negative example is always the positive class. Mirroring
        # it supplies the negative class without changing the ranking objective.
        self.model.fit(np.vstack((x, -x)), np.r_[np.ones(len(x)), np.zeros(len(x))])
        self._fitted = True
        return self

    def validate_candidate_schema(self, candidate: CandidateFeatures) -> None:
        """Validate that candidate values contain required features without unexpected schema errors."""
        missing = [f for f in self.feature_names if f not in candidate.values]
        if missing:
            raise ValueError(f"Feature schema mismatch for candidate {candidate.entity}: missing {missing}")

    def rank(self, candidates: Sequence[CandidateFeatures]) -> tuple[str, ...]:
        if not self._fitted:
            raise RuntimeError("PairwiseRanker must be fitted before inference")
        for c in candidates:
            self.validate_candidate_schema(c)
        matrix = np.asarray(
            [[candidate.values[f] for f in self.feature_names] for candidate in candidates],
            dtype=float,
        )
        scores = self.model.decision_function(matrix)
        ordered = sorted(
            zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0].entity),
        )
        return tuple(candidate.entity for candidate, _ in ordered)

    def save(self, path: Path | str, extra_metadata: dict | None = None) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.MODEL_VERSION,
            "feature_names": self.feature_names,
            "metadata": {**self.metadata, **(extra_metadata or {})},
            "model": self.model,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path | str) -> PairwiseRanker:
        import pickle
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "model" not in payload or "feature_names" not in payload:
            raise ValueError(f"Invalid model artifact structure in {path}")
        instance = cls(payload["feature_names"])
        instance.model = payload["model"]
        instance._fitted = payload.get("fitted", True)
        instance.metadata = payload.get("metadata", {})
        return instance
