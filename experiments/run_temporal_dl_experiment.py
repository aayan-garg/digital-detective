"""One small temporal complement to the frozen Model B ranker."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "experiments")]

from eval.stats import compare_methods_paired
from ml_rca_ranker import CandidateFeatures, MODEL_B_FEATURES


DATASET = Path.home() / ".cache" / "rcaeval_validation"
CACHE_DIR = DATASET / ".ml_rca_features_v1"
SEED = 42
SEQUENCE_LENGTH = 721
RESULT_PATH = ROOT / "eval" / "results" / "temporal_dl_experiment_v1.json"
_SEQUENCE_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, np.ndarray]] = {}


def _seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _index() -> list[dict]:
    return pq.read_table(DATASET / "cases.parquet").to_pylist()


def _cached_features(case_id: str) -> tuple[list[CandidateFeatures], str]:
    with (CACHE_DIR / f"{case_id}.pkl").open("rb") as handle:
        return pickle.load(handle)


def _sequences(case_id: str, entities: list[str]) -> dict[str, np.ndarray]:
    cache_key = (case_id, tuple(entities))
    if cache_key in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[cache_key]
    table = pq.read_table(DATASET / case_id / "metrics.parquet")
    times = np.asarray(table["time"].to_numpy(), dtype=np.int64)
    inject = int((DATASET / case_id / "inject_time.txt").read_text().strip())
    prefix = np.flatnonzero(times <= inject)
    if len(prefix) != SEQUENCE_LENGTH:
        raise ValueError(f"{case_id}: expected {SEQUENCE_LENGTH} causal rows, got {len(prefix)}")
    sequences = {}
    for entity in entities:
        columns = [name for name in table.column_names if name.startswith(entity + "_")]
        if not columns:
            continue
        preferred = [name for name in columns if name == f"{entity}_cpu"]
        column = preferred[0] if preferred else sorted(columns)[0]
        values = np.asarray(table[column].to_numpy(), dtype=np.float32)[prefix]
        mean = float(np.nanmean(values))
        scale = float(np.nanstd(values))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        sequences[entity] = np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    _SEQUENCE_CACHE[cache_key] = sequences
    return sequences


class TemporalRanker(nn.Module):
    """Conv1d -> ReLU -> global average pool -> embedding plus Model B."""

    def __init__(self, static_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.embedding = nn.Linear(4, 4)
        self.score = nn.Linear(static_size + 4, 1)

    def forward(self, static: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence.unsqueeze(1)).mean(dim=2)
        temporal = self.embedding(encoded)
        return self.score(torch.cat((static, temporal), dim=1)).squeeze(1)


def _metrics(ranking: tuple[str, ...], root: str) -> dict[str, float | bool | int | None]:
    rank = next((i for i, name in enumerate(ranking, 1) if name == root), None)
    return {
        "top1": rank == 1,
        "top3": bool(rank and rank <= 3),
        "top5": bool(rank and rank <= 5),
        "mrr": 1.0 / rank if rank else 0.0,
        "target_rank": rank,
    }


def _families(rows: list[dict], datasets: set[str], repetitions: set[int] | None = None) -> list[dict]:
    return [
        row for row in rows
        if row["dataset"] in datasets and (repetitions is None or row["repetition"] in repetitions)
    ]


def main() -> int:
    started = time.perf_counter()
    _seed()
    rows = _index()
    dev = _families(rows, {"RE2-SS", "RE2-TT"})
    test = _families(rows, {"RE2-OB"}, {2, 3})
    families = sorted(
        {(r["suite"], r["system"], r["root_cause_service"], r["fault"]) for r in dev}
    )
    validation_families = set(families[::5])
    train = [r for r in dev if tuple((r["suite"], r["system"], r["root_cause_service"], r["fault"])) not in validation_families]
    validation = [r for r in dev if tuple((r["suite"], r["system"], r["root_cause_service"], r["fault"])) in validation_families]

    train_cases = []
    for row in train:
        candidates, root = _cached_features(row["case"])
        train_cases.append((row["case"], candidates, root))
    validation_cases = []
    for row in validation:
        candidates, root = _cached_features(row["case"])
        validation_cases.append((row["case"], candidates, root))

    static_size = len(MODEL_B_FEATURES)
    model = TemporalRanker(static_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_state = None
    best_mrr = -1.0
    stale = 0
    epochs = 0
    for epoch in range(10):
        model.train()
        order = list(train_cases)
        random.shuffle(order)
        for case_id, candidates, root in order:
            by_entity = {c.entity: c for c in candidates}
            entities = sorted(by_entity)
            static = torch.tensor(
                [[by_entity[e].values.get(f, 0.0) for f in MODEL_B_FEATURES] for e in entities],
                dtype=torch.float32,
            ).to(device)
            sequence_map = _sequences(case_id, entities)
            sequences = torch.from_numpy(np.stack([sequence_map[e] for e in entities])).to(device)
            root_i = entities.index(root)
            negative_i = [i for i in range(len(entities)) if i != root_i]
            optimizer.zero_grad()
            scores = model(static, sequences)
            differences = scores[root_i] - scores[negative_i]
            loss = nn.functional.binary_cross_entropy_with_logits(
                differences, torch.ones_like(differences)
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_mrr = 0.0
            for case_id, candidates, root in validation_cases:
                entities = sorted(c.entity for c in candidates)
                by_entity = {c.entity: c for c in candidates}
                static = torch.tensor(
                    [[by_entity[e].values.get(f, 0.0) for f in MODEL_B_FEATURES] for e in entities],
                    dtype=torch.float32,
                ).to(device)
                sequence_map = _sequences(case_id, entities)
                sequences = torch.from_numpy(np.stack([sequence_map[e] for e in entities])).to(device)
                scores = model(static, sequences).tolist()
                ranking = tuple(e for e, _ in sorted(zip(entities, scores), key=lambda x: (-x[1], x[0])))
                validation_mrr += _metrics(ranking, root)["mrr"]
            validation_mrr /= len(validation_cases)
        epochs = epoch + 1
        if validation_mrr > best_mrr:
            best_mrr, stale, best_state = validation_mrr, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 2:
                break
    if best_state is None:
        raise RuntimeError("No development checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()

    dl_results: dict[str, list[dict]] = defaultdict(list)
    model_b_results: dict[str, list[dict]] = defaultdict(list)
    locked_baseline = json.loads(
        (ROOT / "eval" / "results" / "ml_rca_locked_test_v1.json").read_text()
    )["results"]["Model B"]
    baseline_by_case = {entry["case_id"]: entry for entry in locked_baseline}
    with torch.no_grad():
        for row in test:
            case_id = row["case"]
            candidates, root = _cached_features(case_id)
            entities = sorted(c.entity for c in candidates)
            by_entity = {c.entity: c for c in candidates}
            static = torch.tensor(
                [[by_entity[e].values.get(f, 0.0) for f in MODEL_B_FEATURES] for e in entities],
                dtype=torch.float32,
            ).to(device)
            sequence_map = _sequences(case_id, entities)
            sequences = torch.from_numpy(np.stack([sequence_map[e] for e in entities])).to(device)
            scores = model(static, sequences).tolist()
            ranking = tuple(e for e, _ in sorted(zip(entities, scores), key=lambda x: (-x[1], x[0])))
            dl_results["DL + Model B"].append({"case_id": case_id, "family": row, "ground_truth": root, **_metrics(ranking, root)})
            model_b_results["Model B"].append(
                {"case_id": case_id, "family": row, "ground_truth": root, **baseline_by_case[case_id]}
            )

    comparisons = {}
    for metric in ("top1", "top3", "top5", "mrr"):
        comparisons[metric] = compare_methods_paired(
            "DL + Model B", "Model B", metric,
            dl_results["DL + Model B"], model_b_results["Model B"],
            {row["case"]: (row["suite"], row["system"], row["root_cause_service"], row["fault"]) for row in test},
            randomization_replicates=10_000, bootstrap_replicates=10_000, seed=SEED,
        ).to_dict()
    aggregate = {
        name: {metric: float(np.mean([entry[metric] for entry in entries])) for metric in ("top1", "top3", "top5", "mrr")}
        for name, entries in {**dl_results, **model_b_results}.items()
    }
    report = {
        "experiment": "temporal_dl_experiment_v1",
        "temporal_sequences_available": True,
        "development_cases": len(dev),
        "development_families": len(families),
        "locked_test_cases": len(test),
        "sequence_length": SEQUENCE_LENGTH,
        "architecture": "Conv1d(1->4,kernel=5) -> ReLU -> global average pool -> Linear(4->4), concatenated with Model B static features -> Linear",
        "seed": SEED,
        "epochs": epochs,
        "development_mrr": best_mrr,
        "aggregate": aggregate,
        "clustered_comparisons": comparisons,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
