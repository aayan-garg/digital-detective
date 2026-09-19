"""Train on grouped RE2-SS/RE2-TT and evaluate once on locked RE2-OB 2/3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
import pickle
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.rca import rank_with_s_comb
from digital_detective.topology import Dependency, build_entity_graph
from digital_detective.trace_attribution import rank_with_trace_elevation
from digital_detective.traces import extract_trace_latency_evidence
from eval.manifest import BenchmarkManifest, ManifestCase
from eval.universe import resolve_candidate_universe
from ml_rca_ranker import CandidateFeatures, MODEL_A_FEATURES, MODEL_B_FEATURES, PairwiseRanker

DATASET = Path.home() / ".cache" / "rcaeval_validation"
CACHE_DIR = DATASET / ".ml_rca_features_v1"
MANIFEST = ROOT / "eval" / "manifests" / "re2_ob_all_cases.json"


def _extract(case_meta: ManifestCase) -> tuple[list[CandidateFeatures], str]:
    case = load_rcaeval_case(
        DATASET,
        case_meta.case_id,
        include_logs=False,
        trace_columns=(
            "spanID", "parentSpanID", "traceID", "serviceName",
            "operationName", "startTime", "duration", "statusCode",
        ),
    )
    candidates = resolve_candidate_universe(case_meta.system, case=case)
    det = detect_metric_anomalies(case)
    trace_lat = (
        extract_trace_latency_evidence(
            case, service_aliases={"frontendservice": "frontend"}, expected_services=candidates
        )
        if case.traces is not None
        else None
    )
    deps = (
        [Dependency(source=edge.caller_service, target=edge.callee_service) for edge in trace_lat.edges]
        if trace_lat is not None
        else ()
    )
    graph = build_entity_graph(det.metric_names, dependencies=deps)
    episodes = aggregate_entity_episodes(det, graph, EpisodeConfig(persistence=3, consensus=2))
    s_scores = {x.entity: x for x in rank_with_s_comb(det, episodes, graph=graph, candidate_universe=candidates)}
    if trace_lat is not None:
        t_scores = {x.entity: x for x in rank_with_trace_elevation(trace_lat, candidate_universe=candidates)}
    else:
        t_scores = {}
    trace_entities = {}
    first_ts = [e.first_episode_start_ts for e in episodes.values() if e.first_episode_start_ts is not None]
    first_min = min(first_ts) if first_ts else None
    first_max = max(first_ts) if first_ts else None
    result: list[CandidateFeatures] = []
    for entity in candidates:
        s = s_scores.get(entity)
        episode = episodes.get(entity)
        trace = trace_entities.get(entity)
        timing = 0.0
        if episode is not None and episode.first_episode_start_ts is not None and first_max != first_min:
            timing = 1.0 - (episode.first_episode_start_ts - first_min) / (first_max - first_min)
        values = {
            "s_comb": float(s.score if s else 0.0),
            "e_elev": float(t_scores.get(entity).score if entity in t_scores else 0.0),
            "r_strength": float(s.r_strength if s else 0.0),
            "r_early": float(s.r_early if s else 0.0),
            "r_coverage": float(s.r_coverage if s else 0.0),
            "r_prop": float(s.r_prop if s else 0.0),
            "peak_anomaly_strength": float(episode.peak_active_metrics if episode else 0.0),
            "anomalous_metric_count": float(len(episode.all_contributing_metrics) if episode else 0.0),
            "episode_active": float(bool(episode and episode.has_episode)),
            "episode_timing": float(timing),
            "trace_wait_concentration": float(trace.primary_edge.caller_wait_concentration if trace and trace.primary_edge else 0.0),
            "trace_terminal_factor": float(trace.primary_edge.terminal_factor if trace and trace.primary_edge else 0.0),
        }
        result.append(CandidateFeatures(case_meta.case_id, entity, values))
    return result, case_meta.root_cause_service


def _cached_extract(case_meta: ManifestCase) -> tuple[list[CandidateFeatures], str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{case_meta.case_id}.pkl"
    if path.is_file():
        with path.open("rb") as handle:
            return pickle.load(handle)
    result = _extract(case_meta)
    with path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"extracted {case_meta.case_id}", flush=True)
    return result


def _metrics(ranking: Sequence[str], root: str) -> dict[str, float | bool | int | None]:
    rank = next((i for i, x in enumerate(ranking, 1) if x == root), None)
    return {"top1": rank == 1, "top3": bool(rank and rank <= 3), "top5": bool(rank and rank <= 5),
            "mrr": 1.0 / rank if rank else 0.0, "target_rank": rank}


def _aggregate(entries: Sequence[dict]) -> dict[str, float]:
    return {key: float(np.mean([e[key] for e in entries])) for key in ("top1", "top3", "top5", "mrr")}


def main() -> int:
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
    dev = [c for c in all_cases if c.dataset in {"RE2-SS", "RE2-TT"}]
    test = [c for c in all_cases if c.dataset == "RE2-OB" and c.repetition in {2, 3}]
    missing_dev = [
        c.case_id
        for c in dev
        if not (DATASET / c.case_id / "metrics.parquet").is_file()
    ]
    if missing_dev:
        report = {
            "experiment": "ml_rca_development_v1",
            "status": "BLOCKED",
            "reason": "Required RE2-SS/RE2-TT development telemetry is absent from the local cache.",
            "development_case_count": len(dev),
            "missing_development_case_count": len(missing_dev),
            "missing_development_cases_sample": missing_dev[:10],
            "locked_test_not_run": True,
            "feature_sets": {"Model A": list(MODEL_A_FEATURES), "Model B": list(MODEL_B_FEATURES)},
        }
        (ROOT / "eval/results/ml_rca_development_v1.json").write_text(json.dumps(report, indent=2))
        (ROOT / "eval/results/ml_rca_locked_test_v1.json").write_text(
            json.dumps(
                {
                    "experiment": "ml_rca_locked_test_v1",
                    "status": "BLOCKED",
                    "reason": "Locked test was not run because development training data was unavailable.",
                    "locked_test_not_run": True,
                },
                indent=2,
            )
        )
        print(json.dumps(report, indent=2))
        return 0
    cache: dict[str, tuple[list[CandidateFeatures], str]] = {}
    started = time.perf_counter()
    for meta in dev + test:
        cache[meta.case_id] = _cached_extract(meta)
    families = sorted({c.scenario_family for c in dev})
    folds = [{f for i, f in enumerate(families) if i % 3 == fold} for fold in range(3)]
    validation: dict[str, list[dict]] = {"Model A": [], "Model B": []}
    for fold_families in folds:
        train_meta = [c for c in dev if c.scenario_family not in fold_families]
        val_meta = [c for c in dev if c.scenario_family in fold_families]
        for name, features in (("Model A", MODEL_A_FEATURES), ("Model B", MODEL_B_FEATURES)):
            model = PairwiseRanker(features).fit([cache[c.case_id][0] for c in train_meta], [c.root_cause_service for c in train_meta])
            validation[name].extend(
                [{**_metrics(model.rank(cache[c.case_id][0]), c.root_cause_service), "case_id": c.case_id, "fold": len(validation[name])}
                 for c in val_meta]
            )
    validation_scores = {
        name: _aggregate(entries)
        for name, entries in validation.items()
    }
    selected = (
        "Model A"
        if validation_scores["Model A"]["mrr"] >= validation_scores["Model B"]["mrr"]
        else "Model B"
    )
    models = {"Model A": MODEL_A_FEATURES, "Model B": MODEL_B_FEATURES}
    trained = {name: PairwiseRanker(features).fit([cache[c.case_id][0] for c in dev], [c.root_cause_service for c in dev])
               for name, features in models.items()}
    locked: dict[str, list[dict]] = defaultdict(list)
    for meta in test:
        base = tuple(x.entity for x in sorted(cache[meta.case_id][0], key=lambda x: (-x.values["s_comb"], x.entity)))
        for name, model in trained.items():
            ranking = model.rank(cache[meta.case_id][0])
            locked[name].append({"case_id": meta.case_id, "family": meta.scenario_family, "ground_truth": meta.root_cause_service,
                                 "ranking": list(ranking), **_metrics(ranking, meta.root_cause_service)})
        locked["Digital Detective frozen RCA"].append({"case_id": meta.case_id, "family": meta.scenario_family,
            "ground_truth": meta.root_cause_service, "ranking": list(base), **_metrics(base, meta.root_cause_service)})
    dev_report = {"experiment": "ml_rca_development_v1", "development_datasets": ["RE2-SS", "RE2-TT"],
                  "family_count": len(families), "case_count": len(dev), "grouped_folds": 3,
                  "validation": {k: {"aggregate": validation_scores[k], "cases": v} for k, v in validation.items()},
                  "selected_model": selected, "feature_sets": {"Model A": list(MODEL_A_FEATURES), "Model B": list(MODEL_B_FEATURES)},
                  "elapsed_seconds": time.perf_counter() - started}
    test_report = {"experiment": "ml_rca_locked_test_v1", "locked_dataset": "RE2-OB repetitions 2 and 3",
                   "case_count": len(test), "selected_model": selected,
                   "aggregate": {k: _aggregate(v) for k, v in locked.items()},
                   "results": dict(locked), "no_post_test_tuning": True}
    (ROOT / "eval/results/ml_rca_development_v1.json").write_text(json.dumps(dev_report, indent=2))
    (ROOT / "eval/results/ml_rca_locked_test_v1.json").write_text(json.dumps(test_report, indent=2))
    print(json.dumps({"selected_model": selected, "development": dev_report["validation"], "locked": test_report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
