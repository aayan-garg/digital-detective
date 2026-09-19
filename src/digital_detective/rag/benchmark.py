"""Leakage-controlled DD-IR-60 benchmark construction and annotation pooling."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa

from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.rcaeval import load_rcaeval_case

from .models import KnowledgeDocument

PROHIBITED_FIELDS = (
    "ground_truth",
    "root_cause_service",
    "fault_type",
    "inject_time",
    "injection_time",
    "post_recovery",
    "remediation",
    "final_rca",
)


def load_target_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if len(cases) != 60:
        raise ValueError(f"DD-IR-60 requires 60 cases, found {len(cases)}.")
    if {case["repetition"] for case in cases} != {2, 3}:
        raise ValueError("DD-IR-60 requires repetitions 2 and 3.")
    return payload


def build_queries(
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Construct one query per manifest case using only observed prefix telemetry."""

    queries = []
    for manifest_case in manifest["cases"]:
        case_id = str(manifest_case["case_id"])
        case = load_rcaeval_case(dataset_root, case_id)
        queries.append(_build_case_query(case, manifest_case))
    return tuple(queries)


def _build_case_query(case: Any, manifest_case: Mapping[str, Any]) -> dict[str, Any]:
    anomaly = detect_metric_anomalies(case, window_size=60, threshold=3.0)
    boundary = anomaly.summary["first_anomaly_timestamp"]
    if boundary is None:
        raise ValueError(f"No causally detected metric anomaly for {case.metadata.case_id}.")
    index = next(i for i, timestamp in enumerate(anomaly.timestamps) if timestamp == boundary)
    metric_table = case.metrics.raw_data
    metric_names = _observable_anomalies(anomaly, index)
    metric_phrases = [_metric_phrase(metric, metric_table, index) for metric in metric_names]
    log_phrases = _prefix_log_phrases(case, boundary)
    trace_phrases = _prefix_trace_phrases(case, boundary)
    observations = [phrase for phrase in metric_phrases + log_phrases + trace_phrases if phrase]
    if not observations:
        raise ValueError(f"No observable prefix signals for {case.metadata.case_id}.")
    query = (
        "Observed service telemetry shows "
        + "; ".join(observations[:8])
        + ". What operational guidance is relevant to investigating this condition?"
    )
    validate_query_leakage(query)
    return {
        "query_id": f"ddir60-{case.metadata.case_id}",
        "case_id": case.metadata.case_id,
        "scenario_family": "|".join(
            (
                str(manifest_case["suite"]),
                str(manifest_case["system"]),
                str(manifest_case["root_cause_service"]),
                str(manifest_case["fault"]),
            )
        ),
        "repetition": int(manifest_case["repetition"]),
        "split": "final_test",
        "query": query,
        "causal_prefix_boundary": int(boundary),
        "provenance": [
            {
                "kind": "metric_anomaly",
                "source": "metrics.parquet",
                "fields": metric_names,
                "boundary": int(boundary),
            },
            *(
                [
                    {
                        "kind": "log_observation",
                        "source": "logs.parquet",
                        "fields": ["timestamp", "container_name", "message"],
                        "boundary": int(boundary),
                    }
                ]
                if log_phrases
                else []
            ),
            *(
                [
                    {
                        "kind": "trace_observation",
                        "source": "traces.parquet",
                        "fields": ["startTimeMillis", "serviceName", "operationName", "statusCode"],
                        "boundary": int(boundary),
                    }
                ]
                if trace_phrases
                else []
            ),
        ],
        "observable_fields": ["metrics.parquet", "logs.parquet", "traces.parquet"],
    }


def _observable_anomalies(anomaly: Any, index: int) -> list[str]:
    names = [
        metric
        for metric in anomaly.metric_names
        if anomaly.anomalies[metric][index]
    ]
    return sorted(names)[:6]


def _metric_phrase(metric: str, table: pa.Table, index: int) -> str:
    values = table[metric].to_pylist()
    current = values[index]
    history = [value for value in values[max(0, index - 60):index] if value is not None]
    direction = "changed"
    if history and current is not None:
        baseline = sum(history) / len(history)
        direction = "increased" if current > baseline else "decreased"
    label = metric.replace("_", " ").replace("-", " ")
    return f"{label} {direction}"


def _prefix_log_phrases(case: Any, boundary: int) -> list[str]:
    if case.logs is None:
        return []
    rows = case.logs.raw_data.filter(
        pa.compute.less_equal(case.logs.raw_data["timestamp"], boundary)
    ).to_pylist()
    signals = []
    for row in rows:
        message = str(row.get("message", ""))
        if re.search(r"\b(error|exception|timeout|timed out|failed|failure|unavailable|refused)\b", message, re.I):
            normalized = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", message, flags=re.I)
            normalized = re.sub(r"\b\d+\b", "<n>", normalized)
            signals.append(f"{row.get('container_name', 'service')} logs report {normalized[:120]}")
    return list(dict.fromkeys(signals))[:3]


def _prefix_trace_phrases(case: Any, boundary: int) -> list[str]:
    if case.traces is None:
        return []
    table = case.traces.raw_data
    mask = pa.compute.less_equal(table["startTimeMillis"], boundary * 1000)
    rows = table.filter(mask).to_pylist()
    services = sorted({str(row.get("serviceName")) for row in rows if row.get("serviceName")})
    failures = sorted({str(row.get("statusCode")) for row in rows if row.get("statusCode") not in (None, 0, "0")})
    phrases = []
    if services:
        phrases.append("traces include services " + ", ".join(services[:6]))
    if failures:
        phrases.append("traces include non-success status codes " + ", ".join(failures[:4]))
    return phrases


def validate_query_leakage(query: str) -> None:
    lowered = query.lower()
    violations = [field for field in PROHIBITED_FIELDS if field in lowered]
    if violations:
        raise ValueError(f"Prohibited query fields detected: {violations}")


def corpus_audit(documents: Sequence[KnowledgeDocument]) -> tuple[dict[str, Any], ...]:
    """Represent the frozen corpus with explicit provenance/admission decisions."""

    audited = []
    for document in documents:
        metadata = dict(document.metadata)
        complete = all(
            metadata.get(field)
            for field in ("source", "version", "updated_at", "license", "document_type")
        )
        audited.append(
            {
                "document_id": document.document_id,
                "title": document.title,
                "text": document.text,
                "metadata": metadata,
                "source": metadata.get("source", ""),
                "version": metadata.get("version", ""),
                "updated_at": metadata.get("updated_at", ""),
                "license": metadata.get("license", ""),
                "document_type": metadata.get("document_type", ""),
                "temporal_admission": "admitted" if complete else "excluded_unknown_provenance",
                "admission_reason": "complete provenance" if complete else "required source/version/date/license/type unavailable",
            }
        )
    return tuple(audited)


def build_pool(
    queries: Sequence[Mapping[str, Any]],
    retrieval_records: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    *,
    background_count: int = 5,
    seed: int = 60,
) -> tuple[dict[str, Any], ...]:
    """Build a blinded deterministic pool from existing C0-C4 output records."""

    by_query: dict[str, list[str]] = {}
    corpus_ids = [str(item["document_id"]) for item in corpus]
    for query in queries:
        query_id = str(query["query_id"])
        ids = []
        records = [r for r in retrieval_records if r["query_id"] == query_id]
        for record in records:
            ids.extend(item["document_id"] for item in record.get("retrieved", [])[:5])
        rng = random.Random(f"{seed}:{query_id}")
        remaining = [doc_id for doc_id in corpus_ids if doc_id not in ids]
        ids.extend(rng.sample(remaining, min(background_count, len(remaining))))
        by_query[query_id] = list(dict.fromkeys(ids))
    pooled = []
    for query_id, document_ids in by_query.items():
        order = list(document_ids)
        random.Random(f"{seed}:presentation:{query_id}").shuffle(order)
        for position, document_id in enumerate(order, 1):
            pooled.append({"query_id": query_id, "presentation_id": f"{query_id}-p{position:03d}", "document_id": document_id})
    return tuple(pooled)


def manifest_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
