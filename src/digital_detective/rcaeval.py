"""Local, preservation-first loading for the current RCAEval Parquet layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)


_INDEX_FILE = "cases.parquet"
_GROUND_TRUTH_FIELDS = (
    "root_cause_service",
    "fault",
    "fault_description",
    "repetition",
    "inject_time",
)
_INCIDENT_METADATA_FIELDS = (
    "n_metrics",
    "n_timesteps",
    "time_start",
    "time_end",
    "duration_minutes",
    "normal_timesteps",
    "faulty_timesteps",
    "has_logs",
    "n_logs",
    "has_traces",
    "n_traces",
    "has_root_cause_file",
)


def load_rcaeval_case(
    dataset_root: str | Path,
    case_id: str,
    source_revision: str | None = None,
) -> TelemetryCase:
    """Load one current-layout RCAEval case from an already local dataset root.

    The returned modality data are ``pyarrow.Table`` objects, kept in their
    source field order. This function performs no telemetry normalization.
    """

    root = Path(dataset_root)
    index = _read_parquet(root / _INDEX_FILE, "RCAEval case index")
    matching_rows = [row for row in index.to_pylist() if row.get("case") == case_id]
    if not matching_rows:
        raise ValueError(f"RCAEval case {case_id!r} was not found in {root / _INDEX_FILE}")
    if len(matching_rows) != 1:
        raise ValueError(f"RCAEval case {case_id!r} occurs {len(matching_rows)} times in {root / _INDEX_FILE}")
    index_row = matching_rows[0]

    case_directory = root / case_id
    metrics = _load_required_modality(
        case_directory / "metrics.parquet",
        "metrics",
        index_row,
        case_id,
        source_revision,
        timestamp_field="time",
        identity_fields=(),
    )
    _verify_inject_time(case_directory / "inject_time.txt", index_row.get("inject_time"))

    return TelemetryCase(
        metadata=CaseMetadata(
            case_id=case_id,
            dataset=index_row.get("dataset"),
            suite=index_row.get("suite"),
            system=index_row.get("system"),
            system_name=index_row.get("system_name"),
            incident_metadata={
                field: index_row[field]
                for field in _INCIDENT_METADATA_FIELDS
                if field in index_row
            },
        ),
        ground_truth=GroundTruth(
            values={
                field: index_row[field]
                for field in _GROUND_TRUTH_FIELDS
                if field in index_row
            }
        ),
        metrics=metrics,
        logs=_load_optional_modality(
            case_directory / "logs.parquet",
            "logs",
            index_row,
            case_id,
            source_revision,
            timestamp_field="timestamp",
            identity_fields=("container_name",),
        ),
        traces=_load_optional_modality(
            case_directory / "traces.parquet",
            "traces",
            index_row,
            case_id,
            source_revision,
            timestamp_field="time",
            identity_fields=("traceID", "spanID", "serviceName"),
        ),
    )


def _load_required_modality(
    path: Path,
    modality: str,
    index_row: dict[str, Any],
    case_id: str,
    source_revision: str | None,
    *,
    timestamp_field: str,
    identity_fields: tuple[str, ...],
) -> TelemetryModality:
    if not path.is_file():
        raise FileNotFoundError(f"Required RCAEval {modality} file is missing: {path}")
    return _make_modality(
        _read_parquet(path, f"RCAEval {modality}"),
        index_row,
        case_id,
        source_revision,
        timestamp_field,
        identity_fields,
        original_format=path.name,
    )


def _load_optional_modality(
    path: Path,
    modality: str,
    index_row: dict[str, Any],
    case_id: str,
    source_revision: str | None,
    *,
    timestamp_field: str,
    identity_fields: tuple[str, ...],
) -> TelemetryModality | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"RCAEval optional {modality} path is not a file: {path}")
    return _make_modality(
        _read_parquet(path, f"RCAEval {modality}"),
        index_row,
        case_id,
        source_revision,
        timestamp_field,
        identity_fields,
        original_format=path.name,
    )


def _make_modality(
    table: Any,
    index_row: dict[str, Any],
    case_id: str,
    source_revision: str | None,
    timestamp_field: str,
    identity_fields: tuple[str, ...],
    *,
    original_format: str,
) -> TelemetryModality:
    return TelemetryModality(
        raw_data=table,
        provenance=ModalityProvenance(
            source_dataset=index_row["dataset"],
            source_case=case_id,
            source_revision=source_revision,
            original_format=original_format,
            original_field_names=tuple(table.column_names),
            timestamp_field=timestamp_field,
            identity_fields=identity_fields,
        ),
    )


def _verify_inject_time(path: Path, index_value: Any) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required RCAEval injection-time file is missing: {path}")
    try:
        text_value = path.read_text(encoding="utf-8").strip()
        file_value = int(text_value)
        expected_value = int(index_value)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Malformed RCAEval injection time in {path}") from error
    if file_value != expected_value:
        raise ValueError(f"RCAEval injection time in {path} does not match the case index")


def _read_parquet(path: Path, description: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} file is missing: {path}")
    try:
        return pq.read_table(path)
    except Exception as error:
        raise ValueError(f"Could not read {description} Parquet file: {path}") from error
