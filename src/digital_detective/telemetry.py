"""Preservation-first containers for incident telemetry.

The model deliberately keeps modality data in its source-shaped form. Loading,
normalization, and interpretation belong to later, separately scoped work.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class CaseMetadata:
    """Verified source-level identity and incident metadata for one case."""

    case_id: str
    dataset: str | None = None
    suite: str | None = None
    system: str | None = None
    system_name: str | None = None
    incident_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    """Raw source ground truth, deliberately separate from telemetry."""

    values: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModalityProvenance:
    """Source information shared by all records in one telemetry modality."""

    source_dataset: str
    source_case: str
    original_format: str
    original_field_names: tuple[str, ...]
    timestamp_field: str | None = None
    identity_fields: tuple[str, ...] = ()
    source_revision: str | None = None
    transformation_status: str = "raw"


@dataclass
class TelemetryModality:
    """Raw data and its modality-level provenance, without interpretation."""

    raw_data: Any
    provenance: ModalityProvenance


@dataclass
class TelemetryCase:
    """One incident case with independently optional telemetry modalities."""

    metadata: CaseMetadata
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    metrics: TelemetryModality | None = None
    logs: TelemetryModality | None = None
    traces: TelemetryModality | None = None
