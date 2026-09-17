"""Evaluation infrastructure for Digital Detective Stage 0 and Stage 1 research.
"""

from .models import (
    CaseIdentity,
    CaseMetrics,
    IncidentWindow,
    MethodRankingResult,
    ModalityAvailability,
    RankedEntity,
)
from .windows import resolve_incident_window
from .universe import resolve_candidate_universe, CANONICAL_UNIVERSES
from .manifest import BenchmarkManifest, ManifestCase

__all__ = [
    "BenchmarkManifest",
    "CANONICAL_UNIVERSES",
    "CaseIdentity",
    "CaseMetrics",
    "IncidentWindow",
    "ManifestCase",
    "MethodRankingResult",
    "ModalityAvailability",
    "RankedEntity",
    "resolve_candidate_universe",
    "resolve_incident_window",
]
