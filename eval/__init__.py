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
from .stats import (
    PairedCaseObservation,
    StatisticalComparisonResult,
    align_paired_observations,
    apply_holm_correction,
    cluster_bootstrap_ci,
    compare_methods_paired,
    compute_quantile,
    paired_cluster_randomization_test,
    validate_scenario_family,
)

__all__ = [
    "BenchmarkManifest",
    "CANONICAL_UNIVERSES",
    "CaseIdentity",
    "CaseMetrics",
    "IncidentWindow",
    "ManifestCase",
    "MethodRankingResult",
    "ModalityAvailability",
    "PairedCaseObservation",
    "RankedEntity",
    "StatisticalComparisonResult",
    "align_paired_observations",
    "apply_holm_correction",
    "cluster_bootstrap_ci",
    "compare_methods_paired",
    "compute_quantile",
    "paired_cluster_randomization_test",
    "resolve_candidate_universe",
    "resolve_incident_window",
    "validate_scenario_family",
]

