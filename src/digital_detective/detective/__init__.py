"""Digital Detective Core Investigation Engine.

Provides deterministic autonomous exploration under strict query budgets,
multi-modal evidence fusion, causal consistency validation, safe remediation,
and post-remediation recovery verification.
"""

from .investigator import InvestigationEngine
from .models import (
    CausalConsistencyResult,
    EvidenceItem,
    InterventionValidationResult,
    InvestigationState,
    RankedHypothesis,
    RecoveryVerificationResult,
    RemediationAction,
    RemediationExecutionResult,
    RootCauseDecision,
    ToolQueryRecord,
)
from .remediation import (
    execute_simulated_remediation,
    propose_remediation,
    validate_remediation,
)
from .scoring import EvidenceFusion
from .tools import DetectiveTools
from .validation import CausalConsistencyValidator
from .verification import SystemStateSnapshot, validate_intervention, verify_recovery

__all__ = [
    "CausalConsistencyResult",
    "CausalConsistencyValidator",
    "DetectiveTools",
    "EvidenceFusion",
    "EvidenceItem",
    "InterventionValidationResult",
    "InvestigationEngine",
    "InvestigationState",
    "RankedHypothesis",
    "RecoveryVerificationResult",
    "RemediationAction",
    "RemediationExecutionResult",
    "RootCauseDecision",
    "SystemStateSnapshot",
    "ToolQueryRecord",
    "execute_simulated_remediation",
    "propose_remediation",
    "validate_intervention",
    "validate_remediation",
    "verify_recovery",
]
