"""Shared data models for Digital Detective Stage 0 & Stage 1 evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CaseIdentity:
    """Metadata identifying a specific benchmark incident case."""

    case_id: str
    dataset: str
    system: str
    fault_type: str
    root_cause_service: str
    repetition: int


@dataclass(frozen=True)
class ModalityAvailability:
    """Explicit tracking of available telemetry modalities for a case."""

    has_metrics: bool = True
    has_traces: bool = False
    has_logs: bool = False


@dataclass(frozen=True)
class IncidentWindow:
    """Formal representation of the incident temporal window.

    Distinguishes oracle injection-time window from estimated/detected window.
    """

    onset_ts: int
    end_ts: int
    mode: str  # "oracle" | "detected"
    source_description: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {"oracle", "detected"}:
            raise ValueError(f"Invalid window mode: {self.mode!r}. Must be 'oracle' or 'detected'.")
        if self.end_ts < self.onset_ts:
            raise ValueError(f"end_ts ({self.end_ts}) cannot precede onset_ts ({self.onset_ts}).")


@dataclass(frozen=True)
class RankedEntity:
    """A candidate entity with its score and deterministic 1-indexed rank."""

    entity: str
    score: float
    rank: int


@dataclass(frozen=True)
class CaseMetrics:
    """Standard evaluation metrics for a single case prediction."""

    top1: bool
    top3: bool
    top5: bool
    mrr: float
    ac1: float
    ac2: float
    ac3: float
    ac4: float
    ac5: float
    avg3: float
    avg5: float
    target_rank: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_case_metrics(
    ranking: Sequence[RankedEntity | str],
    ground_truth_root_cause: str | Sequence[str],
) -> CaseMetrics:
    """Compute per-case evaluation metrics over the predicted ranking.

    Supports single or multiple valid root causes. The first hit determines rank.
    """
    targets = {ground_truth_root_cause} if isinstance(ground_truth_root_cause, str) else set(ground_truth_root_cause)

    entities: list[str] = []
    for item in ranking:
        if isinstance(item, RankedEntity):
            entities.append(item.entity)
        elif hasattr(item, "entity"):
            entities.append(getattr(item, "entity"))
        else:
            entities.append(str(item))

    target_rank: int | None = None
    for idx, ent in enumerate(entities, start=1):
        if ent in targets:
            target_rank = idx
            break

    top1 = target_rank is not None and target_rank <= 1
    top3 = target_rank is not None and target_rank <= 3
    top5 = target_rank is not None and target_rank <= 5
    mrr = 1.0 / target_rank if target_rank is not None else 0.0

    ac1 = 1.0 if (target_rank is not None and target_rank <= 1) else 0.0
    ac2 = 1.0 if (target_rank is not None and target_rank <= 2) else 0.0
    ac3 = 1.0 if (target_rank is not None and target_rank <= 3) else 0.0
    ac4 = 1.0 if (target_rank is not None and target_rank <= 4) else 0.0
    ac5 = 1.0 if (target_rank is not None and target_rank <= 5) else 0.0

    avg3 = (ac1 + ac2 + ac3) / 3.0
    avg5 = (ac1 + ac2 + ac3 + ac4 + ac5) / 5.0

    return CaseMetrics(
        top1=top1,
        top3=top3,
        top5=top5,
        mrr=mrr,
        ac1=ac1,
        ac2=ac2,
        ac3=ac3,
        ac4=ac4,
        ac5=ac5,
        avg3=avg3,
        avg5=avg5,
        target_rank=target_rank,
    )


@dataclass
class MethodRankingResult:
    """Full ranking and execution outcome for one method evaluated on one case."""

    case_id: str
    method_name: str
    window_mode: str
    ranking: tuple[RankedEntity, ...]
    candidate_universe: tuple[str, ...]
    status: str = "SUCCESS"  # "SUCCESS" | "UNAVAILABLE" | "ERROR"
    exclusion_reason: str | None = None
    metrics: CaseMetrics | None = None
    runtime_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "method_name": self.method_name,
            "window_mode": self.window_mode,
            "status": self.status,
            "exclusion_reason": self.exclusion_reason,
            "runtime_sec": round(self.runtime_sec, 6),
            "candidate_universe": list(self.candidate_universe),
            "ranking": [
                {"rank": r.rank, "entity": r.entity, "score": round(r.score, 6)}
                for r in self.ranking
            ],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AggregateMethodMetrics:
    """Aggregated evaluation metrics across a set of cases."""

    method_name: str
    total_cases: int
    evaluated_cases: int
    excluded_cases: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float
    ac1_accuracy: float
    ac2_accuracy: float
    ac3_accuracy: float
    ac4_accuracy: float
    ac5_accuracy: float
    avg3_accuracy: float
    avg5_accuracy: float
    mean_runtime_sec: float
    is_oracle: bool = False
    abstention_rate: float | str = "N/A"
    false_remediation_rate: float | str = "N/A"
    recovery_success_rate: float | str = "N/A"
    regression_rate: float | str = "N/A"
    query_efficiency: float | str = "N/A"

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        for k in (
            "top1_accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "mrr",
            "ac1_accuracy",
            "ac2_accuracy",
            "ac3_accuracy",
            "ac4_accuracy",
            "ac5_accuracy",
            "avg3_accuracy",
            "avg5_accuracy",
            "mean_runtime_sec",
        ):
            res[k] = round(res[k], 4)
        for k in (
            "abstention_rate",
            "false_remediation_rate",
            "recovery_success_rate",
            "regression_rate",
            "query_efficiency",
        ):
            if isinstance(res[k], float):
                res[k] = round(res[k], 4)
        return res


def aggregate_case_metrics(
    method_name: str,
    results: Sequence[MethodRankingResult],
    is_oracle: bool = False,
) -> AggregateMethodMetrics:
    """Aggregate a sequence of MethodRankingResults into summary metrics.

    Only evaluates cases where status == "SUCCESS" and metrics is not None.
    Computes safety, remediation, and efficiency metrics alongside existing RCA metrics.
    If a metric lacks a valid denominator, 'N/A' is reported rather than zero.
    """
    total = len(results)
    successful = [r for r in results if r.status == "SUCCESS" and r.metrics is not None]
    evaluated_count = len(successful)
    excluded_count = total - evaluated_count

    # 1. Abstention Rate:
    # Denominator: total_cases.
    # Numerator: cases where method explicitly abstained or failed to provide a diagnosis/ranking.
    if total > 0:
        abstained_cases = sum(
            1 for r in results
            if r.status != "SUCCESS"
            or r.metadata.get("abstained") is True
            or r.metadata.get("status") in {"ABSTAIN", "STOP"}
        )
        abstention_rate: float | str = abstained_cases / total
    else:
        abstention_rate = "N/A"

    # 2. Remediation Safety & Verification Metrics:
    # Denominator: authorized_remediations (cases where remediation was actually authorized).
    authorized_cases = [
        r for r in results
        if r.metadata.get("remediation_authorized") is True
    ]
    auth_count = len(authorized_cases)

    if auth_count > 0:
        # false_remediation_rate: fraction of authorized remediations targeted at an incorrect entity
        false_count = sum(
            1 for r in authorized_cases
            if not (r.metrics and r.metrics.top1)
        )
        false_remediation_rate: float | str = false_count / auth_count

        # recovery_success_rate: fraction of authorized remediations with verified recovery
        recovered_count = sum(
            1 for r in authorized_cases
            if r.metadata.get("recovery_verified") is True
            or r.metadata.get("verification_status") == "RESOLVED"
        )
        recovery_success_rate: float | str = recovered_count / auth_count

        # regression_rate: fraction of authorized remediations that induced regressions
        regression_count = sum(
            1 for r in authorized_cases
            if r.metadata.get("regression_detected") is True
        )
        regression_rate: float | str = regression_count / auth_count
    else:
        false_remediation_rate = "N/A"
        recovery_success_rate = "N/A"
        regression_rate = "N/A"

    # 3. Query Efficiency:
    # Denominator: total queries consumed across evaluated cases.
    # Numerator: total Top@1 hits.
    # Represents Top@1 accuracy achieved per telemetry query unit.
    total_queries = sum(r.metadata.get("queries_executed", 0) for r in results)
    if total_queries > 0:
        top1_hits = sum(1 for r in successful if r.metrics.top1)
        query_efficiency: float | str = top1_hits / total_queries
    else:
        query_efficiency = "N/A"

    if evaluated_count == 0:
        return AggregateMethodMetrics(
            method_name=method_name,
            total_cases=total,
            evaluated_cases=0,
            excluded_cases=excluded_count,
            top1_accuracy=0.0,
            top3_accuracy=0.0,
            top5_accuracy=0.0,
            mrr=0.0,
            ac1_accuracy=0.0,
            ac2_accuracy=0.0,
            ac3_accuracy=0.0,
            ac4_accuracy=0.0,
            ac5_accuracy=0.0,
            avg3_accuracy=0.0,
            avg5_accuracy=0.0,
            mean_runtime_sec=0.0,
            is_oracle=is_oracle,
            abstention_rate=abstention_rate,
            false_remediation_rate=false_remediation_rate,
            recovery_success_rate=recovery_success_rate,
            regression_rate=regression_rate,
            query_efficiency=query_efficiency,
        )

    top1 = sum(1 for r in successful if r.metrics.top1) / evaluated_count
    top3 = sum(1 for r in successful if r.metrics.top3) / evaluated_count
    top5 = sum(1 for r in successful if r.metrics.top5) / evaluated_count
    mrr = sum(r.metrics.mrr for r in successful) / evaluated_count

    ac1 = sum(r.metrics.ac1 for r in successful) / evaluated_count
    ac2 = sum(r.metrics.ac2 for r in successful) / evaluated_count
    ac3 = sum(r.metrics.ac3 for r in successful) / evaluated_count
    ac4 = sum(r.metrics.ac4 for r in successful) / evaluated_count
    ac5 = sum(r.metrics.ac5 for r in successful) / evaluated_count

    avg3 = sum(r.metrics.avg3 for r in successful) / evaluated_count
    avg5 = sum(r.metrics.avg5 for r in successful) / evaluated_count
    mean_time = sum(r.runtime_sec for r in successful) / evaluated_count

    return AggregateMethodMetrics(
        method_name=method_name,
        total_cases=total,
        evaluated_cases=evaluated_count,
        excluded_cases=excluded_count,
        top1_accuracy=top1,
        top3_accuracy=top3,
        top5_accuracy=top5,
        mrr=mrr,
        ac1_accuracy=ac1,
        ac2_accuracy=ac2,
        ac3_accuracy=ac3,
        ac4_accuracy=ac4,
        ac5_accuracy=ac5,
        avg3_accuracy=avg3,
        avg5_accuracy=avg5,
        mean_runtime_sec=mean_time,
        is_oracle=is_oracle,
        abstention_rate=abstention_rate,
        false_remediation_rate=false_remediation_rate,
        recovery_success_rate=recovery_success_rate,
        regression_rate=regression_rate,
        query_efficiency=query_efficiency,
    )
