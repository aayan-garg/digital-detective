"""Evaluation helpers for LLM diagnosis runs separate from deterministic RCA."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from .models import LLMExperimentRecord


@dataclass(frozen=True)
class LLMEvaluationSummary:
    """Aggregate metrics for a set of LLM diagnosis runs."""

    total_cases: int
    top1: float
    top3: float
    abstention_rate: float
    mean_query_count: float
    mean_query_cost: float
    mean_latency_sec: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_cases": self.total_cases,
            "top1": self.top1,
            "top3": self.top3,
            "abstention_rate": self.abstention_rate,
            "mean_query_count": self.mean_query_count,
            "mean_query_cost": self.mean_query_cost,
            "mean_latency_sec": self.mean_latency_sec,
        }


def llm_topk_accuracy(records: Sequence[LLMExperimentRecord], k: int = 1) -> float:
    """Return the fraction of non-abstaining LLM diagnoses that land in top-k."""
    if k < 1:
        raise ValueError("k must be >= 1")
    valid = [r for r in records if not r.abstain and r.llm_diagnosis]
    if not valid:
        return 0.0
    hits = 0
    for record in valid:
        rankings = record.deterministic_rankings or (
            record.deterministic_top_service,
        )
        if not rankings:
            continue
        if record.llm_diagnosis in rankings[:k]:
            hits += 1
    return hits / len(valid)


def llm_abstention_rate(records: Sequence[LLMExperimentRecord]) -> float:
    """Return the fraction of LLM runs that abstained."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.abstain) / len(records)


def summarize_llm_records(records: Sequence[LLMExperimentRecord]) -> LLMEvaluationSummary:
    """Aggregate LLM diagnosis metrics across a set of runs."""
    if not records:
        return LLMEvaluationSummary(
            total_cases=0,
            top1=0.0,
            top3=0.0,
            abstention_rate=0.0,
            mean_query_count=0.0,
            mean_query_cost=0.0,
            mean_latency_sec=0.0,
        )

    total = len(records)
    valid = [r for r in records if not r.abstain and r.llm_diagnosis]
    return LLMEvaluationSummary(
        total_cases=total,
        top1=llm_topk_accuracy(records, k=1),
        top3=llm_topk_accuracy(records, k=3),
        abstention_rate=llm_abstention_rate(records),
        mean_query_count=mean(r.query_count for r in records),
        mean_query_cost=mean(r.query_cost for r in records),
        mean_latency_sec=mean(r.latency_sec for r in records),
    )


def summarize_condition_records(records_by_condition: dict[str, Sequence[LLMExperimentRecord]]) -> dict[str, LLMEvaluationSummary]:
    """Summarize records by condition label such as 'A' or 'C'."""
    return {condition: summarize_llm_records(record_list) for condition, record_list in records_by_condition.items()}
