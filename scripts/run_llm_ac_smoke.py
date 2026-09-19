#!/usr/bin/env python3
"""Small smoke comparison of LLM A-vs-C behavior on matched RE2-OB cases."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from digital_detective.agent.evaluation import summarize_condition_records
from digital_detective.agent.models import LLMExperimentRecord


CASE_IDS = (
    "re2ob_checkoutservice_cpu_2",
    "re2ob_currencyservice_cpu_2",
    "re2ob_emailservice_cpu_2",
    "re2ob_checkoutservice_delay_2",
    "re2ob_currencyservice_delay_2",
)


def _record(
    case_id: str,
    *,
    condition: str,
    model: str,
    llm_service: str,
    abstain: bool,
    query_count: int,
    query_cost: int,
    latency: float,
    deterministic_rankings: tuple[str, ...],
    evidence_ids: tuple[str, ...] = (),
    reasoning: str = "",
) -> LLMExperimentRecord:
    return LLMExperimentRecord(
        case_id=case_id,
        condition=condition,
        model=model,
        llm_diagnosis=llm_service,
        abstain=abstain,
        reasoning=reasoning or ("Abstained to avoid unsupported final diagnosis." if abstain else f"LLM selected {llm_service}."),
        evidence_ids=evidence_ids,
        query_count=query_count,
        query_cost=query_cost,
        termination_action="STOP" if abstain else "FINAL_DIAGNOSIS",
        latency_sec=latency,
        deterministic_top_service=deterministic_rankings[0],
        deterministic_rankings=deterministic_rankings,
        detected_window={"mode": "detected", "onset_ts": 1000, "end_ts": 2000},
        metadata={"matched_case": True},
    )


def _condition_a_records() -> list[LLMExperimentRecord]:
    rankings = {
        "re2ob_checkoutservice_cpu_2": ("checkoutservice", "paymentservice", "currencyservice"),
        "re2ob_currencyservice_cpu_2": ("currencyservice", "checkoutservice", "paymentservice"),
        "re2ob_emailservice_cpu_2": ("emailservice", "checkoutservice", "paymentservice"),
        "re2ob_checkoutservice_delay_2": ("checkoutservice", "emailservice", "currencyservice"),
        "re2ob_currencyservice_delay_2": ("currencyservice", "checkoutservice", "emailservice"),
    }
    diagnosis = {
        "re2ob_checkoutservice_cpu_2": "checkoutservice",
        "re2ob_currencyservice_cpu_2": "currencyservice",
        "re2ob_emailservice_cpu_2": "emailservice",
        "re2ob_checkoutservice_delay_2": "STOP",
        "re2ob_currencyservice_delay_2": "currencyservice",
    }
    abstain = {
        "re2ob_checkoutservice_cpu_2": False,
        "re2ob_currencyservice_cpu_2": False,
        "re2ob_emailservice_cpu_2": False,
        "re2ob_checkoutservice_delay_2": True,
        "re2ob_currencyservice_delay_2": False,
    }
    return [
        _record(
            cid,
            condition="A",
            model="mock-llm",
            llm_service=diagnosis[cid],
            abstain=abstain[cid],
            query_count=0,
            query_cost=0,
            latency=0.42 + idx * 0.05,
            deterministic_rankings=rankings[cid],
            evidence_ids=(f"q_{idx+1}",),
            reasoning="Single-shot diagnosis from observed evidence.",
        )
        for idx, cid in enumerate(CASE_IDS)
    ]


def _condition_c_records() -> list[LLMExperimentRecord]:
    rankings = {
        "re2ob_checkoutservice_cpu_2": ("checkoutservice", "paymentservice", "currencyservice"),
        "re2ob_currencyservice_cpu_2": ("currencyservice", "checkoutservice", "paymentservice"),
        "re2ob_emailservice_cpu_2": ("emailservice", "checkoutservice", "paymentservice"),
        "re2ob_checkoutservice_delay_2": ("checkoutservice", "emailservice", "currencyservice"),
        "re2ob_currencyservice_delay_2": ("currencyservice", "checkoutservice", "emailservice"),
    }
    return [
        _record(
            cid,
            condition="C",
            model="mock-llm",
            llm_service=rankings[cid][0],
            abstain=False,
            query_count=2 + idx,
            query_cost=3 + idx,
            latency=0.75 + idx * 0.12,
            deterministic_rankings=rankings[cid],
            evidence_ids=(f"q_{idx+1}", f"q_{idx+2}"),
            reasoning="Iterative diagnosis after bounded query loop.",
        )
        for idx, cid in enumerate(CASE_IDS)
    ]


def main() -> None:
    condition_records = {
        "A": _condition_a_records(),
        "C": _condition_c_records(),
    }
    summaries = summarize_condition_records(condition_records)

    print("=" * 80)
    print("LLM A vs C smoke experiment (5 matched RE2-OB cases)")
    print("=" * 80)
    for condition in ("A", "C"):
        summary = summaries[condition]
        print(f"Condition {condition}:")
        print(f"  Top@1: {summary.top1:.3f}")
        print(f"  Top@3: {summary.top3:.3f}")
        print(f"  Abstention rate: {summary.abstention_rate:.3f}")
        print(f"  Mean query count: {summary.mean_query_count:.2f}")
        print(f"  Mean query cost: {summary.mean_query_cost:.2f}")
        print(f"  Mean latency (s): {summary.mean_latency_sec:.3f}")
        print()


if __name__ == "__main__":
    main()
