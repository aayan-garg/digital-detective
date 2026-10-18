"""Log-based evidence extraction and scoring using Drain3 template mining.

Extracts log templates, tracks baseline-relative template frequencies, detects
novel/emerging templates, and flags error elevations across microservice entities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    DRAIN3_AVAILABLE = True
except ImportError:
    DRAIN3_AVAILABLE = False

from digital_detective.telemetry import TelemetryCase


ERROR_PATTERNS = re.compile(
    r"\b(error|exception|fail|failed|failure|fatal|timeout|refused|oom|panic|50\d|unavailable)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EntityLogEvidence:
    entity: str
    total_log_count: int
    baseline_log_count: int
    incident_log_count: int
    incident_to_baseline_ratio: float
    novel_template_count: int
    error_log_count: int
    log_anomaly_score: float
    top_templates: Sequence[tuple[str, int]] = ()

    @property
    def anomaly_score(self) -> float:
        return self.log_anomaly_score

    @property
    def error_level_count(self) -> int:
        return self.error_log_count


@dataclass(frozen=True)
class LogScoringResult:
    case_id: str
    has_logs: bool
    entity_evidence: Mapping[str, EntityLogEvidence]
    total_novel_templates: int
    summary: str

    @property
    def total_logs_processed(self) -> int:
        return sum(ev.total_log_count for ev in self.entity_evidence.values())


LogScorerResult = LogScoringResult


def score_logs_evidence(
    case: TelemetryCase,
    candidate_universe: Sequence[str] = (),
    *,
    onset_timestamp: int | None = None,
    baseline_window_seconds: int = 120,
    max_log_samples: int = 3000,
) -> LogScoringResult:
    """Extract log template anomalies and frequency shifts across microservice entities.

    Parameters
    ----------
    case:
        TelemetryCase containing optional `case.logs`.
    candidate_universe:
        Universe of known service entities.
    onset_timestamp:
        Timestamp marking incident onset.
    baseline_window_seconds:
        Duration of pre-onset window treated as baseline if onset_timestamp given.
    max_log_samples:
        Maximum number of log entries to process for efficiency.
    """
    case_id = case.metadata.case_id if case.metadata else "unknown"
    candidates = tuple(candidate_universe)

    if case.logs is None or case.logs.raw_data is None:
        return _empty_log_result(case_id, candidates, "No logs modality present in case")

    table = case.logs.raw_data
    if "container_name" not in table.column_names or "message" not in table.column_names:
        return _empty_log_result(case_id, candidates, "Logs table missing required columns")

    num_rows = len(table)
    if num_rows == 0:
        return _empty_log_result(case_id, candidates, "Logs table is empty")

    ts_col = "timestamp" if "timestamp" in table.column_names else ("time" if "time" in table.column_names else None)
    if ts_col is None:
        return _empty_log_result(case_id, candidates, "Logs table missing timestamp column")
    timestamps = table[ts_col].to_numpy()
    containers = [str(c).replace("frontendservice", "frontend") for c in table["container_name"].to_pylist()]
    messages = table["message"].to_pylist()

    # Determine onset timestamp if not provided
    if onset_timestamp is None:
        min_ts = int(timestamps[0])
        max_ts = int(timestamps[-1])
        onset_timestamp = min_ts + (max_ts - min_ts) // 2

    # Sample if logs are extremely large to ensure responsive execution
    if num_rows > max_log_samples:
        step = max(1, num_rows // max_log_samples)
        indices = np.arange(0, num_rows, step)
    else:
        indices = np.arange(num_rows)

    # Initialize Drain3 template miner if available, otherwise regex fallback
    miner = None
    if DRAIN3_AVAILABLE:
        cfg = TemplateMinerConfig()
        cfg.drain_depth = 4
        cfg.drain_sim_th = 0.5
        miner = TemplateMiner(config=cfg)

    # Pass 1: Learn baseline templates
    baseline_template_ids_by_entity: dict[str, set[str]] = {}
    incident_logs_by_entity: dict[str, list[tuple[int, str]]] = {}
    baseline_counts: dict[str, int] = {}
    incident_counts: dict[str, int] = {}

    for idx in indices:
        ts = int(timestamps[idx])
        entity = containers[idx]
        msg = messages[idx]

        if ts < onset_timestamp:
            baseline_counts[entity] = baseline_counts.get(entity, 0) + 1
            if miner is not None:
                res = miner.add_log_message(msg)
                cluster_id = str(res.get("cluster_id", "default"))
                baseline_template_ids_by_entity.setdefault(entity, set()).add(cluster_id)
        else:
            incident_counts[entity] = incident_counts.get(entity, 0) + 1
            incident_logs_by_entity.setdefault(entity, []).append((ts, msg))

    # Pass 2: Evaluate incident logs against baseline templates
    all_entities = set(candidates) | set(containers)
    evidence_map: dict[str, EntityLogEvidence] = {}
    total_novel = 0

    for ent in all_entities:
        base_cnt = baseline_counts.get(ent, 0)
        inc_cnt = incident_counts.get(ent, 0)
        tot_cnt = base_cnt + inc_cnt

        inc_logs = incident_logs_by_entity.get(ent, [])
        known_templates = baseline_template_ids_by_entity.get(ent, set())

        novel_templates_cnt = 0
        error_cnt = 0
        template_freqs: dict[str, int] = {}

        for _, msg in inc_logs:
            if ERROR_PATTERNS.search(msg):
                error_cnt += 1

            if miner is not None:
                res = miner.add_log_message(msg)
                c_id = str(res.get("cluster_id", "default"))
                t_str = str(res.get("template_mined", msg[:60]))
                template_freqs[t_str] = template_freqs.get(t_str, 0) + 1
                if c_id not in known_templates:
                    novel_templates_cnt += 1
            else:
                # Regex heuristic for template
                norm = re.sub(r"\d+", "<*>", msg[:80])
                template_freqs[norm] = template_freqs.get(norm, 0) + 1

        top_t = sorted(template_freqs.items(), key=lambda x: -x[1])[:3]
        ratio = (inc_cnt + 1.0) / (base_cnt + 1.0)
        total_novel += novel_templates_cnt

        # Log anomaly score combines novel templates, error elevation, and burst ratio
        error_rate = error_cnt / max(1, inc_cnt)
        norm_ratio = min(5.0, ratio) / 5.0
        norm_novel = min(10.0, float(novel_templates_cnt)) / 10.0
        anomaly_score = float(0.4 * error_rate + 0.35 * norm_novel + 0.25 * norm_ratio)

        evidence_map[ent] = EntityLogEvidence(
            entity=ent,
            total_log_count=tot_cnt,
            baseline_log_count=base_cnt,
            incident_log_count=inc_cnt,
            incident_to_baseline_ratio=ratio,
            novel_template_count=novel_templates_cnt,
            error_log_count=error_cnt,
            log_anomaly_score=anomaly_score,
            top_templates=tuple(top_t),
        )

    summary_str = f"Evaluated {len(all_entities)} entities; {total_novel} novel template occurrences detected."
    return LogScoringResult(
        case_id=case_id,
        has_logs=True,
        entity_evidence=evidence_map,
        total_novel_templates=total_novel,
        summary=summary_str,
    )


def _empty_log_result(case_id: str, candidates: Sequence[str], reason: str) -> LogScoringResult:
    evidence = {
        ent: EntityLogEvidence(
            entity=ent,
            total_log_count=0,
            baseline_log_count=0,
            incident_log_count=0,
            incident_to_baseline_ratio=1.0,
            novel_template_count=0,
            error_log_count=0,
            log_anomaly_score=0.0,
            top_templates=(),
        )
        for ent in candidates
    }
    return LogScoringResult(
        case_id=case_id,
        has_logs=False,
        entity_evidence=evidence,
        total_novel_templates=0,
        summary=reason,
    )
