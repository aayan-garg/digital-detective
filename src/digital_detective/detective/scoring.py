"""Evidence fusion for Digital Detective hypothesis scoring.

Combines heterogeneous telemetry evidence (metrics, traces, logs, topology, and temporal ordering)
into normalized composite hypothesis scores. Does not replace S_comb or E_elev; reuses their
proven formulations as primary evidence primitives.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .models import EvidenceItem, RankedHypothesis

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "metric": 0.30,
    "trace": 0.30,
    "change": 0.15,
    "log": 0.10,
    "topology": 0.05,
    "temporal": 0.10,
}


class EvidenceFusion:
    """Combines structured evidence items across modalities into normalized candidate rankings."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def rank_hypotheses(
        self,
        candidate_universe: Sequence[str],
        evidence_items: Sequence[EvidenceItem],
        initial_metric_scores: Mapping[str, float] | None = None,
        initial_trace_scores: Mapping[str, float] | None = None,
        consistency_scores: Mapping[str, float] | None = None,
    ) -> tuple[RankedHypothesis, ...]:
        """Compute composite scores and rank all candidate services in the closed universe.

        Scoring Formula:
            score(s) = w_metric * norm_metric(s)
                     + w_trace  * norm_trace(s)
                     + w_log    * norm_log(s)
                     + w_topo   * norm_topo(s)
                     + w_temp   * norm_temp(s)

        Parameters
        ----------
        candidate_universe:
            Deterministic sequence of candidate services to be ranked.
        evidence_items:
            Collected discrete evidence items.
        initial_metric_scores:
            Optional baseline metric scores (e.g. from S_comb).
        initial_trace_scores:
            Optional baseline trace scores (e.g. from E_elev).

        Returns
        -------
        tuple[RankedHypothesis, ...]
            Deterministically ranked candidate entities with component scores.
        """
        candidates = sorted(set(candidate_universe))

        # Collect raw values per candidate
        metric_raw: dict[str, float] = {s: 0.0 for s in candidates}
        trace_raw: dict[str, float] = {s: 0.0 for s in candidates}
        log_raw: dict[str, float] = {s: 0.0 for s in candidates}
        topo_raw: dict[str, float] = {s: 0.0 for s in candidates}
        change_raw: dict[str, float] = {s: 0.0 for s in candidates}
        earliest_ts: dict[str, int] = {}
        conf_sum: dict[str, float] = {s: 0.0 for s in candidates}

        # Seed from baseline rankers if available
        if initial_metric_scores is not None:
            for s, val in initial_metric_scores.items():
                if s in metric_raw:
                    metric_raw[s] = max(metric_raw[s], float(val))

        if initial_trace_scores is not None:
            for s, val in initial_trace_scores.items():
                if s in trace_raw:
                    trace_raw[s] = max(trace_raw[s], float(val))

        # Ingest discrete evidence items
        for ev in evidence_items:
            s = ev.service
            if s not in candidates:
                continue

            conf_sum[s] += ev.confidence_contribution

            if ev.timestamp is not None:
                if s not in earliest_ts or ev.timestamp < earliest_ts[s]:
                    earliest_ts[s] = ev.timestamp

            if ev.modality == "metrics":
                metric_raw[s] = max(metric_raw[s], ev.magnitude)
            elif ev.modality == "traces":
                trace_raw[s] = max(trace_raw[s], ev.magnitude)
            elif ev.modality == "logs":
                log_raw[s] += ev.magnitude
            elif ev.modality == "topology":
                topo_raw[s] = max(topo_raw[s], ev.magnitude)
            elif ev.modality == "recent_change":
                change_raw[s] = max(change_raw[s], ev.magnitude)
            elif ev.modality == "health":
                metric_raw[s] = max(metric_raw[s], 0.5 * ev.magnitude)

        # Normalization across candidate universe
        max_m = max(metric_raw.values()) if metric_raw else 0.0
        max_t = max(trace_raw.values()) if trace_raw else 0.0
        log_log1p = {s: math.log1p(log_raw[s]) for s in candidates}
        max_l = max(log_log1p.values()) if log_log1p else 0.0
        max_topo = max(topo_raw.values()) if topo_raw else 0.0

        norm_m = {s: (metric_raw[s] / max_m if max_m > 0 else 0.0) for s in candidates}
        norm_t = {s: (trace_raw[s] / max_t if max_t > 0 else 0.0) for s in candidates}
        norm_l = {s: (log_log1p[s] / max_l if max_l > 0 else 0.0) for s in candidates}
        norm_topo = {s: (topo_raw[s] / max_topo if max_topo > 0 else 0.0) for s in candidates}
        norm_ch = {s: change_raw[s] for s in candidates}

        # Temporal precedence scoring: earlier onset gets higher score
        min_ts = min(earliest_ts.values()) if earliest_ts else None
        norm_temp: dict[str, float] = {}
        for s in candidates:
            if s in earliest_ts and min_ts is not None:
                dt = earliest_ts[s] - min_ts
                norm_temp[s] = max(0.0, 1.0 - (dt / 300.0))
            else:
                norm_temp[s] = 0.0

        # Compute evidence scores and composite scores
        w_m = self.weights.get("metric", 0.30)
        w_t = self.weights.get("trace", 0.30)
        w_ch = self.weights.get("change", 0.15)
        w_l = self.weights.get("log", 0.10)
        w_topo = self.weights.get("topology", 0.05)
        w_temp = self.weights.get("temporal", 0.10)

        evidence_scores: dict[str, float] = {}
        composite_scores: dict[str, float] = {}
        cc_scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}

        for s in candidates:
            ev_score = (
                w_m * norm_m[s]
                + w_t * norm_t[s]
                + w_ch * norm_ch[s]
                + w_l * norm_l[s]
                + w_topo * norm_topo[s]
                + w_temp * norm_temp[s]
            )
            evidence_scores[s] = ev_score

            cc_val = consistency_scores.get(s, 1.0) if consistency_scores is not None else 1.0
            cc_scores[s] = cc_val

            # Consistency multiplier in [0.4, 1.0]
            c_mult = 0.4 + 0.6 * max(0.0, min(1.0, cc_val))
            composite_scores[s] = ev_score * c_mult

            components[s] = {
                "metric": norm_m[s],
                "trace": norm_t[s],
                "change": norm_ch[s],
                "log": norm_l[s],
                "topology": norm_topo[s],
                "temporal": norm_temp[s],
            }

        # Deterministic sorting: (-composite_score, service_name)
        sorted_candidates = sorted(
            candidates,
            key=lambda s: (-composite_scores[s], s),
        )

        # Compute margin between #1 and #2 for decision_confidence
        runner_up_score = composite_scores[sorted_candidates[1]] if len(sorted_candidates) > 1 else 0.0
        top_score = composite_scores[sorted_candidates[0]] if sorted_candidates else 0.0
        top_margin = max(0.0, min(1.0, (top_score - runner_up_score) / max(top_score, 1e-6)))

        ranked: list[RankedHypothesis] = []
        for rank_idx, s in enumerate(sorted_candidates, start=1):
            c_sc = composite_scores[s]
            ev_sc = evidence_scores[s]
            cc_sc = cc_scores[s]

            if rank_idx == 1:
                # Top hypothesis decision confidence combines composite score, consistency, and separation margin
                dec_conf = min(0.99, max(0.05, c_sc * (0.75 + 0.25 * top_margin))) if c_sc > 0 else 0.0
            else:
                dec_conf = min(0.99, max(0.0, c_sc * cc_sc)) if c_sc > 0 else 0.0

            ranked.append(
                RankedHypothesis(
                    service=s,
                    score=c_sc,
                    confidence=dec_conf,
                    rank=rank_idx,
                    evidence_score=ev_sc,
                    consistency_score=cc_sc,
                    component_scores=components[s],
                )
            )

        return tuple(ranked)
