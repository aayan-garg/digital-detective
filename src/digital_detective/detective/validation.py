"""Lightweight causal-consistency checker.

Audits suspected root causes for:
A. Temporal ordering (suspected cause becomes anomalous before downstream symptoms).
B. Topology consistency (suspected cause is upstream or connected to affected entities).
C. Propagation consistency (downstream symptoms are compatible with the cause).
D. Evidence independence (confidence not inflated by redundant counts of identical signals).

Note: This verifies operational causal consistency under stated domain assumptions;
it does NOT perform formal interventional causal inference.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from eval.universe import normalize_service_name
from .models import CausalConsistencyResult, EvidenceItem


class CausalConsistencyValidator:
    """Validates operational causal consistency for suspected root causes."""

    def __init__(self, graph: Any | None = None) -> None:
        self.graph = graph

    def evaluate_consistency(
        self,
        target_service: str,
        evidence_items: Sequence[EvidenceItem],
        earliest_onsets: Mapping[str, int | None] | None = None,
    ) -> CausalConsistencyResult:
        """Evaluate temporal, topological, and propagation consistency for the target.

        Parameters
        ----------
        target_service:
            Suspected root cause service entity.
        evidence_items:
            All evidence items collected across services during investigation.
        earliest_onsets:
            Mapping from service name to earliest anomaly timestamp if known.

        Returns
        -------
        CausalConsistencyResult
            Structured scores and pass/fail verdict.
        """
        target = normalize_service_name(target_service)
        warnings: list[str] = []
        onsets = earliest_onsets or {}

        # 1. Temporal Ordering Check
        target_onset = onsets.get(target)
        downstream_onsets = [
            ts for s, ts in onsets.items()
            if s != target and ts is not None
        ]

        if target_onset is not None and downstream_onsets:
            min_downstream = min(downstream_onsets)
            if target_onset <= min_downstream:
                temporal_score = 1.0
            elif target_onset - min_downstream <= 15:
                # Within rolling detection window tolerance
                temporal_score = 0.85
                warnings.append(
                    f"Target onset (t={target_onset}) is within 15s of earliest symptom (t={min_downstream})."
                )
            else:
                temporal_score = 0.40
                warnings.append(
                    f"Temporal inversion: Target onset (t={target_onset}) occurred after downstream symptom (t={min_downstream})."
                )
        elif target_onset is not None:
            temporal_score = 0.90
        else:
            temporal_score = 0.70  # Neutral when timestamps unobserved

        # 2. Topology Consistency Check
        # Check if target is connected to services reporting degradation
        affected_services = {
            normalize_service_name(ev.service)
            for ev in evidence_items
            if ev.magnitude > 0 and normalize_service_name(ev.service) != target
        }

        connected_to_affected = False
        target_neighbors: set[str] = set()

        if self.graph is not None:
            if hasattr(self.graph, "callees_of") and hasattr(self.graph, "callers_of"):
                for c in self.graph.callees_of(target):
                    target_neighbors.add(normalize_service_name(c))
                for c in self.graph.callers_of(target):
                    target_neighbors.add(normalize_service_name(c))
            elif hasattr(self.graph, "dependencies"):
                for dep in self.graph.dependencies:
                    u_n = normalize_service_name(getattr(dep, "caller", ""))
                    v_n = normalize_service_name(getattr(dep, "callee", ""))
                    if u_n == target:
                        target_neighbors.add(v_n)
                    if v_n == target:
                        target_neighbors.add(u_n)

            # Check direct or 2-hop connection to affected services
            if affected_services:
                overlap = target_neighbors & affected_services
                if overlap:
                    connected_to_affected = True
                else:
                    # Check 2-hop reachability
                    two_hop: set[str] = set()
                    for n in target_neighbors:
                        if hasattr(self.graph, "callees_of"):
                            for c in self.graph.callees_of(n):
                                two_hop.add(normalize_service_name(c))
                        if hasattr(self.graph, "callers_of"):
                            for c in self.graph.callers_of(n):
                                two_hop.add(normalize_service_name(c))
                    if two_hop & affected_services:
                        connected_to_affected = True

        if not affected_services:
            # Only target is degraded: topologically isolated fault
            topology_score = 0.90
        elif connected_to_affected:
            topology_score = 1.0
        else:
            topology_score = 0.50
            warnings.append(
                f"Target {target} has no direct topology edge to affected services: {sorted(affected_services)[:3]}."
            )

        # 3. Propagation Consistency Check
        # If target has high metric anomaly / CPU / error, caller latency elevation is expected
        target_evs = [ev for ev in evidence_items if normalize_service_name(ev.service) == target]
        has_internal_anomaly = any(ev.modality in {"metrics", "logs", "health"} and ev.magnitude > 0 for ev in target_evs)
        has_latency_elevation = any(ev.modality == "traces" and ev.magnitude > 0.4 for ev in evidence_items)

        if has_internal_anomaly and has_latency_elevation:
            propagation_score = 1.0
        elif has_internal_anomaly:
            propagation_score = 0.85
        elif has_latency_elevation:
            propagation_score = 0.80
        else:
            propagation_score = 0.50
            warnings.append("Weak propagation signal: Neither internal anomaly nor latency propagation detected.")

        # 4. Evidence Independence Check
        modalities_for_target = {ev.modality for ev in target_evs if ev.magnitude > 0}
        if len(modalities_for_target) < 2:
            warnings.append(
                f"Evidence for target {target} relies on single modality ({list(modalities_for_target)})."
            )

        # Composite consistency score
        consistency_score = (
            0.35 * temporal_score
            + 0.35 * topology_score
            + 0.30 * propagation_score
        )
        passed = consistency_score >= 0.65

        return CausalConsistencyResult(
            target_service=target,
            temporal_score=temporal_score,
            topology_score=topology_score,
            propagation_score=propagation_score,
            consistency_score=consistency_score,
            passed=passed,
            warnings=tuple(warnings),
        )
