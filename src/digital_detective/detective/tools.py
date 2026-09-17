"""Deterministic telemetry query tools with explicit query costs and budget auditing.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import pyarrow.compute as pc

from eval.models import IncidentWindow
from eval.universe import normalize_service_name
from .models import EvidenceItem, InvestigationState, ToolQueryRecord

DEFAULT_TOOL_COSTS: Mapping[str, int] = {
    "get_metrics": 1,
    "get_logs": 2,
    "get_traces": 3,
    "get_neighbors": 1,
    "get_service_health": 1,
    "get_recent_change": 2,
}


class DetectiveTools:
    """Tool execution interface providing deterministic access to incident telemetry."""

    def __init__(
        self,
        case: Any,
        candidate_universe: Sequence[str],
        graph: Any | None = None,
        anomaly_result: Any | None = None,
        ep_evidence: Mapping[str, Any] | None = None,
        trace_latency: Any | None = None,
        costs: Mapping[str, int] | None = None,
        change_events: Mapping[str, Any] | None = None,
    ) -> None:
        self.case = case
        self.candidate_universe = tuple(sorted(set(candidate_universe)))
        self.graph = graph
        self.anomaly_result = anomaly_result
        self.ep_evidence = ep_evidence or {}
        self.trace_latency = trace_latency
        self.costs = dict(DEFAULT_TOOL_COSTS if costs is None else costs)
        self.change_events = dict(change_events or {})

    def execute_query(
        self,
        state: InvestigationState,
        tool_name: str,
        service: str,
        window: IncidentWindow | None = None,
        rationale: str = "",
        **kwargs: Any,
    ) -> tuple[bool, str, list[EvidenceItem]]:
        """Execute a tool query against the case, strictly enforcing remaining budget.

        Parameters
        ----------
        state:
            Active InvestigationState tracking budget and queries.
        tool_name:
            Name of tool to execute ('get_metrics', 'get_traces', etc.).
        service:
            Target service name.
        window:
            Optional temporal IncidentWindow.

        Returns
        -------
        tuple[bool, str, list[EvidenceItem]]
            (success, summary, evidence_items)
        """
        service = normalize_service_name(service)
        cost = self.costs.get(tool_name, 1)
        timestamp = time.time()
        query_id = f"q_{len(state.queries_executed) + 1}_{tool_name}_{service}"

        # 1. Strict Query Budget Check
        if state.remaining_budget < cost:
            summary = (
                f"Tool '{tool_name}' requires cost {cost}, but remaining budget is {state.remaining_budget}."
            )
            state.add_query_record(
                ToolQueryRecord(
                    query_id=query_id,
                    tool_name=tool_name,
                    service=service,
                    cost=cost,
                    timestamp=timestamp,
                    status="REJECTED_BUDGET",
                    parameters={"service": service, **kwargs},
                    result_summary=summary,
                    rationale=rationale,
                )
            )
            if state.remaining_budget <= 0:
                state.status = "BUDGET_EXHAUSTED"
            return False, summary, []

        # 2. Deduct Budget
        state.remaining_budget -= cost
        if state.remaining_budget <= 0:
            state.status = "BUDGET_EXHAUSTED"

        # 3. Dispatch to deterministic tool implementation
        evidence_items: list[EvidenceItem] = []
        try:
            if tool_name == "get_metrics":
                summary, evidence_items = self._get_metrics(service, window)
            elif tool_name == "get_traces":
                summary, evidence_items = self._get_traces(service, window)
            elif tool_name == "get_logs":
                summary, evidence_items = self._get_logs(service, window)
            elif tool_name == "get_neighbors":
                summary, evidence_items = self._get_neighbors(service)
            elif tool_name == "get_service_health":
                summary, evidence_items = self._get_service_health(service, window)
            elif tool_name == "get_recent_change":
                summary, evidence_items = self._get_recent_change(service)
            else:
                summary = f"Unknown tool: {tool_name}"
                state.add_query_record(
                    ToolQueryRecord(
                        query_id=query_id,
                        tool_name=tool_name,
                        service=service,
                        cost=cost,
                        timestamp=timestamp,
                        status="ERROR",
                        parameters={"service": service, **kwargs},
                        result_summary=summary,
                        rationale=rationale,
                    )
                )
                return False, summary, []

            # 4. Record Success & Store Evidence
            state.add_query_record(
                ToolQueryRecord(
                    query_id=query_id,
                    tool_name=tool_name,
                    service=service,
                    cost=cost,
                    timestamp=timestamp,
                    status="SUCCESS",
                    parameters={"service": service, **kwargs},
                    result_summary=summary,
                    rationale=rationale,
                )
            )
            for ev in evidence_items:
                state.add_evidence(ev)

            return True, summary, evidence_items

        except Exception as err:
            summary = f"Error executing {tool_name} on {service}: {err}"
            state.add_query_record(
                ToolQueryRecord(
                    query_id=query_id,
                    tool_name=tool_name,
                    service=service,
                    cost=cost,
                    timestamp=timestamp,
                    status="ERROR",
                    parameters={"service": service, **kwargs},
                    result_summary=summary,
                    rationale=rationale,
                )
            )
            return False, summary, []

    def _get_metrics(self, service: str, window: IncidentWindow | None) -> tuple[str, list[EvidenceItem]]:
        """Query metric timeseries anomalies and episodes for service."""
        evidence: list[EvidenceItem] = []
        ev_ep = self.ep_evidence.get(service)
        has_ep = getattr(ev_ep, "has_episode", False) if ev_ep else False
        peak_active = getattr(ev_ep, "peak_active_metrics", 0) if ev_ep else 0
        # Calculate first episode start timestamp within or after window onset
        first_ts = None
        if ev_ep and hasattr(ev_ep, "episodes"):
            valid_eps = [
                ep for ep in ev_ep.episodes
                if window is None or window.onset_ts is None or ep.start_timestamp >= window.onset_ts
            ]
            if valid_eps:
                first_ts = valid_eps[0].start_timestamp
            elif hasattr(ev_ep, "first_episode_start_ts") and (window is None or window.onset_ts is None or ev_ep.first_episode_start_ts >= window.onset_ts):
                first_ts = ev_ep.first_episode_start_ts
        elif ev_ep:
            ts_cand = getattr(ev_ep, "first_episode_start_ts", None)
            if ts_cand is not None and (window is None or window.onset_ts is None or ts_cand >= window.onset_ts):
                first_ts = ts_cand
        contrib_metrics = getattr(ev_ep, "all_contributing_metrics", ()) if ev_ep else ()

        # Calculate peak magnitude
        peak_z = 0.0
        if self.anomaly_result is not None:
            for m in contrib_metrics:
                scores = self.anomaly_result.anomaly_scores.get(m, ())
                for s_val in scores:
                    if s_val is not None and float(s_val) > peak_z:
                        peak_z = float(s_val)

        # Normalized magnitude in [0, 1] combining peak active and log-magnitude
        norm_count = min(1.0, peak_active / 3.0) if has_ep else 0.0
        norm_mag = min(1.0, math.log1p(peak_z) / 15.0) if has_ep else 0.0
        magnitude = (0.5 * norm_count + 0.5 * norm_mag) if has_ep else 0.0
        conf = 0.35 if has_ep and peak_active >= 2 else (0.15 if has_ep else 0.0)

        item = EvidenceItem(
            service=service,
            modality="metrics",
            signal="metric_anomaly_episode",
            magnitude=magnitude,
            timestamp=first_ts,
            source="detect_metric_anomalies/episodes",
            confidence_contribution=conf,
            metadata={
                "has_episode": has_ep,
                "peak_active_metrics": peak_active,
                "contributing_metrics": list(contrib_metrics),
                "peak_z": round(peak_z, 2),
            },
        )
        evidence.append(item)
        summary = (
            f"Metrics for {service}: has_episode={has_ep}, peak_active={peak_active}, "
            f"contributing={len(contrib_metrics)}, peak_z={peak_z:.1f}"
        )
        return summary, evidence

    def _get_traces(self, service: str, window: IncidentWindow | None) -> tuple[str, list[EvidenceItem]]:
        """Query trace latency elevation and edge calls for service."""
        evidence: list[EvidenceItem] = []
        max_elev = 0.0
        incoming_edges = []
        outgoing_edges = []

        if self.trace_latency is not None:
            for edge in self.trace_latency.edges:
                p90 = float(edge.caller_duration_dist.p90)
                p10 = float(edge.caller_duration_dist.p10)
                elev = max(0.0, (p90 - p10) / p90) if p90 > 1e-9 else 0.0

                if edge.callee_service == service:
                    incoming_edges.append((edge.caller_service, round(elev, 3), round(p90, 1)))
                    if elev > max_elev:
                        max_elev = elev
                if edge.caller_service == service:
                    outgoing_edges.append((edge.callee_service, round(elev, 3), round(p90, 1)))

        conf = 0.35 if max_elev >= 0.8 else (0.20 if max_elev >= 0.5 else 0.0)
        item = EvidenceItem(
            service=service,
            modality="traces",
            signal="trace_latency_elevation",
            magnitude=max_elev,
            timestamp=None,
            source="trace_attribution/E_elev",
            confidence_contribution=conf,
            metadata={
                "max_incoming_elevation": round(max_elev, 3),
                "incoming_callers": incoming_edges,
                "outgoing_callees": outgoing_edges,
            },
        )
        evidence.append(item)
        summary = (
            f"Traces for {service}: incoming_edges={len(incoming_edges)}, "
            f"max_incoming_elevation={max_elev:.3f}"
        )
        return summary, evidence

    def _get_logs(self, service: str, window: IncidentWindow | None) -> tuple[str, list[EvidenceItem]]:
        """Query error log entries matching ERROR / FAIL / EXCEPTION for service."""
        evidence: list[EvidenceItem] = []
        error_count = 0
        samples: list[str] = []

        if hasattr(self.case, "logs") and self.case.logs is not None:
            table = getattr(self.case.logs, "table", None)
            if table is not None and table.num_rows > 0:
                container_col = "container_name" if "container_name" in table.column_names else "serviceName"
                ts_col = "timestamp" if "timestamp" in table.column_names else "time"
                msg_col = "message" if "message" in table.column_names else "log"

                # Temporal filter if window provided
                if window is not None and window.onset_ts is not None and window.end_ts is not None and ts_col in table.column_names:
                    t_arr = table[ts_col]
                    w_mask = pc.and_(pc.greater_equal(t_arr, window.onset_ts), pc.less_equal(t_arr, window.end_ts))
                    table = table.filter(w_mask)

                if table.num_rows > 0 and container_col in table.column_names and msg_col in table.column_names:
                    srv_mask = pc.equal(table[container_col], service)
                    if service == "frontend":
                        srv_mask = pc.or_(srv_mask, pc.equal(table[container_col], "frontendservice"))
                    srv_table = table.filter(srv_mask)

                    if srv_table.num_rows > 0:
                        msg_l = pc.utf8_lower(srv_table[msg_col])
                        err_m = pc.or_(
                            pc.match_substring(msg_l, "error"),
                            pc.or_(pc.match_substring(msg_l, "fail"), pc.match_substring(msg_l, "exception")),
                        )
                        err_table = srv_table.filter(err_m)
                        error_count = err_table.num_rows
                        if error_count > 0:
                            samples = [str(x) for x in err_table[msg_col][:2].to_pylist()]

        conf = 0.20 if error_count >= 5 else (0.10 if error_count > 0 else 0.0)
        item = EvidenceItem(
            service=service,
            modality="logs",
            signal="error_log_count",
            magnitude=float(error_count),
            timestamp=window.onset_ts if window else None,
            source="logs_search/keywords",
            confidence_contribution=conf,
            metadata={"error_count": error_count, "sample_messages": samples},
        )
        evidence.append(item)
        summary = f"Logs for {service}: error_keywords_count={error_count}, samples={len(samples)}"
        return summary, evidence

    def _get_neighbors(self, service: str) -> tuple[str, list[EvidenceItem]]:
        """Query dependency topology callers and callees."""
        evidence: list[EvidenceItem] = []
        callers: list[str] = []
        callees: list[str] = []

        if self.graph is not None and hasattr(self.graph, "edges"):
            for u, v in self.graph.edges:
                u_norm = normalize_service_name(u)
                v_norm = normalize_service_name(v)
                if v_norm == service and u_norm != service:
                    callers.append(u_norm)
                if u_norm == service and v_norm != service:
                    callees.append(v_norm)
        elif self.trace_latency is not None:
            for edge in self.trace_latency.edges:
                u_norm = normalize_service_name(edge.caller_service)
                v_norm = normalize_service_name(edge.callee_service)
                if v_norm == service and u_norm != service:
                    callers.append(u_norm)
                if u_norm == service and v_norm != service:
                    callees.append(v_norm)

        callers = sorted(set(callers))
        callees = sorted(set(callees))
        total_deg = len(callers) + len(callees)

        item = EvidenceItem(
            service=service,
            modality="topology",
            signal="neighbor_dependencies",
            magnitude=float(total_deg),
            source="entity_graph/trace_topology",
            confidence_contribution=0.05 if total_deg > 0 else 0.0,
            metadata={"callers": callers, "callees": callees, "degree": total_deg},
        )
        evidence.append(item)
        summary = f"Topology for {service}: callers={callers}, callees={callees}, total_degree={total_deg}"
        return summary, evidence

    def _get_service_health(self, service: str, window: IncidentWindow | None) -> tuple[str, list[EvidenceItem]]:
        """Evaluate synthetic health status (HEALTHY, DEGRADED, CRITICAL)."""
        ev_ep = self.ep_evidence.get(service)
        has_ep = getattr(ev_ep, "has_episode", False) if ev_ep else False
        peak_active = getattr(ev_ep, "peak_active_metrics", 0) if ev_ep else 0

        # Check trace latency elevation if present
        max_elev = 0.0
        if self.trace_latency is not None:
            for edge in self.trace_latency.edges:
                if edge.callee_service == service:
                    p90 = float(edge.caller_duration_dist.p90)
                    p10 = float(edge.caller_duration_dist.p10)
                    elev = max(0.0, (p90 - p10) / p90) if p90 > 1e-9 else 0.0
                    if elev > max_elev:
                        max_elev = elev

        if has_ep and peak_active >= 2 and max_elev > 0.5:
            health = "CRITICAL"
            health_score = 1.0
        elif has_ep or max_elev > 0.3:
            health = "DEGRADED"
            health_score = 0.6
        else:
            health = "HEALTHY"
            health_score = 0.0

        item = EvidenceItem(
            service=service,
            modality="health",
            signal="service_health_status",
            magnitude=health_score,
            source="health_check/multi_signal",
            confidence_contribution=0.10 if health == "CRITICAL" else 0.0,
            metadata={"status": health, "has_episode": has_ep, "latency_elevation": round(max_elev, 3)},
        )
        summary = f"Health for {service}: status={health} (score={health_score})"
        return summary, [item]

    def _get_recent_change(self, service: str) -> tuple[str, list[EvidenceItem]]:
        """Query recent configuration / deployment change events for service.

        Checks explicit change_events feed if provided, or inspects container logs
        for deployment, initialization, and restart events. Never accesses ground truth.
        """
        is_changed = False
        change_type = "none"

        if service in self.change_events:
            is_changed = bool(self.change_events[service])
            change_type = str(self.change_events[service])
        elif hasattr(self.case, "logs") and self.case.logs is not None:
            table = getattr(self.case.logs, "table", None)
            if table is not None and table.num_rows > 0:
                container_col = "container_name" if "container_name" in table.column_names else "serviceName"
                msg_col = "message" if "message" in table.column_names else "log"
                if container_col in table.column_names and msg_col in table.column_names:
                    srv_mask = pc.equal(table[container_col], service)
                    if service == "frontend":
                        srv_mask = pc.or_(srv_mask, pc.equal(table[container_col], "frontendservice"))
                    srv_table = table.filter(srv_mask)
                    if srv_table.num_rows > 0:
                        msg_l = pc.utf8_lower(srv_table[msg_col])
                        change_m = pc.or_(
                            pc.match_substring(msg_l, "deploy"),
                            pc.or_(
                                pc.match_substring(msg_l, "restarting"),
                                pc.or_(
                                    pc.match_substring(msg_l, "configuration changed"),
                                    pc.match_substring(msg_l, "started container"),
                                ),
                            ),
                        )
                        matched = srv_table.filter(change_m)
                        if matched.num_rows > 0:
                            is_changed = True
                            change_type = "container_restart_or_deploy"

        magnitude = 1.0 if is_changed else 0.0
        conf = 0.15 if is_changed else 0.0
        item = EvidenceItem(
            service=service,
            modality="recent_change",
            signal="deployment_change_record",
            magnitude=magnitude,
            source="audit_log/deployments",
            confidence_contribution=conf,
            metadata={
                "has_recent_change": is_changed,
                "change_type": change_type,
            },
        )
        summary = f"Recent changes for {service}: has_change={is_changed} (type={change_type})"
        return summary, [item]
