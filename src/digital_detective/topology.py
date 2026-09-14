"""Dependency-aware telemetry representation and entity topology.

Provides a minimal, preservation-first entity model for mapping metric identifiers
to logical entities, representing directional dependencies, and aggregating
metric-level anomaly evidence to entity-level timelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .anomaly import MetricAnomalyResult
from .telemetry import TelemetryCase

KNOWN_ENTITY_TYPES: Mapping[str, str] = {
    "redis": "datastore",
    "frontend-external": "gateway",
    "PassthroughCluster": "infrastructure",
    "main": "process",
}
DEFAULT_ENTITY_TYPE = "service"


@dataclass(frozen=True)
class MetricIdentifier:
    """Dissected identity of a telemetry metric while preserving the raw name."""

    raw_name: str
    entity: str
    signal: str


@dataclass(frozen=True)
class EntityNode:
    """Logical entity in the system topology."""

    name: str
    entity_type: str
    metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class Dependency:
    """Directional dependency edge from caller (source) to callee (target)."""

    source: str
    target: str


@dataclass(frozen=True)
class TraceDependencyObservation:
    """An observed directed caller -> callee relationship extracted from trace spans."""

    source: str
    target: str
    observation_count: int

    def to_dependency(self) -> Dependency:
        """Convert to an EntityGraph Dependency object."""
        return Dependency(source=self.source, target=self.target)


@dataclass(frozen=True)
class EntityGraph:
    """Immutable entity topology containing nodes, metrics, and dependencies."""

    entities: Mapping[str, EntityNode]
    dependencies: tuple[Dependency, ...]
    _metric_to_entity: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index: dict[str, str] = {}
        for entity_name, node in self.entities.items():
            for m in node.metrics:
                index[m] = entity_name
        object.__setattr__(self, "_metric_to_entity", index)

    def get_entity_for_metric(self, metric_name: str) -> str | None:
        """Return the owning entity name for a given metric, or None if unmapped."""
        return self._metric_to_entity.get(metric_name)

    def callers_of(self, entity: str) -> tuple[str, ...]:
        """Return distinct entities that directly call the given entity."""
        seen: set[str] = set()
        callers: list[str] = []
        for dep in self.dependencies:
            if dep.target == entity and dep.source not in seen:
                seen.add(dep.source)
                callers.append(dep.source)
        return tuple(callers)

    def callees_of(self, entity: str) -> tuple[str, ...]:
        """Return distinct entities that are directly called by the given entity."""
        seen: set[str] = set()
        callees: list[str] = []
        for dep in self.dependencies:
            if dep.source == entity and dep.target not in seen:
                seen.add(dep.target)
                callees.append(dep.target)
        return tuple(callees)


@dataclass(frozen=True)
class EntityAnomalyEvidence:
    """Immutable entity-level anomaly evidence aggregated from MetricAnomalyResult."""

    case_id: str
    entity: str
    timestamps: tuple[Any, ...]
    active_metric_counts: tuple[int, ...]
    is_anomalous: tuple[bool, ...]
    first_anomaly_ts: int | None
    anomalous_metrics: tuple[str, ...]


def parse_rcaeval_metric_identifier(metric_name: str) -> MetricIdentifier:
    """Parse a metric name using the verified RCAEval <component>_<signal> structure.

    Preserves the raw metric name without modification.
    """
    if not isinstance(metric_name, str) or "_" not in metric_name:
        raise ValueError(f"Malformed RCAEval metric name (missing underscore): {metric_name!r}")

    parts = metric_name.split("_", 1)
    entity, signal = parts[0], parts[1]
    if not entity or not signal:
        raise ValueError(f"Malformed RCAEval metric name (empty entity or signal): {metric_name!r}")

    return MetricIdentifier(raw_name=metric_name, entity=entity, signal=signal)


def build_entity_graph(
    metric_names: Sequence[str],
    dependencies: Sequence[Dependency | tuple[str, str]] = (),
    entity_type_overrides: Mapping[str, str] | None = None,
) -> EntityGraph:
    """Build an EntityGraph from metric names and optional explicit dependencies.

    Entities are grouped by their parsed component prefix. Entity types are assigned
    using KNOWN_ENTITY_TYPES for verified non-service entities, falling back to 'service'.
    Duplicate exact dependency edges are eliminated deterministically while preserving order.
    """
    entity_metrics: dict[str, list[str]] = {}
    for m in metric_names:
        ident = parse_rcaeval_metric_identifier(m)
        entity_metrics.setdefault(ident.entity, []).append(m)

    # Process dependencies and ensure all referenced entities exist in the graph
    parsed_deps: list[Dependency] = []
    seen_deps: set[tuple[str, str]] = set()
    for item in dependencies:
        if isinstance(item, Dependency):
            dep = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            dep = Dependency(source=str(item[0]), target=str(item[1]))
        else:
            raise ValueError(f"Invalid dependency specification: {item!r}")

        pair = (dep.source, dep.target)
        if pair not in seen_deps:
            seen_deps.add(pair)
            parsed_deps.append(dep)
            entity_metrics.setdefault(dep.source, [])
            entity_metrics.setdefault(dep.target, [])

    type_mapping = dict(KNOWN_ENTITY_TYPES)
    if entity_type_overrides:
        type_mapping.update(entity_type_overrides)

    entities: dict[str, EntityNode] = {}
    for name, metrics in entity_metrics.items():
        etype = type_mapping.get(name, DEFAULT_ENTITY_TYPE)
        entities[name] = EntityNode(name=name, entity_type=etype, metrics=tuple(metrics))

    return EntityGraph(
        entities=entities,
        dependencies=tuple(parsed_deps),
    )


def aggregate_entity_anomaly_evidence(
    anomaly_result: MetricAnomalyResult,
    graph: EntityGraph,
) -> Mapping[str, EntityAnomalyEvidence]:
    """Aggregate metric-level anomalies into per-entity anomaly evidence.

    Operates purely downstream on MetricAnomalyResult without recalculating scores.
    Non-anomaly, warmup, missing, and insufficient-history states do not contribute to anomaly counts.
    Causality is strictly preserved.
    """
    timestamps = anomaly_result.timestamps
    num_steps = len(timestamps)
    evidence: dict[str, EntityAnomalyEvidence] = {}

    available_metrics = set(anomaly_result.metric_names)

    for entity_name, entity_node in graph.entities.items():
        # Owned metrics present in the anomaly result
        entity_metrics = [m for m in entity_node.metrics if m in available_metrics]

        active_counts = [0] * num_steps
        anomalous_metric_set: set[str] = set()

        for m in entity_metrics:
            statuses = anomaly_result.evaluation_statuses[m]
            for idx in range(num_steps):
                if statuses[idx] == "anomaly":
                    active_counts[idx] += 1
                    anomalous_metric_set.add(m)

        is_anom = tuple(c > 0 for c in active_counts)

        # Causally find first anomalous timestamp
        first_ts = None
        for idx in range(num_steps):
            if is_anom[idx]:
                first_ts = int(timestamps[idx])
                break

        evidence[entity_name] = EntityAnomalyEvidence(
            case_id=anomaly_result.case_id,
            entity=entity_name,
            timestamps=timestamps,
            active_metric_counts=tuple(active_counts),
            is_anomalous=is_anom,
            first_anomaly_ts=first_ts,
            anomalous_metrics=tuple(sorted(anomalous_metric_set)),
        )

    return evidence


def extract_trace_dependencies(
    case: TelemetryCase,
    *,
    service_aliases: Mapping[str, str] | None = None,
) -> tuple[TraceDependencyObservation, ...]:
    """Extract directly observed caller -> callee service edges from case traces.

    Operates strictly over observed parent-child span relationships.
    An observed edge represents an observed execution relationship,
    not complete system topology, not causality, and not root cause.

    Parameters
    ----------
    case:
        The TelemetryCase containing optional traces modality.
    service_aliases:
        Optional explicit mapping from trace serviceName to canonical entity name
        (e.g. {"frontendservice": "frontend"}). No fuzzy matching is performed.

    Returns
    -------
    tuple[TraceDependencyObservation, ...]
        Deterministically sorted tuple of unique caller -> callee edges with observation counts.
    """
    if case.traces is None:
        return ()

    table = case.traces.raw_data
    if hasattr(table, "column_names"):
        columns = set(table.column_names)
        get_col = lambda name: table.column(name).to_pylist()
    elif isinstance(table, Mapping):
        columns = set(table.keys())
        get_col = lambda name: list(table[name])
    else:
        raise ValueError(f"Unsupported trace raw_data type: {type(table)!r}")

    required_cols = ("spanID", "parentSpanID", "serviceName")
    missing = [c for c in required_cols if c not in columns]
    if missing:
        raise ValueError(f"Trace data missing required column(s): {missing}")

    span_ids = get_col("spanID")
    parent_ids = get_col("parentSpanID")
    service_names = get_col("serviceName")

    # Build spanID -> serviceName/parentSpanID lookup, allowing exact duplicate
    # records (e.g. from network retry/loss telemetry duplication) while strictly
    # rejecting duplicate spanIDs that map to conflicting services or parents.
    span_to_service: dict[str, str] = {}
    span_to_parent: dict[str, str | None] = {}
    for sid, pid, sname in zip(span_ids, parent_ids, service_names):
        if sid is None:
            raise ValueError("Trace spanID cannot be null")
        if sid in span_to_service:
            if span_to_service[sid] != sname or span_to_parent.get(sid) != pid:
                raise ValueError(f"Duplicate spanID detected in trace data: {sid!r}")
            continue
        span_to_service[sid] = sname
        span_to_parent[sid] = pid

    edge_counts: dict[tuple[str, str], int] = {}

    for sid, child_svc in span_to_service.items():
        pid = span_to_parent.get(sid)
        if pid is None or pid == "":
            continue
        if pid not in span_to_service:
            continue

        parent_svc = span_to_service[pid]
        if parent_svc is None or child_svc is None:
            continue
        if parent_svc == "" or child_svc == "":
            continue

        if service_aliases:
            parent_svc = service_aliases.get(parent_svc, parent_svc)
            child_svc = service_aliases.get(child_svc, child_svc)

        if parent_svc == child_svc:
            continue

        edge = (parent_svc, child_svc)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

    sorted_edges = sorted(edge_counts.keys(), key=lambda e: (e[0], e[1]))
    return tuple(
        TraceDependencyObservation(source=u, target=v, observation_count=edge_counts[(u, v)])
        for u, v in sorted_edges
    )
