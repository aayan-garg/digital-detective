"""Structured tool schemas for the Digital Detective agent layer.

Exposes the six deterministic detective tools as a structured catalog that:
  - documents each tool's name, description, parameters, and cost
  - provides schema validation helpers
  - supports compact prompt-ready formatting

The adapter is a READ-ONLY catalog — it does not execute queries.
All execution remains inside ``DetectiveTools.execute_query()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Tool schema definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolParameter:
    """A single parameter in a tool's input schema."""

    name: str
    type: str  # "string" | "integer" | "float"
    description: str
    required: bool = True


@dataclass(frozen=True)
class ToolSchema:
    """Schema for a single deterministic detective tool.

    Attributes
    ----------
    name:
        Canonical tool name (matches ``DetectiveTools`` dispatch keys).
    description:
        What the tool returns and when to use it.
    parameters:
        Ordered list of accepted parameters.
    cost:
        Query budget units consumed by one invocation.
    returns:
        Description of what evidence / summary is returned.
    """

    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    cost: int
    returns: str

    def prompt_description(self) -> str:
        """One-line description suitable for embedding in a system prompt."""
        param_strs = ", ".join(
            f"{p.name}: {p.type}{'*' if p.required else '?'}"
            for p in self.parameters
        )
        return (
            f"  {self.name}({param_strs})  [cost={self.cost}]\n"
            f"    {self.description}\n"
            f"    Returns: {self.returns}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "cost": self.cost,
            "returns": self.returns,
            "parameters": [
                {"name": p.name, "type": p.type, "required": p.required, "description": p.description}
                for p in self.parameters
            ],
        }


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------

_SERVICE_PARAM = ToolParameter(
    name="service",
    type="string",
    description="Target microservice name from the candidate universe.",
    required=True,
)

TOOL_CATALOG: tuple[ToolSchema, ...] = (
    ToolSchema(
        name="get_metrics",
        description=(
            "Retrieve anomaly scores and metric signals for a service during the incident window. "
            "Use to detect CPU, memory, latency, or error-rate anomalies."
        ),
        parameters=(_SERVICE_PARAM,),
        cost=1,
        returns="Per-metric anomaly magnitudes and timestamps.",
    ),
    ToolSchema(
        name="get_service_health",
        description=(
            "Check the overall health status of a service (HEALTHY / DEGRADED / CRITICAL) "
            "based on aggregated episode data."
        ),
        parameters=(_SERVICE_PARAM,),
        cost=1,
        returns="Health status, episode count, peak anomaly magnitude.",
    ),
    ToolSchema(
        name="get_recent_change",
        description=(
            "Detect recent deployments, restarts, or configuration changes for a service. "
            "Use to distinguish fault injection from natural drift."
        ),
        parameters=(_SERVICE_PARAM,),
        cost=2,
        returns="Boolean has_change, change type, and approximate timestamp if found.",
    ),
    ToolSchema(
        name="get_traces",
        description=(
            "Retrieve distributed trace latency evidence: p50/p90 latency deltas and "
            "call-path error signals involving a service. Only available when trace data is present."
        ),
        parameters=(_SERVICE_PARAM,),
        cost=3,
        returns="Latency elevation magnitude, upstream/downstream call statistics.",
    ),
    ToolSchema(
        name="get_neighbors",
        description=(
            "List direct upstream and downstream service dependencies in the call graph. "
            "Use to reason about propagation paths and rule out bystander services."
        ),
        parameters=(_SERVICE_PARAM,),
        cost=1,
        returns="Upstream and downstream service names.",
    ),
    ToolSchema(
        name="get_logs",
        description=(
            "Retrieve aggregated log error signals for a service during the incident window. "
            "Only available when log data is present."
        ),
        parameters=(_SERVICE_PARAM,),
        cost=2,
        returns="Error log count, error rate, representative error message samples.",
    ),
)

# Name → schema lookup
_TOOL_BY_NAME: dict[str, ToolSchema] = {t.name: t for t in TOOL_CATALOG}


def get_tool_schema(name: str) -> ToolSchema | None:
    """Return the tool schema for ``name``, or ``None`` if not found."""
    return _TOOL_BY_NAME.get(name)


def is_valid_tool(name: str) -> bool:
    """Return ``True`` if ``name`` is a registered detective tool."""
    return name in _TOOL_BY_NAME


def get_tool_cost(name: str) -> int:
    """Return the query cost for ``name``.  Returns ``1`` for unknown tools."""
    schema = _TOOL_BY_NAME.get(name)
    return schema.cost if schema else 1


def catalog_prompt_block(available_tools: Sequence[str] | None = None) -> str:
    """Return a compact prompt-ready block describing all (or available) tools.

    Parameters
    ----------
    available_tools:
        If provided, only describe tools in this list.
    """
    tools = TOOL_CATALOG
    if available_tools is not None:
        available_set = set(available_tools)
        tools = tuple(t for t in TOOL_CATALOG if t.name in available_set)
    lines = ["AVAILABLE TOOLS:"]
    for t in tools:
        lines.append(t.prompt_description())
    return "\n".join(lines)
