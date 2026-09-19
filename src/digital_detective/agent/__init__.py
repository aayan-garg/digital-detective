"""Digital Detective Agent Layer.

Provides an orchestration layer that operates the deterministic detective core
tools to conduct incident investigations via a structured agent reasoning loop.

The language model acts as an orchestrator that decides WHICH tools to invoke.
The deterministic core tools execute the queries and return structured evidence.
The agent never has access to ground-truth root cause information.

Public API
----------
AgentModel
    Abstract interface for agent decision-making models.
MockAgentModel
    Deterministic rule-based agent. No network required.
OllamaAgentModel
    Local Ollama backend (default model: qwen3:8b). No API key required.
GrokAgentModel
    xAI Grok backend. Requires XAI_API_KEY.
OpenAIChatAdapter
    OpenAI-compatible backend. Requires OPENAI_API_KEY.
build_agent_model
    Factory function: selects implementation based on mode + environment.
AgentOrchestrator
    Bounded reasoning loop driving investigations via agent decisions.
InvestigationTrajectory
    Structured, exportable record of a complete agent-driven investigation.
AgentDecision
    Structured control signal emitted by the agent model each step.
"""

from .evaluation import (
    LLMEvaluationSummary,
    llm_abstention_rate,
    llm_topk_accuracy,
    summarize_condition_records,
    summarize_llm_records,
)
from .models import (
    AgentDecision,
    AgentModel,
    GrokAgentModel,
    InvestigationTrajectory,
    LLMExperimentRecord,
    MockAgentModel,
    OllamaAgentModel,
    OpenAIChatAdapter,
    build_agent_model,
)
from .orchestrator import AgentOrchestrator

__all__ = [
    "AgentDecision",
    "AgentModel",
    "AgentOrchestrator",
    "GrokAgentModel",
    "InvestigationTrajectory",
    "LLMEvaluationSummary",
    "LLMExperimentRecord",
    "MockAgentModel",
    "OllamaAgentModel",
    "OpenAIChatAdapter",
    "build_agent_model",
    "llm_abstention_rate",
    "llm_topk_accuracy",
    "summarize_condition_records",
    "summarize_llm_records",
]
