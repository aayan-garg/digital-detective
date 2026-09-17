# Digital Detective Agent Layer Design

## Overview

Digital Detective is structured as a **two-layer system**:

1. **Deterministic Core** — anomaly detection, S_comb/E_elev ranking, causal-consistency validation, remediation safety gate, recovery verification. These components are authoritative and immutable from the agent's perspective.

2. **Agent / Planner Layer** — a bounded reasoning loop that decides *which* telemetry tools to invoke and *when* to stop. The agent is an orchestrator, not an RCA engine.

> **Explicit design statement:**  
> *The language model is an orchestration layer. Deterministic RCA, budget enforcement, causal-consistency validation, remediation authorization, and recovery verification remain outside the language model.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│               AgentOrchestrator                         │
│  ┌──────────────┐    ┌──────────────────────────────┐   │
│  │  AgentModel  │    │  Policy / Tool Validator     │   │
│  │  (decide())  │───▶│  validate_decision()          │   │
│  └──────────────┘    └──────────────────────────────┘   │
│         │                        │                      │
│         │            ┌───────────▼──────────────────┐   │
│         │            │  DetectiveTools.execute_query │   │
│         │            │  (budget enforced internally) │   │
│         │            └───────────┬──────────────────┘   │
│         │                        │                      │
│         │            ┌───────────▼──────────────────┐   │
│         │            │  InvestigationState           │   │
│         │            │  (evidence, rankings, budget) │   │
│         └────────────┘                               │   │
│  InvestigationTrajectory (exported decision log)     │   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               Deterministic Core                        │
│  detect_metric_anomalies → aggregate_entity_episodes    │
│  → rank_with_s_comb → rank_with_trace_elevation        │
│  → EvidenceFusion.rank_hypotheses                       │
│  → CausalConsistencyValidator                           │
│  → propose_remediation → execute_simulated_remediation  │
│  → verify_recovery                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Model Implementations

### MockAgentModel

A deterministic rule-based agent. **No network required.** Always works.

**Strategy:**
1. If top hypothesis has `confidence >= threshold` AND at least `min_evidence_queries` queries were made → emit `FINAL_DIAGNOSIS`.
2. If `remaining_budget <= stop_budget_reserve` → emit `STOP` or accept best hypothesis.
3. Otherwise → pick the highest-suspicion service with an un-queried tool (following `_TOOL_PRIORITY_ORDER`) and emit `QUERY`.

**Use for:** local testing, CI/CD, development without an LLM.

```bash
python -m digital_detective.agent.demo --model mock
```

---

### OllamaAgentModel (Primary Local Backend)

Uses Ollama's OpenAI-compatible endpoint at `http://localhost:11434/v1`.

**Why Ollama?**  
- No API key required — runs entirely locally
- Privacy-preserving — telemetry never leaves the machine
- Works offline
- Supports many open models (Qwen3, Llama3, Mistral, etc.)

**Default model:** `qwen3:8b`

**Configuration:**

| Variable | Default | Purpose |
|---|---|---|
| `DIGITAL_DETECTIVE_OLLAMA_MODEL` | `qwen3:8b` | Model name |
| `DIGITAL_DETECTIVE_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Endpoint |

**Qwen3 considerations:**  
Qwen3 emits `<think>...</think>` blocks in its output. The adapter strips these before JSON parsing. No model-specific flags needed.

**Setup:**
```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull qwen3:8b

# Run the demo
python -m digital_detective.agent.demo --model ollama --verbose
```

**If Ollama is unavailable:**
```
Ollama unavailable: <reason>
Start Ollama and pull the configured model, or use --model mock.
```

---

### GrokAgentModel (Optional xAI Backend)

Uses xAI's API (`https://api.x.ai/v1`). Optional — only available if `XAI_API_KEY` is set.

**Default model:** `grok-3-mini`

**Configuration:**

| Variable | Default | Purpose |
|---|---|---|
| `XAI_API_KEY` | *(required)* | xAI API key |
| `DIGITAL_DETECTIVE_GROK_MODEL` | `grok-3-mini` | Model name |

```bash
export XAI_API_KEY=xai-...
python -m digital_detective.agent.demo --model grok
```

**If key is missing:**
```
ValueError: mode='grok' requires XAI_API_KEY environment variable.
```

---

### OpenAIChatAdapter (Generic Fallback)

Generic OpenAI-compatible endpoint. Requires `OPENAI_API_KEY`.

**For development, prefer OllamaAgentModel.** This adapter exists for compatibility with any OpenAI-compatible service.

---

## Model Abstraction

All backends implement the same interface:

```python
class AgentModel(abc.ABC):
    model_name: str

    @abc.abstractmethod
    def decide(
        self,
        state: InvestigationState,
        trajectory: InvestigationTrajectory,
        available_tools: Sequence[str],
        already_queried: set[tuple[str, str]],
    ) -> AgentDecision: ...
```

The `AgentOrchestrator` has no knowledge of whether the model is Mock, Ollama, or Grok. All backends are interchangeable.

---

## Tool-Calling Architecture

The agent **requests** tool calls; the deterministic core **executes** them.

```
Agent emits:  AgentDecision(action="QUERY", tool_name="get_traces", service="checkoutservice")
                                 │
                        policy.validate_decision()
                                 │
                    ┌────────────▼────────────────┐
                    │  DetectiveTools.execute_query│
                    │  - Checks remaining budget   │
                    │  - Dispatches to telemetry   │
                    │  - Records ToolQueryRecord   │
                    │  - Updates InvestigationState│
                    └─────────────────────────────┘
```

**Available tools:**

| Tool | Cost | Returns |
|---|---|---|
| `get_metrics` | 1 | Anomaly scores and metric signals |
| `get_service_health` | 1 | Aggregated health status |
| `get_recent_change` | 2 | Deployment/restart/config changes |
| `get_traces` | 3 | Trace latency elevation evidence |
| `get_neighbors` | 1 | Upstream/downstream dependencies |
| `get_logs` | 2 | Error log signals |

The agent may only call tools that are `available_tools` for the current investigation (e.g. `get_traces` is unavailable if no trace data exists).

---

## Evidence Grounding

Every collected telemetry response is assigned a stable E-ID for the LLM to cite:

```
E1: service=checkoutservice  modality=metrics  signal=cpu_anomaly  magnitude=0.92
E2: service=checkoutservice  modality=health   signal=status        magnitude=0.80
E3: service=currencyservice  modality=metrics  signal=mem_anomaly   magnitude=0.45
```

`FINAL_DIAGNOSIS` decisions **must** cite at least one E-ID:

```json
{
  "action": "FINAL_DIAGNOSIS",
  "reasoning": "E1 shows CPU saturation on checkoutservice consistent with a CPU fault. E2 confirms CRITICAL health status.",
  "evidence_ids": ["E1", "E2"]
}
```

Evidence citation is validated by `policy.validate_agent_claim()`. A diagnosis citing `E99` when only 3 evidence items exist is rejected with `INVALID_EVIDENCE_ID`.

---

## Ground-Truth Isolation

**The agent model never receives ground truth.**

| Information | Agent sees | Evaluation layer sees |
|---|---|---|
| `case.ground_truth` | ❌ Never | ✓ Post-hoc comparison only |
| `inject_time` | ❌ Never | ✓ For oracle window mode |
| `root_cause_service` | ❌ Never | ✓ For accuracy calculation |
| Current rankings | ✓ (deterministic scores) | ✓ |
| Evidence items (E-IDs) | ✓ | ✓ |
| Causal consistency scores | ✓ | ✓ |

**Incident window:** The orchestrator always uses `mode='detected'` (based on observed episode timestamps). Oracle mode is only available in the `InvestigationEngine` (evaluation/demo layer).

**Forbidden patterns in LLM output:**  
The policy validator rejects any reasoning containing: `ground truth`, `ground_truth`, `root_cause_service`, `fault_type`, `inject_time`, `injection_time`.

---

## Budget Enforcement

The query budget is controlled **exclusively** by `DetectiveTools.execute_query()`.

```
Agent sees:  state.remaining_budget   (read-only display)
Agent can:   REQUEST a tool           (QUERY action)
Agent can't: Modify remaining_budget
             Reset budget
             Request an unaffordable tool (policy rejects it)
```

Budget enforcement layers:
1. `policy.validate_decision()` checks `remaining_budget >= tool_cost` before executing
2. `DetectiveTools.execute_query()` enforces hard budget cap
3. `AgentOrchestrator` terminates loop when `remaining_budget <= 0`

---

## Safety Boundaries

```
Agent emits:  REQUEST_REMEDIATION
                   │
         No shortcut possible.
                   │
    ┌──────────────▼──────────────────────┐
    │  execute_simulated_remediation()    │
    │  Checks: confidence >= threshold    │
    │  Checks: causal_consistency.passed  │
    └──────────────┬──────────────────────┘
                   │
          threshold met?
           │            │
          YES           NO
           │            │
    Remediation    remediation_authorized=False
    executes       (proposal stored, not executed)
           │
    ┌──────▼──────────────┐
    │  verify_recovery()  │
    └─────────────────────┘
```

The LLM **cannot**:
- Execute remediation directly
- Bypass the confidence threshold check
- Bypass causal consistency validation
- Modify `remediation_authorized`
- Modify `budget`

---

## Bounded Loop

The agent loop is bounded by:

| Condition | Trigger |
|---|---|
| `action == "FINAL_DIAGNOSIS"` | Immediate termination |
| `action == "STOP"` | Immediate termination |
| `action == "REQUEST_REMEDIATION"` | Immediate termination, then safety gate |
| `remaining_budget <= 0` | `BUDGET_EXHAUSTED` |
| `step >= max_steps` (default 30) | `MAX_STEPS` |
| 3 consecutive policy-rejected decisions | Automatic `STOP` |

---

## Demo Usage

```bash
# Mock agent (always works)
python -m digital_detective.agent.demo --model mock

# Ollama/Qwen3 (local, no key)
python -m digital_detective.agent.demo --model ollama --verbose

# Grok (requires XAI_API_KEY)
python -m digital_detective.agent.demo --model grok

# Side-by-side: agent + deterministic
python -m digital_detective.agent.demo --model ollama --mode both

# All default cases
python -m digital_detective.agent.demo --model mock --all-cases

# Multi-model comparison (skips unavailable)
python -m digital_detective.agent.demo --compare-models --all-cases

# Custom case
python -m digital_detective.agent.demo --model ollama --case re2ob_currencyservice_delay_1
```

**Demo output format (agent mode):**
```
TURN 1
  Model   : Ollama/qwen3:8b
  Action  : QUERY
  Tool    : get_traces(checkoutservice)
  Reason  : Rank-1 suspect; querying traces to detect latency propagation.
  Remaining budget: 17

TURN 2
  ...

FINAL DIAGNOSIS
  Root cause          : checkoutservice
  Confidence          : 58.3%
  Causal consistency  : temporal=0.91 topology=0.87 overall=0.89 passed=YES
  Budget used         : 8
  Evidence collected  : 4

REMEDIATION AUTHORIZATION : NO (held)

Ground truth (post-hoc display only): checkoutservice (cpu)
```

---

## Limitations

1. **Evidence grounding is citation-based.** The validator checks that cited E-IDs exist and that referenced modalities were actually collected. It does not verify that the LLM's *interpretation* of the evidence is correct — deterministic scores are the authoritative RCA signal.

2. **Ollama performance varies by model and hardware.** `qwen3:8b` on CPU can be slow (>30s first token). Use `--model mock` for fast iteration.

3. **The agent cannot improve scores.** Scores are computed deterministically from telemetry. The agent can only direct which telemetry is collected and when to stop.

4. **No persistent memory across incidents.** Each investigation starts fresh. RAG-based historical context is explicitly out of scope.

5. **Simulated remediation only.** No live Kubernetes, AWS, or GCP operations are performed.
