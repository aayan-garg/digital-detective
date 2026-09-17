# Digital Detective

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-301%20passed-brightgreen.svg)]()
[![Frozen Baselines](https://img.shields.io/badge/baselines-frozen-success.svg)]()

Digital Detective is an autonomous AIOps incident investigation and root-cause analysis (RCA) research system designed for transient, cascading failures in microservice environments.

---

## 1. What Problem Does It Solve?

In distributed microservices, failures (such as CPU saturation, memory exhaustion, or network packet drops) cascade downstream across service dependencies, creating hundreds of noisy alarms across metrics, logs, and traces.

Human operators face telemetry overload and uncertainty. Digital Detective addresses this challenge by:
- Ingesting heterogeneous telemetry (metrics, logs, traces) into a strict canonical representation without losing provenance.
- Using deterministic statistical algorithms ($S_{\text{comb}}$ and $E_{\text{elev}}$) to rank candidate root causes.
- Auditing candidate diagnoses against topological reachability, temporal precedence, and propagation consistency.
- Providing an autonomous investigation agent that dynamically queries discriminative telemetry under a strict budget constraint.
- Safety-gating remediation interventions and verifying multi-symptom recovery post-intervention.

---

## 2. Architecture Overview

The system strictly enforces separation of concerns across 15 pipeline stages:

```
[ Telemetry Ingestion (Metrics, Logs, Traces) ]
                      ↓
[ Streaming Anomaly Detection (Rolling Z-Score) ]
                      ↓
[ Entity Anomaly Episodes (Persistence K=3, Consensus M=2) ]
                      ↓
[ Trace Latency & Dynamic Topology Extraction ]
                      ↓
[ Deterministic Baseline Scoring (S_comb & E_elev) ]
                      ↓
[ Multi-Modal Evidence Fusion & Confidence Estimation ]
                      ↓
[ Observational Causal Consistency Audit ]
                      ↓
┌─────────────────────────────────────────────────────────┐
│ Autonomous Investigation Loop (Budget: 20 units)        │
│  - Deterministic Investigation Engine                   │
│  - OR LLM Agent Orchestrator (Ollama / Qwen3 / Grok)   │
│  - Modality Dispersion Query Acquisition Heuristic      │
└─────────────────────────────────────────────────────────┘
                      ↓
[ Root-Cause Diagnostic Decision ]
                      ↓
[ Safety Gate (Confidence Threshold >= 0.80) ]
       ├── If BLOCKED → Hold for operator override
       └── If AUTHORIZED → Simulated Sandbox Remediation
                                ↓
[ Multi-Symptom Recovery Verification ]
                                ↓
[ Post-Intervention Causal Support Validation ]
                                ↓
[ Benchmark Evaluation & Headroom Analysis ]
```

> [!NOTE]
> **Core Principle**: The LLM agent is an orchestrator and explorer, never the ungrounded source of truth. All scoring, causal consistency audits, and recovery verifications remain in the deterministic core.

---

## 3. Installation

Requires **Python 3.11 or 3.12**.

```powershell
# Clone repository
git clone https://github.com/aayan-garg/digital-detective.git
cd digital-detective

# Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Install in editable mode with test dependencies
pip install --editable ".[test]"
```

---

## 4. How to Run Tests

Run the full automated test suite (307 collected items: 301 passed, 6 skipped, 0 failures):

```powershell
# Via convenience script
python scripts/run_tests.py

# Or via pytest directly
pytest tests/
```

---

## 5. How to Run Deterministic RCA

To run deterministic RCA algorithms over benchmark cases or individual telemetry cases:

```python
from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import EpisodeConfig, aggregate_entity_episodes
from digital_detective.rca import rank_with_s_comb
from digital_detective.topology import build_entity_graph, extract_trace_dependencies
from digital_detective.rcaeval import load_rcaeval_case
from pathlib import Path

# Load an incident case
case = load_rcaeval_case(Path("~/.cache/rcaeval_validation"), "re2ob_checkoutservice_cpu_1")

# Detect anomalies & construct graph
det_res = detect_metric_anomalies(case)
deps = [td.to_dependency() for td in extract_trace_dependencies(case)]
graph = build_entity_graph(det_res.metric_names, dependencies=deps)
ep_evidence = dict(aggregate_entity_episodes(det_res, graph, EpisodeConfig(persistence=3, consensus=2)))

# Rank candidates using frozen S_comb
rankings = rank_with_s_comb(det_res, ep_evidence, graph=graph)
for rank, item in enumerate(rankings[:3], 1):
    print(f"Top {rank}: {item.entity} (Score: {item.score:.4f})")
```

---

## 6. How to Run the Investigation Demos

### Deterministic Demo
Demonstrates the full autonomous loop (exploration, hypothesis convergence, causal audit, safety gating, and recovery verification):

```powershell
python scripts/run_demo.py --mode deterministic
# Or directly:
python -m digital_detective.detective.demo
```

### Agentic Demo (Side-by-Side Comparison)
Compares the deterministic engine against the agentic orchestrator:

```powershell
# Run with fast deterministic Mock agent across all demo cases:
python scripts/run_demo.py --mode agent -- --model mock --all-cases

# View side-by-side comparison:
python scripts/run_demo.py --mode both -- --model mock --case re2ob_checkoutservice_cpu_1
```

---

## 7. How to Run Local Ollama / Qwen3 Agent

To run the agent with a local LLM via [Ollama](https://ollama.ai/):

```powershell
# 1. Pull the model in Ollama
ollama pull qwen3:8b

# 2. Run the agent demo
python -m digital_detective.agent.demo --model ollama --case re2ob_checkoutservice_cpu_1 --verbose
```

---

## 8. How to Run Evaluation

Run the evaluation harness over the 30-case RE2-OB benchmark smoke suite:

```powershell
# Via convenience script
python scripts/run_eval.py

# Or via CLI module
python -m eval.harness --smoke-re2-ob --window-mode oracle
```

### Frozen Regression Baselines (30-case RE2-OB Repetition 1)
| Method | Modality | Top@1 | Top@3 | Top@5 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$S_{\text{comb}}$** | Metrics | **70.0%** (21/30) | **86.7%** (26/30) | **93.3%** (28/30) | **0.8011** |
| **$E_{\text{elev}}$** | Traces | **60.0%** (18/30) | **96.7%** (29/30) | **100.0%** (30/30) | **0.7567** |
| **Fixed Fusion** | Multi-Modal | **86.7%** (26/30) | **96.7%** (29/30) | **100.0%** (30/30) | **0.9194** |

---

## 9. Important Source Files

```
src/digital_detective/
├── telemetry.py              <- Canonical telemetry classes (MetricSeries, LogRecord, TraceSpan)
├── rcaeval.py                <- Lossless RCAEval dataset adapter
├── anomaly.py                <- Streaming rolling z-score anomaly detector
├── episodes.py               <- Multi-metric persistent episode aggregation
├── topology.py               <- Dynamic service dependency graph extraction
├── traces.py                 <- Trace span duration & latency profiling
├── rca.py                    <- S_comb metric RCA formulation
├── trace_attribution.py      <- E_elev trace edge elevation formulation
├── detective/
│   ├── investigator.py       <- Deterministic InvestigationEngine & adaptive acquisition heuristic
│   ├── scoring.py            <- EvidenceFusion & hypothesis confidence scoring
│   ├── validation.py         <- CausalConsistencyValidator (temporal, topological, propagation)
│   ├── remediation.py        <- Safety-gated remediation & blast-radius simulation
│   └── verification.py       <- Multi-symptom recovery & post-intervention causal validation
└── agent/
    ├── orchestrator.py       <- AgentOrchestrator bounded reasoning loop
    ├── models.py             <- AgentModel interfaces (Mock, Ollama, Grok)
    ├── policy.py             <- Decision validation guardrails
    └── prompts.py            <- Compact prompt construction & observation templates
```

---

## 10. Research Documentation & Provenance

- **`docs/PROJECT_STATE.md`**: Complete milestone log, current baseline freeze status, and roadmap.
- **`docs/design/digital-detective-core.md`**: Formal specification of the 15 pipeline stages, exact mathematical formulas, and causal validation rules.
- **`docs/design/digital-detective-agent.md`**: LLM agent orchestrator specification, prompt formats, and latency profiling.
- **`docs/research/established_rca_methods.md`**: Deep empirical study and reproduction of the RCD algorithm and ablation analyses.
- **`scratch/README.md`**: Provenance guide for historical research ablation scripts and diagnostic dumps.

---

## 11. Methodological Boundaries

To ensure scientific rigor, Digital Detective explicitly disclaims the following:
1. **No Calibrated Probability Claim**: Hypothesis confidence scores are normalized ranking separations, not Bayesian posterior probabilities.
2. **No Formal Causal Inference Claim**: Observational causal audits check empirical temporal and reachability constraints; they do not perform Structural Causal Modeling (SCM) or Pearl's do-calculus.
3. **No Optimal Information Gain Claim**: Adaptive query selection is a deterministic modality-dispersion heuristic, not formal Bayesian information gain.
4. **Sandboxed Remediation**: Remediation actions execute inside an in-memory simulation model, not in live production infrastructure.
5. **Smoke Benchmark Scope**: The 30-case RE2-OB benchmark is an internal regression suite, not an official competition split.

---

## 12. How to Start Modifying the System

1. **Verify Baseline**: Always verify that `python scripts/run_tests.py` and `python scripts/run_eval.py` pass before making changes.
2. **Preserve Invariants**: Do not alter frozen scoring formulas ($S_{\text{comb}}$, $E_{\text{elev}}$), tie-breaking keys, or test assertions without explicit approval.
3. **Configuration First**: Adjust experiment parameters in `configs/` rather than scattering magic constants in source code.
4. **Keep Core Independent**: Keep agent and LLM experiments strictly isolated from the deterministic core library.
