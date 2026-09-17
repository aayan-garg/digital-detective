# Digital Detective — Project State

## Project Objective

Digital Detective is a research-oriented AIOps system for investigating transient cascading failures in microservice systems.

The long-term goal is an autonomous digital detective that can:

- inspect metrics, logs, and traces
- detect abnormal behavior
- construct and reason over service dependencies
- perform causal root-cause analysis
- retrieve historical and operational knowledge using RAG
- investigate incidents using an LLM/ReAct agent
- operate under a strict query budget
- evaluate remediation safety and blast radius
- perform remediation
- verify that the incident actually recovered
- be evaluated rigorously against public benchmarks

Core architectural principle:

The LLM/agent is an investigator and orchestrator, not the sole source of truth.

Specialized statistical, graph, and causal tools provide evidence to the agent.

RAG provides historical and operational context; it does not replace causal RCA.

Deep learning is an experimental component and must earn inclusion through evaluation.

---

## Engineering Principles

The project follows the rules in `AGENTS.md`.

Important principles:

- correctness over speed
- smallest correct system over speculative infrastructure
- inspect before implementing
- verify external APIs, datasets, papers, and claims
- avoid unnecessary dependencies
- avoid dead code and duplicate abstractions
- keep experiments isolated from stable/core code
- test meaningful behavior
- never fabricate benchmark results or research claims
- do not implement future milestones prematurely
- stop when the requested milestone is complete

---

## Current Development Roadmap

### Core

1. Repository / Codex engineering foundation
2. RCAEval discovery
3. Canonical telemetry representation
4. Basic anomaly detection
5. Trace processing and dependency graph
6. Causal RCA
7. Hybrid incident RAG
8. ReAct investigation agent + query budget
9. Safety / guardrails
10. Remediation
11. Recovery verification
12. Evaluation

### Research Enhancements

- deep-learning anomaly detection
- learned incident embeddings
- hybrid / graph-aware retrieval
- process mining
- conformal prediction
- counterfactual reasoning
- cascade / blast-radius forecasting

### Stretch

- multimodal telemetry foundation models
- learned causal models
- full OpenRCA-scale experiments
- advanced survival analysis

Future components must be introduced only when the relevant milestone is reached.

---

## Completed Milestones

### Milestone 0 — Repository foundation

Status: COMPLETE

Created:

- Python project scaffold
- `pyproject.toml`
- minimal package
- test infrastructure
- `.gitignore`
- README
- initial AGENTS.md

Python requirement:

`>=3.11,<3.13`

Validated with Python 3.12.

Runtime dependencies at initialization:

None.

---

### Milestone 0.1 — Codex engineering rules

Status: COMPLETE

`AGENTS.md` now defines:

- inspect-before-modify workflow
- milestone boundaries
- simplicity / minimal-change rules
- dependency discipline
- testing and validation rules
- Git safety
- environment discipline
- research correctness
- dataset discipline
- experimental isolation
- external research / API verification
- implementation protocol

---

### Milestone 0.2 — GitHub setup

Status: COMPLETE

Remote:

`origin`

Primary branch:

`main`

Local and remote are synchronized.

---

### Milestone 1 — RCAEval discovery

Status: COMPLETE

Authoritative sources were inspected.

Repository:

`phamquiluan/RCAEval`

Inspected live repository commit:

`bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90`

Current Hugging Face dataset distribution was inspected.

Verified benchmark structure:

- 735 cases
- 3 suites: RE1, RE2, RE3
- 9 datasets across:
  - Online Boutique
  - Sock Shop
  - Train Ticket

Verified suite characteristics:

- RE1: metric-only
- RE2: multi-source
- RE3: multi-source/code-level faults

A real RE2 Online Boutique case was inspected:

`re2ob_checkoutservice_cpu_1`

Observed:

- metrics
- logs
- traces
- injection time
- root-cause service
- fault type

The exact Parquet schemas and observed field names are documented in:

`docs/research/rcaeval.md`

Important verified observation:

Metrics, logs, and traces have different timestamp fields / representations and should not be collapsed prematurely into a single assumed schema.

Important evaluator observation:

The current evaluator uses ranked outputs under a `ranks` key and evaluates top-k accuracy and Avg@k.

---

## Important RCAEval Research Note

The authoritative discovery note is:

`docs/research/rcaeval.md`

Do not duplicate or contradict its facts.

When implementing the RCAEval adapter:

- inspect the actual files again
- preserve raw source semantics
- do not assume timestamp units that were not verified
- preserve raw root-cause service/fault/injection information
- verify benchmark evaluator behavior against the actual current evaluator

---

### Milestone 2 — Canonical Telemetry Representation
Status: COMPLETE (commit `a4cf4fd`)
- Implemented `src/digital_detective/telemetry.py` with `TelemetryCase`, `MetricSeries`, `LogRecord`, `TraceSpan`.
- Unit tests: `tests/test_telemetry.py`.

### Milestone 3 — RCAEval Local Adapter
Status: COMPLETE (commit `0dbe93e`, `d7e7e1c`)
- Implemented `src/digital_detective/rcaeval.py` parsing raw parquet telemetry without schema loss.
- Unit & integration tests: `tests/test_rcaeval.py`, `tests/test_rcaeval_integration.py`.

### Milestone 4 — Metric Anomaly Detection
Status: COMPLETE (commit `35c5819`)
- Implemented `src/digital_detective/anomaly.py` streaming rolling z-score detector.
- Unit tests: `tests/test_anomaly.py`.

### Milestone 5 — Anomaly Aggregation & Evaluation
Status: COMPLETE (commit `b3a24bf`, `3c000fe`)
- Implemented `src/digital_detective/aggregation.py` and `src/digital_detective/evaluation.py`.
- Unit tests: `tests/test_aggregation.py`, `tests/test_evaluation.py`.

### Milestone 6 — Trace Processing & Topology Extraction
Status: COMPLETE (commit `e0903c5`, `2b6b2c9`, `b838a3b`)
- Implemented `src/digital_detective/topology.py` dynamic entity graph extraction from trace spans.
- Unit tests: `tests/test_topology.py`.

### Milestone 7 — Entity Anomaly Episodes
Status: COMPLETE (commit `cca8de3`)
- Implemented `src/digital_detective/episodes.py` with temporal persistence ($K=3$) and metric consensus ($M=2$).
- Unit tests: `tests/test_episodes.py`.

### Milestone 8 — Trace Latency Evidence Layer
Status: COMPLETE (commit `5d0fb33`)
- Implemented `src/digital_detective/traces.py` span duration extraction and edge latency evidence.
- Unit tests: `tests/test_traces.py`.

### Milestone 9 — Deterministic RCA Baselines (S_comb and E_elev)
Status: COMPLETE (commit `cf6680c`, `afedb6d`)
- Implemented metric anomaly RCA baseline ($S_{\text{comb}}$) in `src/digital_detective/rca.py`.
- Implemented trace edge elevation baseline ($E_{\text{elev}}$) in `src/digital_detective/trace_attribution.py`.
- Frozen benchmark results verified:
  - $S_{\text{comb}}$: Top@1 = 70.0%, Top@3 = 86.7%, Top@5 = 93.3%, MRR = 0.8011.
  - $E_{\text{elev}}$: Top@1 = 60.0%, Top@3 = 96.7%, Top@5 = 100.0%, MRR = 0.7567.

### Milestone 10 — Multi-Modal Evidence Fusion & Causal-Consistency Validation
Status: COMPLETE
- Implemented `src/digital_detective/detective/scoring.py` (EvidenceFusion) and `src/digital_detective/detective/validation.py` (temporal, topological, propagation consistency).

### Milestone 11 — Autonomous Investigation Engine & Adaptive Evidence Acquisition
Status: COMPLETE
- Implemented `src/digital_detective/detective/investigator.py` with budget enforcement and modality dispersion heuristic.
- Added deterministic demo: `python -m digital_detective.detective.demo`.

### Milestone 12 — LLM Agent Orchestrator Layer
Status: COMPLETE
- Implemented `src/digital_detective/agent/` (`AgentOrchestrator`, `AgentModel`, `policy`, `prompts`).
- Support for `MockAgentModel`, `OllamaAgentModel` (`qwen3:8b`), `GrokAgentModel`.
- Added multi-mode demo: `python -m digital_detective.agent.demo`.

### Milestone 13 — Remediation, Recovery Verification & Post-Intervention Validation
Status: COMPLETE
- Implemented `src/digital_detective/detective/remediation.py` and `verification.py`.
- Evaluates multi-symptom recovery and post-hoc causal support (`INTERVENTION_SUPPORTS_HYPOTHESIS`, etc.).

### Milestone 14 — Benchmark Evaluation Harness & Safety Metrics
Status: COMPLETE
- Implemented `eval/` (`harness.py`, `models.py`, `manifest.py`, `universe.py`, `windows.py`).
- Added 5 safety/efficiency metrics: `abstention_rate`, `false_remediation_rate`, `recovery_success_rate`, `regression_rate`, `query_efficiency`.

### Milestone 15 — Repository Cleanup & Baseline Freeze
Status: COMPLETE
- Cleaned dependencies in `pyproject.toml`.
- Added explicit experiment configurations in `configs/`.
- Created maintained utility scripts in `scripts/`.
- Documented exploratory provenance in `scratch/README.md`.
- All 307 tests pass (301 passed, 6 skipped, 0 failures).

---

## Current State & Frozen Baselines

The repository has reached the **Baseline Freeze**. Core deterministic RCA algorithms, scoring formulas, causal-consistency audits, and frozen benchmarks remain strictly frozen.

### Frozen 30-Case RE2-OB Repetition-1 Smoke Baseline:
- **$S_{\text{comb}}$**: Top@1 = 70.0% (21/30), Top@3 = 86.7% (26/30), Top@5 = 93.3% (28/30), MRR = 0.8011
- **$E_{\text{elev}}$**: Top@1 = 60.0% (18/30), Top@3 = 96.7% (29/30), Top@5 = 100.0% (30/30), MRR = 0.7567
- **Fixed Equal-Weight Fusion**: Top@1 = 86.7% (26/30), Top@3 = 96.7% (29/30), Top@5 = 100.0% (30/30), MRR = 0.9194

---

## Methodological Boundaries & Non-Claims

1. **Confidence is NOT Calibrated Probability**: Confidence scores represent empirical relative separation between top hypotheses, not Bayesian posterior probabilities.
2. **Causal Consistency is NOT Formal Causal Discovery**: Observational causal audits verify consistency against empirical graph reachability and temporal orderings; they do not perform Structural Causal Modeling (SCM) or do-calculus.
3. **Adaptive Query Selection is a Deterministic Heuristic**: Query selection uses modality dispersion over top-$K$ candidates divided by tool cost; it is not formal information gain or POMDP belief-state modeling.
4. **Remediation is Simulated**: Interventions execute within a mathematical simulation sandbox model; they do not interact with live Kubernetes clusters.
5. **Regression Set Scope**: The 30-case RE2-OB benchmark is an internal regression and development smoke subset, not an official complete competition partition.

---

## Handoff Rule

When continuing this project in a new session:
1. Read `AGENTS.md`.
2. Read `docs/PROJECT_STATE.md`.
3. Read `README.md`.
4. Run tests with `python -m pytest tests/` or `python scripts/run_tests.py`.
5. Run evaluation with `python scripts/run_eval.py`.
6. Maintain frozen baseline invariants. Never introduce numerical drift.