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

## Current State

The repository is currently at the end of RCAEval discovery.

No actual AIOps pipeline has been implemented yet.

No canonical telemetry schema has been finalized.

No RAG system has been implemented.

No deep-learning model has been implemented.

No LLM agent has been implemented.

No Kubernetes remediation has been implemented.

This is intentional.

---

## Current Git State

Latest known commit:

`1bc11be`

Commit message:

`docs: document RCAEval discovery`

Expected branch:

`main`

Expected remote:

`origin/main`

Working tree should remain clean after this state file is committed.

---

## Immediate Next Milestone

### Milestone 2 — Canonical Telemetry Representation

Goal:

Design the smallest useful internal representation that allows Digital Detective to consume different benchmark telemetry formats without destroying source-specific semantics.

Before implementation:

1. inspect the actual RCAEval telemetry schemas already discovered
2. identify genuinely common concepts
3. preserve modality-specific fields where semantics differ
4. define explicit timestamp / identity semantics
5. avoid building a large generic telemetry framework
6. design tests around the representation
7. implement only what is needed for the next milestone

Do NOT yet implement:

- anomaly detection
- RAG
- causal RCA
- agents
- remediation
- deep-learning models

The next milestone should first produce a clear design and acceptance criteria, then implementation.

---

## Handoff Rule

When continuing this project in a new ChatGPT/Codex session:

1. Read `AGENTS.md`.
2. Read `docs/PROJECT_STATE.md`.
3. Read the relevant research notes under `docs/research/`.
4. Inspect the current Git state.
5. Continue only from the current milestone.
6. Do not assume previous conversational context exists.