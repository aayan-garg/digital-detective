# Digital Detective

Digital Detective is a research-grade prototype for autonomous root-cause analysis (RCA), investigative orchestration, and safety-gated remediation of transient, cascading microservice failures in cloud-native distributed systems.

---

## 1. Problem Statement

Microservice architectures suffer from transient, cascading performance degradations where initial anomalies propagate rapidly across service call graphs. Pinpointing the true root-cause service from multimodal telemetry (streaming metrics, distributed traces, and service topologies) requires causal disambiguation rather than simple anomaly correlation. Digital Detective establishes a causal evidence hierarchy that prevents ungrounded hallucinations while orchestrating autonomous investigations.

---

## 2. System Architecture & Authority Boundaries

The logical pipeline separates causal reasoning from generative hypothesis investigation:

```
Telemetry Ingestion
       │
       ▼
Anomaly Detection (Streaming rolling z-score / BOCPD / MAD)
       │
       ▼
Sequential Confirmation (Multi-metric persistent episode aggregation & TCEC)
       │
       ▼
Deterministic RCA (S_comb metric scoring & E_elev trace attribution)
       │
       ▼
ML Candidate Ranking (Model B supervised pairwise candidate ranker)
       │
       ▼
LLM Investigator (Bounded tool-augmented investigation loop)
       │
       ├── Hybrid RAG (Operational knowledge retrieval context)
       ▼
Safety Gate Policy (Pre-execution Blast-radius & risk validation)
       │
       ▼
Simulated Sandbox Remediation
       │
       ▼
Multi-Symptom Recovery Verification
       │
       ▼
Post-Intervention Validation
```

### Strict Authority Boundaries

* **Deterministic RCA ($S_{\text{comb}}$, $E_{\text{elev}}$)**: Sole causal ground truth authority for candidate scoring and baseline ordering.
* **ML Candidate Ranking (Model B)**: Supervised ranking enhancement on causal-prefix features; operates under strict family-aware split boundaries.
* **LLM Investigator**: Bounded hypothesis explorer and orchestrator. It collects observations and proposes diagnoses, but cannot override deterministic RCA authority.
* **RAG Operational Layer**: Operational knowledge retrieval supplying reference-only context; strictly prohibited from altering scoring formulas or safety verdicts.
* **Safety Gate Policy**: Deterministic policy enforcement controlling simulated remediation actions and verifying multi-symptom recovery.

---

## 3. Main Contributions & Research Results

### A. Accepted Supervised Candidate Ranking (Model B)
Model B uses pairwise candidate differences on causal-prefix features (combining $S_{\text{comb}}$, $E_{\text{elev}}$, anomaly coverage, and topological distance) trained with family-grouped cross-validation:

**Locked RE2-OB Repetitions 2–3 Evaluation (60 Cases):**
* **Top@1**: 0.8167 (49/60)
* **Top@3**: 0.9667 (58/60)
* **Top@5**: 1.0000 (60/60)
* **MRR**: 0.8916667

### B. Temporal Deep Learning Complement (Evaluated and Rejected)
A compact temporal deep learning model operating over causal-prefix metric sequences was evaluated as a candidate ranker complement:
* **Development Validation MRR**: Model B = 0.623 vs DL + Model B = 0.587
* **Locked RE2-OB Test**: Model B (0.817 / 0.967 / 1.000 / 0.892) vs DL + Model B (0.767 / 0.883 / 0.933 / 0.846)
* **Decision**: **REJECTED**. The temporal DL enhancement degraded ranking performance on both development and locked-test evaluations and was therefore rejected. Model B is retained as the frozen ranker.

### C. Operational Knowledge RAG
* **Operational Corpus**: Provenance-backed microservice operational runbooks, architecture specifications, and troubleshooting guides (536 documents).
* **Role**: Injected as `OPERATIONAL KNOWLEDGE (RAG – REFERENCE ONLY)` to provide system context to the investigator agent without altering deterministic scoring.
* **Validation**: RAG retrieval and context injection were end-to-end verified. The separate RAG-2 hybrid BM25 + BGE-small + RRF implementation was validated through its controlled retrieval experiments.
* *Note*: Full human/expert RAG relevance labeling was not completed before freeze; RAG remains an operational/reference knowledge layer.

### D. LLM Investigator & Safety Gate
* **Investigator**: Bounded reasoning loop supporting tool actions (`QUERY`, `FINAL_DIAGNOSIS`, `STOP`) across Conditions A and C.
* **Local Backend**: Evaluated with `qwen3:8b` via Ollama OpenAI-compatible endpoints with deterministic authority preservation and structured JSON schema enforcement.

---

## 4. Installation

Requires **Python 3.11 or 3.12**.

```powershell
# Clone repository
git clone https://github.com/aayan-garg/digital-detective.git
cd digital-detective

# Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Install package in editable mode with test dependencies
pip install -e ".[test]"
```

*Note on RAG-2 dependencies*: The heavier RAG-2 dense retrieval and reranker stack (e.g. `sentence-transformers`, `torch`) may require additional runtime packages beyond the core package dependencies declared in `pyproject.toml`.

---

## 5. Running the System

### Automated Test Suite
```powershell
$env:PYTHONPATH='src;.'
& '.venv/Scripts/python.exe' -m unittest discover -s tests
```

### ML Candidate Ranking Experiment
```powershell
$env:PYTHONPATH='src;.'
& '.venv/Scripts/python.exe' experiments/run_ml_rca_experiment.py
```

### RAG Smoke & Audit
```powershell
$env:PYTHONPATH='src;.'
& '.venv/Scripts/python.exe' scripts/run_rag2_smoke.py
& '.venv/Scripts/python.exe' scripts/audit_rag2_corpus.py
```

### End-to-End Investigation Demo
```powershell
# Stable deterministic/mock agent demo
$env:PYTHONPATH='src;.'
& '.venv/Scripts/python.exe' -m digital_detective.agent.demo --model mock --case re2ob_checkoutservice_cpu_1

# End-to-end RAG + Investigator execution
$env:PYTHONPATH='src;.'
& '.venv/Scripts/python.exe' scripts/run_rag_investigator_e2e.py
```

---

## 6. Reproducibility & Leakage Controls

To support reproducibility and reduce leakage risk:
* **Family-Aware Partitioning**: Split allocations are grouped strictly by scenario family (`RE2-SS` and `RE2-TT` for development, `RE2-OB` repetitions 2–3 locked).
* **Causal-Prefix Isolation**: All features and sequential anomalies are extracted strictly prior to or at the confirmed injection boundary to prevent future-leakage.
* **Fixed Seeds**: All stochastic processes (cross-validation folds, ranker initialization) use locked seeds (`seed=42`).

---

## 7. Known Runtime Limitations

* **Local Ollama / Windows GPU Instability**: Under Windows with CUDA-enabled local Ollama instances, GPU memory allocation exceptions may occur under sustained load. The orchestrator includes graceful fallbacks and timeout guards.
* **Simulation Sandboxing**: Remediation actions execute inside an in-memory simulation model rather than live cloud infrastructure.
* **RAG Relevance Benchmarking**: Full human/expert relevance labeling on the operational benchmark pool was not finalized prior to freeze.

---

## 8. Project Status

**Final frozen research prototype (v1.0-final)**.
