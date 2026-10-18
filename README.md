# Digital Detective

Digital Detective is an end-to-end, research-grade microservice Root Cause Analysis (RCA), investigative orchestration, and safety-gated remediation system for cloud-native distributed systems.

---

## 1. Problem Statement

Microservice architectures suffer from transient, cascading performance degradations where initial metric anomalies propagate rapidly across service call graphs. Pinpointing the true root-cause service from multimodal telemetry (streaming metrics, distributed traces, and container logs) requires causal disambiguation rather than simple correlation. Digital Detective establishes a strict causal evidence hierarchy that prevents ungrounded hallucinations while orchestrating autonomous, grounded investigations.

---

## 2. System Architecture & Authority Boundaries

The complete end-to-end pipeline operates through 13 deterministic, verified stages:

```
Incident Telemetry (Metrics, Traces, Logs, Topology)
       │
       ▼
Multivariate Changepoint Detection (BOCPD / TCEC)
       │
       ▼
Topology Graph & Call-Edge Reconstruction
       │
       ▼
Metric Anomaly Attribution (S_comb) & Trace Latency Elevation (E_elev)
       │
       ▼
Structural Causal Residual Modeling (Standardized caller-expectation z-scores)
       │
       ▼
Log Template Mining & Anomaly Scoring (Drain3 parsing & error elevation)
       │
       ▼
Supervised Pairwise Candidate Ranker (Model B)
       │
       ▼
Uncertainty Resolution (Margin-triggered disambiguation: Delta = s_(1) - s_(2) < tau)
       │
       ▼
Hybrid Runbook RAG (Structure-aware chunking + BM25 + Dense BGE-small + RRF)
       │
       ▼
Grounded Investigator (LLM / Bounded Heuristic with strict Candidate Universe)
       │
       ▼
Deterministic Safety Gate Policy (Legality, prerequisites, risk, approval)
       │
       ▼
Remediation Planning (Dry-run recommendation & Human-in-the-Loop execution)
       │
       ▼
Recovery & Post-Intervention Verification
```

### Strict Authority Boundaries

* **Deterministic RCA ($S_{\text{comb}}$, $E_{\text{elev}}$)**: Sole baseline causal authority for candidate scoring and baseline ordering.
* **Supervised Candidate Ranker (Model B)**: Pairwise gradient-boosted ranker operating on causal-prefix features; preserves locked baselines.
* **Structural Causal Residuals**: Pre-incident baseline ridge regression estimating expected child metric responses from topological caller parents to derive standardized residual $z$-scores.
* **Drain3 Log Template Miner**: Offline log parsing extracting baseline template clusters, detecting novel post-onset error templates and burst ratios.
* **Disambiguator**: When the ranker confidence margin $\Delta = s_{(1)} - s_{(2)} < \tau$, a bounded investigator evaluates only the Top-3 candidates against multi-modal evidence. If confidence is sufficient ($\Delta \ge \tau$), deterministic ranking is preserved.
* **Operational Knowledge RAG**: Structure-aware chunking with hybrid BM25 + Dense BGE-small + Reciprocal Rank Fusion (RRF); provides reference-only context and enforces grounded citation verification.
* **Safety Gate Policy**: Deterministic policy enforcement verifying action permissions, prerequisites, degradation evidence, and rollback procedures. Potentially destructive actions default to dry-run/recommendation and require explicit operator approval.

---

## 3. Measured Benchmark Results

### A. RCA Ranking & Cross-Application Generalization

Model B is trained on Online Boutique (`RE2-OB`, 90 cases) using causal-prefix features and evaluated zero-shot across distinct distributed microservice architectures (`RE2-SS` and `RE2-TT`):

| Evaluation Set | Target Architecture | Cases ($N$) | Top@1 | Top@3 | Top@5 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Locked In-Domain Test** | Online Boutique (`RE2-OB` Reps 2–3) | 60 | **0.8167** | **0.9667** | **1.0000** | **0.8917** |
| **Zero-Shot Transfer** | Sock Shop (`RE2-SS`) | 90 | **0.5000** | **0.9222** | **0.9667** | **0.7066** |
| **Zero-Shot Transfer** | Train Ticket (`RE2-TT`) | 90 | **0.4556** | **0.7111** | **0.7333** | **0.5911** |
| **Combined Out-of-Domain** | All Transfer Cases (`SS + TT`) | 180 | **0.4778** | **0.8167** | **0.8500** | **0.6489** |

*Note*: The temporal deep learning complement was previously evaluated and rejected due to lower MRR (0.587 vs 0.623).

### B. Hybrid RAG Retrieval & Citation Grounding Evaluation

Evaluated across all 60 benchmark queries in the `DD-IR-60` suite over the 536-document operational runbook corpus:

| Retrieval Condition | Retriever Architecture | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | Lexical TF-IDF Baseline | 0.0500 | 0.0833 | 0.1200 | 0.0797 | 0.0838 | 2.97 |
| **C1** | Lexical BM25 (Okapi $k_1=1.5, b=0.75$) | **0.1000** | **0.1267** | **0.1467** | **0.1269** | **0.1803** | **1.33** |
| **C2** | Dense BGE-small-en-v1.5 (Cosine Sim) | 0.0000 | 0.0167 | 0.0333 | 0.0167 | 0.0210 | 288.14 |
| **C3** | Hybrid RRF (BM25 + BGE + Category Boost) | 0.0000 | 0.0833 | 0.1233 | 0.0439 | 0.0747 | 30.98 |

* **Citation Verification Guard**: 100% precision on grounded references. Injected phantom citations (e.g. `[ddir60-fake-doc-999999]`) are actively detected, flagged as hallucinations, and blocked before response delivery.
* **Dense Embedding Caching**: Runbook embeddings are cached persistently to `.npy` on disk, reducing subsequent index reload time from 35 seconds to under 3 milliseconds.

---

## 4. Installation & Environment Setup

Requires **Python 3.11 or 3.12**.

```powershell
# 1. Clone repository
git clone https://github.com/aayan-garg/digital-detective.git
cd digital-detective

# 2. Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies in editable mode
pip install -e ".[test]"

# 4. Copy environment configuration
cp .env.example .env
```

---

## 5. Running the System

### A. Run Automated Tests
```powershell
# Run the complete test suite (475 passed tests)
pytest -v

# Run the advanced RCA, RAG, and FastAPI integration test suites
pytest tests/test_features_advanced.py tests/test_disambiguator.py tests/test_rag_advanced.py tests/test_pipeline_and_api.py -v
```

### B. Launch FastAPI Backend & Replay Web UI
```powershell
# Start the API service and Web UI server
python -m uvicorn digital_detective.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Once started, visit:
* **Interactive Incident Replay Dashboard**: [http://127.0.0.1:8000/app/index.html](http://127.0.0.1:8000/app/index.html)
* **API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **System Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### C. Run Standalone Evaluations & Pipeline Walkthroughs
```powershell
# 1. End-to-end pipeline CLI run on a real incident
python scratch/test_pipeline_e2e.py

# 2. Cross-application generalization evaluation (OB -> SS, TT)
python scripts/run_cross_application_generalization.py

# 3. Systematic 60-query RAG benchmark evaluation
python scripts/evaluate_rag_system.py
```

---

## 6. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns service status, cached incident counts, and RAG readiness. |
| `GET` | `/api/incidents` | Lists cached incidents with optional `system` filter (`ob`, `ss`, `tt`). |
| `GET` | `/api/incidents/{case_id}` | Retrieves metadata and telemetry modality availability for an incident. |
| `POST` | `/api/investigate` | Executes complete 13-stage investigation and returns Pydantic `CanonicalIncident`. |
| `POST` | `/api/remediation/action` | Human-in-the-loop action endpoint to explicitly approve, reject, or test remediations. |
| `POST` | `/api/rag/query` | Direct hybrid runbook search across 536 operational documents. |

---

## 7. Project Structure

```
digital-detective/
├── data/
│   └── rag/
│       ├── runbooks_canonical.jsonl     # 536 operational runbooks
│       └── cache/                       # Cached BGE dense vector embeddings
├── eval/
│   ├── rag_evaluation_results.json      # DD-IR-60 RAG benchmark metrics
│   └── results/                         # Cross-application generalization logs
├── src/
│   ├── digital_detective/
│   │   ├── schema.py                    # Canonical Incident Pydantic schema
│   │   ├── pipeline.py                  # Unified 13-stage orchestration pipeline
│   │   ├── agent/
│   │   │   └── disambiguator.py         # Margin-triggered uncertainty resolution
│   │   ├── features/
│   │   │   ├── causal_residuals.py      # Pre-incident topological ridge residuals
│   │   │   └── log_scorer.py            # Drain3 log template miner
│   │   ├── rag/
│   │   │   ├── ingestion.py             # Structure-aware document chunking
│   │   │   ├── bm25.py                  # Lexical BM25 retriever
│   │   │   ├── dense.py                 # Dense BGE-small retriever
│   │   │   ├── hybrid.py                # Reciprocal Rank Fusion retriever
│   │   │   └── context_assembly.py      # Grounded context formatting & citation guard
│   │   ├── api/
│   │   │   └── main.py                  # FastAPI service
│   │   └── web/                         # Modern dark-mode incident replay dashboard
│   │       ├── index.html
│   │       ├── style.css
│   │       └── app.js
├── tests/                               # 475 automated tests
└── pyproject.toml
```

---

## 8. Reproducibility & Leakage Controls

* **Strict Temporal Partitioning**: All causal-prefix features and BOCPD changepoints are computed strictly prior to the confirmed injection timestamp to prevent future-data leakage.
* **Deterministic Tie-Breaking & Seed Control**: All stochastic processes (cross-validation folds, ranker initialization) use fixed seeds (`seed=42`).
* **Candidate Restriction**: The investigator agent is strictly bounded to the candidate universe identified by deterministic RCA and Model B, prohibiting phantom candidate invention.
* **Safety Isolation**: Remediation actions default to dry-run simulation with human approval gates, preventing unauthorized operations.

---

## 9. Status & Portfolio Verification

**Fully Tested & Integrated (v1.0-complete)**.
* **Test Suite**: 475 passed, 11 skipped, 0 failed.
* **Measured Baselines**: Preserved and verified.
