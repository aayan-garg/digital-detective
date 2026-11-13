# Digital Detective — Architecture, Baseline, and Reproducibility Audit

**Audit Date:** 2026-11-13  
**Audited Commit SHA:** `391438f8e59f4c4ae4de956265b51934f1a5a279`  
**Audited Branch:** `agent/issue-3-audit-architecture` (tracking `origin/main`)  
**Auditor:** Antigravity Agent  
**Audit Purpose:** Comprehensive, source-verified baseline snapshot of implemented architecture, test suite health, benchmark reproducibility, and documentation consistency for GitHub Issue #3.

---

## 1. Executive Summary

Digital Detective has evolved through multiple research and engineering phases:
1. **Phases 0–15 (Foundational RCA & Modular Detective):** Documented in `docs/PROJECT_STATE.md`, culminating in the frozen baseline for metric scoring ($S_{\text{comb}}$), trace latency elevation ($E_{\text{elev}}$), and equal-weight fusion on the 30-case RE2-OB repetition-1 smoke benchmark.
2. **Phase 16+ (Supervised Ranking, Advanced Multimodal Features, Hybrid RAG, & Web Deployment):** Implemented in `src/digital_detective/pipeline.py`, `experiments/ml_rca_ranker.py`, `src/digital_detective/api/main.py`, and `src/digital_detective/web/`, featuring Model B pairwise linear ranking, Drain3 log template scoring, topological ridge residuals, hybrid RRF runbook retrieval, and a public-facing FastAPI/Vercel demonstration layer.

### Key Audit Findings

* **Frozen Baselines Verified:** The core deterministic baselines ($S_{\text{comb}}$, $E_{\text{elev}}$, Fixed Equal-Weight Fusion) were **100% reproduced** from the local dataset cache (`~/.cache/rcaeval_validation`) matching documented values to the 4th decimal place.
* **Test Suite Health:** The test suite contains **522 collected tests** across 56 test files. When executed under Python 3.12.10:
  * **510 tests passed**.
  * **11 tests skipped** (5 due to optional `pyrca` dependency, 6 due to unset dataset environment variables / optional fixtures).
  * **1 test failed** (`tests/test_ml_rca_ranker.py::test_pairwise_inference_is_deterministic`), caused by a test-fixture schema mismatch introduced when candidate schema validation was hardened in commit `5ad22ff`.
* **Documentation Discrepancies:**
  * Test count: `docs/PROJECT_STATE.md` records 307 tests; `README.md` records 475 tests; the actual test inventory is 522 tests.
  * Training split direction: `docs/FINAL_PROJECT_REPORT.md` and `eval/results/canonical_locked_test_results.json` document Model B trained on RE2-SS + RE2-TT (180 cases) and tested on RE2-OB Reps 2–3 (60 cases). In contrast, `README.md` and `eval/results/cross_application_generalization.json` document Model B trained on RE2-OB (90 cases) and transferred zero-shot to RE2-SS + RE2-TT (180 cases). Both are valid experiments, but their coexistence must be explicitly delineated.
  * Remediation scope: The README refers to "Remediation Planning & Execution"; this is strictly an in-memory mathematical simulation model without live Kubernetes/cloud infrastructure integration.

---

## 2. Source-Verified Architecture and Runtime Flow

### 2.1 The Production Pipeline (`src/digital_detective/pipeline.py`)

The active runtime orchestrator is `DigitalDetectivePipeline` in `src/digital_detective/pipeline.py`, executed via `/api/investigate` in `src/digital_detective/api/main.py` or standalone in `scratch/test_pipeline_e2e.py`. It executes 13 sequential stages:

```
[1. Telemetry Case Ingestion (Metrics, Logs, Traces, Injection Meta)]
                              │
                              ▼
[2. Multivariate Changepoint Detection (BOCPD + Sequential TCEC)]
                              │
                              ▼
[3. Topology Graph & Dynamic Call-Edge Reconstruction]
                              │
                              ▼
[4. Metric Anomaly Attribution (S_comb)] & [5. Trace Latency Elevation (E_elev)]
                              │
                              ▼
[6. Structural Causal Residuals (Caller-expectation ridge regression)]
                              │
                              ▼
[7. Log Template Mining & Error Elevation (Drain3)]
                              │
                              ▼
[8. Supervised Pairwise Candidate Ranker (Model B, loaded from PKL)]
                              │
                              ▼
[9. Uncertainty Resolution (Disambiguator margin check: Δ = s1 - s2 < 0.05)]
                              │
                              ▼
[10. Operational Knowledge RAG (Hybrid BM25 + Dense BGE-small + RRF)]
                              │
                              ▼
[11. Grounded Investigator (Candidate-universe-bounded evidence appraisal)]
                              │
                              ▼
[12. Deterministic Safety Gate Policy (Legality, degradation evidence, rollback)]
                              │
                              ▼
[13. Simulated Sandbox Remediation & Post-Intervention Recovery Verification]
```

### 2.2 Authority Hierarchy

| Level | Component | Authority Scope | Overridable by Agent/LLM? | Source Location |
|---|---|---|---|---|
| **Causal Foundation** | $S_{\text{comb}}$, $E_{\text{elev}}$ | Deterministic score assignment, topological graph traversal | **Strictly No** | `src/digital_detective/rca.py`, `trace_attribution.py` |
| **Supervised Ranker** | Model B | Causal-prefix candidate reordering | **Strictly No** | `experiments/ml_rca_ranker.py`, `models/model_b_ranker_v1.pkl` |
| **Disambiguator** | Margin Guard ($\tau = 0.05$) | Resolves close candidate ties ($\Delta < \tau$); otherwise preserves Model B order | **Bounded** | `src/digital_detective/agent/disambiguator.py` |
| **Context Layer** | Hybrid RAG | Operational runbooks, reference knowledge | **Reference Only** | `src/digital_detective/rag/` |
| **Safety Gate** | Safety Policy | Evaluates action prerequisites, degradation evidence, and rollback plans | **Strictly No** | `src/digital_detective/detective/remediation.py` |
| **Recovery** | Recovery Verifier | Proves post-intervention metric symptom alleviation | **Strictly No** | `src/digital_detective/detective/verification.py` |

---

## 3. Component Status Matrix

| Component | Implementation File | Status | Validation / Evidence |
|---|---|---|---|
| **Canonical Telemetry Schema** | `src/digital_detective/telemetry.py`, `schema.py` | **Implemented & Active** | Used across all ingestion, pipeline stages, and FastAPI endpoints |
| **RCAEval Local Adapter** | `src/digital_detective/rcaeval.py` | **Implemented & Active** | Ingests raw Parquet metrics, logs, traces, and injection metadata without schema loss |
| **Metric Anomaly Detector (Rolling Z)** | `src/digital_detective/anomaly.py` | **Implemented & Active** | Evaluated and unit-tested in `tests/test_anomaly.py` |
| **BOCPD Incident Detector** | `src/digital_detective/bocpd.py` | **Implemented & Active** | Fast vectorized Bayesian online changepoint detector |
| **Sequential TCEC** | `src/digital_detective/sequential_tcec.py`, `tcec.py` | **Implemented & Active** | Topology-coherent episode confirmation for changepoint validation |
| **Trace Topology Extraction** | `src/digital_detective/topology.py` | **Implemented & Active** | Extracts directed service call graphs from span parent-child relationships |
| **Trace Latency Attribution ($E_{\text{elev}}$)** | `src/digital_detective/trace_attribution.py`, `traces.py` | **Implemented & Active** | Span duration elevation and caller concentration scoring |
| **Metric RCA Baseline ($S_{\text{comb}}$)** | `src/digital_detective/rca.py` | **Implemented & Active** | Multi-factor metric scoring (strength, earliness, coverage, propagation) |
| **Structural Causal Residuals** | `src/digital_detective/features/causal_residuals.py` | **Implemented & Active** | Pre-incident ridge regression residual modeling against caller parents |
| **Drain3 Log Scorer** | `src/digital_detective/features/log_scorer.py` | **Implemented & Active** | Drain3 template clustering and novel error burst scoring |
| **Supervised Ranker (Model B)** | `experiments/ml_rca_ranker.py`, `models/model_b_ranker_v1.pkl` | **Implemented & Active** | Loaded in `pipeline.py`; trained on causal features |
| **Uncertainty Disambiguator** | `src/digital_detective/agent/disambiguator.py` | **Implemented & Active** | Margin-triggered verification of top-3 candidates |
| **Hybrid Runbook RAG** | `src/digital_detective/rag/` (`bm25.py`, `dense.py`, `hybrid.py`, `context_assembly.py`) | **Implemented & Active** | Indexes 536 runbooks; evaluated on DD-IR-60; caches BGE dense embeddings |
| **FastAPI Backend Service** | `src/digital_detective/api/main.py` | **Implemented & Active** | Fully functioning API with `/api/health`, `/api/investigate`, CORS, and demo cache |
| **Web UI Dashboard** | `src/digital_detective/web/` (`index.html`, `app.js`, `styles.css`) | **Implemented & Active** | Incident replay UI; deployed on Vercel |
| **Interactive ReAct Agent** | `src/digital_detective/agent/orchestrator.py`, `models.py` | **Implemented (Standalone/Demo)** | Multi-turn LLM investigation loop; used in `agent/demo.py`, not in `pipeline.py` |
| **Adaptive Query Heuristic** | `src/digital_detective/detective/investigator.py` | **Implemented (Standalone/Demo)** | Modality dispersion / query budget; used in `detective/demo.py` |
| **PyRCA Hypothesis Testing** | `src/digital_detective/pyrca_ht.py` | **Implemented (Experimental)** | Optional comparison backend; requires uninstalled `pyrca` package |
| **Counterfactual Removal (CRV)** | `src/digital_detective/crv.py` | **Implemented (Experimental)** | Stage 7 ablation experiment |
| **Temporal Deep Learning** | `experiments/temporal_dl_experiment.py` | **Evaluated & Formally Rejected** | Degraded MRR (-0.0458 on locked test); isolated from core pipeline |
| **Live Kubernetes Remediation** | N/A | **Documentation-Only / Simulated** | Simulated state transformations only; no live cluster integration |
| **Streaming Telemetry Ingestion** | N/A | **Documentation-Only** | Offline file ingestion only; no Kafka/OTel collector |

---

## 4. Frozen Invariants and Source Locations

The following components represent core project invariants:

1. **$S_{\text{comb}}$ Metric Scoring Formula:**  
   `src/digital_detective/rca.py` (`rank_with_s_comb`, lines 300–450).  
   Formula: $S_{\text{comb}}(v) = w_{\text{str}} R_{\text{str}}(v) + w_{\text{early}} R_{\text{early}}(v) + w_{\text{cov}} R_{\text{cov}}(v) + w_{\text{prop}} R_{\text{prop}}(v)$. Weights and normalizations are frozen.
2. **$E_{\text{elev}}$ Trace Duration Elevation:**  
   `src/digital_detective/trace_attribution.py` (`compute_trace_attribution`, lines 180–310).  
   Aggregates caller edge elevation, wait-time concentration, and terminal factors.
3. **Candidate Universe Policy:**  
   `eval/universe.py` (`resolve_candidate_universe`).  
   Restricts candidates strictly to services observed in telemetry and topology; forbids phantom candidate generation.
4. **Temporal Partitioning / Anti-Leakage Gate:**  
   `src/digital_detective/tcec.py`, `src/digital_detective/rca.py`, `src/digital_detective/pipeline.py`.  
   Telemetry extraction for anomaly baselines and causal residuals strictly respects the pre-injection window boundary.
5. **Fixed Equal-Weight Fusion:**  
   `eval/harness.py` (`FixedEqualWeightFusion`).  
   Arithmetic mean of normalized $S_{\text{comb}}$ and $E_{\text{elev}}$ scores.

---

## 5. Test Suite Health and Environment Details

### 5.1 Test Environment

* **Operating System:** Windows 11 (build 10.0.26100)
* **Python Runtime:** Python 3.12.10 (AMD64)
* **Active Environment:** `.\.venv\` (declared in `pyproject.toml`)
* **Test Framework:** `pytest` 9.1.1, `pluggy` 1.6.0, `anyio` 4.15.1

### 5.2 Test Execution Results

#### Full Test Suite

```powershell
.\.venv\Scripts\pytest.exe -q
```

* **Collected:** 522 items across 56 test modules
* **Passed:** 510
* **Failed:** 1
* **Skipped:** 11
* **Warnings:** 1 (`StarletteDeprecationWarning` regarding `httpx` in `starlette.testclient`)
* **Runtime:** 260.48s (4m 20s)

#### Focused Pipeline & Advanced Test Suite (from README §5.A)

```powershell
pytest tests/test_features_advanced.py tests/test_disambiguator.py tests/test_rag_advanced.py tests/test_pipeline_and_api.py -v
```

* **Collected:** 28 items
* **Passed:** 28 (100%)
* **Failed:** 0
* **Skipped:** 0
* **Runtime:** 37.02s

### 5.3 Detailed Test Failure Analysis

* **Failing Test:** `tests/test_ml_rca_ranker.py::test_pairwise_inference_is_deterministic`
* **Root Cause:** In commit `5ad22ff501` (`build: configure Render and Vercel free-tier public deployment`), `experiments/ml_rca_ranker.py` was hardened with a new method `validate_candidate_schema(candidate)` requiring candidates to contain all features defined in `self.feature_names`. The test fixture `_case("other")` in `tests/test_ml_rca_ranker.py` only populated `{"s_comb": 1.0}`, causing a `ValueError: Feature schema mismatch for candidate root: missing ['e_elev', 'r_strength', 'r_early', 'r_coverage', 'r_prop']`.
* **Remediation Plan:** Update the test fixture in `tests/test_ml_rca_ranker.py` to populate all feature keys in `MODEL_A_FEATURES`.

### 5.4 Skipped Tests Accounting

The 11 skipped tests fall into two distinct, documented categories:
1. **Missing Optional Dependency (`pyrca`):** 5 tests in `tests/test_pyrca_ht.py` skip gracefully via `@unittest.skipUnless(PYRCA_AVAILABLE)`.
2. **Missing Environment Variable / Specific Dataset Paths:** 6 tests skip gracefully when run without `RCAEval_DATASET_ROOT` set or when specific secondary fixtures are absent (`tests/test_rcaeval_integration.py`, `tests/test_topology.py`, `tests/test_detective_engine.py`).

---

## 6. Benchmark Reproducibility and Metric Audit

| Benchmark / Evaluation | Method | Reported Metric | Provenance Classification | Verified Location | Notes |
|---|---|---|---|---|---|
| **RE2-OB Rep 1 Smoke (30 cases)** | $S_{\text{comb}}$ (Metrics) | Top@1: 0.7000, Top@3: 0.8667, Top@5: 0.9333, MRR: 0.8011 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (21/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | $E_{\text{elev}}$ (Traces) | Top@1: 0.6000, Top@3: 0.9667, Top@5: 1.0000, MRR: 0.7567 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (18/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | Fixed Fusion | Top@1: 0.8667, Top@3: 0.9667, Top@5: 1.0000, MRR: 0.9194 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (26/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | SimpleRCA Baseline | Top@1: 0.2667, Top@3: 0.5000, Top@5: 0.8000, MRR: 0.4522 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (8/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | Random Baseline | Top@1: 0.0333, Top@3: 0.3667, Top@5: 0.4667, MRR: 0.2482 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (1/30 Top@1) |
| **Model B Locked Test (60 cases)** | Model B (Pairwise) | Top@1: 0.8167, Top@3: 0.9667, Top@5: 1.0000, MRR: 0.8917 | **Artifact- & Test-Verified** | `eval/results/canonical_locked_test_results.json`, `tests/test_model_b_evaluation.py` | 49/60 Top@1; tests passed 4/4 |
| **Model B Locked Test (60 cases)** | Baseline $S_{\text{comb}}$ | Top@1: 0.6167, Top@3: 0.8167, Top@5: 0.8667, MRR: 0.7384 | **Artifact-Verified** | `eval/results/canonical_locked_test_results.json` | 37/60 Top@1 |
| **Zero-Shot Transfer: RE2-SS (90 cases)** | Model B | Top@1: 0.5000, Top@3: 0.9222, Top@5: 0.9667, MRR: 0.7066 | **Artifact-Verified** | `eval/results/cross_application_generalization.json` | Trained on RE2-OB (90) |
| **Zero-Shot Transfer: RE2-TT (90 cases)** | Model B | Top@1: 0.4556, Top@3: 0.7111, Top@5: 0.7333, MRR: 0.5911 | **Artifact-Verified** | `eval/results/cross_application_generalization.json` | Trained on RE2-OB (90) |
| **Zero-Shot Combined (180 cases)** | Model B | Top@1: 0.4778, Top@3: 0.8167, Top@5: 0.8500, MRR: 0.6489 | **Artifact-Verified** | `eval/results/cross_application_generalization.json` | Matches README table |
| **Temporal DL Enhancement** | 1D-CNN/GRU + Model B | MRR: 0.8458 ($\Delta = -0.0458$) | **Artifact-Verified** | `eval/results/temporal_dl_experiment_v1.json`, `docs/FINAL_PROJECT_REPORT.md` | Formally rejected |
| **RAG DD-IR-60: C0 (TF-IDF)** | Lexical TF-IDF | Hit@1: 0.05, Hit@3: 0.0833, Hit@5: 0.15, MRR: 0.0797, Latency: 2.95ms | **Artifact-Verified** | `eval/rag_evaluation_results.json` | README lists Recall@5 as 0.1200 |
| **RAG DD-IR-60: C1 (BM25 raw)** | BM25 ($k_1=1.5, b=0.75$) | Hit@1: 0.10, Hit@3: 0.15, Hit@5: 0.1833, MRR: 0.1269, Latency: 1.23ms | **Artifact-Verified** | `eval/rag_evaluation_results.json` | README lists Recall@3 as 0.1267, Recall@5 as 0.1467 |
| **RAG DD-IR-60: C2 (Dense BGE-small)** | Cosine Sim | Hit@1: 0.00, Hit@3: 0.0333, Hit@5: 0.0333, MRR: 0.0167, Latency: 134.52ms | **Artifact-Verified** | `eval/rag_evaluation_results.json` | README lists latency as 288.14ms |
| **RAG DD-IR-60: C3 (Hybrid RRF)** | BM25 + BGE RRF | Hit@1: 0.00, Hit@3: 0.0833, Hit@5: 0.15, MRR: 0.0439, Latency: 30.84ms | **Artifact-Verified** | `eval/rag_evaluation_results.json` | README lists Recall@5 as 0.1233 |
| **RAG Incident-Aware: C1_incident** | BM25 Incident-Aware | Hit@1: 0.1667, Hit@3: 0.2000, Hit@5: 0.3667, MRR: 0.2128 | **Artifact-Verified** | `eval/rag_evaluation_results.json` | Omitted from README table |
| **RAG Incident-Aware: C3_incident** | Tuned Hybrid | Hit@1: 0.2833, Hit@3: 0.2833, Hit@5: 0.6333, MRR: 0.3617 | **Artifact-Verified** | `eval/rag_evaluation_results.json` | Omitted from README table |

---

## 7. Documentation Discrepancies and Reconciliation

### 7.1 Test Count Claims
* **Discrepancy:** `docs/PROJECT_STATE.md` states **307 tests** (301 passed, 6 skipped). `README.md` states **475 automated tests** (475 passed, 11 skipped).
* **Source-Verified Fact:** The repository currently collects **522 tests** across 56 test files. Under Python 3.12.10, 510 pass, 11 skip, and 1 fails.
* **Resolution:** Both documentation files need to be updated to reflect the actual 522-test inventory and current execution counts once the single failing test is resolved.

### 7.2 Training vs. Evaluation Split Direction
* **Discrepancy:**
  * `docs/FINAL_PROJECT_REPORT.md` and `eval/results/canonical_locked_test_results.json` record:
    $$\text{Train on } \text{RE2-SS} + \text{RE2-TT} \ (180 \text{ cases}) \longrightarrow \text{Test on } \text{RE2-OB} \ (\text{Reps 2–3}, 60 \text{ cases})$$
    yielding Model B Top@1 = 0.8167, MRR = 0.8917.
  * `README.md` §3.A and `eval/results/cross_application_generalization.json` record:
    $$\text{Train on } \text{RE2-OB} \ (90 \text{ cases}) \longrightarrow \text{Test zero-shot on } \text{RE2-SS} \ (90) + \text{RE2-TT} \ (90) \ (180 \text{ cases})$$
    yielding Combined Top@1 = 0.4778, MRR = 0.6489.
* **Source-Verified Fact:** Both experiments were conducted and have corresponding artifact logs in `eval/results/`. The former evaluates in-domain performance on the primary evaluation benchmark (Online Boutique), while the latter evaluates cross-system zero-shot transfer from Online Boutique to Sock Shop and Train Ticket.
* **Resolution:** Add clear explanatory headers in `README.md` and `docs/PROJECT_STATE.md` distinguishing Experiment 1 (In-domain benchmark evaluation: Train SS+TT $\rightarrow$ Test OB) from Experiment 2 (Cross-application domain transfer: Train OB $\rightarrow$ Test SS+TT).

### 7.3 RAG Benchmark Metrics
* **Discrepancy:** Slight metric variations exist between the markdown table in `README.md` §3.B and the raw dictionary in `eval/rag_evaluation_results.json` (e.g. C1 Recall@3 listed as 0.1267 vs hit rate 0.1500; C2 latency listed as 288ms vs 134ms). In addition, the much stronger incident-aware retrieval conditions (`C1_bm25_incident` MRR 0.2128, `C3_hybrid_incident` MRR 0.3617) were omitted from the README summary table.
* **Resolution:** Reconcile the README table with the exact values in `eval/rag_evaluation_results.json` and present both raw-query and incident-aware conditions.

---

## 8. Limitations and Unknowns

1. **Test Regression in `experiments/ml_rca_ranker.py`:** The unit test `test_pairwise_inference_is_deterministic` is broken due to a schema mismatch in its test fixture.
2. **Missing `pyrca` Dependency:** `pyrca` is listed in `requirements.txt` under ML/evaluation dependencies but is not installed in the default `.venv` or declared in `pyproject.toml`. As a result, 5 PyRCA tests skip.
3. **Simulated Sandbox Boundaries:** The "remediation" subsystem simulates recovery using in-memory state mutations. It does not interact with Kubernetes APIs, Helm, ArgoCD, or live cloud infrastructure.
4. **Offline Ingestion Boundary:** All telemetry ingestion operates against offline Parquet files. No live streaming collector (OTel Collector, Prometheus remote write, Vector) is implemented.
5. **RAG Ground-Truth Annotations:** Ground-truth relevance judgments in `eval/rag2_benchmark/rag2_qrels.tsv` are partially synthetic/heuristic; full human annotation was not conducted prior to freezing.

---

## 9. Prioritized Recommendations for Next Milestones

### Milestone P0: Fix ML Ranker Unit Test Regression
* **Objective:** Restore 100% passing status across all executable unit tests.
* **Scope:** Edit `tests/test_ml_rca_ranker.py` to populate all feature keys in `MODEL_A_FEATURES` within the test helper `_case()`.
* **Acceptance Criteria:** `pytest tests/test_ml_rca_ranker.py` passes 4/4; full test suite passes 511 tests (with 11 expected skips).

### Milestone P1: Reconcile Documentation Inconsistencies
* **Objective:** Align `README.md` and `docs/PROJECT_STATE.md` with source reality.
* **Scope:**
  * Update test counts to 522 collected / 511 passed.
  * Clarify the dual split directions for Model B (In-Domain Locked Test vs. Cross-Application Zero-Shot Transfer).
  * Align RAG benchmark numbers with `eval/rag_evaluation_results.json` and highlight incident-aware retrieval.
* **Acceptance Criteria:** Zero discrepancies between documentation metrics and retained JSON evaluation artifacts.

### Milestone P2: Optional Dependency Packaging Clean-up
* **Objective:** Provide explicit optional dependency groups for evaluation backends.
* **Scope:** Add `[project.optional-dependencies] evaluation = ["pandas", "scikit-learn", "pyrca", "networkx"]` and `rag = ["sentence-transformers", "torch"]` in `pyproject.toml`.
* **Acceptance Criteria:** Clean environment installation with clear distinction between runtime, testing, evaluation, and RAG dependencies.

### Milestone P3: Expert RAG Benchmark Annotation
* **Objective:** Establish formal human-annotated relevance judgments for runbook retrieval.
* **Scope:** Execute the protocol in `eval/rag2_benchmark/ANNOTATION_PROTOCOL.md` to produce multi-annotator evaluated qrels.
* **Acceptance Criteria:** Verified Cohen's kappa $\ge 0.70$ between annotators; official RAG retrieval benchmark locked.
