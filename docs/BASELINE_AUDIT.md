# Digital Detective — Architecture, Baseline, and Reproducibility Audit

**Audit Date:** 2026-11-13  
**Audited Commit SHA:** `391438f8e59f4c4ae4de956265b51934f1a5a279`  
**Audited Branch:** `agent/issue-3-audit-architecture` (tracking `origin/main`)  
**Auditor:** Antigravity Agent  
**Revision:** v1.1 (incorporating Architect review findings from PR #5 / Issue #3)  
**Audit Purpose:** Comprehensive, source-verified baseline snapshot of implemented architecture, active runtime execution flow, test suite health, benchmark reproducibility, and documentation consistency for GitHub Issue #3.

---

## 1. Executive Summary

Digital Detective has evolved across distinct research and engineering phases:
1. **Phases 0–15 (Foundational RCA & Modular Detective):** Documented in `docs/PROJECT_STATE.md`, culminating in the frozen baseline for metric scoring ($S_{\text{comb}}$), trace latency elevation ($E_{\text{elev}}$), and equal-weight fusion on the 30-case RE2-OB repetition-1 smoke benchmark.
2. **Phase 16+ (Supervised Ranking, Multimodal Features, Hybrid RAG, & Web Deployment):** Implemented in `src/digital_detective/pipeline.py`, `experiments/ml_rca_ranker.py`, `src/digital_detective/api/main.py`, and `src/digital_detective/web/`, featuring Model B pairwise ranking, Drain3 log template scoring, topological ridge residuals, hybrid RRF runbook retrieval, and a public-facing FastAPI/Vercel demonstration layer.

### Key Audit Findings

* **Frozen Baselines Verified:** The core deterministic baselines ($S_{\text{comb}}$, $E_{\text{elev}}$, Fixed Equal-Weight Fusion) were **100% reproduced** from the local dataset cache (`~/.cache/rcaeval_validation`) matching documented values to the 4th decimal place.
* **Test Suite Health:** The test suite contains **522 collected tests** across 56 test modules. When executed under Python 3.12.10:
  * **510 tests passed**.
  * **11 tests skipped** (5 due to optional `pyrca` dependency, 6 due to unset dataset environment variables / optional fixtures).
  * **1 test failed** (`tests/test_ml_rca_ranker.py::test_pairwise_inference_is_deterministic`), caused by a test-fixture schema mismatch introduced when candidate schema validation was hardened in commit `5ad22ff`.
* **Runtime Verification & Discrepancies:**
  * **Detector Runtime Wiring:** In the default active runtime (`pipeline.py` and API request model), `use_bocpd=False` by default; changepoints are detected via streaming rolling z-scores. `BOCPD` is implemented but gated behind `use_bocpd=True`. `SequentialTCEC` exists as an experimental module but is **not wired into `pipeline.py`**.
  * **Execution Order:** In `pipeline.py`, Operational Knowledge RAG retrieval is executed at **Stage 8**, *prior* to Margin-Triggered Disambiguation at **Stage 9**, allowing RAG context to inform disambiguation and agent reasoning.
  * **Disambiguation Default Threshold:** The active default threshold in both `Disambiguator` and `InvestigationPipeline` is **$\tau = 0.15$** (not 0.05, which was merely an arbitrary override in unit test mocks).
  * **Model B Identity Mismatch:** The active source code (`experiments/ml_rca_ranker.py`) and serialized model artifact (`models/model_b_ranker_v1.pkl`) use **LogisticRegression** (`solver="liblinear"`). The cross-application metadata labels it a "Pairwise Linear SVM", while the README describes it as a "pairwise gradient-boosted ranker".
  * **RAG Metric Definitions:** The README labels its RAG metrics as "Recall@k", but the underlying evaluation script computes both `hit@k` (success rate) and `std_recall@k` (standard multi-document recall). Neither metric exactly matches the README table values, classifying those values as unresolved provenance.
  * **Split Direction:** `FINAL_PROJECT_REPORT.md` (Train SS+TT $\rightarrow$ Test OB) and `README.md` (Train OB $\rightarrow$ Transfer SS+TT) represent two distinct valid experiments that were previously conflated.

---

## 2. Source-Verified Architecture and Runtime Flow

### 2.1 The Production Pipeline (`src/digital_detective/pipeline.py`)

The active runtime orchestrator is `DigitalDetectivePipeline` in `src/digital_detective/pipeline.py`, invoked via `/api/investigate` in `src/digital_detective/api/main.py`. Tracing the exact source comments and execution lines reveals 13 sequential stages:

```
[Stage 1: Telemetry Ingestion (Metrics, Logs, Traces, Injection Meta)]
                              │
                              ▼
[Stage 2: Anomaly Detection & Incident Window]
 (Default: streaming_zscore; Optional if use_bocpd=True: causal_bocpd)
                              │
                              ▼
[Stage 3: Trace Latency Evidence & Service Topology Graph]
                              │
                              ▼
[Stage 4: Entity Episodes (K=3 persistence, M=2 consensus)]
                              │
                              ▼
[Stage 5: Deterministic RCA Scoring (S_comb and E_elev)]
                              │
                              ▼
[Stage 6: Structural Causal Residuals & Log Evidence (Ridge + Drain3)]
                              │
                              ▼
[Stage 7: Model B Feature Construction & Pairwise Ranking]
                              │
                              ▼
[Stage 8: Operational Knowledge RAG Retrieval (BM25 + Dense BGE + RRF)]
                              │
                              ▼
[Stage 9: Margin-Triggered Disambiguation (Triggered if Δ = s1 - s2 < 0.15)]
                              │
                              ▼
[Stage 10: Agent Trajectory & Reasoning (Bounded ReAct / Heuristic)]
                              │
                              ▼
[Stage 11: Safety Policy & Remediation Plan]
                              │
                              ▼
[Stage 12: Recovery Verification Simulation (Post-intervention check)]
                              │
                              ▼
[Stage 13: Canonical Schema Serialization (Pydantic CanonicalIncident)]
```

### 2.2 Authority Hierarchy and Boundary Mapping

| Level | Component | Authority Scope | Overridable by Agent/LLM? | Active Source Location |
|---|---|---|---|---|
| **Causal Foundation** | $S_{\text{comb}}$, $E_{\text{elev}}$ | Deterministic score assignment, topological graph traversal | **Strictly No** | `src/digital_detective/rca.py`, `trace_attribution.py` |
| **Supervised Ranker** | Model B (`LogisticRegression`) | Causal-prefix candidate reordering | **Strictly No** | `experiments/ml_rca_ranker.py`, `models/model_b_ranker_v1.pkl` |
| **Context Layer** | Hybrid RAG (Stage 8) | Operational runbooks, reference knowledge | **Reference Only** | `src/digital_detective/rag/` |
| **Disambiguator** | Margin Guard ($\tau = 0.15$, Stage 9) | Resolves close candidate ties ($\Delta < 0.15$); otherwise preserves Model B order | **Bounded** | `src/digital_detective/agent/disambiguator.py` |
| **Safety Gate** | Safety Policy (Stage 11) | Evaluates action prerequisites, degradation evidence, and rollback plans | **Strictly No** | `src/digital_detective/detective/remediation.py` |
| **Recovery** | Recovery Verifier (Stage 12) | Proves post-intervention metric symptom alleviation | **Strictly No** | `src/digital_detective/detective/verification.py` |

---

## 3. Component Status Matrix

| Component | Implementation File | Status | Traced Runtime Wiring & Evidence |
|---|---|---|---|
| **Canonical Telemetry Schema** | `src/digital_detective/telemetry.py`, `schema.py` | **Implemented & Active** | Central data models across pipeline, API, and serialization |
| **RCAEval Local Adapter** | `src/digital_detective/rcaeval.py` | **Implemented & Active** | Ingests raw Parquet telemetry without schema loss |
| **Metric Anomaly Detector (Rolling Z)** | `src/digital_detective/anomaly.py` | **Implemented & Active (Default)** | Default Stage 2 detector in `pipeline.py` (`cp_method = "streaming_zscore"`) |
| **BOCPD Incident Detector** | `src/digital_detective/bocpd.py` | **Implemented & Optional** | Invoked in Stage 2 only when `use_bocpd=True` (API request default is `False`) |
| **Sequential TCEC** | `src/digital_detective/sequential_tcec.py`, `tcec.py` | **Implemented (Disconnected)** | Standalone module evaluated in Stage 9 research; **not wired into `pipeline.py`** |
| **Trace Topology Extraction** | `src/digital_detective/topology.py` | **Implemented & Active** | Dynamic graph construction from trace spans (Stage 3) |
| **Trace Duration Elevation ($E_{\text{elev}}$)** | `src/digital_detective/trace_attribution.py` | **Implemented & Active** | Trace edge duration elevation and caller concentration (Stage 5) |
| **Metric RCA Baseline ($S_{\text{comb}}$)** | `src/digital_detective/rca.py` | **Implemented & Active** | Multi-factor metric scoring (strength, earliness, coverage, propagation; Stage 5) |
| **Structural Causal Residuals** | `src/digital_detective/features/causal_residuals.py` | **Implemented & Active** | Pre-incident topological ridge residuals against caller parents (Stage 6) |
| **Drain3 Log Scorer** | `src/digital_detective/features/log_scorer.py` | **Implemented & Active** | Drain3 template clustering and novel error burst scoring (Stage 6) |
| **Model B Ranker** | `experiments/ml_rca_ranker.py`, `models/model_b_ranker_v1.pkl` | **Implemented & Active** | Pairwise `LogisticRegression` loaded in Stage 7 |
| **Hybrid Runbook RAG** | `src/digital_detective/rag/` (`bm25.py`, `dense.py`, `hybrid.py`, `context_assembly.py`) | **Implemented & Active** | Ingests 536 runbooks; BM25 + Dense BGE + RRF executed at Stage 8 |
| **Uncertainty Disambiguator** | `src/digital_detective/agent/disambiguator.py` | **Implemented & Active** | Evaluated at Stage 9 with active runtime default threshold $\tau = 0.15$ |
| **Grounded Investigator** | `src/digital_detective/agent/orchestrator.py`, `detective/investigator.py` | **Implemented (Bounded/Demo)** | Stage 10 bounded execution; full multi-turn ReAct runs in demo scripts |
| **Safety Gate Policy** | `src/digital_detective/detective/remediation.py` | **Implemented & Active** | Deterministic validation of action type, rollback, target, degradation (Stage 11) |
| **Recovery Verifier** | `src/digital_detective/detective/verification.py` | **Implemented & Active** | Post-intervention simulation and symptom resolution verification (Stage 12) |
| **FastAPI Backend Service** | `src/digital_detective/api/main.py` | **Implemented & Active** | Exposes `/api/health`, `/api/incidents`, `/api/investigate`, `/api/remediation/action` |
| **Web UI Dashboard** | `src/digital_detective/web/` (`index.html`, `app.js`, `styles.css`) | **Implemented & Active** | Incident replay UI deployed on Vercel |
| **PyRCA Hypothesis Testing** | `src/digital_detective/pyrca_ht.py` | **Implemented (Experimental)** | Optional comparison backend; requires uninstalled `pyrca` package |
| **Counterfactual Removal (CRV)** | `src/digital_detective/crv.py` | **Implemented (Experimental)** | Stage 7 research ablation; not in active pipeline |
| **Temporal Deep Learning** | `experiments/temporal_dl_experiment.py` | **Evaluated & Formally Rejected** | Degraded MRR (-0.0458 on locked test); isolated from core pipeline |
| **Live Kubernetes Driver** | N/A | **Documentation-Only / Simulated** | In-memory simulated state transitions only; no cluster driver |
| **Streaming Telemetry Ingestion** | N/A | **Documentation-Only** | Offline file ingestion only; no live Kafka or OTel collector |

---

## 4. Frozen Invariants and Source Locations

1. **$S_{\text{comb}}$ Metric Scoring Formula:**  
   `src/digital_detective/rca.py` (`rank_with_s_comb`, lines 300–450).  
   Formula: $S_{\text{comb}}(v) = w_{\text{str}} R_{\text{str}}(v) + w_{\text{early}} R_{\text{early}}(v) + w_{\text{cov}} R_{\text{cov}}(v) + w_{\text{prop}} R_{\text{prop}}(v)$. Weights and normalizations are frozen.
2. **$E_{\text{elev}}$ Trace Duration Elevation:**  
   `src/digital_detective/trace_attribution.py` (`compute_trace_attribution`, lines 180–310).  
   Combines edge duration elevation, caller wait concentration, and terminal factor.
3. **Candidate Universe Policy:**  
   `eval/universe.py` (`resolve_candidate_universe`).  
   Restricts candidates strictly to services observed in telemetry and topology; forbids phantom candidate generation.
4. **Temporal Partitioning / Anti-Leakage Gate:**  
   `src/digital_detective/tcec.py`, `src/digital_detective/rca.py`, `src/digital_detective/pipeline.py`.  
   Feature extraction strictly respects pre-injection windows.
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
* **Root Cause:** In commit `5ad22ff501` (`build: configure Render and Vercel free-tier public deployment`), `experiments/ml_rca_ranker.py` added `validate_candidate_schema(candidate)` requiring candidates to contain all features defined in `self.feature_names`. The test fixture `_case("other")` in `tests/test_ml_rca_ranker.py` only populated `{"s_comb": 1.0}`, causing a `ValueError: Feature schema mismatch for candidate root: missing ['e_elev', 'r_strength', 'r_early', 'r_coverage', 'r_prop']`.
* **Remediation Plan:** Update the test fixture in `tests/test_ml_rca_ranker.py` to populate all feature keys in `MODEL_A_FEATURES`.

### 5.4 Skipped Tests Accounting

The 11 skipped tests fall into two categories:
1. **Missing Optional Dependency (`pyrca`):** 5 tests in `tests/test_pyrca_ht.py` skip gracefully via `@unittest.skipUnless(PYRCA_AVAILABLE)`.
2. **Missing Environment Variable / Specific Dataset Paths:** 6 tests skip gracefully when run without `RCAEval_DATASET_ROOT` set or when specific secondary fixtures are absent (`tests/test_rcaeval_integration.py`, `tests/test_topology.py`, `tests/test_detective_engine.py`).

---

## 6. Benchmark Reproducibility and Metric Audit

### 6.1 RCA Ranking & Baselines

| Benchmark / Evaluation | Method | Reported Metric | Provenance Classification | Verified Location | Notes |
|---|---|---|---|---|---|
| **RE2-OB Rep 1 Smoke (30 cases)** | $S_{\text{comb}}$ (Metrics) | Top@1: 0.7000, Top@3: 0.8667, Top@5: 0.9333, MRR: 0.8011 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (21/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | $E_{\text{elev}}$ (Traces) | Top@1: 0.6000, Top@3: 0.9667, Top@5: 1.0000, MRR: 0.7567 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (18/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | Fixed Fusion | Top@1: 0.8667, Top@3: 0.9667, Top@5: 1.0000, MRR: 0.9194 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (26/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | SimpleRCA Baseline | Top@1: 0.2667, Top@3: 0.5000, Top@5: 0.8000, MRR: 0.4522 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (8/30 Top@1) |
| **RE2-OB Rep 1 Smoke (30 cases)** | Random Baseline | Top@1: 0.0333, Top@3: 0.3667, Top@5: 0.4667, MRR: 0.2482 | **Reproduced** | `eval.harness --smoke-re2-ob` | Exact reproduction (1/30 Top@1) |
| **Model B Locked Test (60 cases)** | Model B (`LogisticRegression`) | Top@1: 0.8167, Top@3: 0.9667, Top@5: 1.0000, MRR: 0.8917 | **Artifact- & Test-Verified** | `eval/results/canonical_locked_test_results.json`, `tests/test_model_b_evaluation.py` | 49/60 Top@1; tests passed 4/4 |
| **Model B Locked Test (60 cases)** | Baseline $S_{\text{comb}}$ | Top@1: 0.6167, Top@3: 0.8167, Top@5: 0.8667, MRR: 0.7384 | **Artifact-Verified** | `eval/results/canonical_locked_test_results.json` | 37/60 Top@1 |
| **Zero-Shot Transfer: RE2-SS (90 cases)** | Model B | Top@1: 0.5000, Top@3: 0.9222, Top@5: 0.9667, MRR: 0.7066 | **Artifact-Verified** | `eval/results/cross_application_generalization.json` | Trained on RE2-OB (90) |
| **Zero-Shot Transfer: RE2-TT (90 cases)** | Model B | Top@1: 0.4556, Top@3: 0.7111, Top@5: 0.7333, MRR: 0.5911 | **Artifact-Verified** | `eval/results/cross_application_generalization.json` | Trained on RE2-OB (90) |
| **Zero-Shot Combined (180 cases)** | Model B | Top@1: 0.4778, Top@3: 0.8167, Top@5: 0.8500, MRR: 0.6489 | **Artifact-Verified** | `eval/results/cross_application_generalization.json` | Matches README §3.A |
| **Temporal DL Enhancement** | 1D-CNN/GRU + Model B | MRR: 0.8458 ($\Delta = -0.0458$) | **Artifact-Verified** | `eval/results/temporal_dl_experiment_v1.json`, `docs/FINAL_PROJECT_REPORT.md` | Formally rejected |

---

### 6.2 Hybrid RAG Benchmark (DD-IR-60 on 536 Operational Runbooks)

`scripts/evaluate_rag_system.py` evaluates 60 benchmark queries across two distinct recall formulations stored in `eval/rag_evaluation_results.json`:
* **Hit@k (Success Rate):** Fraction of queries where at least one relevant document appears in top-$k$.
* **Std Recall@k:** Fraction of total relevant documents for the query retrieved in top-$k$.

#### Verified Artifact Results vs. README Claims

| Condition | Description | Hit@1 (Artifact) | Hit@3 (Artifact) | Hit@5 (Artifact) | Std Recall@5 (Artifact) | MRR (Artifact) | nDCG@5 (Artifact) | Latency ms (Artifact) | README Reported Values (§3.B) | Classification |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **C0_tfidf** | Raw TF-IDF | 0.0500 | 0.0833 | 0.1500 | 0.0254 | 0.0797 | 0.0555 | 2.95 | Rec@1: 0.0500, Rec@3: 0.0833, Rec@5: 0.1200, nDCG: 0.0838, Lat: 2.97 | **Unresolved Provenance** (Recall@5 & nDCG differ) |
| **C1_bm25** | Raw BM25 | 0.1000 | 0.1500 | 0.1833 | 0.0284 | 0.1269 | 0.0696 | 1.23 | Rec@1: 0.1000, Rec@3: 0.1267, Rec@5: 0.1467, nDCG: 0.1803, Lat: 1.33 | **Unresolved Provenance** (Recall@3/5 & nDCG differ) |
| **C2_dense** | Raw Dense BGE | 0.0000 | 0.0333 | 0.0333 | 0.0030 | 0.0167 | 0.0071 | 134.52 | Rec@1: 0.0000, Rec@3: 0.0167, Rec@5: 0.0333, nDCG: 0.0210, Lat: 288.14 | **Unresolved Provenance** (Latency & Recall@3 differ) |
| **C3_hybrid_raw** | Raw Hybrid RRF | 0.0000 | 0.0833 | 0.1500 | 0.0163 | 0.0439 | 0.0282 | 30.84 | Rec@1: 0.0000, Rec@3: 0.0833, Rec@5: 0.1233, nDCG: 0.0747, Lat: 30.98 | **Unresolved Provenance** (Recall@5 & nDCG differ) |
| **C1_bm25_incident** | BM25 Incident-Aware | 0.1667 | 0.2000 | 0.3667 | 0.0815 | 0.2128 | 0.1729 | 1.24 | *(Omitted from README)* | **Artifact-Verified** |
| **C3_hybrid_incident** | Tuned Hybrid Incident | 0.2833 | 0.2833 | 0.6333 | 0.1082 | 0.3617 | 0.2297 | 26.92 | *(Omitted from README)* | **Artifact-Verified** |

*Analysis:* The README claims under §3.B do not match either `hit@k` or `std_recall@k` in the retained JSON artifact. They appear to derive from an earlier or alternate configuration run whose raw artifact is not checked into the repository. The verified JSON artifact demonstrates that incident-aware retrieval significantly outperforms raw-query retrieval (C3 tuned hybrid achieves Hit@5 = 0.6333 and MRR = 0.3617).

---

## 7. Documentation Discrepancies and Reconciliation

### 7.1 Model B Identity and Model Family Provenance
* **Discrepancy:**
  * `README.md` §2 & §3.A describes Model B as a **"pairwise gradient-boosted ranker"**.
  * `eval/results/cross_application_generalization.json` records model as **"Model B (Pairwise Linear SVM)"**.
  * `models/model_b_ranker_v1.pkl` and `experiments/ml_rca_ranker.py` implement **`LogisticRegression(solver='liblinear')`**.
* **Source-Verified Fact:** The repository contains no gradient-boosted decision tree libraries (no XGBoost, LightGBM, or CatBoost in `requirements.txt` or `pyproject.toml`). The actual active model loaded by `pipeline.py` is a binary Logistic Regression model trained on pairwise feature differences.
* **Resolution:** Correct `README.md` and evaluation notes to accurately name Model B as a **pairwise Logistic Regression ranker**.

### 7.2 Training vs. Evaluation Split Direction
* **Discrepancy:**
  * `docs/FINAL_PROJECT_REPORT.md` and `eval/results/canonical_locked_test_results.json`:
    $$\text{Train on } \text{RE2-SS} + \text{RE2-TT} \ (180 \text{ cases}) \longrightarrow \text{Test on } \text{RE2-OB} \ (\text{Reps 2–3}, 60 \text{ cases})$$
    yielding Model B Top@1 = 0.8167, MRR = 0.8917.
  * `README.md` §3.A and `eval/results/cross_application_generalization.json`:
    $$\text{Train on } \text{RE2-OB} \ (90 \text{ cases}) \longrightarrow \text{Test zero-shot on } \text{RE2-SS} \ (90) + \text{RE2-TT} \ (90) \ (180 \text{ cases})$$
    yielding Combined Top@1 = 0.4778, MRR = 0.6489.
* **Source-Verified Fact:** Both experiments exist as distinct artifacts in `eval/results/`. The former assesses in-domain performance on Online Boutique, while the latter assesses zero-shot cross-system transfer to Sock Shop and Train Ticket.
* **Resolution:** Delineate both experiments with explicit headings in `README.md` and `docs/PROJECT_STATE.md`.

### 7.3 Test Count Inventory
* **Discrepancy:** `docs/PROJECT_STATE.md` records **307 tests**. `README.md` records **475 tests**.
* **Source-Verified Fact:** The test suite currently collects **522 tests** across 56 modules (510 passed, 1 failed, 11 skipped).
* **Resolution:** Update both documents to reflect the true count of 522 tests once the single test failure is resolved.

### 7.4 Disambiguation Default Threshold
* **Discrepancy:** Earlier audit notes cited $\tau = 0.05$.
* **Source-Verified Fact:** Both `Disambiguator` in `src/digital_detective/agent/disambiguator.py` (line 69) and `InvestigationPipeline` in `src/digital_detective/pipeline.py` (line 102) declare default `margin_threshold: float = 0.15`.
* **Resolution:** Document $\tau = 0.15$ as the canonical default.

---

## 8. Limitations and Unknowns

1. **Test Regression in `experiments/ml_rca_ranker.py`:** The unit test `test_pairwise_inference_is_deterministic` fails due to mock fixture schema incompatibility.
2. **Missing `pyrca` Dependency:** `pyrca` is listed in `requirements.txt` under evaluation dependencies but is not installed in `.venv`. 5 PyRCA tests skip.
3. **Simulated Sandbox Boundaries:** The remediation subsystem simulates recovery via in-memory state mutations; it does not connect to Kubernetes or cloud providers.
4. **Offline Ingestion Boundary:** Telemetry is ingested from offline Parquet files; live streaming ingestion is not implemented.
5. **RAG Ground-Truth Annotations:** Annotations in `eval/rag2_benchmark/rag2_qrels.tsv` are partially synthetic/heuristic; expert human annotation remains unfinalized.

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
  * Correct Model B description from "gradient-boosted" to "pairwise Logistic Regression".
  * Clarify dual split directions (In-Domain Locked Test vs. Cross-Application Zero-Shot Transfer).
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
