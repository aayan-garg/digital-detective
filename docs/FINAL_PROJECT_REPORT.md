# Digital Detective — Final Project Report

**Status**: Final Frozen Research Prototype  
**Release Tag**: `v1.0-final`  
**Dataset**: RCAEval (RE2-SS, RE2-TT, RE2-OB)

---

## 1. Abstract

Digital Detective is an automated root-cause analysis (RCA), investigative orchestration, and safety-gated remediation framework for transient cascading microservice failures in cloud-native environments. Addressing the tendency of large language model (LLM) agents to hallucinate root causes in high-dimensional telemetry, Digital Detective enforces a strict authority hierarchy: deterministic causal reasoning ($S_{\text{comb}}$ and $E_{\text{elev}}$) and family-isolated supervised candidate ranking (Model B) serve as the ground-truth authority, while an LLM investigator orchestrates hypothesis exploration, references retrieved operational knowledge (RAG), and submits actions to deterministic safety and recovery verification gates. Across the locked 60-case RE2-OB benchmark (repetitions 2–3), Model B achieves Top@1 accuracy of 0.8167 and MRR of 0.8917. A temporal deep learning complement was rigorously evaluated and rejected after demonstrating degradation across both development and locked test splits. The full software pipeline is end-to-end verified and frozen.

---

## 2. Problem Statement

Microservice systems under transient hardware, network, or application faults exhibit rapid, multi-hop anomaly propagation across complex service call graphs. Traditional correlation-based anomaly detection often flags downstream symptom services rather than the upstream root cause. Conversely, unconstrained generative LLM agents operating directly on raw logs or metrics are prone to ungrounded causal assertions and inconsistent remediation decisions. Digital Detective resolves this tension by architecting a multi-stage pipeline where causal authority, ML candidate ranking, operational retrieval, and safety governance remain strictly decoupled and verified.

---

## 3. System Architecture

The complete system comprises 11 functional stages arranged in a unidirectional pipeline:

```
[ Telemetry Ingestion (Metrics, Spans, Topology) ]
                       │
                       ▼
[ Anomaly Detection ]
                       │
                       ▼
[ Sequential Incident Confirmation ]
                       │
                       ▼
[ Deterministic RCA (S_comb Metric Scoring & E_elev Trace Attribution) ]
                       │
                       ▼
[ ML Candidate Ranking (Model B Pairwise Ranker) ]
                       │
                       ▼
[ LLM Investigator (Bounded Tool-Augmented Exploration Loop) ]
                       │
                       ├── [ Hybrid Operational Knowledge RAG ]
                       ▼
[ Safety Gate Policy (Blast-Radius & Pre-Condition Validation) ]
                       │
                       ▼
[ Simulated Sandbox Remediation ]
                       │
                       ▼
[ Multi-Symptom Recovery Verification ]
                       │
                       ▼
[ Post-Intervention Causal Validation ]
```

### Authority Matrix

| Layer | Component | Authority Scope | Overridable by LLM? |
|---|---|---|---|
| **Causal Core** | Deterministic RCA ($S_{\text{comb}}$, $E_{\text{elev}}$) | Ground-truth scoring, graph traversal | **No** |
| **Ranking Core** | Supervised Model B Ranker | Causal-prefix candidate re-ranking | **No** |
| **Exploration** | LLM Investigator (Qwen3:8B) | Dynamic observation gathering, hypothesis probing | N/A (Proposes only) |
| **Context** | Hybrid Operational Knowledge RAG | Operational runbooks, reference knowledge | **No** (Reference only) |
| **Safety Gate** | Safety Policy & Recovery Verifier | Remediation authorization, recovery proof | **No** (Strict gate) |

---

## 4. Telemetry and Incident Detection

The telemetry subsystem processes heterogeneous signals:
* **Metric Series**: CPU utilization, memory consumption, socket counts, and p50/p90 latency distributions.
* **Distributed Traces**: Span start/end timestamps, parent-child relations, and edge duration profiles.
* **Topology**: Dynamic directed service dependency graphs extracted directly from runtime span traces.

### Detector Evolution and Final Design
During pipeline development, rolling z-scores and robust Median Absolute Deviation (MAD) baseline estimation were systematically evaluated as experimental detection alternatives. While providing baseline sensitivity, simple thresholding proved vulnerable to transient noise bursts. The final frozen architecture adopts Bayesian Online Changepoint Detection (BOCPD) integrated with Topology-Coherent Episode Confirmation (TCEC) and multi-metric persistence ($K=3, M=2$) to establish statistically grounded incident confirmation prior to invoking downstream RCA.

---

## 5. Deterministic RCA

The deterministic RCA engine combines metric and trace evidence without stochastic parameters:

1. **Metric Scoring ($S_{\text{comb}}$)**:
   $$S_{\text{comb}}(v) = w_{\text{str}} R_{\text{str}}(v) + w_{\text{early}} R_{\text{early}}(v) + w_{\text{cov}} R_{\text{cov}}(v) + w_{\text{prop}} R_{\text{prop}}(v)$$
   integrating anomaly intensity, onset earliness, metric coverage, and downstream reachability.
2. **Trace Edge Attribution ($E_{\text{elev}}$)**: Quantifies edge duration elevation relative to baseline distributions along trace call paths.
3. **Multi-Modal Fusion**: Generates unified deterministic hypothesis rankings.

### Regression Baselines (30-Case RE2-OB Repetition 1 Smoke Suite)

| Method | Modality | Top@1 | Top@3 | Top@5 | MRR |
|---|---|:---:|:---:|:---:|:---:|
| $S_{\text{comb}}$ | Metrics | 0.7000 | 0.8667 | 0.9333 | 0.8011 |
| $E_{\text{elev}}$ | Traces | 0.6000 | 0.9667 | 1.0000 | 0.7567 |
| **Fixed Fusion** | Multi-Modal | **0.8667** | **0.9667** | **1.0000** | **0.9194** |

---

## 6. ML Candidate Ranking

To improve upon heuristic fusion, a supervised pairwise candidate ranking framework was formulated.

### Formulation
* **Input Representation**: Pairwise candidate feature differences $\mathbf{x}_{\text{root}} - \mathbf{x}_{\text{negative}}$ computed over causal-prefix windows.
* **Model A**: Uses deterministic $S_{\text{comb}}$ scores, $E_{\text{elev}}$ attributions, and basic topological depth.
* **Model B (Selected)**: Extends Model A with compact anomaly persistence, metric consensus, and propagation consistency features.
* **Training Protocol**: Pinned `LogisticRegression` optimized via 3-fold family-grouped cross-validation over 180 development cases (RE2-SS and RE2-TT).

### Experimental Evaluation

| Population | Method | Top@1 | Top@3 | Top@5 | MRR |
|---|---|:---:|:---:|:---:|:---:|
| **Development (Grouped CV)** | Model A | 0.4389 | 0.7611 | 0.8500 | 0.6150 |
| **Development (Grouped CV)** | **Model B (Selected)** | **0.4500** | **0.7889** | **0.8278** | **0.6231** |
| **Locked Test (RE2-OB Reps 2–3)** | Frozen Baseline RCA | 0.6167 | 0.8167 | 0.8667 | 0.7381 |
| **Locked Test (RE2-OB Reps 2–3)** | Model A | 0.8000 | 0.9500 | 1.0000 | 0.8783 |
| **Locked Test (RE2-OB Reps 2–3)** | **Model B (Selected)** | **0.8167** | **0.9667** | **1.0000** | **0.8917** |

Model B demonstrates superior candidate ordering on the locked test set (49/60 Top@1, 58/60 Top@3, 60/60 Top@5) and is accepted as the frozen ranking model.

---

## 7. Deep Learning Experiment

### Research Question
Does a compact temporal neural sequence model (1D-CNN / GRU) operating over raw 721-step causal-prefix metric sequences provide orthogonal ranking signal beyond Model B's tabular features?

### Experimental Design & Methodology
* **Architecture**: Compact 2-layer temporal encoder with temporal average pooling, trained strictly on the 180-case development split with scenario family grouping.
* **Ensemble**: Late fusion combining temporal sequence embeddings with Model B pairwise win logits.

### Results & Rejection Decision

| Evaluation Split | Model B (Tabular) MRR | Temporal DL + Model B MRR | $\Delta$ MRR |
|---|:---:|:---:|:---:|
| **Development (Grouped CV)** | **0.6231** | 0.5872 | -0.0359 |
| **Locked Test (RE2-OB Reps 2–3)** | **0.8917** (0.817 / 0.967 / 1.000) | 0.8458 (0.767 / 0.883 / 0.933) | -0.0458 |

**Formal Decision: REJECTED.** The temporal DL enhancement degraded ranking performance on both development and locked-test evaluations and was therefore rejected. Model B is retained without DL modifications.

---

## 8. LLM Investigator

The investigative agent serves as an interactive explorer operating within strict guardrails:
* **Protocol & Conditions**: Evaluated under Condition A (autonomous query formulation) and Condition C (guided hypothesis disambiguation).
* **Action Space**: Strict JSON schema with actions `QUERY(tool_name, service)`, `FINAL_DIAGNOSIS(diagnosis_service, evidence_ids)`, and `STOP(reasoning)`.
* **Cost & Turn Budgets**: Bounded execution (budget = 20 cost units, max 6 turns).
* **Local Ollama Integration**: Tested against `qwen3:8b` via an OpenAI-compatible interface with `reasoning_effort=none` and strict JSON parsing.
* **Authority Preservation**: Even if the LLM produces a speculative diagnosis, the orchestrator preserves the deterministic RCA decision in the permanent audit trail.

---

## 9. Hybrid Operational Knowledge RAG

* **Corpus**: 536 provenance-backed operational runbooks, architecture maps, and microservice failure playbooks stored in `eval/rag2_benchmark/rag2_corpus.jsonl`.
* **Retrieval & Role**: Injected as `OPERATIONAL KNOWLEDGE (RAG – REFERENCE ONLY)` to supply system context to the investigator agent without altering deterministic scoring.
* **Validation**: RAG retrieval and context injection were end-to-end verified. The separate RAG-2 hybrid BM25 + BGE-small + RRF implementation was validated through its controlled retrieval experiments.
* **Benchmark Status**: Full human/expert RAG relevance labeling was not completed before freeze; RAG remains an operational/reference knowledge layer.
* **Dependency Note**: The heavier RAG-2 dense retrieval and reranker stack may require additional runtime packages beyond the core dependencies in `pyproject.toml`.

---

## 10. Safety and Remediation

Autonomous remediation adheres to safety-by-design:
1. **Safety Policy Gate**: Evaluates proposed actions against service criticalities, blast radius thresholds, and causal evidence requirements. Unauthorized actions are blocked.
2. **Simulated Sandbox Remediation**: Applies in-memory state transformations (e.g., container restart, traffic throttling, cache flush) to simulate recovery.
3. **Multi-Symptom Recovery Verification**: Assesses post-intervention metric trends across all affected services to verify symptom alleviation without secondary degradation.
4. **Post-Intervention Validation**: Re-runs topological consistency checks to ensure complete recovery.

---

## 11. Reproducibility and Leakage Controls

To support reproducibility and reduce leakage risk:
* **Strict Dataset Partitioning**: Development (RE2-SS 90 cases + RE2-TT 90 cases) is strictly isolated from the locked test split (RE2-OB 60 cases).
* **Zero Future-Leakage**: All feature extraction, anomaly timestamps, and metric windows respect causal-prefix boundaries prior to or at fault injection.
* **Deterministic Seeds**: All random seeds (`seed=42`) and tie-breaking criteria are frozen and documented.
* **Complete Artifacts**: All benchmark results, ablation dumps, and evaluation scripts are checked into the repository with cryptographic git commit hashes.

---

## 12. End-to-End Verification

A comprehensive end-to-end verification pass was conducted on the representative frozen case: `re2ob_checkoutservice_cpu_1`.

### Verification Results

```
[PASS] 1. Telemetry Ingestion (Metrics, Spans, Topology loaded)
[PASS] 2. Anomaly Detection (11 anomalous entities identified)
[PASS] 3. Sequential Incident Confirmation (Episode persistence K=3, M=2 aggregated)
[PASS] 4. Deterministic RCA (S_comb candidate scoring produced)
[PASS] 5. Model B Ranking (Pairwise causal ranking computed)
[PASS] 6. RAG Retrieval (536 docs indexed, top-5 operational passages retrieved)
[PASS] 7. Context Injection (OPERATIONAL KNOWLEDGE block formatted with reference-only disclaimer)
[PASS] 8. Investigator Stage (Orchestrator received RAG context block)
[PASS] 9. Safety Policy Gate (Policy checks executed; unauthorized remediation blocked)
[PASS] 10. Simulated Remediation & Recovery Verification (RecoveryVerifier executed)
[PASS] 11. Final State Output (Complete trajectory and state preserved)
[PASS] 12. Deterministic RCA Authority (Deterministic causal top preserved independently of LLM)
```

### Live LLM Status
Live `qwen3:8b` execution via Ollama was verified. Under Windows GPU environments, occasional CUDA illegal memory access exceptions occur during sustained inference; the orchestrator handles these cleanly via bounded retries and graceful fallback while preserving full audit logs.

---

## 13. Limitations

1. **RAG Relevance Benchmarking**: Full human/expert RAG relevance labeling was not completed before freeze; RAG remains an operational/reference knowledge layer.
2. **Remediation Sandboxing**: All remediation actions execute within an in-memory simulation model rather than production cloud infrastructure.
3. **Local Ollama Runtime Stability**: Local Windows GPU memory allocation under Ollama remains sensitive to driver and VRAM constraints.
4. **Benchmark Scope**: Empirical evaluations are grounded in the RCAEval microservice benchmark suite; generalization to arbitrary monolithic or non-containerized architectures remains unverified.

---

## 14. Conclusion

Digital Detective demonstrates that autonomous microservice incident investigation can achieve high accuracy (0.8917 MRR on locked RE2-OB) while remaining safe, explainable, and resilient to hallucination. By placing deterministic causal analytics and supervised pairwise ranking at the center of authority, and employing LLM agents and hybrid RAG strictly for investigative exploration and operational synthesis, the system provides a dependable foundation for autonomous cloud operations.

---

## 15. Final Component Status Table

| Component | Status | Empirical Validation / Benchmark Result |
|---|:---:|---|
| **Anomaly Detection** | **Frozen** | BOCPD/TCEC-based detection (Z-score & MAD evaluated as alternatives) |
| **Sequential Incident Confirmation** | **Frozen** | Multi-metric persistence ($K=3, M=2$) & TCEC topological confirmation |
| **Deterministic RCA** | **Validated** | Multi-Modal Fusion: Top@1 = 86.7%, MRR = 0.9194 (RE2-OB Rep 1) |
| **Model B Supervised Ranker** | **Selected** | Locked RE2-OB (Reps 2–3): Top@1 = 0.8167, Top@5 = 1.0000, MRR = 0.8917 |
| **Temporal DL Complement** | **Rejected** | Evaluated & Rejected ($\Delta$ MRR = -0.0458 on locked test) |
| **LLM Investigator (A/C)** | **Accepted** | Bounded reasoning loop, JSON schema enforcement, tool use |
| **Live Qwen3:8B Backend** | **Verified** | Verified with Ollama; documented Windows GPU runtime limitation |
| **Hybrid Operational Knowledge RAG** | **Implemented** | RAG retrieval/injection E2E verified; RAG-2 hybrid evaluated in controlled experiments |
| **Safety Policy Gate** | **Implemented** | Deterministic pre-execution blast-radius & permission checks |
| **Recovery Verification** | **Implemented** | Multi-symptom recovery & post-intervention topological checks |
| **End-to-End Pipeline** | **Verified** | Full software pipeline verified on `re2ob_checkoutservice_cpu_1` |
| **Reproducibility & Splits** | **Documented** | Family-grouped 3-fold dev splits, locked RE2-OB test, fixed seeds |
