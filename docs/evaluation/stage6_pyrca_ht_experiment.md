# Stage 6 Statistical Evaluation: PyRCA Hypothesis Testing (HT) vs Digital Detective S_comb

**Evaluation Mode:** Paired Cluster-Level Analysis (Stage 1C Protocol)  
**Analysis Date:** 2026-09-18 / 2026-09-19  
**Experiment Deliverable:** [`eval/results/stage6_re2ob_pyrca_ht_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage6_re2ob_pyrca_ht_treatment_v1.json)  
**Statistical Comparison Artifact:** [`eval/results/stage6_pyrca_ht_statistical_comparison_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage6_pyrca_ht_statistical_comparison_v1.json)  

---

## Result Statement

**Statistical Outcome:** **Digital Detective $S_{\text{comb}}$ outperforms PyRCA Hypothesis Testing (HT) across all evaluated ranking metrics on the frozen RE2-OB benchmark; differences are not statistically significant after Holm multiplicity correction ($p > 0.30$). $S_{\text{comb}}$ is retained as the primary metric-RCA backend.**

1. **Top@1 Accuracy:** Digital Detective $S_{\text{comb}}$ achieves **25.00% (15/60)** vs PyRCA HT **18.33% (11/60)** ($\Delta = -6.67$ percentage points, 95% cluster-bootstrap CI: `[-21.67%, +8.33%]`, paired cluster randomization $p = 0.5111$, Holm-adjusted $p = 1.0000$).
2. **Top@3 Accuracy:** $S_{\text{comb}}$ achieves **40.00% (24/60)** vs PyRCA HT **31.67% (19/60)** ($\Delta = -8.33$ percentage points, 95% cluster-bootstrap CI: `[-26.67%, +11.67%]`, $p = 0.4960$, Holm-adjusted $p = 1.0000$).
3. **Top@5 Accuracy:** $S_{\text{comb}}$ achieves **55.00% (33/60)** vs PyRCA HT **46.67% (28/60)** ($\Delta = -8.33$ percentage points, 95% cluster-bootstrap CI: `[-26.67%, +10.00%]`, $p = 0.4906$, Holm-adjusted $p = 1.0000$).
4. **Mean Reciprocal Rank (MRR):** $S_{\text{comb}}$ achieves **0.4084** vs PyRCA HT **0.3444** ($\Delta = -0.0640$, 95% cluster-bootstrap CI: `[-0.1894, +0.0606]`, $p = 0.3228$, Holm-adjusted $p = 1.0000$).

---

## 1. Provenance and Experimental Population

| Dimension | Specification / Value |
| :--- | :--- |
| **Control Baseline ($S_{\text{comb}}$)** | Stage 5 TCEC Treatment Artifact ([`stage5_tcec_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage5_tcec_treatment_v1.json)) |
| **Control Artifact SHA-256** | `F4FD25219D6D0EDC60A8781CE23E2BE58C6C936EA115C39A218BB82B8A0646F5` |
| **Treatment Artifact (PyRCA HT)** | [`eval/results/stage6_re2ob_pyrca_ht_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage6_re2ob_pyrca_ht_treatment_v1.json) |
| **Treatment Artifact SHA-256** | `3EF7AEBEC9CBBEF329AF5EBBC4792C40E6D6D8B50D1049CA48B3EBA99E430248` |
| **Benchmark Manifest** | [`eval/manifests/re2_ob_all_cases.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/manifests/re2_ob_all_cases.json) (`stage2_re2ob_rep2_rep3_baseline_v1`) |
| **Manifest Hash** | `7f82ef5d38185ad20152e3db5dab45ecab3df2a59ad518d9d1681c3ef3f15659` |
| **Dataset Population** | RE2-OB, Repetitions 2 and 3 strictly (Repetition 1 completely excluded) |
| **Executions / Cases ($N$)** | 60 executions (measurement observations) |
| **Scenario Families ($K$)** | 30 unique scenario families (inferential clusters; 2 executions per family) |
| **PyRCA Version** | `sfr-pyrca==1.0.1` |
| **Python Version** | 3.11.9 (in dedicated `.venv_pyrca` isolation environment) |
| **Statistical Framework** | `eval.stats` (paired cluster sign-swapping randomization + cluster bootstrap) |
| **Randomization Replicates** | 10,000 Monte Carlo sign-swaps over whole scenario families |
| **Bootstrap Replicates** | 10,000 cluster resamples with replacement |
| **Confidence Level & Seed** | 95% two-sided percentile CI ($\alpha = 0.05$), `seed = 42` |

---

## 2. Methodology & Experimental Controls

### 2.1 Why Hypothesis Testing (HT) Was Selected
Salesforce PyRCA's Hypothesis Testing (HT) method represents the canonical regression-residual root cause analysis approach:
- Given an existing causal directed acyclic graph (DAG), each node $j$ is predicted via linear regression from its direct parents: $\hat{X}_j = \sum_{i \in \text{Pa}(j)} \beta_i X_i$.
- Under normal operation, the residuals $\epsilon_j = X_j - \hat{X}_j$ follow a baseline error distribution.
- When an incident occurs, anomalously large standardized residuals indicate that the normal invariant between parent and child has broken down, localizing the root cause to node $j$.
- HT was chosen because it accepts a pre-existing graph without running expensive, non-deterministic graph discovery (e.g., PC, GES, LiNGAM) and evaluates strictly metric anomalies.

### 2.2 Frozen Detector & Incident Windows
- Candidate Detection: Causal BOCPD ($\lambda=100.0, \text{threshold}=0.5, \text{min\_warmup}=60$).
- Incident Confirmation: Topology-Coherent Episode Confirmation (TCEC, persistence $K=3$, consensus $M=2$, topological propagation gate).
- Incident Boundary: Strictly causal confirmation timestamp $t_{\text{confirm}}$.
- All 60 cases use the exact confirmed incident window resolved in Stage 5. Zero future telemetry ($t > t_{\text{confirm}}$) is observed.

### 2.3 Graph Source & Adjacency Orientation
- The call topology graph is constructed deterministically from observed trace service dependencies up to $t_{\text{confirm}}$ (`build_entity_graph`).
- Adjacency matrix orientation strictly follows PyRCA requirements:
  $$G[i, j] = 1 \iff i \to j \quad (\text{node } j \text{ has parent } i)$$
- Graph DAG check: Verified that all 60 RE2-OB dependency graphs are valid DAGs with zero cycles.

### 2.4 Training Data Policy
- Normal training data $N_i$: Full permitted historical telemetry strictly preceding the confirmed incident boundary ($t < t_{\text{confirm}}$).
- Standardized using normal-period mean and standard deviation.
- Incident test data $I_i$: Incident telemetry evaluated at $[t_{\text{confirm}}, t_{\text{confirm}}]$.
- Zero ground truth: `inject_time` and root-cause service labels are never accessed or observed by the model.

### 2.5 Candidate Universe Projection
- PyRCA evaluates all graph nodes (including non-candidate infrastructure nodes such as `frontend-external`).
- The reported ranking is projected strictly onto the canonical candidate universe (`eval/universe.py`), omitting non-candidate entities.
- Unobserved candidates receive a score of 0.0.
- Candidates are sorted descending by HT residual score with deterministic lexical tie-breaking: `(-score, candidate_name)`.

### 2.6 Dependency Isolation
- PyRCA has legacy dependency pins (Python $\le 3.11$, specific versions of scikit-learn and pandas).
- Environment isolation: Installed in dedicated `.venv_pyrca` (Python 3.11.9, `sfr-pyrca==1.0.1`).
- Main Digital Detective environment remains Python 3.12 without PyRCA dependency conflicts.

---

## 3. Paired Statistical Comparison: PyRCA HT vs Digital Detective $S_{\text{comb}}$

$$\Delta = \text{PyRCA HT (Arm B)} - S_{\text{comb}} \text{ (Arm A)}$$

| Metric | Digital Detective $S_{\text{comb}}$ | PyRCA HT | Paired Diff ($\Delta$) | 95% Cluster-Bootstrap CI | Paired Randomization $p$-value | Holm-Adjusted $p$-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top@1 Accuracy** | **25.00%** (15/60) | 18.33% (11/60) | **-6.67%** | `[-21.67%, +8.33%]` | 0.5111 | 1.0000 | Not Significant |
| **Top@3 Accuracy** | **40.00%** (24/60) | 31.67% (19/60) | **-8.33%** | `[-26.67%, +11.67%]` | 0.4960 | 1.0000 | Not Significant |
| **Top@5 Accuracy** | **55.00%** (33/60) | 46.67% (28/60) | **-8.33%** | `[-26.67%, +10.00%]` | 0.4906 | 1.0000 | Not Significant |
| **MRR** | **0.4084** | 0.3444 | **-0.0640** | `[-0.1894, +0.0606]` | 0.3228 | 1.0000 | Not Significant |
| **Avg@5 Accuracy** | **40.33%** | 32.00% | **-8.33%** | — | — | — | — |

---

## 4. Scientific Interpretation

1. **Residual-Based vs Episode-Topology RCA:**
   - PyRCA HT relies exclusively on linear regression residual deviations from graph parents. In microservice environments with bursty traffic and non-linear metric interactions, linear regressions on parent entity signals often exhibit high residual variance across multiple downstream services simultaneously, diluting root-cause localization.
   - Digital Detective $S_{\text{comb}}$ leverages persistent episode confirmation ($K=3, M=2$) coupled with log-anomaly magnitude and topological propagation. This discrete episode filtering effectively eliminates noise in non-faulty downstream nodes that would otherwise produce spurious regression residuals.

2. **Statistical Equivalence vs Practical Superiority:**
   - The 95% cluster-bootstrap confidence intervals for all metrics cross zero, meaning the difference is not statistically significant at $\alpha = 0.05$ after multiplicity correction.
   - However, $S_{\text{comb}}$ numerically outperforms PyRCA HT on every metric (+4 Top@1 diagnoses, +5 Top@3 diagnoses, +5 Top@5 diagnoses, +0.0640 MRR).
   - Furthermore, $S_{\text{comb}}$ requires no external C/Fortran regression solver dependencies, executes in $<1$ ms, and naturally integrates with multi-metric episode aggregation.

3. **Conclusion:**
   - PyRCA HT does not exceed $S_{\text{comb}}$ on the RE2-OB benchmark.
   - $S_{\text{comb}}$ remains the verified, frozen primary metric RCA backend for Digital Detective.
