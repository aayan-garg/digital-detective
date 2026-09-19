# Stage 10 Statistical Evaluation: Observational Propagation Consistency

**Evaluation Mode:** Paired Cluster-Level Randomization and Bootstrap Analysis  
**Analysis Date:** 2026-09-19  
**Experiment Deliverable:** [`eval/results/stage10_re2ob_propagation_consistency_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage10_re2ob_propagation_consistency_treatment_v1.json)  
**Statistical Comparison Artifact:** [`eval/results/stage10_propagation_consistency_statistical_comparison_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage10_propagation_consistency_statistical_comparison_v1.json)  

---

## Result Statement

**Experimental Verdict:** **FAILURE.**

**Summary:** The parameter-free temporal-topological propagation consistency treatment significantly degraded primary RCA accuracy relative to the frozen Stage-9 control. MRR decreased by **-0.1515** (from 0.5479 down to 0.3964; 95% clustered bootstrap CI `[-0.2412, -0.0638]`, Holm-adjusted $p = 0.0104$, statistically significant degradation). Top@1 accuracy decreased from 35.00% to **18.33%** (-16.67 percentage points, 10 fewer correct top-ranked cases). Ranking-change analysis reveals that downstream symptoms and unrelated nodes with earlier detected metric anomaly onsets systematically displaced the true root cause (true root demoted in 27/60 cases, while promoted in only 9/60).

---

## 1. Provenance and Experimental Population

| Dimension | Specification / Value |
| :--- | :--- |
| **Control Baseline Artifact** | [`eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json) |
| **Control Artifact SHA-256** | `B8E9F3EEFB608A9A57B39F84BC8CAFDD9C54C3E51951595DE80652611C615A32` |
| **Treatment Artifact** | [`eval/results/stage10_re2ob_propagation_consistency_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage10_re2ob_propagation_consistency_treatment_v1.json) |
| **Treatment Artifact SHA-256** | `5B18F004F8F2885B472E70BEE0803D26F0BE8133226B423CA3FB50E42200887C` |
| **Comparison Artifact** | [`eval/results/stage10_propagation_consistency_statistical_comparison_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage10_propagation_consistency_statistical_comparison_v1.json) |
| **Comparison Artifact SHA-256** | `4C8F838C5585FCF14819A149B04B931DF9F0EDF65D8CFBC98413C221BC5F85E1` |
| **Benchmark Manifest** | [`eval/manifests/re2_ob_all_cases.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/manifests/re2_ob_all_cases.json) (`stage2_re2ob_rep2_rep3_baseline_v1`) |
| **Manifest Hash** | `7f82ef5d38185ad20152e3db5dab45ecab3df2a59ad518d9d1681c3ef3f15659` |
| **Dataset Population** | RE2-OB, Repetitions 2 and 3 strictly (Repetition 1 completely excluded) |
| **Executions / Cases ($N$)** | 60 executions across 30 scenario families (2 executions per family) |
| **Statistical Methodology** | Paired cluster-level randomization (10,000 sign-swaps) + cluster bootstrap (10,000 resamples, seed=42) |

---

## 2. Treatment Definition and Causal Semantics

### Frozen Control (Stage 9 RCA)
- Streaming confirmation cutoff $\tau_{\text{confirm}}$ derived from Sequential TCEC.
- Strictly causal telemetry prefix $X_{\le \tau_{\text{confirm}}}$ (no future telemetry, no inject_time fallback).
- Metric score $S_{\text{comb}}$ and trace elevation $E_{\text{elev}}$ combined with fixed equal-weight fusion:
  $$F(c) = 0.5 \cdot \text{norm}(S_{\text{comb}}(c)) + 0.5 \cdot \text{norm}(E_{\text{elev}}(c))$$
- Deterministic lexicographic tie-breaking: $(-F(c), c)$.

### Stage-10 Treatment: Observational Propagation Consistency
1. **Anomalous Entities ($A$):**
   Entities in the candidate universe with a confirmed anomaly episode within $X_{\le \tau_{\text{confirm}}}$. For each $v \in A$, let $\tau_v$ be its first observed episode onset.
2. **Static Adjacency Orientation:**
   The observed causal dependency graph (truncated at $\tau_{\text{confirm}}$) defines an undirected adjacency relation. For each connected pair $\{u, v\}$:
   - If $\tau_u < \tau_v$: directed temporal propagation edge $u \to v$.
   - If $\tau_v < \tau_u$: directed temporal propagation edge $v \to u$.
   - If $\tau_u == \tau_v$: no temporal propagation edge.
   The resulting graph is strictly time-increasing and acyclic (DAG).
3. **Propagation Coverage ($P(c)$):**
   Number of distinct anomalous entities reachable from $c$ along directed temporal edges (excluding $c$).
   If $c \notin A$ or $|A| < 2$, $P(c) = 0$.
4. **Lexicographic Treatment Ordering:**
   - Higher $P(c)$ first.
   - If $P(c)$ ties, higher frozen $F(c)$ first.
   - If both tie, preserve existing frozen baseline tie-break.
5. **Zero Parameters:**
   No weights, no decay constants, no PageRank, no thresholds, no tuning.

### Directionality Limitation
Temporal orientation reflects **observational propagation consistency** along static adjacency, not proven causal influence. Early symptom emergence on an upstream orchestrator or downstream dependency due to differential metric sensitivity can invert the temporal arrow.

---

## 3. Paired Statistical Comparison (N=60, K=30 Families)

$$\Delta = \text{Treatment (Stage 10)} - \text{Control (Stage 9)}$$

| Endpoint | Treatment | Control | Diff ($\Delta$) | 95% Clustered CI | Randomization $p$-value | Holm-Adjusted $p$-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top@1** | 0.1833 | **0.3500** | **-0.1667** | `[-0.2833, -0.0500]` | 0.0195 | 0.0585 | Substantial Negative Trend |
| **Top@3** | 0.4500 | **0.6500** | **-0.2000** | `[-0.3500, -0.0333]` | 0.0340 | 0.0680 | Substantial Negative Trend |
| **Top@5** | 0.7500 | **0.8333** | **-0.0833** | `[-0.1833, +0.0000]` | 0.1819 | 0.1819 | Degradation |
| **MRR** | 0.3964 | **0.5479** | **-0.1515** | `[-0.2412, -0.0638]` | **0.0026** | **0.0104** | **Statistically Significant Degradation ($p < 0.05$)** |
| **Avg@5** | 0.4600 | **0.6300** | **-0.1700** | `[-0.2767, -0.0633]` | 0.0058 | — | Secondary Degradation |

---

## 4. Family-Level Consistency Analysis

Across the 30 scenario families:
- **Families with Treatment MRR > Control MRR:** 5 / 30 (16.7%)
- **Families with Treatment MRR < Control MRR:** 19 / 30 (63.3%)
- **Families with Treatment MRR == Control MRR:** 6 / 30 (20.0%)

Nearly 4 times as many scenario families suffered degradation under observational propagation consistency as benefited from it.

---

## 5. Ranking-Change Analysis

| Shift Type | Count | Percentage | Description |
|---|:---:|:---:|---|
| **True Root Promoted** | 9 / 60 | 15.0% | True root cause achieved a higher rank in treatment than in control |
| **True Root Demoted** | 27 / 60 | 45.0% | True root cause dropped to a lower rank in treatment |
| **True Root Unchanged** | 24 / 60 | 40.0% | True root cause rank was identical |
| **Winner Changed** | 33 / 60 | 55.0% | The #1 predicted entity changed |
| **Downstream Symptom Demoted** | 27 / 60 | 45.0% | An entity reachable from the true root was pushed down |
| **Unrelated Candidate Promoted** | 27 / 60 | 45.0% | An entity outside the true root's propagation path was promoted |

### Mechanism of Degradation
In microservice architectures, metric anomaly detection sensitivity is heterogeneous:
1. Orchestrators (such as `frontend` or `checkoutservice`) aggregate traffic across multiple dependencies and frequently cross anomaly thresholds (e.g., latency spikes, request retries) **before** the root-cause service's resource metrics (CPU, memory) reach sustained anomaly episode criteria ($K=3, M=2$).
2. High-volume leaf services (such as `productcatalogservice`) exhibit rapid metric shifts due to downstream call volume surges, creating early anomaly onsets ($\tau_{\text{leaf}} < \tau_{\text{root}}$).
3. Orienting static edges by $\tau_u < \tau_v$ inverted the propagation direction, assigning the highest reachability $P(c)$ to symptoms or central transit nodes rather than the true injection point.
4. Unrelated candidates with early baseline fluctuations gained reachability through transitive hub nodes, resulting in 27 cases where unrelated entities were promoted above the true root.

---

## 6. Predeclared Acceptance Criteria and Decision

### Acceptance Criteria
- [ ] At least one primary RCA metric improves under paired clustered analysis. *(Failed: all primary metrics decreased)*
- [x] No primary metric significantly degrades. *(Failed: MRR degraded by -0.1515, Holm-adjusted $p = 0.0104$)*
- [ ] Ranking-change analysis shows improvement is primarily true-root promotion. *(Failed: 27 demotions vs. 9 promotions)*
- [x] No causal leakage (verified: 100% causal cutoffs preserved).
- [x] No evaluation-derived tuning (verified: parameter-free formulation).

### Decision
**FAILURE.**
The observational propagation consistency treatment is rejected. In accordance with project discipline, the result is frozen. No post-hoc tuning or alternative propagation heuristics will be applied to the evaluation population.
