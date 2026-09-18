# Stage 7 Experiment: Counterfactual Removal Validation (CRV)

## 1. Scientific Motivation & Hypothesis

Counterfactual Removal Validation (CRV) evaluates whether model-based counterfactual disturbance removal provides incremental diagnostic value when applied to the top candidates produced by the frozen multimodal RCA pipeline ($S_{\text{comb}}$).

### Hypothesis ($H_1$)
> Among the frozen top-5 candidates produced by $S_{\text{comb}}$, candidate-specific counterfactual removal provides incremental discriminatory information that improves RCA ranking quality.

Formally evaluated via:
$$\Delta \text{MRR} = \text{MRR}_{\text{CRV\_shadow}} - \text{MRR}_{S\_\text{comb}}$$

### Predeclared Stopping & Acceptance Criteria
- **Support $H_1$**: The 95% clustered bootstrap confidence interval for $\Delta \text{MRR}$ is **entirely above zero** ($\text{CI}_{\text{lower}} > 0$).
- **Falsify $H_1$**: The 95% clustered bootstrap CI for $\Delta \text{MRR}$ is **entirely below zero** ($\text{CI}_{\text{upper}} < 0$).
- **Inconclusive**: The 95% clustered bootstrap CI **crosses zero** ($\text{CI}_{\text{lower}} \le 0 \le \text{CI}_{\text{upper}}$).

---

## 2. Distinction from Prior Backends

- **PyRCA HT / CIRCA**: Fit regression models on reference data and rank candidates by static residual magnitude $|\hat{\epsilon}_i(t_{\text{incident}})|$.
- **Counterfactual Removal Validation (CRV)**: A **second-stage validator** operating strictly within the top-5 candidates of $S_{\text{comb}}$. It sets candidate disturbance $\epsilon_c^{\text{cf}}(t) = 0$, propagates the intervention forward through the directed entity graph, and measures downstream relief on a predeclared incident symptom target.

---

## 3. Frozen Conditions & Leakage Safeguards

1. **Incident Boundary**: Strictly identical to the frozen Stage 5 TCEC confirmed incident onset ($t_{\text{confirm}}$).
2. **Reference Window**: Structural causal models $X_i(t) = f_i(\text{Pa}_i(t)) + \epsilon_i(t)$ are fitted strictly on telemetry preceding $t_{\text{confirm}}$ ($t < t_{\text{confirm}}$).
3. **Candidate Universe**: Strictly restricted to the top-5 candidates from the frozen $S_{\text{comb}}$ ranking ($C_i = \text{Top5}(S_{\text{comb}})$). Tail candidates (ranks 6..N) preserve their relative ordering.
4. **Target Source**: Predeclared incident symptom target is `confirming_entity` from the Stage 5 TCEC incident confirmation record (fallback to `frontend` gateway if unavailable).
5. **Score Fusion**: Zero score fusion (no $S_{\text{comb}} + \lambda R_c$). The research shadow ranking is ordered strictly by descending relief $R_c$, tie-broken by original $S_{\text{comb}}$ rank.
6. **No Oracle Leakage**: Zero consumption of ground truth `inject_time` or root-cause labels during model fitting, target selection, candidate selection, or propagation.

---

## 4. Mathematical Model & Propagation

### Structural Model
For each entity node $i$ with parents $\text{Pa}_i$:
$$X_i(t) = f_i(\text{Pa}_i(t)) + \epsilon_i(t)$$
where $f_i$ is estimated via regularized linear regression on reference data ($t < t_{\text{confirm}}$).

### Incident Disturbance
$$\hat{\epsilon}_i(t) = X_i(t) - \hat{f}_i(\text{Pa}_i(t)) \quad \text{for } t \in [t_{\text{confirm}}, t_{\text{confirm}}]$$

### Candidate Intervention & Forward Propagation
For candidate $c$:
$$\epsilon_c^{\text{cf}}(t) = 0, \quad \epsilon_j^{\text{cf}}(t) = \hat{\epsilon}_j(t) \text{ for } j \neq c$$
$$X_j^{\text{cf}}(t) = \hat{f}_j(\text{Pa}_j^{\text{cf}}(t)) + \epsilon_j^{\text{cf}}(t)$$
evaluated in topological order.

### Target Relief
For incident target $Y$ with reference statistics $(\mu_Y, \sigma_Y)$:
$$R_c = \text{mean}_t \left[ \frac{|Y(t) - \mu_Y| - |Y_c^{\text{cf}}(t) - \mu_Y|}{\sigma_Y} \right]$$
If candidate $c$ has no directed path to $Y$ in the DAG: $R_c = 0.0$.

---

## 5. Benchmark Population & Execution

- **Suite**: `RE2-OB`
- **Repetitions**: `2` and `3` strictly (60 executions, 30 scenario families, 2 executions per family).
- **Repetition 1**: Excluded completely from all tuning and validation.

---

## 6. Empirical Results

### Descriptive Metrics (N=60)
| Metric | Control ($S_{\text{comb}}$) | Treatment (CRV Shadow) | Difference ($\Delta$) | 95% Clustered CI | Raw $p$ | Holm-Adjusted $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Top@1 Accuracy** | 25.0% (15/60) | 18.3% (11/60) | -6.7 pp | [-18.3%, +3.3%] | 0.4057 | 0.8113 |
| **Top@3 Accuracy** | 40.0% (24/60) | 31.7% (19/60) | -8.3 pp | [-16.7%, +0.0%] | 0.1248 | 0.4992 |
| **Top@5 Accuracy** | 55.0% (33/60) | 55.0% (33/60) | +0.0 pp | [+0.0%, +0.0%] | 1.0000 | 1.0000 |
| **MRR** | **0.4084** | **0.3509** | **-0.0575** | **[-0.1314, +0.0078]** | **0.1278** | **0.4992** |

### Execution Diagnostics
- **Top-5 Order Changed**: 53 / 60 executions (88.3%)
- **Top-1 Candidate Changed**: 43 / 60 executions (71.7%)
- **Top-5 Candidates with No Directed Target Path**: 212 / 300 candidates (70.7%)

---

## 7. Scientific Conclusion & Interpretation

1. **Hypothesis Decision: INCONCLUSIVE**
   - The clustered 95% bootstrap confidence interval for $\Delta \text{MRR}$ is $[-0.1314, +0.0078]$, which crosses zero.
   - Following the predeclared scientific stopping criteria, we report the result as **inconclusive** (no statistically significant incremental improvement).
2. **Diagnostic Assessment**:
   - In microservice call graphs, 70.7% of candidate-to-target pairs lack a forward directed topological path from root fault to observed symptom under trace-derived call graphs (e.g. caller $\to$ callee edge orientations vs symptom propagation in reverse).
   - Point estimates for Top@1 (-6.7 pp) and MRR (-0.0575) show nominal degradation when reranking strictly by downstream linear disturbance relief without bidirectional topology or multimodal evidence.
3. **Epistemic Constraints**:
   - CRV is a model-based counterfactual validation under frozen graph assumptions, not physical experimental intervention.
   - The production RCA ranking remains $S_{\text{comb}}$, which remains unchanged and fully preserved.

---

## 8. Artifact Provenance
- Benchmark Artifact: `eval/results/stage7_re2ob_crv_treatment_v1.json` (SHA-256: `77CFDBB984F77A480DC69E728DCC9F13DF921B29984FCA87B7D98269AD82B364`)
- Statistical Artifact: `eval/results/stage7_crv_statistical_comparison_v1.json` (SHA-256: `04537013F9A43D588B52DCD0F408BB0152FFCAC7C2E76FA60FE5FB27E971EAF7`)
