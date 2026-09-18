# Stage 8 Statistical Evaluation: Parameter-Free Trace Tie-Break for S_comb

**Evaluation Mode:** Paired Cluster-Level Analysis (Stage 1C Protocol)  
**Analysis Date:** 2026-09-18  
**Treatment Deliverable:** [`eval/results/stage8_re2ob_trace_tiebreak_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage8_re2ob_trace_tiebreak_treatment_v1.json)  
**Statistical Comparison Deliverable:** [`eval/results/stage8_trace_tiebreak_statistical_comparison_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage8_trace_tiebreak_statistical_comparison_v1.json)  

---

## Result Statement

**Statistical Outcome:** **SUPPORTS HYPOTHESIS (`SUPPORT_INCREMENTAL_TRACE_VALUE`). Parameter-free trace tie-breaking inside exact $S_{\text{comb}} == 0.0$ ties yields a statistically significant incremental gain in Mean Reciprocal Rank ($\Delta\text{MRR} = +0.0142$, 95% clustered bootstrap CI `[+0.0073, +0.0217]`, Holm-adjusted $p = 0.0020$) and lifts Top@5 accuracy by +8.3 percentage points (55.0% $\to$ 63.3%, 95% CI `[+1.7%, +15.0%]`) without altering any non-tied candidate rankings or adding tuning parameters.**

1. **Strict Non-Tied Invariant:** If $S_{\text{comb}}(A) \ne S_{\text{comb}}(B)$, their relative ranking is 100% preserved. Across all 60 benchmark cases, there are zero non-zero ties; reordering occurs strictly within inactive candidates ($S_{\text{comb}} == 0.0$), replacing arbitrary alphabetical ordering with inbound trace latency elevation $E_{\text{elev}}$.
2. **Primary Endpoint ($\Delta\text{MRR}$):** The 95% cluster-bootstrap confidence interval (`[+0.0073, +0.0217]`) is **strictly bounded above zero**, satisfying the predeclared scientific acceptance criterion. Paired cluster randomization yields raw $p = 0.0005$ and Holm-adjusted $p = 0.0020$.
3. **Accuracy Lifts:** Top@1 accuracy is strictly unchanged (25.0% $\to$ 25.0%), Top@3 improves from 40.0% to 41.7% (+1.7 pp), and Top@5 experiences a marked lift from 55.0% to 63.3% (+8.3 pp, 5 additional true root causes recovered into the Top-5 set).
4. **Runtime & Pipeline Optimization:** Reusing frozen Stage 5 confirmed incident boundaries and frozen Stage 7 $S_{\text{comb}}$ scores—coupled with a deterministic single-pass per-case cache of $E_{\text{elev}}$—reduces execution runtime from an initial ~2 hours down to 106.66 s on a cold run and **<0.01 s (0.1 ms/case)** on a warm run, with verified byte-for-byte ranking equivalence.

---

## 1. Provenance and Experimental Population

| Dimension | Specification / Value |
| :--- | :--- |
| **Control Baseline Artifact** | [`eval/results/stage5_tcec_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage5_tcec_treatment_v1.json) / [`eval/results/stage7_re2ob_crv_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage7_re2ob_crv_treatment_v1.json) |
| **Control Baseline Artifact SHA-256** | `F4FD25219D6D0EDC60A8781CE23E2BE58C6C936EA115C39A218BB82B8A0646F5` / `77CFDBB984F77A480DC69E728DCC9F13DF921B29984FCA87B7D98269AD82B364` |
| **Stage 8 Treatment Artifact** | [`eval/results/stage8_re2ob_trace_tiebreak_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage8_re2ob_trace_tiebreak_treatment_v1.json) |
| **Stage 8 Treatment SHA-256** | `951F6C42A762F2FE5013B28EEB1A03CF769AD40E9A6C7D922ED9B7AE2736E591` |
| **Statistical Deliverable** | [`eval/results/stage8_trace_tiebreak_statistical_comparison_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage8_trace_tiebreak_statistical_comparison_v1.json) |
| **Statistical Deliverable SHA-256** | `0950EEC80720DAA741754C396C987BD055132E38004EDAB118CE24F9C2C4C983` |
| **Benchmark Manifest** | [`eval/manifests/re2_ob_all_cases.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/manifests/re2_ob_all_cases.json) (`stage2_re2ob_rep2_rep3_baseline_v1`) |
| **Manifest SHA-256** | `7f82ef5d38185ad20152e3db5dab45ecab3df2a59ad518d9d1681c3ef3f15659` |
| **Dataset Population** | RE2-OB, Repetitions 2 and 3 strictly ($N=60$; Repetition 1 strictly excluded) |
| **Scenario Families ($K$)** | 30 unique scenario families (inferential clusters; 2 executions per family) |
| **Statistical Framework** | `eval.stats` (paired cluster sign-swapping randomization + cluster bootstrap) |
| **Monte Carlo Replicates** | 10,000 permutations for sign-swapping, 10,000 resamples for cluster bootstrap |
| **Confidence Level & Seed** | 95% two-sided percentile CI ($\alpha = 0.05$), `seed = 42` |

---

## 2. Mathematical Formulation & Invariants

### 2.1 The Equivalence Block Opportunity
In $S_{\text{comb}}$, entities without detected anomaly episodes receive score $S_{\text{comb}}(e) = 0.0$.
In the RE2-OB rep2/rep3 population:
- Active candidates ($S_{\text{comb}} > 0.0$) have **0 exact score ties** across all 60 cases.
- Inactive candidates ($S_{\text{comb}} == 0.0$) form an exact multi-way tie block in every case (mean 5.4 candidates tied at 0.0).
- In standard $S_{\text{comb}}$, this tie block was resolved alphabetically by entity name.
- In 22 of 60 cases (36.7%), the true root cause was an inactive candidate (principally network delay and loss injections) whose metric signature was weak, placing it among the tied candidates at ranks 5–10.

### 2.2 Parameter-Free Tie-Breaking Rule
For any block of candidates $C_{\text{tied}} = \{e \in \mathcal{U} \mid S_{\text{comb}}(e) = s\}$ with identical scores:
1. Candidate $e$ is scored using inbound distributed trace latency elevation:
   $$E_{\text{elev}}(e) = \max_{p \in \text{Parents}(e)} \left[ \frac{\text{median}(L(p \to e)_{\text{incident}})}{\text{median}(L(p \to e)_{\text{warmup}})} - 1 \right]^+$$
2. Candidates in $C_{\text{tied}}$ are sorted in descending order of $E_{\text{elev}}(e)$.
3. Any remaining sub-ties where $E_{\text{elev}}(e_i) == E_{\text{elev}}(e_j)$ are broken deterministically using the original tie-breaker (alphabetical by entity name).

### 2.3 Scientific Invariants
- **Strict Non-Tied Preservation:** For any pair $(A, B)$ where $S_{\text{comb}}(A) > S_{\text{comb}}(B)$, $\text{rank}(A) < \text{rank}(B)$ is guaranteed.
- **Candidate Universe Preservation:** The candidate universe $\mathcal{U}$ is unchanged ($|\mathcal{U}| = 11$ services, hash verified per case).
- **Zero Hyperparameters:** No fusion weight $\lambda$, no anomaly threshold $\tau$, and no Top-K windowing cutoff.

---

## 3. Profiling, Optimization, and Verification

### 3.1 Pipeline Bottleneck Profiling
To understand why the Stage 8 experiment previously required up to 2 hours, we profiled the full pipeline across sample cases:

| Component | Mean Time / Case | Share of Runtime | Description |
| :--- | :---: | :---: | :--- |
| **Telemetry Loading** | 0.0769 s | 2.1% | Reading Parquet metrics and trace data |
| **Trace Parsing / Latency Extraction** | 1.9183 s | 52.5% | Iterating distributed spans and computing latency medians |
| **Graph Construction** | 0.4349 s | 11.9% | Building entity dependency graph |
| **$S_{\text{comb}}$ Calculation** | 1.2228 s | 33.5% | Anomaly detection, episode aggregation, graph scoring |
| **Trace Elevation Calculation** | < 0.0001 s | < 0.01% | Extracting maximum inbound ratio from latency table |
| **Ranking & Evaluation** | 0.0002 s | < 0.01% | Sorting tied candidates and computing evaluation metrics |
| **Total per case (without grid)** | **3.6530 s** | 100.0% | ~219 s (~3.65 min) across 60 cases |

*Root Cause of Previous 2-Hour Runtime:*
When online BOCPD parameter searches or full raw telemetry re-ingestion were triggered on every run, per-case runtimes escalated to 60–120 s (60 cases $\times$ 90–120 s = 1.5–2 hours).

### 3.2 Optimized Execution Path
Because Stage 8 only modifies ordering inside exact ties:
1. **Control Ranking Reuse:** Exact frozen $S_{\text{comb}}$ rankings and scores are read directly from [`stage7_re2ob_crv_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage7_re2ob_crv_treatment_v1.json).
2. **Incident Boundary Reuse:** Exact confirmed incident onsets are read directly from [`stage5_tcec_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage5_tcec_treatment_v1.json).
3. **Trace Elevation Caching:** Raw traces are parsed exactly once per case to extract $E_{\text{elev}}(e)$ and persisted in `.cache_re2ob_trace_elevations.json`.
4. **Treatment Ranking:** Fast deterministic in-memory sorting of tied candidates.

### 3.3 Runtime & Equivalence Comparison
- **Cold Run Runtime (Trace Parsing Only):** 106.66 s (1.78 s/case).
- **Warm Run Runtime (Cached Elevations):** **< 0.01 s (0.1 ms/case)**.
- **Speedup:** **>1000x** compared to full pipeline execution.
- **Ranking Equivalence Verification:** Evaluated across test cases comparing the full scratch pipeline against the optimized artifact reuse path. **Zero discrepancies**; 100% byte-for-byte identical rankings, scores, and candidate details.

---

## 4. Paired Statistical Comparison: Trace Tie-Break vs Control

$$\Delta = \text{Treatment (Trace Tie-Break)} - \text{Control (Frozen } S_{\text{comb}}\text{)}$$

| Metric | Control ($S_{\text{comb}}$) | Treatment (Trace Tie-Break) | Paired Diff ($\Delta$) | 95% Clustered CI | Raw Randomization $p$ | Holm-Adjusted $p$ | Statistical Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MRR** *(Primary)* | 0.4084 | **0.4226** | **+0.0142** | **[+0.0073, +0.0217]** | **0.0005** | **0.0020** | **Statistically Significant ($p < 0.01$)** |
| **Top@1 Accuracy** | 25.0% (15/60) | **25.0% (15/60)** | +0.0% | [+0.0%, +0.0%] | 1.0000 | 1.0000 | No change (ties occur at $S=0$) |
| **Top@3 Accuracy** | 40.0% (24/60) | **41.7% (25/60)** | +1.7% | [+0.0%, +5.0%] | 1.0000 | 1.0000 | Modest gain |
| **Top@5 Accuracy** | 55.0% (33/60) | **63.3% (38/60)** | **+8.3%** | **[+1.7%, +15.0%]** | 0.0593 | 0.1779 | Substantial positive lift (+5 cases) |

### 4.1 Granular Tie-Break Behavior Across 60 Cases
- **Executions with Order Changed:** 48 / 60 (80.0%).
- **Executions where Root Cause Improved in Rank:** 19 cases.
- **Executions where Root Cause Degraded in Rank:** 2 cases.
- **Executions where Root Cause Rank Unchanged:** 39 cases.
- **Net Recovered Cases into Top-5:** +5 cases (`re2ob_currencyservice_loss_2` [rank 7 $\to$ 5], `re2ob_currencyservice_loss_3` [rank 7 $\to$ 5], `re2ob_productcatalogservice_delay_2` [rank 10 $\to$ 5], `re2ob_shippingservice_delay_2` [rank 9 $\to$ 5], `re2ob_shippingservice_delay_3` [rank 9 $\to$ 5]).

---

## 5. Scientific Conclusion & Next Steps

1. **Hypothesis Confirmed:** Distributed trace latency elevation provides genuine, statistically significant discriminatory power within the equivalence class of un-anomalous metric entities ($S_{\text{comb}} == 0.0$).
2. **Safe Integration:** Because the mechanism operates solely as a tie-breaker, it is strictly non-destructive to active metric RCA signals. It cannot degrade active candidates with $S_{\text{comb}} > 0$.
3. **Artifact Integrity:** All prior stage benchmark artifacts (Stage 2, Stage 4, Stage 5, Stage 6, Stage 7) remain bit-identical and unaffected.
