# Stage 2 Statistical Comparison: Causal Median/MAD vs. Frozen Mean/Std Baseline

**Evaluation Mode:** Paired Cluster-Level Analysis (Stage 1C Protocol)  
**Analysis Date:** 2026-09-18  
**Analysis Output Artifact:** [`eval/results/stage2_re2ob_mad_vs_meanstd_statistical_comparison_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage2_re2ob_mad_vs_meanstd_statistical_comparison_v1.json)  

---

## 1. Provenance and Experimental Population

| Dimension | Specification / Value |
| :--- | :--- |
| **Control Baseline Artifact** | [`eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json) |
| **Control Artifact SHA-256** | `3C01E2CA28F89AA805A8109152E0B8D992F32330C3E70CDADC19A645C1110157` |
| **Control Git Commit** | `72739dcd95bf8adbefe8ceeab71989f405b120b3` |
| **Treatment Artifact** | [`eval/results/stage2_re2ob_detected_mad_rep2_rep3_treatment_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage2_re2ob_detected_mad_rep2_rep3_treatment_v1.json) |
| **Treatment Artifact SHA-256** | `052DDF40635148C2530EADBB3B5B69620DF5C83B412A0A546B0372F6F7EF19DF` |
| **Treatment Git Commit** | `f98f835f42de31bfe8074e756e08c2ea0bacc22a` (frozen in `ac63237a9d4ac5894442607f8ec4f08a135c06ae`) |
| **Benchmark Manifest** | [`eval/manifests/re2_ob_all_cases.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/manifests/re2_ob_all_cases.json) (`manifest_id: stage2_re2ob_rep2_rep3_baseline_v1`) |
| **Manifest Hash** | `7f82ef5d38185ad20152e3db5dab45ecab3df2a59ad518d9d1681c3ef3f15659` |
| **Dataset Population** | RE2-OB, Repetitions 2 and 3 strictly (Repetition 1 completely excluded) |
| **Executions / Cases ($N$)** | 60 executions (measurement observations) |
| **Scenario Families ($K$)** | 30 unique scenario families (inferential clusters; 2 executions per family) |
| **Statistical Framework** | `eval.stats` (paired cluster sign-swapping randomization + cluster bootstrap) |
| **Randomization Replicates** | 10,000 Monte Carlo sign-swaps over whole scenario families |
| **Bootstrap Replicates** | 10,000 cluster resamples with replacement |
| **Confidence Level & Seed** | 95% two-sided percentile CI ($\alpha = 0.05$), `seed = 42` |

---

## 2. Confirmatory Primary Outcome: Downstream RCA MRR

The primary efficacy metric specified by the Stage 2 evaluation protocol is the **Mean Reciprocal Rank (MRR)** of Digital Detective's principal metric-based RCA algorithm (`s_comb`).

$$\Delta_{\text{MRR}} = \text{MAD MRR} - \text{Control Mean/Std MRR}$$

| Metric | Control (Mean/Std) | Treatment (MAD) | Paired Difference ($\Delta$) | Relative Difference | 95% Cluster-Bootstrap CI | Paired Randomization $p$-value | Holm-Adjusted $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_{\text{comb}}$ MRR** | **0.3134** | **0.2788** | **-0.0346** (-3.46 pp) | -11.04% | **[-0.1486, +0.0708]** | **0.5505** | **1.0000** |

- **Inference:** The observed paired difference in $S_{\text{comb}}$ MRR is **negative** ($\Delta = -0.0346$). The 95% cluster-bootstrap confidence interval spans zero ($[-0.1486, +0.0708]$), and the paired cluster-level randomization test indicates no statistically significant difference ($p = 0.5505$).

---

## 3. Primary Safety Outcome: False-Early Detection Rate

A detector variant must not trigger prematurely on non-incident telemetry. In detected window mode, an episode onset timestamp earlier than ground-truth fault injection ($t_{\text{onset}} < T_{\text{inject}}$) is classified as a **false-early detection**.

$$\Delta_{\text{FE}} = \text{MAD False-Early Rate} - \text{Control False-Early Rate}$$

| Safety Dimension | Control (Mean/Std) | Treatment (MAD) | Paired Difference | Decision Threshold | Threshold Satisfied |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **False-Early Count** | 60 / 60 | 60 / 60 | 0 cases | N/A | N/A |
| **False-Early Rate** | **100.0%** | **100.0%** | **0.00 pp** | $\le +2.0\,\text{pp}$ increase | **YES** |
| **Discordance ($N_{10}, N_{01}$)** | $N_{10} = 0$, $N_{01} = 0$ | $N_{11} = 60$, $N_{00} = 0$ | Zero discordance | No discordant pairs | **YES** |

- **Note on Binary Invariance:** Because both arms trigger false-early on 100% of cases in this dataset, there is zero variance in the paired binary outcome ($\Delta = 0.0\,\text{pp}$). Per the evaluation protocol, no inferential test is forced onto this invariant outcome.

---

## 4. Secondary Detection Outcomes

### Detection Delay
Detection delay measures the temporal offset between confirmed incident onset and ground-truth fault injection: $\text{delay} = t_{\text{onset}} - T_{\text{inject}}$. Negative values indicate detection prior to injection.

$$\Delta_{\text{delay}} = \text{MAD Delay} - \text{Control Delay}$$

| Arm | Mean Delay | Median Delay | Min Delay | Max Delay | Detection Rate | No-Detection Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control (Mean/Std)** | -621.83 s | -626.50 s | -658 s | -491 s | 100.0% (60/60) | 0.0% (0/60) |
| **Treatment (MAD)** | -657.10 s | -658.00 s | -658 s | -644 s | 100.0% (60/60) | 0.0% (0/60) |
| **Paired Difference ($\Delta$)** | **-35.27 s** | **-31.50 s** | N/A | N/A | **0.00 pp** | **0.00 pp** |

- **Statistical Significance of Delay Difference:**
  - Mean Paired Difference: **-35.27 s** (MAD triggers earlier)
  - 95% Cluster-Bootstrap CI: **[-44.37 s, -27.40 s]**
  - Paired Cluster Randomization $p$-value: **0.0001** ($p < 0.001$)
- **Interpretation:** MAD triggers significantly earlier relative to fault injection than Mean/Std (mean shift of $-35.27\,\text{s}$). As established in the audit, this shift is driven by MAD's narrower dispersion scale locking onto routine background fluctuations as soon as warmup concludes ($k = 62$, delay $-658\,\text{s}$).

---

## 5. Secondary RCA Outcomes Across Evaluated Methods

Paired statistical comparisons across all 5 evaluated RCA methods on RE2-OB:

### Compact Results Table

| Method | Metric | Control Mean | Treatment Mean | Paired Diff ($\Delta$) | 95% Cluster CI | $N_{\text{trt}> \text{ctrl}}$ ($N_{10}$) | $N_{\text{ctrl}> \text{trt}}$ ($N_{01}$)| Cluster Rand. $p$ | Exploratory Holm $p$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S_{\text{comb}}$** *(Primary)* | **Top@1** | 0.1667 | 0.1000 | -0.0667 | [-0.2167, +0.0667] | 6 | 10 | 0.4913 | 1.0000 |
| | **Top@3** | 0.2167 | 0.2167 | 0.0000 | [-0.1667, +0.1667] | 10 | 10 | 1.0000 | 1.0000 |
| | **Top@5** *(Safeguard)*| 0.5500 | 0.5333 | -0.0167 | [-0.1333, +0.1000] | 6 | 7 | 1.0000 | 1.0000 |
| | **MRR** *(Primary)* | 0.3134 | 0.2788 | -0.0346 | [-0.1486, +0.0708] | N/A | N/A | 0.5505 | 1.0000 |
| **Trace Elevation** | **Top@1** | 0.2000 | 0.2000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **Top@3** | 0.5667 | 0.5833 | +0.0167 | [0.0000, +0.0500] | 1 | 0 | 1.0000 | 1.0000 |
| | **Top@5** | 0.8000 | 0.8167 | +0.0167 | [-0.0500, +0.0833] | 3 | 2 | 1.0000 | 1.0000 |
| | **MRR** | 0.4406 | 0.4425 | +0.0019 | [-0.0028, +0.0069] | N/A | N/A | 0.6247 | 1.0000 |
| **Fixed Fusion** | **Top@1** | 0.1500 | 0.1667 | +0.0167 | [-0.1167, +0.1500] | 9 | 8 | 1.0000 | 1.0000 |
| | **Top@3** | 0.5000 | 0.4833 | -0.0167 | [-0.1333, +0.1000] | 6 | 7 | 1.0000 | 1.0000 |
| | **Top@5** | 0.8000 | 0.7333 | -0.0667 | [-0.2000, +0.0504] | 7 | 11 | 0.4680 | 1.0000 |
| | **MRR** | 0.4029 | 0.4029 | +0.00002| [-0.0986, +0.0924] | N/A | N/A | 1.0000 | 1.0000 |
| **Simple RCA** | **Top@1** | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **Top@3** | 0.2000 | 0.2000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **Top@5** | 0.6000 | 0.6000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **MRR** | 0.2039 | 0.2039 | 0.0000 | [0.0000, 0.0000] | N/A | N/A | 1.0000 | 1.0000 |
| **Random** | **Top@1** | 0.0667 | 0.0667 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **Top@3** | 0.1333 | 0.1333 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **Top@5** | 0.3167 | 0.3167 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | 1.0000 | 1.0000 |
| | **MRR** | 0.2222 | 0.2222 | 0.0000 | [0.0000, 0.0000] | N/A | N/A | 1.0000 | 1.0000 |

---

## 6. Multiple Comparisons and Confirmatory Logic

Under the Stage 2 confirmatory framework:
1. **Primary Efficacy Test:** $S_{\text{comb}}$ MRR improvement ($\text{raw } p = 0.5505 \rightarrow \text{Holm-adjusted } p = 1.0000$).
2. **Safeguard Test:** $S_{\text{comb}}$ Top@5 non-degradation ($\text{raw } p = 1.0000 \rightarrow \text{Holm-adjusted } p = 1.0000$).

Neither test achieves statistical significance. There is no evidence of efficacy improvement on the primary RCA outcome, nor is there statistically significant degradation on the Top@5 safeguard.

---

## 7. Family-Level Robustness

Across the 30 scenario-family clusters (evaluating paired differences in $S_{\text{comb}}$ MRR):

- **Positive Family Effect ($\text{MAD} > \text{Control}$):** **10 families** (33.3%)
- **Negative Family Effect ($\text{MAD} < \text{Control}$):** **14 families** (46.7%)
- **Tied Family Effect ($\text{MAD} == \text{Control}$):** **6 families** (20.0%)
- **Distribution Range:** $\min = -0.8167$ (`emailservice:mem`), $\text{median} = -0.0063$, $\max = +0.4833$ (`emailservice:disk`).

The family-level direction of effect is net-negative (14 negative vs. 10 positive), confirming that the lack of improvement is consistent across clusters rather than an artifact of one or two outlier families.

---

## 8. Mechanical Acceptance / Falsification Decision

Evaluating the six predeclared project criteria:

| # | Project Decision Criterion | Predeclared Threshold | Observed Value | Status |
|---|:---|:---:|:---:|:---:|
| 1 | Primary paired cluster-level MRR difference is positive | $\Delta_{\text{MRR}} > 0$ | $-0.0346$ | **FAIL (Falsified)** |
| 2 | 95% cluster-bootstrap CI entirely above zero | $\text{CI}_{\text{lower}} > 0$ | $[-0.1486, +0.0708]$ | **FAIL (Falsified)** |
| 3 | Absolute MRR improvement is at least +0.05 | $\Delta_{\text{MRR}} \ge +0.05$ | $-0.0346$ | **FAIL (Falsified)** |
| 4 | False-early rate increase is no more than +2 pp | $\Delta_{\text{FE}} \le +2.0\,\text{pp}$ | $+0.00\,\text{pp}$ | **PASS** |
| 5 | No Holm-adjusted significant degradation in Top@5 | Not ($\Delta < 0$ and $p_{\text{adj}} < 0.05$) | $\Delta = -0.0167, p_{\text{adj}} = 1.0$ | **PASS** |
| 6 | Improvement is not confined to a tiny subset of families | Positive families > Negative families | 10 positive vs 14 negative | **FAIL (Falsified)** |

### Overall Decision: **FALSIFIED / TREATMENT NOT RETAINED**
The causal rolling median/MAD detector treatment fails Criteria 1, 2, 3, and 6. The treatment is not retained as a candidate replacement for the baseline detector.

---

## 9. Frozen Regression Protection

- **Repetition 1 Smoke Artifact:** Verified untouched. [`configs/eval_smoke_re2_ob.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/configs/eval_smoke_re2_ob.json) remains clean and unmodified from its frozen commit (`70922bf`).
- **No Repetition 1 Execution:** Repetition 1 was not re-executed or used in this evaluation.

---

## 10. Distinctions

- **Measured Facts:**
  - Both arms report 100% false-early rate (60/60 cases) on RE2-OB.
  - MAD detects earlier by a mean paired difference of $-35.27\,\text{s}$ ($p = 0.0001$).
  - MAD $S_{\text{comb}}$ MRR is $0.2788$ vs. Control $0.3134$ ($\Delta = -0.0346$).
- **Statistical Results:**
  - The paired cluster-level randomization test on primary MRR yields $p = 0.5505$ (Holm-adjusted $p = 1.0000$).
  - The 95% cluster-bootstrap confidence interval is $[-0.1486, +0.0708]$.
  - The difference in RCA efficacy between MAD and Mean/Std is not statistically significant.
- **Project Decision Criteria:**
  - The project mandates a minimum $+0.05$ absolute MRR improvement with CI strictly above zero.
  - Because the observed difference is negative ($-0.0346$), the treatment is falsified and rejected.
