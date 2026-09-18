# Stage 3 Diagnostic: Oracle vs. Detected Incident Window Headroom Analysis

**Status:** DIAGNOSTIC HEADROOM COMPARISON ONLY  
**Evaluation Label:** `ORACLE — theoretical headroom only`  
**Purpose:** Determine whether premature detected incident window boundaries are the primary bottleneck in Stage 2 RCA performance.  
**Analysis Date:** 2026-09-18  
**Output Artifact:** [`eval/results/stage3_oracle_vs_detected_headroom_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage3_oracle_vs_detected_headroom_v1.json)  

> [!WARNING]
> **DISCLAIMER: ORACLE — THEORETICAL HEADROOM ONLY**  
> The oracle condition uses ground-truth `inject_time` strictly to define the evaluation window. It does not represent an operational or deployable method. It establishes the theoretical performance ceiling achievable by the existing downstream RCA algorithms if incident windowing were perfect.

---

## 1. Provenance and Evaluated Population

| Parameter | Specification |
| :--- | :--- |
| **Control Baseline Artifact** | [`eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json) |
| **Control Artifact SHA-256** | `3C01E2CA28F89AA805A8109152E0B8D992F32330C3E70CDADC19A645C1110157` (verified byte-for-byte unchanged) |
| **Benchmark Manifest** | [`eval/manifests/re2_ob_all_cases.json`](file:///c:/Users/Aayan/Desktop/Project/MidNightCluster/digital-detective/eval/manifests/re2_ob_all_cases.json) (`stage2_re2ob_rep2_rep3_baseline_v1`) |
| **Manifest Hash** | `7f82ef5d38185ad20152e3db5dab45ecab3df2a59ad518d9d1681c3ef3f15659` |
| **Scientific Population** | RE2-OB, Repetitions 2 & 3 strictly (Repetition 1 completely excluded) |
| **Executions / Cases ($N$)** | 60 executions |
| **Scenario Families ($K$)** | 30 scenario families (2 executions per family) |
| **Evaluated Methods** | `s_comb`, `trace_elevation`, `fixed_equal_weight_fusion`, `simple_rca`, `random` |
| **Candidate Universe** | Closed canonical universe (11 services) across all cases and methods |

---

## 2. Window Semantics: Condition A vs. Condition B

| Dimension | Condition A: Detected Window (Frozen Stage 2) | Condition B: Oracle Window (Theoretical Headroom) |
| :--- | :--- | :--- |
| **Window Mode** | `mode="detected"` | `mode="oracle"` |
| **Window Onset ($t_{\text{onset}}$)** | Earliest confirmed entity episode timestamp ($T_{\text{inject}} - 621\,\text{s}$ mean) | Ground-truth fault injection timestamp ($T_{\text{inject}}$) |
| **Window Termination ($t_{\text{end}}$)** | $t_{\text{onset}}$ (strictly causal pre-injection snapshot) | Telemetry end timestamp ($T_{\text{inject}} + \approx 210\,\text{s}$) |
| **Evidence Boundary** | All metric series, trace dependencies, and trace latencies strictly truncated at $t_{\text{onset}}$ | Complete incident telemetry spanning post-injection window |
| **Access to Ground Truth** | Zero access to ground truth | `inject_time` used **ONLY** for window definition; zero access during anomaly detection, scoring, or ranking |

---

## 3. Aggregate Headroom Results

$$\text{Headroom Gain} = \text{Metric}_{\text{Oracle}} - \text{Metric}_{\text{Detected}}$$

| RCA Method | Metric | Condition A (Detected) | Condition B (Oracle Headroom) | Absolute Headroom Gain | Relative Headroom Gain | Family Direction (Pos / Neg / Tie) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$S_{\text{comb}}$** *(Primary)* | **Top@1** | 0.1667 (10/60) | **0.6167** (37/60) | **+0.4500** (+45.0 pp) | +270.0% | 25 pos / 2 neg / 3 tie |
| | **Top@3** | 0.2167 (13/60) | **0.8167** (49/60) | **+0.6000** (+60.0 pp) | +276.9% | 25 pos / 2 neg / 3 tie |
| | **Top@5** | 0.5500 (33/60) | **0.8667** (52/60) | **+0.3167** (+31.7 pp) | +57.6% | 25 pos / 2 neg / 3 tie |
| | **MRR** | 0.3134 | **0.7384** | **+0.4250** | **+135.6%** | **25 pos / 2 neg / 3 tie** |
| **Trace Elevation** | **Top@1** | 0.2000 (12/60) | **0.6500** (39/60) | **+0.4500** (+45.0 pp) | +225.0% | 19 pos / 2 neg / 9 tie |
| | **Top@3** | 0.5667 (34/60) | **0.9167** (55/60) | **+0.3500** (+35.0 pp) | +61.8% | 19 pos / 2 neg / 9 tie |
| | **Top@5** | 0.8000 (48/60) | **1.0000** (60/60) | **+0.2000** (+20.0 pp) | +25.0% | 19 pos / 2 neg / 9 tie |
| | **MRR** | 0.4406 | **0.7903** | **+0.3497** | **+79.4%** | **19 pos / 2 neg / 9 tie** |
| **Fixed Fusion** | **Top@1** | 0.1500 (9/60) | **0.8167** (49/60) | **+0.6667** (+66.7 pp) | +444.4% | 27 pos / 2 neg / 1 tie |
| | **Top@3** | 0.5000 (30/60) | **0.9667** (58/60) | **+0.4667** (+46.7 pp) | +93.3% | 27 pos / 2 neg / 1 tie |
| | **Top@5** | 0.8000 (48/60) | **1.0000** (60/60) | **+0.2000** (+20.0 pp) | +25.0% | 27 pos / 2 neg / 1 tie |
| | **MRR** | 0.4029 | **0.8936** | **+0.4908** | **+121.8%** | **27 pos / 2 neg / 1 tie** |
| **Simple RCA** | **Top@1** | 0.0000 (0/60) | **0.2167** (13/60) | **+0.2167** (+21.7 pp) | N/A | 13 pos / 10 neg / 7 tie |
| | **Top@3** | 0.2000 (12/60) | **0.4833** (29/60) | **+0.2833** (+28.3 pp) | +141.7% | 13 pos / 10 neg / 7 tie |
| | **Top@5** | 0.6000 (36/60) | **0.7833** (47/60) | **+0.1833** (+18.3 pp) | +30.6% | 13 pos / 10 neg / 7 tie |
| | **MRR** | 0.2039 | **0.4215** | **+0.2176** | **+106.7%** | **13 pos / 10 neg / 7 tie** |
| **Random** | **Top@1** | 0.0667 (4/60) | **0.0667** (4/60) | **+0.0000** | +0.0% | 0 pos / 0 neg / 30 tie |
| | **Top@3** | 0.1333 (8/60) | **0.1333** (8/60) | **+0.0000** | +0.0% | 0 pos / 0 neg / 30 tie |
| | **Top@5** | 0.3167 (19/60) | **0.3167** (19/60) | **+0.0000** | +0.0% | 0 pos / 0 neg / 30 tie |
| | **MRR** | 0.2222 | **0.2222** | **+0.0000** | +0.0% | 0 pos / 0 neg / 30 tie |

---

## 4. Scenario-Family Robustness

### $S_{\text{comb}}$ MRR Direction:
- **Positive Families ($\text{Oracle} > \text{Detected}$):** **25 families** (83.3%)
- **Negative Families ($\text{Oracle} < \text{Detected}$):** **2 families** (6.7%)
- **Tied Families ($\text{Oracle} == \text{Detected}$):** **3 families** (10.0%)
- **Family Effect Range:** Min = $-0.4190$, Median = $+0.5694$, Max = $+0.8889$

### Fixed Equal-Weight Fusion MRR Direction:
- **Positive Families ($\text{Oracle} > \text{Detected}$):** **27 families** (90.0%)
- **Negative Families ($\text{Oracle} < \text{Detected}$):** **2 families** (6.7%)
- **Tied Families ($\text{Oracle} == \text{Detected}$):** **1 family** (3.3%)
- **Family Effect Range:** Min = $-0.0986$, Median = $+0.5619$, Max = $+0.8889$

---

## 5. Diagnostic Interpretation

The experimental diagnostic yields an unambiguous, decisive finding:

### Decision Pathway: **LARGE ORACLE IMPROVEMENT**

1. **Downstream RCA Algorithms Function Exceptionally Well:**
   When given telemetry encompassing the actual incident (Condition B), the existing deterministic RCA algorithms achieve outstanding localization accuracy on RE2-OB:
   - Fixed Equal-Weight Fusion achieves **81.7% Top@1**, **96.7% Top@3**, **100.0% Top@5**, and **0.8936 MRR**.
   - $S_{\text{comb}}$ achieves **61.7% Top@1**, **81.7% Top@3**, and **0.7384 MRR**.
   - Trace Elevation achieves **65.0% Top@1**, **91.7% Top@3**, and **0.7903 MRR**.
2. **The Incident Window is the Primary Bottleneck:**
   In Condition A (detected mode), premature detector triggering on normal background fluctuations ($\approx 621\,\text{s}$ before injection) causes causal truncation to discard the entire incident period. As a result, the downstream rankers are evaluated on non-incident background noise, causing $S_{\text{comb}}$ Top@1 to collapse from $61.7\%$ down to $16.7\%$, and Fusion Top@1 to collapse from $81.7\%$ down to $15.0\%$.
3. **Research Implication:**
   Downstream scoring, graph construction, and attribution logic are **not** the limiting factor in Digital Detective's performance. The system's performance ceiling is constrained almost entirely by **incident window resolution and detector false-alarm resistance**.
   Subsequent research must focus on incident onset estimation, window anchoring, or false-alarm suppression rather than fine-tuning downstream RCA rankers.
