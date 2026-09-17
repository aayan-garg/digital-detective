# Anomaly Detection Evaluation Methodology

This document specifies the evaluation methodology for assessing the performance of Digital Detective's anomaly detection baseline on benchmark telemetry (primarily the RCAEval Parquet distribution). It is a design document only; it does not implement evaluation scripts or alter the detector implementation.

---

## 1. Scope and Objective

### Evaluation Question
The central research question answered by this evaluation methodology is:
> **"Given the detector's anomaly outputs, how reliably and how quickly does it detect the injected fault period, without producing premature alarms during normal operation?"**

### Separation of Concerns
In accordance with `AGENTS.md` and `docs/PROJECT_STATE.md`:
- **The detector is blind to ground truth:** The anomaly detector (`src/digital_detective/anomaly.py`) processes metric time-series chronologically and has no access to `inject_time`, `root_cause_service`, or `fault`.
- **The evaluator consumes evidence and ground truth:** The evaluator receives the immutable `MetricAnomalyResult` and compares its timestamped anomaly outputs against `case.ground_truth.values["inject_time"]`.
- **Explicit separation between metric-level and case-level results:**
  - *Metric-level results* are the direct output of the detector: individual metric anomaly scores ($|z_t|$), signed scores ($z_t$), evaluation statuses (`"warmup"`, `"missing_observation"`, `"insufficient_history"`, `"normal"`, `"anomaly"`), and valid history counts.
  - *Case-level results* are the outputs of the evaluator: applying an aggregation rule across metric alarms to determine case detection timestamps, latency, false alarm rates, and case outcomes.
- **No service attribution at this stage:** Metric-to-service attribution and root-cause localization belong to subsequent dependency-graph and causal RCA milestones. This evaluation focuses strictly on **detection fidelity, false alarm rates, and detection latency**.

---

## 2. Temporal Windows and Exact Definitions

```text
Time ─────────────────────────────────────────────────────────────────────────────►
[      Warmup Window      )[    Pre-Injection Normal    )[      Post-Injection Evaluation Window      ]
t_start                   t_warmup                      inject_time                                    t_eval_end
(Detector initializing)   (FAR evaluated here)          (Fault active; Latency evaluated here)
```

For any evaluated case, the timeline is partitioned into three mutually exclusive intervals:
1. **Warmup Window $[t_{start}, t_{warmup})$:**
   - Indices $0 \le i < W_{warmup}$.
   - The detector accumulates initial rolling history. All observations receive `status = "warmup"` and `anomaly_score = None`.
   - **Excluded from all evaluation metrics and denominators.**
2. **Pre-Injection Normal Window $[t_{warmup}, \text{inject\_time})$:**
   - The system operates under normal, un-faulted conditions.
   - Evaluates false alarms and normal-state stability.
3. **Post-Injection Evaluation Window $[\text{inject\_time}, t_{eval\_end}]$:**
   - The fault has been injected and is active or propagating.
   - **Determination of $t_{eval\_end}$:** Verified from the RCAEval distribution, each case index record specifies `time_end`, and `metrics.parquet` records timesteps through the end of the case. By default, $t_{eval\_end}$ is the timestamp of the final observation in the case telemetry:
     $$t_{eval\_end} = \text{timestamps}[-1]$$
     No synthetic or arbitrary interval is assumed.

### Exact Event Definitions

- **Point-wise Pre-Injection False Alarm:**
  An observation $t \in [t_{warmup}, \text{inject\_time})$ where at least one metric has `evaluation_status == "anomaly"`.

- **Pre-Injection False Alarm Rate (FAR):**
  The proportion of validly evaluated pre-injection observations containing at least one anomaly:
  $$\text{FAR}_{\text{case}} = \frac{\sum_{t \in [t_{warmup}, \text{inject\_time})} \mathbf{1}(\exists m : \text{status}_{m, t} = \text{"anomaly"})}{N_{\text{evaluated\_pre\_injection}}}$$
  where $N_{\text{evaluated\_pre\_injection}}$ is the count of pre-injection timesteps where at least one metric was validly evaluated (`status in ("normal", "anomaly")`).

- **Case-Level Early Alarm:**
  A case where the earliest detected anomaly across all metrics occurs strictly before the fault was injected:
  $$t_{first\_alarm} < \text{inject\_time} \quad (t_{first\_alarm} \ge t_{warmup})$$
  In real production, an early alarm represents a premature operational alert fired during normal conditions.

- **Post-Injection Detection Timestamp ($t_{detect}$):**
  The earliest timestamp at or after fault injection where a case-level anomaly alert is triggered according to the baseline aggregation rule:
  $$t_{detect} = \min \{ t \ge \text{inject\_time} \mid \text{case\_anomaly}(t) = \text{True} \}$$
  If no alert triggers in $[\text{inject\_time}, t_{eval\_end}]$, $t_{detect} = \text{None}$.

- **Detection Latency ($\Delta t_{detect}$):**
  Detection latency is **strictly defined only for detections occurring at or after fault injection**:
  $$\Delta t_{detect} = t_{detect} - \text{inject\_time} \quad (\text{defined only when } t_{detect} \ge \text{inject\_time})$$
  - Units: seconds (integer Unix-seconds difference).
  - If $t_{detect} = \text{inject\_time}$, $\Delta t_{detect} = 0\,\text{s}$ (immediate detection).
  - If $t_{detect} > \text{inject\_time}$, $\Delta t_{detect} > 0\,\text{s}$ (positive detection delay).
  - If $t_{detect} = \text{None}$, $\Delta t_{detect} = \text{None}$ (undefined; recorded as missed detection).
  - Pre-injection alarms do **not** define a negative latency. They are classified as early alarms.

- **Missed Detection (False Negative):**
  A case where $t_{detect} = \text{None}$ (no metric triggered an anomaly anywhere in $[\text{inject\_time}, t_{eval\_end}]$).

- **Mutually Exclusive Case-Level Outcomes:**
  Each evaluated case is classified into exactly one of four distinct outcomes:
  1. **Clean Detection:** Detected post-injection ($t_{detect} \ge \text{inject\_time}$) **with zero pre-injection false alarms** ($\text{FAR} = 0$).
  2. **Noisy Detection:** Detected post-injection ($t_{detect} \ge \text{inject\_time}$), but **pre-injection false alarms occurred** ($\text{FAR} > 0$).
  3. **Early Alarm Only:** Pre-injection false alarm occurred, but **no anomaly was detected post-injection** ($t_{detect} = \text{None}, \Delta t_{detect} = \text{None}$).
  4. **Complete Miss:** Neither pre-injection alarms nor post-injection detections occurred ($t_{detect} = \text{None}, \Delta t_{detect} = \text{None}, \text{FAR} = 0$).

---

## 3. Warmup and Missing Observation Semantics in Denominators

### Warmup Period Impact
- Warmup observations ($t < t_{warmup}$) are strictly excluded from both the numerator and denominator of FAR.
- **Warmup Validity Pre-Condition:** The evaluation protocol requires $t_{warmup} < \text{inject\_time}$. If a case has $\text{inject\_time} \le t_{warmup}$, the pre-injection baseline is truncated, and the case must be flagged as `"invalid_warmup_window"` rather than scored. In validated RCAEval cases, hundreds of normal observations precede $\text{inject\_time}$ (e.g., 720 seconds in RE2-OB), satisfying this condition.

### Handling Missing / Non-Numeric Observations in Denominators
As implemented in `MetricAnomalyResult`, each observation per metric has an explicit `evaluation_status`:
`"warmup"`, `"missing_observation"`, `"insufficient_history"`, `"normal"`, or `"anomaly"`.

- **Denominator integrity:** Observations marked `"missing_observation"` (current observation is NaN/null) or `"insufficient_history"` (preceding window had $< W_{min\_valid}$ valid samples) were not evaluated. They must **not** be counted in the denominator as normal, and must **not** be counted in the numerator as anomalous.
- **Metric-level denominator:** For metric $m$, the pre-injection denominator is:
  $$N_{\text{eval}, m} = \sum_{t \in [t_{warmup}, \text{inject\_time})} \mathbf{1}(\text{status}_{m, t} \in \{\text{"normal"}, \text{"anomaly"}\})$$
- **Case-level denominator:** A timestep $t$ enters $N_{\text{evaluated\_pre\_injection}}$ if at least one metric was evaluated (`status in ("normal", "anomaly")`). If all metrics in a case are missing or insufficient at step $t$, that step is excluded from the denominator.

---

## 4. Case-Level Anomaly Aggregation Rules

### Primary Baseline Rule: Any-Metric OR-Gate
$$\text{case\_anomaly}_{\text{OR}}(t) = \bigvee_{m} (\text{status}_{m, t} == \text{"anomaly"})$$
- **Specification:** Triggers a case alert immediately when any metric flags an anomaly at timestep $t$.
- **Role:** This is the pre-specified, frozen primary baseline aggregation rule for the benchmark. It provides a simple, reproducible lower bound on detection latency.

### Secondary Comparison Experiments (Later Milestones)
Subsequent research experiments will compare this baseline against structured aggregation filters:
1. *Temporal Persistence ($k$-of-$N$ window):* Requiring $k$ consecutive alarms on the same metric (e.g., $k = 3$) to filter isolated sensor spikes.
2. *Multi-Metric Consensus ($M$-metric coincidence):* Requiring $\ge M$ simultaneous metric alarms (e.g., $M \ge 2$) to reflect microservice cascade behavior.

---

## 5. Configuration and Hyperparameter Protocol

### Pre-Specified Primary Baseline Configuration
The primary baseline is pre-specified and frozen before evaluating the benchmark:
- Observation window size: $W = 60$
- Anomaly threshold: $\tau = 3.0$ standard deviations
- Minimum valid history: $W_{min\_valid} = 30$
- Numerical epsilon: $\epsilon = 10^{-6}$

This baseline configuration is evaluated directly on all cases without tuning.

### Optional Calibration Experiment (Secondary Study)
If hyperparameter calibration is studied as an experimental comparison:
- **Partitioning:** Repetition 1 cases from RCAEval form the development/calibration set. Repetitions 2 and 3 remain the held-out test set.
- **Formal Objective Function:** Parameters $(W, \tau)$ must be selected by optimizing an explicit objective on the calibration set, never by informal inspection. The pre-specified objective is to maximize Clean Detection Rate subject to a bounded False Alarm Rate:
  $$\max_{W \in \{30, 60, 120\}, \, \tau \in [2.0, 4.0]} \text{Recall}_{\text{clean}}(W, \tau) \quad \text{subject to} \quad \overline{\text{FAR}}(W, \tau) \le 0.05$$
- The resulting tuned configuration $(W^*, \tau^*)$ is then evaluated once on the unseen test repetitions to measure generalizability.

---

## 6. Staged Evaluation Plan

Evaluation proceeds in three distinct, reproducible stages:

### Stage 1: Sanity Verification (2 Cases)
- **Cases:** The two already-validated cases:
  1. `re1ob_adservice_cpu_1` (RE1-OB, metric-only)
  2. `re2ob_checkoutservice_cpu_1` (RE2-OB, multi-source)
- **Purpose:** Verify that the evaluator computes $\Delta t_{detect}$, $\text{FAR}$, and classification states without errors.

### Stage 2: Controlled Validation Subset (15 Deterministic Cases)
To prevent selection bias, the 15-case validation subset is selected deterministically from `cases.parquet` using the following exact query:
```text
Filter criteria:
- system == "ob" (Online Boutique)
- repetition == 1
- suite in ("RE1", "RE2")
Ordering:
- Sort alphabetically by case identifier ascending.
Selection:
- Select the first 15 matching cases.
```
This query yields a fixed, reproducible set covering varied services and fault types across RE1 and RE2 without manual post-hoc filtering.

### Stage 3: Full Benchmark Evaluation (All 735 Cases)
- **Scope:** Complete RCAEval benchmark across all 3 suites (RE1, RE2, RE3) and all 3 microservice systems (Online Boutique, Sock Shop, Train Ticket).
- **Purpose:** Publish full benchmark tables, fault breakdowns, and latency distributions.

---

## 7. Exact Reporting Quantities

The future evaluator must report the following quantities:

### Per-Case Reporting Quantities
| Field | Type | Description |
| :--- | :--- | :--- |
| `case_id` | string | Unique case identifier |
| `suite` | string | Benchmark suite (`RE1`, `RE2`, `RE3`) |
| `system` | string | Microservice system (`ob`, `ss`, `tt`) |
| `fault_type` | string | Injected fault type (`cpu`, `mem`, etc.) |
| `inject_time` | integer | Unix timestamp of fault injection |
| `t_eval_end` | integer | Final observation timestamp of the case |
| `outcome` | enum | `Clean_Detection`, `Noisy_Detection`, `Early_Alarm_Only`, or `Complete_Miss` |
| `first_pre_alarm_ts` | int \| None | Earliest alarm in $[t_{warmup}, \text{inject\_time})$, or `None` |
| `t_detect` | int \| None | Earliest alarm in $[\text{inject\_time}, t_{eval\_end}]$, or `None` |
| `detection_latency_sec`| int \| None | $t_{detect} - \text{inject\_time}$ (defined only if $t_{detect} \ge \text{inject\_time}$) |
| `pre_injection_far` | float | Fraction of evaluated pre-injection steps with an alarm |
| `n_eval_pre_steps` | integer | Count of evaluated pre-injection timesteps (denominator) |
| `n_missing_pre_steps` | integer | Count of un-evaluated pre-injection timesteps |
| `post_injection_density`| float | Fraction of evaluated post-injection steps with active alarms |
| `top_anomalous_metrics`| list[str] | Up to 5 metrics with the highest peak $|z_t|$ scores |

### Aggregate Benchmark Reporting Quantities
| Metric | Type | Calculation / Formula |
| :--- | :--- | :--- |
| `total_cases` | integer | Total number of cases evaluated ($C$) |
| `overall_detection_rate` | float | $(C_{\text{Clean}} + C_{\text{Noisy}}) / C$ |
| `clean_detection_rate` | float | $C_{\text{Clean}} / C$ |
| `early_alarm_rate` | float | $(C_{\text{Noisy}} + C_{\text{Early\_Only}}) / C$ |
| `missed_detection_rate` | float | $C_{\text{Complete\_Miss}} / C$ |
| `median_latency_sec` | float | Median $\Delta t_{detect}$ over all detected cases ($t_{detect} \ge \text{inject\_time}$) |
| `p90_latency_sec` | float | 90th percentile $\Delta t_{detect}$ over detected cases |
| `mean_latency_sec` | float | Mean $\Delta t_{detect} \pm \text{std}$ over detected cases |
| `mean_pre_injection_far`| float | Mean $\text{FAR}_{\text{case}}$ across all evaluated cases |

All aggregate quantities are reported overall and broken down by:
1. Suite (`RE1`, `RE2`, `RE3`)
2. System (`Online Boutique`, `Sock Shop`, `Train Ticket`)
3. Fault category (`CPU`, `Memory`, `Disk`, `Delay`, `Loss`, `Socket`, `Code`)

---

## 8. Planned Research Visualizations

For publication and milestone reports, the evaluation pipeline will support:
1. **Case Anomaly Timeline:** Time-series plot of metric scores $|z_t|$ relative to threshold $\tau$, with vertical marker at $\text{inject\_time}$ and evaluation-status color ribbons.
2. **Detection Latency Distribution:** Boxplot and CDF of $\Delta t_{detect}$ across fault types.
3. **FAR vs. Recall Operating Curve:** Sweeping threshold $\tau$ to demonstrate the empirical trade-off between false alarms and detection sensitivity.

---

## 9. Verification Classification

In accordance with `AGENTS.md`:

### Verified Facts
- In RCAEval Parquet distribution, `cases.parquet` contains `case`, `dataset`, `suite`, `system`, `root_cause_service`, `fault`, `inject_time`, and `time_end`.
- In `re1ob_adservice_cpu_1` and `re2ob_checkoutservice_cpu_1`, hundreds of normal observations precede `inject_time`.
- In `metrics.parquet`, `time` is an `int64` column recording timestamps through the end of the case.
- `MetricAnomalyResult` provides explicit per-observation `evaluation_statuses` (`"warmup"`, `"missing_observation"`, `"insufficient_history"`, `"normal"`, `"anomaly"`).

### Assumptions
- $t_{eval\_end} = \text{timestamps}[-1]$ represents the natural terminal boundary for each case's telemetry.
- An initial sample window of $W = 60$ observations provides an adequate statistical baseline for sample mean and variance.

### Open Questions
- Whether any specific benchmark cases in Train Ticket or Sock Shop have pre-injection intervals shorter than $W_{warmup}$ (will be checked programmatically in Stage 2/3).
- Whether code-level faults in RE3 produce detectable metric deviations or manifest primarily in logs and traces.

---

## 10. Evaluation Integrity Foundation (Stage 1)

### 10.1 Benchmark Manifest Typology
Digital Detective explicitly distinguishes three types of evaluation manifests:
1. **Regression Benchmark (`SMOKE_REGRESSION`):**
   - Rapid regression verification of frozen research baselines (e.g., 30-case RE2-OB repetition-1 smoke benchmark).
   - Flagged with `is_smoke_test=True` and partition `'smoke'`.
   - Used strictly to confirm that baseline algorithmic components ($S_{\text{comb}}$, $E_{\text{elev}}$, fixed equal-weight fusion) remain bit-for-bit reproducible across updates. Never used to optimize hyperparameters or report general scientific performance.
2. **Scientific Benchmark (`SCIENTIFIC_BENCHMARK`):**
   - Defines the full declared benchmark population used to evaluate fixed, non-learning methods across complete suites or systems (e.g., full RE2 with 270 cases, RE2-OB with 90 cases, RE2-SS with 90 cases, RE2-TT with 90 cases).
   - Not partitioned into train/val/test; all cases belong to the complete evaluation population (`partition='all'`).
   - Embeds native modality availability summaries (`metrics_available`, `logs_available`, `traces_available`, `all_modalities_available`, `missing_logs`, `missing_traces`). Does not manufacture modalities where they do not natively exist (e.g., RE2-SS has 0 traces in RCAEval; RE2-TT has 1 missing log file).
3. **Leakage-Controlled Tuning Split (`TUNING_SPLIT`):**
   - Optional grouped partition (`'dev'`, `'val'`, `'held_out'`) used only when a method learns or tunes parameters using benchmark telemetry.
   - Enforces the repetition-family grouping rule: all repetitions of any `scenario_family = (suite, system, root_cause_service, fault)` must remain in exactly one partition.
   - Configurable ratios (e.g., dev=0.50, val=0.25, held_out=0.25) and reproducible PRNG seed.
   - **Explicit Project Policy:**
     > RCAEval does not supply an official Digital Detective train/validation/test split; any such split created here is a project-defined experimental split.

### 10.2 Oracle vs Detected Incident Window Semantics
- **Oracle Mode (`mode="oracle"`):** Uses ground-truth `inject_time` as the incident onset timestamp ($t_{\text{onset}} = \text{inject\_time}$). Used exclusively to establish theoretical headroom and upper-bound performance under perfect detection.
- **Detected Mode (`mode="detected"`):** Uses only online observations available during monitoring. The incident onset timestamp is determined strictly by the earliest detected anomaly episode across entity metrics ($t_{\text{onset}} = \min_e t_{\text{episode\_start}}$).
- **Detection Failure Behavior:** If no anomaly episodes are detected, `mode="detected"` must **never** fall back to `inject_time` (which leaks ground truth). Instead, it returns an explicit detection failure outcome (`has_detected_window=False`, `onset_ts=None`). Downstream rankers handle this failure gracefully without fabricating temporal boundaries.

### 10.3 Benchmark Provenance Requirements
Every benchmark execution produces a structured `BenchmarkProvenance` record serialized in machine-readable JSON reports. This record captures:
1. `evaluator_schema_version`: Version of the evaluation harness schema (e.g., `1.0.0`).
2. `manifest_id` and `manifest_hash`: Unique manifest identifier and canonical SHA-256 content hash.
3. `window_mode`: `'oracle'` or `'detected'`.
4. `candidate_universe_policy`: Closed universe policy identifier (e.g., `'canonical_v1'`).
5. `dataset_name`, `dataset_artifact_version`, and `dataset_source_doi`: Explicit dataset provenance. Optional fields not explicitly provided default to `'UNSPECIFIED'` without fabrication.
6. `repository_revision`: Git commit hash or `'UNSPECIFIED'`.
7. `experiment_config_id`: Explicit configuration file or experiment identifier.
8. `methods`: List of evaluated ranker methods.
9. `modality_policy`: Policy governing modality usage (e.g., `'all_available'`).
10. `random_seed`: Seed used for randomized baselines.
11. `runtime_info`: Platform and Python runtime environment metadata.

### 10.4 Deterministic Manifest Hashing
Manifest integrity is sealed via a deterministic SHA-256 hash computed over:
1. Manifest Header: `manifest_id:manifest_type:is_smoke_test:total_cases`
2. Canonical Sorted Case Records (ordered by `case_id`):
   `case_id|dataset|system|fault|root_cause_service|repetition|partition`
Modifying any case ID, partition assignment, dataset identity, or grouping metadata alters the hash. Sorting guarantees that permutation of case ordering does not alter the hash for an identical logical case set.

### 10.5 Repetition-Family Grouping
To prevent information leakage between splits in tuning manifests, cases are grouped by scenario family:
$$\text{scenario\_family} = (\text{suite}, \text{system}, \text{root\_cause\_service}, \text{fault})$$
with `repetition` identifying the repeated execution within that family.
All repetitions (e.g., repetitions 1, 2, 3...) of any given $(\text{suite}, \text{system}, \text{root\_cause\_service}, \text{fault})$ combination are deterministically assigned to the same partition (`'dev'`, `'val'`, or `'held_out'`). Tuning split validation verifies that each family's assigned partitions have cardinality exactly 1. Incorporating `suite` prevents accidental cross-suite grouping when a manifest contains cases from multiple RCAEval suites (e.g., RE1 and RE2).
RCAEval does not supply an official Digital Detective train/validation/test split; any such split is a project-defined experimental split.

### 10.6 Candidate-Universe Policy
The evaluation harness strictly enforces closed, deterministic candidate universes resolved once per case prior to method invocation:
- **Policy `canonical_v1`:** Resolves 11 canonical services for Online Boutique (`adservice`, `cartservice`, `checkoutservice`, `currencyservice`, `emailservice`, `frontend`, `paymentservice`, `productcatalogservice`, `recommendationservice`, `shippingservice`, `redis-cart`) and 8 canonical services for Sock Shop.
- Dynamic universe shrinking, ground-truth-derived candidate subsets, or per-method candidate set variations are forbidden to ensure scientific comparability across all rankers.


