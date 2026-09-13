# Basic Anomaly Detection Baseline

This document specifies the design for the first, smallest correct anomaly-detection baseline in Digital Detective. It is a design document only; it does not implement algorithms, alter the canonical telemetry model, or introduce new package dependencies.

---

## 1. Scope and Architectural Role

Digital Detective is designed as an evidence-based investigation system. In the intended pipeline:
$$\text{telemetry} \longrightarrow \text{canonical model} \longrightarrow \mathbf{\text{anomaly detection}} \longrightarrow \text{dependency graph} \longrightarrow \text{causal RCA} \dots$$

The anomaly detection component serves as a deterministic, statistical evidence provider. Its sole responsibility in this milestone is to consume metric time-series from a `TelemetryCase` and detect anomalous behavior across individual metrics over time.

### Boundary with Downstream Stages
- **Metric-level only:** The detector operates strictly on individual metric streams as opaque time-series.
- **No service attribution:** In accordance with the canonical telemetry model (`docs/design/canonical-telemetry.md`), Digital Detective does not impose premature semantic parsing or service-name decomposition onto raw metric column names. Service attribution, fault localization, and root-cause ranking are the explicit responsibilities of later dependency-graph and causal RCA milestones.

### Explicit Non-Goals
This milestone does **not** include:
- Anomaly detection on logs or traces (reserved for subsequent modalities).
- Multimodal correlation or telemetry fusion.
- Service-name parsing or fault-to-service alignment.
- Dependency graph construction or topological fault propagation.
- Causal root-cause scoring.
- RAG, LLM agent integration, or natural language explanations.
- Advanced non-parametric/segmentation algorithms (e.g., Ruptures, SPOT/EVT, Matrix Profile).
- Deep learning or neural forecasting models.

---

## 2. Input and Output Specification

### Input Contract
The detector consumes a `TelemetryCase` (specifically its `metrics: TelemetryModality`):
- `case.metrics.raw_data`: A `pyarrow.Table` (or column-accessible table) preserving raw source telemetry.
- `case.metrics.provenance`: Provides `timestamp_field` (verified as `"time"` in RCAEval Parquet).
- Explicit configuration parameters:
  - Detection threshold $\tau$ (e.g., $\tau = 3.0$ standard deviations).
  - Observation window size $W$ (integer count of preceding observations, e.g., $W = 60$).
  - Minimum warmup observations $W_{warmup}$ (minimum history required before producing valid scores).
  - Minimum valid history $W_{min\_valid}$ (minimum count of non-null/non-NaN samples required within the preceding $W$ rows; defaults to $\max(2, W_{warmup} // 2)$).
  - Numerical stability constant $\epsilon > 0$ (e.g., $10^{-6}$) to guard floating-point division.

**Strict constraint:** The detector does **not** receive `case.ground_truth` or `case.metadata.incident_metadata` (such as `inject_time`, `root_cause_service`, or `faulty_timesteps`).

### Output Contract
The detector produces an immutable, structured result object, named `MetricAnomalyResult`:
- `case_id: str`: Preserved case identifier.
- `timestamps: tuple[int, ...]`: Chronological sequence of metric timestamps.
- `metric_names: tuple[str, ...]`: Exact source metric column names evaluated (preserving order and raw names).
- `anomaly_scores: Mapping[str, tuple[float | None, ...]]`: Per-metric non-negative anomaly scores defined as $|z_t|$ when evaluated, and `None` when un-evaluated (warmup, missing observation, or insufficient history).
- `signed_scores: Mapping[str, tuple[float | None, ...]]`: Per-metric signed z-scores $z_t$ when evaluated, and `None` when un-evaluated.
- `anomalies: Mapping[str, tuple[bool, ...]]`: Per-metric boolean anomaly flags where $|z_t| \ge \tau$.
- `evaluation_statuses: Mapping[str, tuple[str, ...]]`: Per-metric evaluation state at each observation:
  - `"warmup"`: observation index $t < W_{warmup}$.
  - `"missing_observation"`: current observation $x_t$ is `None` or `NaN`.
  - `"insufficient_history"`: fewer than $W_{min\_valid}$ valid samples in preceding $W$ rows.
  - `"normal"`: evaluated, $|z_t| < \tau$.
  - `"anomaly"`: evaluated, $|z_t| \ge \tau$.
- `valid_history_counts: Mapping[str, tuple[int, ...]]`: Effective count of valid (non-null/non-NaN) observations in the preceding $W$-row window.
- `summary: Mapping[str, Any]`: Summary statistics per metric (e.g., peak score, evaluated count, warmup count, missing count, first anomaly timestamp) and case-level detection totals.

The output does not mutate or overwrite the input `TelemetryCase`.

---

## 3. Telemetry Handling

### Observation-Based Windowing and Causality
1. **Chronological observation window:** The detector operates strictly over the previous $W$ observations in chronological order. Causality is strictly defined over this sequence: for observation at index $k$, estimators access only indices $[\max(0, k - W), k - 1]$. No observation at or after index $k$ is ever used in parameter estimation.
2. **Deterministic sorting:** Input records are indexed in ascending chronological order of the isolated timestamp coordinate. Ties in timestamps are broken deterministically using a stable sort that preserves original input relative order.
3. **No assumed regularity:** The baseline does **not** assume regular sampling intervals ($\Delta t = \text{constant}$) and does **not** perform synthetic interpolation, time-binning, or resampling. Telemetry records are evaluated in the order they occur in the source data.
4. **Timestamp isolation:** The column identified by `provenance.timestamp_field` (`"time"`) is isolated as the temporal coordinate. It is never treated as a metric time-series or fed into anomaly calculations.
5. **Schema independence:** The detector discovers metric columns dynamically by inspecting table column names minus the `timestamp_field`. No fixed metric list or naming format is assumed.

### Conceptual and Numerical Zero-Variance Behavior
In microservice telemetry, metrics frequently remain constant over extended windows (e.g., error count = 0, static pool size). When the historical window has zero sample variance ($\sigma_t = 0$), the behavior is defined conceptually:
- **Case 1 (Unchanged invariant):** If the historical window is constant ($x_{t-W}, \dots, x_{t-1} = c$) and the current observation equals that constant ($x_t = c$), the observation is **evaluated normal**:
  $$|z_t| = 0.0, \quad \text{status}_t = \text{"normal"}, \quad \text{anomaly}_t = \text{False}$$
- **Case 2 (Broken invariant):** If the historical window is constant ($x_{t-W}, \dots, x_{t-1} = c$) and the current observation differs ($x_t \neq c$), the observation represents a discrete departure from an invariant state and is **evaluated anomalous**:
  $$|z_t| = \infty \quad (\text{or numerically } \frac{|x_t - c|}{\epsilon} \ge \tau), \quad \text{status}_t = \text{"anomaly"}, \quad \text{anomaly}_t = \text{True}$$

A small numerical epsilon $\epsilon > 0$ may be used in computation to prevent hardware `ZeroDivisionError`, but epsilon is an implementation safeguard, not the conceptual definition.

### Missing Values and Window Preservation
- **W-row window preservation:** The historical window is strictly the preceding $W$ rows $[t-W, t-1]$. The detector does not expand backward beyond $W$ rows to hunt for substitute valid samples.
- **Reporting effective valid sample count:** Within the preceding $W$ rows, valid (non-null, non-NaN) samples are identified, and their count is recorded in `valid_history_counts`.
- **Minimum valid history rule:** If the effective valid sample count is below $W_{min\_valid}$, the observation is marked as `"insufficient_history"` with `anomaly_score = None`. This prevents silently reducing a 60-observation window into an unreliable 2-sample statistic.
- **Missing current observation:** If the current observation $x_t$ is `None` or `NaN`, it is marked as `"missing_observation"` with `anomaly_score = None` and `anomaly = False`.
- **Scale independence:** Microservice metrics span vastly different orders of magnitude (e.g., request count $\sim 10^1$, latency $\sim 10^2$, memory bytes $\sim 10^9$). The z-score normalization inherently scales deviations by the metric's own historical standard deviation, ensuring scale invariance across metrics.

---

## 4. Leakage Prevention and Ground Truth Separation

### The Ground Truth Leakage Hazard
In benchmark datasets such as RCAEval, `inject_time` marks the Unix epoch when a fault was injected.
- If a detector uses `inject_time` to partition data into a "clean baseline period" and a "detection period", it **leaks benchmark ground truth**. In production operation, the anomaly detector has no foreknowledge of failure onset.
- If a detector calculates global statistics across the full case $[t_{start}, t_{end}]$, post-fault anomalies contaminate the baseline mean and variance (anomaly dilution), constituting bidirectional temporal leakage.

### Protocol for Leakage-Free Detection
1. **Causal streaming updates:** At observation index $k$, parameters (running sample mean $\mu_{k}$ and standard deviation $\sigma_{k}$) are estimated exclusively from the previous $W$ observations ($x_{k-W}, \dots, x_{k-1}$).
2. **Fixed warmup threshold:** A global minimum observation count $W_{warmup}$ is required before scores are emitted as valid. This threshold is uniform across the benchmark and never tuned per-case to `inject_time`.
3. **Evaluation-only role of `inject_time`:** `inject_time` is consumed solely by the **evaluator** after anomaly detection has completed.

---

## 5. Algorithmic Baseline Comparison: Rolling Z-Score vs. EWMA

We compare two simple candidates for the initial baseline:

| Criterion | Causal Rolling Window Z-Score | Streaming EWMA |
| :--- | :--- | :--- |
| **Mathematical Formulation** | $\mu_t = \frac{1}{W}\sum_{i=1}^W x_{t-i}$<br>$\sigma_t = \sqrt{\frac{1}{W}\sum_{i=1}^W (x_{t-i} - \mu_t)^2}$<br>$z_t = \frac{x_t - \mu_t}{\sigma_t + \epsilon}$ | $\mu_t = \alpha x_{t-1} + (1-\alpha)\mu_{t-1}$<br>$\sigma_t^2 = \beta (x_{t-1} - \mu_{t-1})^2 + (1-\beta)\sigma_{t-1}^2$<br>$z_t = \frac{x_t - \mu_{t-1}}{\sqrt{\sigma_t^2} + \epsilon}$ |
| **Observation Memory** | Buffers exactly $W$ preceding observations per metric. | Buffers only running $\mu_{t-1}$ and $\sigma_{t-1}^2$ per metric. |
| **Sampling Irregularity** | Operates directly on the previous $W$ discrete samples without assuming time-step uniformity. | Exponential decay implies a continuous time assumption ($\Delta t$ decay weighting), complicating irregular intervals. |
| **Recovery from Transients** | Finite support: an anomalous shock drops completely out of the estimation window after exactly $W$ observations. | Infinite impulse response (IIR): anomalous values decay asymptotically, permanently altering future running variance unless frozen. |
| **Parameter Interpretability** | Single integer parameter $W$ (number of observations). | Two continuous parameters $\alpha, \beta \in (0, 1)$ without obvious physical calibration. |

### Decision for Initial Baseline
A **causal rolling window z-score** over the previous $W$ observations is selected for the first implementation because:
1. It is finite-memory, fully deterministic, and verifiable by manual inspection of any $W$-observation window.
2. It makes no continuous-time or regular-sampling assumptions.
3. It provides an unambiguous baseline against which EWMA or adaptive filtering can be compared in later milestones.

---

## 6. Benchmark Evaluation Protocol

The evaluation protocol defines how anomaly detector outputs are scored against ground truth.

### 1. Metric-Level Anomaly Rule
An individual metric $m$ is anomalous at observation $t$ if its anomaly score meets or exceeds the threshold:
$$\text{anomaly}_{m, t} = (|z_{m, t}| \ge \tau)$$

### 2. Case-Level Detection Rule (Baseline Aggregation)
For this initial benchmark, case-level detection uses a simple first-alarm OR-gate aggregation:
- **Evaluation start ($t_{eval\_start}$):** Evaluation begins at the first observation after the required warmup period ($t_{eval\_start} = t_{W_{warmup}}$).
- **Case detection timestamp ($t_{detected}$):** The earliest timestamp $t \ge t_{eval\_start}$ at which **any** metric is flagged as anomalous:
  $$t_{detected} = \min \{ t \ge t_{eval\_start} \mid \exists m : \text{anomaly}_{m, t} = \text{True} \}$$
  If no metric flags an anomaly for the duration of the case, $t_{detected} = \text{None}$ (undetected / missed case).

*Methodology note:* This first-alarm rule is the baseline aggregation rule for the benchmark. It provides a clean lower bound and will be compared in future work against more robust aggregation strategies (e.g., $k$-of-$N$ persistence windows, multi-metric consensus).

### 3. Quantitative Evaluation Metrics
- **Detection Latency ($\Delta t_{detect}$):**
  $$\Delta t_{detect} = t_{detected} - \text{inject\_time}$$
  - If $t_{detected} \ge \text{inject\_time}$: Successful detection with non-negative delay $\Delta t_{detect} \ge 0$.
  - If $t_{detected} < \text{inject\_time}$: Early detection / false alarm prior to fault injection.
  - If $t_{detected} = \text{inject\_time}$: Immediate detection ($\Delta t_{detect} = 0$).
  - If $t_{detected} = \text{None}$: Missed detection ($\Delta t_{detect}$ is undefined / penalized in evaluation).
- **Pre-Injection False Alarm Rate (FAR):**
  $$\text{FAR} = \frac{\sum_{t \in [t_{eval\_start}, \text{inject\_time})} \mathbf{1}(\exists m : \text{anomaly}_{m, t} = \text{True})}{N_{\text{pre-injection}}}$$
  Quantifies detector precision and false-alert frequency under normal operational conditions.

---

## 7. Concrete Edge Cases and Test Strategy

The implementation must be validated against the following specific edge cases:

| Edge Case | Description | Expected Behavior |
| :--- | :--- | :--- |
| **Insufficient Warmup History** | Total observations $N < W_{warmup}$. | Do not emit speculative scores; gracefully return empty scores or raise a clear, informative error. |
| **Constant Historical Window (Equal)** | Preceding $W$ observations are constant $c$, and current $x_t = c$. | Strictly $|z_t| = 0.0$, $\text{anomaly}_t = \text{False}$. No division by zero. |
| **Constant Historical Window (Different)** | Preceding $W$ observations are constant $c$, and current $x_t \neq c$. | Discrete invariant break: $|z_t| \ge \tau$, $\text{anomaly}_t = \text{True}$. |
| **NaN / Non-numeric Metric Values** | Column contains NaNs, nulls, or invalid entries. | Handled safely without unhandled runtime exceptions or score corruption across other valid columns. |
| **Irregular Timestamps** | Telemetry samples have non-uniform time gaps ($\Delta t_i \neq \Delta t_j$). | Detector steps through previous $W$ discrete observations without synthetic interpolation or resampling. |
| **Timestamp Column Exclusion** | The isolated timestamp column (`"time"`) is present in the table. | Timestamp column is strictly used as temporal index; never evaluated as a metric or included in metric anomaly results. |
| **Metrics with Different Scales** | Case contains simultaneous small ($\in [0, 1]$) and large ($\approx 10^9$) metrics. | Z-score normalizes each metric relative to its own scale without precision loss or overflow. |
| **No Anomaly After Injection** | Fault produces no measurable metric deviation exceeding $\tau$. | Evaluator correctly records a missed detection ($t_{detected} = \text{None}$); detector does not crash or invent alarms. |
| **Anomaly Exactly at Injection Time** | First anomalous metric observation occurs at $t = \text{inject\_time}$. | Correctly flagged with $\Delta t_{detect} = 0$. |

---

## 8. Verification Classification

In accordance with `AGENTS.md`:

### Verified Facts
- In RCAEval Parquet distribution (`re1ob_adservice_cpu_1` and `re2ob_checkoutservice_cpu_1`), `metrics.parquet` is a wide table containing an int64 `time` column and numeric float/double columns.
- The `time` column uses Unix seconds.
- In `re1ob_adservice_cpu_1`, 4,201 observations are present across 49 metric columns + 1 time column.
- In `re2ob_checkoutservice_cpu_1`, 1,441 observations are present across 72 metric columns + 1 time column.
- Case ground truth in `cases.parquet` explicitly separates `root_cause_service`, `fault`, and `inject_time`.
- In both validated cases, hundreds of observations precede `inject_time`, providing a substantial operational pre-injection baseline.

### Assumptions
- A causal observation window of $W = 60$ preceding samples provides a stable initial baseline for sample mean and variance.
- Evaluating metrics independently without initial cross-metric correlation provides a fast, transparent baseline before introducing causal graph and dependency analysis.

### Open Questions
- What empirical threshold $\tau$ achieves the optimal trade-off between detection latency and false alarm rate across the full 735-case benchmark.
- Whether specific non-Online Boutique datasets (e.g. Sock Shop, Train Ticket) exhibit significant bursts of nulls or irregular sample drops requiring special imputers.

---

## 9. Acceptance Criteria for Future Implementation

Implementation of the basic anomaly detector will be considered complete when:
1. **Preservation-first compatibility:** Operates directly on `TelemetryCase.metrics.raw_data` without mutating the canonical case.
2. **Deterministic execution:** Produces identical deterministic scores on repeated runs with identical configuration.
3. **No lookahead / leakage:** Parameters at step $k$ depend exclusively on observations $k-W$ to $k-1$; `inject_time` is never accessed by the detector.
4. **Observation-based windowing:** Evaluates the preceding $W$ discrete observations without temporal resampling or interpolation.
5. **Exact zero-variance semantics:** Identifies $x_t = c$ as non-anomalous and $x_t \neq c$ as anomalous when window is constant.
6. **Schema-agnostic:** Operates correctly on cases regardless of the count or names of metric columns.
7. **Complete edge case test coverage:** Successfully passes unit tests for all 9 scenarios identified in Section 7.
