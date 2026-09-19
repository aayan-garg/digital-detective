# Stage-10 Persistence Calibration Report: Repetition-1 Trace Telemetry Audit

**Date:** 2026-09-19  
**Audit Type:** READ-ONLY Scientific Calibration & Feasibility Audit  
**Calibration Population:** RE2-OB Repetition 1 (30 cases, `re2ob_*_1`)  
**Evaluation Population (Frozen/Untouched):** RE2-OB Repetitions 2 & 3 (60 cases, `re2ob_*_2`, `re2ob_*_3`)  
**Auditor Constraints:** No code modifications, no parameter tuning, no benchmark runs, no use of repetitions 2 or 3 data.

---

## Verdict Summary

| Section | Subject | Finding |
|---|---|---|
| **A. Data Units** | Parquet timestamps & durations | Verified: `startTime` = microseconds ($\mu\text{s}$), `startTimeMillis` = milliseconds ($\text{ms}$), `duration` = microseconds ($\mu\text{s}$), Metric `time` = seconds ($\text{s}$). |
| **B. What Traces Measure** | Physical & architectural semantics | In-flight synchronous RPC execution durations and network return lags ($10^{-4}$ to $10^{-1}\text{ s}$). |
| **C. Causal Temporal Feasibility** | Bounding anomaly episode propagation | **Infeasible.** Severe category mismatch between microsecond-scale RPC execution and tens-of-seconds macroscopic anomaly propagation. |
| **D. Candidate Formalization** | Mathematical persistence rules | Evaluated candidates 1, 2, and 3. Direct rules truncate below metric sampling resolution; scaled rules require arbitrary multipliers ($k \approx 30,000$). |
| **E. Leakage & Independence** | Repetition 1 vs. Evaluation population | Repetition 1 is strictly independent; zero contamination with Stage 2–9 evaluation data. |
| **F. Pre-Registration** | Protocol recommendations | Do not pre-register trace-derived persistence. Maintain Stage-9 hard reset as baseline. |
| **G. Final Recommendation** | GO / NO-GO Decision | **NO-GO** |

---

## Section A: DATA UNITS

An empirical unit audit was conducted on raw parquet telemetry across all 30 repetition-1 cases in `~/.cache/rcaeval_validation/re2ob_*_1/`.

### 1. Parquet Schema and Timestamp Relationship

Inspection of `traces.parquet` across all cases confirms the following column types and numerical properties:
- `startTime`: `int64` (16 digits, e.g., `1705353846065999`).
- `startTimeMillis`: `int64` (13 digits, e.g., `1705353846065`).
- `duration`: `int64` (e.g., `134`, `4133`, `222894`).

We tested the mathematical relationship across 3,000 sampled rows (100 rows per case across all 30 cases):
$$\text{startTimeMillis} = \lfloor \frac{\text{startTime}}{1000} \rfloor$$
$$\text{startTime} \pmod{1000} \in [0, 999]$$

**Result:** Zero discrepancies. `startTimeMillis` is the integer floor of `startTime` divided by 1,000.
- `startTime` is Unix epoch timestamp in **microseconds ($\mu\text{s}$)** ($10^{-6}\text{ s}$).
- `startTimeMillis` is Unix epoch timestamp in **milliseconds ($\text{ms}$)** ($10^{-3}\text{ s}$).

### 2. Duration Units

In `src/digital_detective/traces.py` (lines 73–76):
```python
@property
def end_time(self) -> int:
    """Span end timestamp in microseconds."""
    return self.start_time + self.duration
```
Because `start_time` is assigned directly from the `startTime` column ($\mu\text{s}$) and added directly to `duration`, span duration is confirmed to be in **microseconds ($\mu\text{s}$)**. This matches the standard OpenTelemetry / Jaeger data model for distributed tracing.

Derived trace latency metrics in `traces.py` share these exact units:
- `callee_duration` = microseconds ($\mu\text{s}$)
- `return_lag` ($t_{\text{end, parent}} - t_{\text{end, child}}$) = microseconds ($\mu\text{s}$)
- `start_lag` ($t_{\text{start, child}} - t_{\text{start, parent}}$) = microseconds ($\mu\text{s}$)

### 3. Metric Modality Units

Inspection of `metrics.parquet` (`case.metrics.raw_data`) across repetition-1 cases confirms:
- Column `time`: `int64` (10 digits, e.g., `1705353846` to `1705355286`).
- Sampling resolution: $\Delta t_{\text{metric}} = 1.0\text{ s}$ uniformly (1,441 rows per case = 1,440 seconds = 24.0 minutes).
- Time units: **seconds ($\text{s}$)**.

### 4. Timescale Comparison

| Metric | Raw Unit | Seconds Equivalent |
|---|---|---|
| Trace span median duration | $141 \mu\text{s}$ | $0.000141\text{ s}$ |
| Trace return lag p90 | $2,088.5 \mu\text{s}$ | $0.002088\text{ s}$ |
| Slowest RPC median (`frontend \to checkout`) | $246,290 \mu\text{s}$ | $0.246\text{ s}$ |
| Metric sampling tick ($\Delta t$) | $1,000,000 \mu\text{s}$ | $1.000\text{ s}$ |
| Observed inter-episode gap (Stage 9) | $15,000,000 - 89,000,000 \mu\text{s}$ | $15.0 - 89.0\text{ s}$ |

A single metric observation interval ($1\text{ s}$) is **~7,000 times larger** than the median callee duration ($141 \mu\text{s}$) and **~500 times larger** than the 90th percentile return lag ($2.09\text{ ms}$).

---

## Section B: WHAT TRACE TIMINGS ACTUALLY MEASURE

Distributed trace spans recorded by Jaeger/OpenTelemetry in Online Boutique capture individual RPC/gRPC transactions between microservices.

### 1. Span Semantics in `traces.py`

```
Parent Span (Caller: frontendservice)
[====================================================================]
         |--> start_lag <--|               |--> return_lag <--|
                           [===============]
                           Child Span (Callee: checkoutservice)
                           |<-- callee_duration -->|
```

- **`callee_duration`:** The synchronous execution duration of the downstream RPC within the callee service process (from request receipt to response transmission).
- **`start_lag`:** Time elapsed between parent span initiation and child span invocation (request preparation, serialization, network transit to callee).
- **`return_lag`:** Time elapsed between child span completion and parent span completion (network response transit, deserialization, caller post-processing).
- **`child_covered_duration`:** The total time interval during which one or more child spans were actively executing.

### 2. Empirical Population Statistics (All 30 Repetition-1 Cases)

Extracting trace latency evidence across all 30 repetition-1 cases ($>4.7$ million span relationships across 9 distinct service-to-service communication edges) yields the following distributions:

| Caller Service | Callee Service | Relationship Count | Avg Median Duration | Avg p90 Duration | Avg p90 Return Lag | Max Duration |
|---|---|---|---|---|---|---|
| `checkoutservice` | `currencyservice` | 64,432 | 0.12 ms | 0.18 ms | 89.82 ms | 591.62 ms |
| `checkoutservice` | `emailservice` | 18,594 | 0.26 ms | 0.46 ms | 15.65 ms | 1,681.46 ms |
| `checkoutservice` | `paymentservice` | 18,656 | 0.22 ms | 0.39 ms | 6.54 ms | 98.76 ms |
| `checkoutservice` | `productcatalogservice` | 45,760 | 0.01 ms | 0.02 ms | 8.42 ms | 192.41 ms |
| `frontendservice` | `checkoutservice` | 18,816 | 246.29 ms | 1,079.48 ms | 15.34 ms | 41,265.98 ms |
| `frontendservice` | `currencyservice` | 1,513,450 | 0.12 ms | 0.18 ms | 78.69 ms | 1,095.68 ms |
| `frontendservice` | `productcatalogservice` | 2,410,649 | 0.01 ms | 0.02 ms | 8.81 ms | 2,402.28 ms |
| `frontendservice` | `recommendationservice` | 370,388 | 3.91 ms | 109.15 ms | 15.54 ms | 13,625.30 ms |
| `recommendationservice` | `productcatalogservice` | 369,061 | 0.01 ms | 0.02 ms | 18.72 ms | 1,907.37 ms |

**Cross-Edge Aggregate Summary:**
- **Callee Median Duration:** Min = $9.0 \mu\text{s}$ ($0.01\text{ ms}$), Median = $141.0 \mu\text{s}$ ($0.14\text{ ms}$), Max = $546,973.0 \mu\text{s}$ ($546.97\text{ ms}$).
- **Callee p90 Duration:** Min = $15.0 \mu\text{s}$ ($0.01\text{ ms}$), Median = $201.5 \mu\text{s}$ ($0.20\text{ ms}$), Max = $8,151,946.0 \mu\text{s}$ ($8,151.95\text{ ms}$).
- **Return Lag p90:** Min = $1,323.0 \mu\text{s}$ ($1.32\text{ ms}$), Median = $2,088.5 \mu\text{s}$ ($2.09\text{ ms}$), Max = $408,017.0 \mu\text{s}$ ($408.02\text{ ms}$).

### 3. Key Finding on Semantic Interpretation

These values represent **in-flight synchronous request execution times**. They reflect how long a single thread or coroutine waits for a remote procedure call to complete.
They **do not** represent:
- Inter-incident propagation delays.
- Time required for downstream buffers or worker pools to saturate.
- Interval between cascading failure episodes in distinct services.

---

## Section C: WHETHER THEY SUPPORT CAUSAL TEMPORAL FEASIBILITY

### 1. The Category Error

To use trace latencies to bound the persistence of a dormant incident hypothesis between metric anomaly episodes is a **fundamental category error**:

1. **Microscopic execution vs. Macroscopic state change:**
   An RPC call between `frontendservice` and `checkoutservice` takes $246\text{ ms}$. If `checkoutservice` suffers a CPU hog fault, individual RPCs may slow down or time out within $1\text{ to }10\text{ seconds}$. However, for `frontendservice` or other callers to exhibit a statistically significant metric anomaly episode (e.g., elevated error count or sustained CPU usage from retries), system queues, thread pools, and retry budgets must accumulate strain over macroscopic intervals.
2. **Metric Polling and Sampling Granularity:**
   Metric telemetry is recorded at $1\text{ s}$ intervals. A single metric sample is larger than almost every observed trace span. No event in trace telemetry that lasts $0.14\text{ ms}$ or $2\text{ ms}$ can define a temporal threshold for metric series sampled at $1,000\text{ ms}$.
3. **Statistical Detection Latency:**
   BOCPD and sequential corroboration require multiple successive observations of anomalous metric behavior to overcome noise and reach decision thresholds. The time to detect an anomaly is governed by the detector's run-length posterior and statistical power, not the network transit time of an individual packet.
4. **Intermittent and Cyclic Incident Dynamics:**
   As revealed in the Stage-9 failure analysis, cases that became unconfirmed exhibited multi-cycle behavior (e.g., Kubernetes pod restart loops, container crash-backs, or periodic load cycles) where the dormant interval between corroborating evidence spikes was **15 to 89 seconds**. Trace spans have no knowledge of container lifecycle events or background scheduling loops.

### 2. The Arbitrary Multiplier Trap

If one attempts to use the 90th percentile return lag ($2.09\text{ ms}$) or callee duration ($0.14\text{ ms}$) to create an anomaly persistence window of $\sim 60\text{ seconds}$, one must apply a scaling factor:
$$k = \frac{60\text{ s}}{0.00209\text{ s}} \approx 28,700$$
There is no theoretical, empirical, or physical basis in repetition-1 data to justify $k = 28,700$, $k = 1,000$, or any other multiplier. Selecting such a constant would be arbitrary parameter fabrication.

### 3. Multi-Hop Paths

Summing trace latencies along dependency paths (e.g., `frontendservice \to checkoutservice \to paymentservice`) yields nominal path execution times of $\sim 250\text{ ms}$. Adding $250\text{ ms}$ to an anomaly persistence rule does nothing to resolve an 80-second inter-episode gap. The path latency remains three orders of magnitude smaller than the inter-episode dynamics.

---

## Section D: CANDIDATE FORMALIZATION

We systematically evaluate three candidate formalizations for persistence rules:

### Candidate 1: Direct Trace-Latency Persistence Window

Define the maximum persistence window $\tau_{\text{persist}}(u \to v)$ for a causal hypothesis from entity $u$ to entity $v$ as:
$$\tau_{\text{persist}}(u \to v) = \text{p99}(\text{callee\_duration}_{u \to v}) + \text{p90}(\text{return\_lag}_{u \to v})$$
- For leaf services (`currencyservice`, `productcatalogservice`):
  $$\tau_{\text{persist}} \approx 0.18\text{ ms} + 89.82\text{ ms} \approx 0.09\text{ s}$$
- For orchestrators (`checkoutservice`):
  $$\tau_{\text{persist}} \approx 8,151\text{ ms} + 15\text{ ms} \approx 8.17\text{ s}$$

**Evaluation:**
- For leaf services, $\tau_{\text{persist}} < \Delta t_{\text{metric}}$ ($0.09\text{ s} < 1.0\text{ s}$). The persistence window expires before a single subsequent metric sample can be evaluated.
- Any dormant cycle $> 8.2\text{ s}$ resets the hypothesis.
- **Outcome:** Degrades detection even further than Stage 9 without resolving any multi-cycle abstentions.

### Candidate 2: Empirically Scaled Trace-Latency Window

Define persistence with a scaling multiplier $k$:
$$\tau_{\text{persist}}(u \to v) = k \cdot \tau_{\text{trace}}(u \to v)$$
**Evaluation:**
- $k$ cannot be derived from repetition-1 traces.
- Repetition 1 contains no anomaly propagation labels.
- Tuning $k$ against the unconfirmed cases of Stage 9 would violate the core requirement of pre-specified, independent calibration.
- **Outcome:** Mathematically and methodologically unsound.

### Candidate 3: Structural Reachability via Traces with Static Cooldown

Separate the temporal persistence problem from trace telemetry:
- **Structural association:** Entity $v$ can corroborate a hypothesis originating at entity $u$ if and only if there exists a directed path $u \to^* v$ in the trace-observed dependency graph.
- **Temporal association:** The hypothesis remains active for a fixed operational horizon $\tau_{\text{horizon}} = T_{\text{cooldown}}$ (e.g., standard SRE operational window of $60\text{ s}$ or $120\text{ s}$).

**Evaluation:**
- The trace telemetry in repetition 1 verifies the existence of edges between services. However, this exact topology is already statically defined in `src/digital_detective/topology.py` (`KNOWN_ENTITY_TYPES` and `DependencyGraph`).
- Traces in repetition 1 do not reveal any new or hidden communication pathways not already present in the architecture.
- The temporal window $T_{\text{cooldown}}$ in this formulation is an operational constant, not a trace-derived property.
- **Outcome:** Traces provide zero temporal calibration value.

---

## Section E: LEAKAGE / INDEPENDENCE ANALYSIS

### 1. Partition Verification

| Dataset Partition | Case Identifiers | Count | Status in Prior Stages | Audit Role |
|---|---|---|---|---|
| **RE2-OB Repetition 1** | `re2ob_*_1` | 30 | Untouched (held out) | Calibration Population |
| **RE2-OB Repetition 2** | `re2ob_*_2` | 30 | Official Evaluation | Frozen / Off-Limits |
| **RE2-OB Repetition 3** | `re2ob_*_3` | 30 | Official Evaluation | Frozen / Off-Limits |

- Repetition 1 contains all 5 services (`checkoutservice`, `currencyservice`, `frontend`, `productcatalogservice`, `recommendationservice`) across all 6 fault injection types (`cpu`, `delay`, `loss`, `memory`, `socket`, `synthetic_network_delay`).
- Inspection of `eval/results/.cache_re2ob_trace_elevations.json` confirms **0 of 60** entries are from repetition 1.
- No Stage-9 evaluation artifacts, scores, or false-early labels were used in this audit.

### 2. Independence Finding

Repetition 1 is strictly independent of the evaluation population. The rejection of trace-derived persistence is **not** due to data leakage or lack of an independent split.
It is due entirely to the **lack of physical and statistical validity** of using synchronous RPC execution durations to model macroscopic incident propagation delays.

---

## Section F: PRE-REGISTRATION RECOMMENDATION

1. **Do NOT pre-register a trace-duration-based temporal persistence rule.**
   Attempting to use `callee_duration` or `return_lag` to govern episode persistence creates an illusion of causal grounding while actually deploying an arbitrary, unphysical threshold.
2. **Respect the domain boundaries of telemetry modalities:**
   - Distributed traces provide high-resolution structural information (which services communicate, caller-callee hierarchies, in-flight request latencies).
   - Metrics provide continuous macroscopic state information (resource exhaustion, aggregate error rates).
   - Episode persistence is a macro-state phenomenon governed by metric sampling intervals ($\Delta t = 1\text{ s}$), detector run-lengths, and container lifecycle events.
3. **If Stage 10 proceeds in the future:**
   Any persistence model must be formulated at the macroscopic level—such as an explicit, bounded hypothesis half-life or a fixed operational cooldown window (e.g., $60\text{ s}$) derived from standard SRE monitoring conventions—and must be evaluated against frozen controls without pretending to be calibrated by microsecond-level trace spans.
4. **Current Recommendation for Stage 9:**
   Stage 9's hard reset after inactivity remains the most conservative, scientifically sound treatment currently validated in the repository.

---

## Section G: GO / NO-GO RECOMMENDATION

### **VERDICT: NO-GO**

### Summary of Scientific Justification

1. **Extreme Unit and Timescale Mismatch:**
   Trace span durations and return lags in repetition 1 have median values of $0.14\text{ ms}$ and $2.09\text{ ms}$ ($10^{-4}$ to $10^{-3}\text{ s}$). Metric telemetry is sampled at $1.0\text{ s}$ ($10^0\text{ s}$), and macroscopic inter-episode gaps span $15\text{ to }89\text{ s}$ ($10^1$ to $10^2\text{ s}$).
2. **Absence of Physical Mechanism in Trace Durations:**
   Individual RPC durations measure in-flight transaction time. They do not capture queue saturation, retry backoff, connection pool exhaustion, or pod recovery cycles.
3. **Impossibility of Principled Calibration:**
   Bridging the gap between trace latencies ($2\text{ ms}$) and episode persistence ($60\text{ s}$) requires an ungrounded multiplier ($k \approx 30,000$) that cannot be calibrated from repetition 1 without arbitrary curve fitting.
4. **Research Integrity Standard:**
   In adherence to the Digital Detective Agent Guide (`AGENTS.md`), we refuse to implement an unverified physical proxy or fabricate benchmark heuristics. Repetition-1 trace telemetry cannot defensibly support a causal temporal-feasibility rule for dormant incident hypotheses.
