# Stage-10 Persistence Source Audit

**Date:** 2026-09-19  
**Audit type:** READ-ONLY  
**Git revision inspected:** `08c456584adeb105dc1fe513fb8f4f232ef37070`  
**Auditor constraint:** No code modifications, no parameter tuning, no benchmark runs.

---

## Verdict

## A. AVAILABLE

An independent, pre-existing source of causal/propagation delay information **does exist** in the current repository and local dataset cache. It can defensibly support a pre-specified causal persistence model without touching the Stage-9 60-case evaluation population.

---

## 1. Available Independent Latency / Propagation Sources

### Source 1 — Trace-derived per-edge span duration and return-lag distributions (RE2-OB Repetition 1)

**Location:** Raw telemetry data in `~/.cache/rcaeval_validation/re2ob_*_1/traces.parquet`  
**Processing code:** `src/digital_detective/traces.py` → `extract_trace_latency_evidence()`  
**Confirmed available:** 30 RE2-OB repetition-1 case directories are present in the local cache (verified: `re2ob_checkoutservice_cpu_1` through `re2ob_recommendationservice_socket_1`).

**What it contains (verified from `re2ob_checkoutservice_cpu_1`):**

The `extract_trace_latency_evidence()` function, operating on raw trace spans with **no access to inject_time, root-cause labels, or outcome data**, produces `EdgeLatencyEvidence` objects for every observed caller → callee service pair, each containing:

| Field | Description |
|---|---|
| `callee_duration_dist` | Distribution of callee span durations (count, min, p10, median, p90, p99, max, mean) |
| `return_lag_dist` | Distribution of time from callee end to parent end (round-trip return lag) |
| `caller_duration_dist` | Distribution of parent span durations |
| `child_covered_fraction_dist` | Fraction of parent span duration covered by child execution |
| `relationship_count` | Number of observed caller→callee span relationships |

**Sample output from `re2ob_checkoutservice_cpu_1` (pre-fault baseline, no truncation):**

| Edge | Callee median duration | Return lag p90 |
|---|---|---|
| `frontend → currencyservice` | 97 (raw units) | 94,487 |
| `checkoutservice → currencyservice` | 97 | 94,159 |
| `frontend → productcatalogservice` | 11 | 1,522 |
| `recommendationservice → productcatalogservice` | 12 | 2,249 |
| `frontend → checkoutservice` | 308,024 | 1,719 |

Raw units are consistent with the trace `startTime` field confirmed in `docs/research/rcaeval.md` as `startTime: int64`, with `startTimeMillis` also present. Duration units must be verified before use (likely microseconds given `startTime ≈ 1705353846065999` and `duration ≈ 2301`).

**Pre-fault baseline extraction:** By applying `extract_trace_latency_evidence()` with `max_timestamp` set to the BOCPD candidate onset (or any pre-incident timestamp), a **baseline propagation delay distribution per service edge** can be extracted that reflects nominal system behaviour prior to any anomaly.

---

### Source 2 — Stage-9 confirmed-case inter-evidence-event gap distribution

**Location:** `eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json`  
**Population:** 50 confirmed cases from the Stage-9 evaluation population (repetitions 2 and 3)

**Verified gap distribution (108 inter-event gaps across 50 confirmed cases):**

| Statistic | Value |
|---|---|
| min | 1 s |
| median | 15 s |
| mean | 24.5 s |
| p90 | 60 s |
| max | 89 s |

**Independence status:** ⚠️ **NOT INDEPENDENT of the Stage-9 evaluation population.** This distribution is derived directly from the 60 cases used in the Stage-9 treatment. Using it to set a persistence timeout would constitute post-hoc tuning on the evaluation population. It is documented here for completeness only; it **must not be used** as a basis for a pre-specified causal persistence model.

---

### Source 3 — Existing trace elevation score cache (Stage-9 evaluation cases only)

**Location:** `eval/results/.cache_re2ob_trace_elevations.json`  
**Population:** Exactly the 60 Stage-9 evaluation cases (reps 2 and 3; zero rep-1 entries confirmed)

**Independence status:** ❌ **Entirely within the Stage-9 evaluation population.** Contains pre-computed trace elevation scores per entity per case. Provides no per-edge timing information. Cannot be used as an independent source.

---

## 2. Dataset / Case Population Partitions

| Population | Cases | Repetitions | Status | Role |
|---|---|---|---|---|
| RE2-OB repetition 1 | 30 cases | rep=1 only | **Not used in any Stage 2–9 experiment** | Frozen regression / smoke |
| RE2-OB repetitions 2+3 | 60 cases | rep=2, rep=3 | Stage-2 through Stage-9 treatment evaluation | **Evaluation population — off-limits for calibration** |
| All Stage-2–9 result artifacts | 60 cases | rep=2, rep=3 | Stages 2–9 | Evaluation population |

**Source:** `eval/manifests/re2_ob_all_cases.json` (manifest_id: `stage2_re2ob_rep2_rep3_baseline_v1`); `configs/eval_smoke_re2_ob.json`; verified by inspecting all result artifact case_id suffixes (all end in `_2` or `_3`).

The trace elevation cache (`eval/results/.cache_re2ob_trace_elevations.json`) contains exactly 60 entries, all with rep-2 or rep-3 suffixes — confirmed zero rep-1 entries.

---

## 3. Repetition-1 Designation: Formal Status and Calibration Legitimacy

### Documented role in the repository

From `eval/manifest.py` (lines 4–6, `create_smoke_re2_ob_rep1_manifest` docstring):

> *"Generate the standard 30-case RE2-OB repetition-1 smoke-test manifest. Explicitly flagged as SMOKE / REGRESSION ONLY."*

From `docs/PROJECT_STATE.md` (line 314–315):

> *"Frozen 30-Case RE2-OB Repetition-1 Smoke Baseline"*  
> *"The 30-case RE2-OB benchmark is an internal regression and development smoke subset, not an official complete competition partition."*

From `configs/eval_smoke_re2_ob.json` (line 4):

> *`"description": "30-case RE2-OB repetition-1 smoke test manifest (PROJECT REGRESSION / SMOKE ONLY)"`*

The `BenchmarkManifest` validation logic enforces `partition='smoke'` for all repetition-1 cases and requires `is_smoke_test=True`. The `SMOKE_REGRESSION` type is explicitly distinct from `SCIENTIFIC_BENCHMARK` and `TUNING_SPLIT`.

### Can repetition-1 legitimately serve as a calibration population?

**Conditional yes, with documented caveats.**

Repetition-1 has **not been used in any Stage-2 through Stage-9 experiment.** It is not in the evaluation population. No Stage-2–9 result artifact contains a case ID ending in `_1`. This separation is enforced by the manifest system and verified by artifact inspection.

However, the repository **does not document repetition-1 as a reserved calibration corpus**. Its current formal role is strictly "SMOKE / REGRESSION ONLY" — used to verify that core deterministic algorithms produce their frozen baseline scores. This is a **narrower** designation than calibration.

**For repetition-1 to serve as a legitimate independent calibration corpus for a Stage-10 persistence model, the following conditions must all hold:**

1. **The calibration derivation must not use inject_time or root-cause labels.** This audit confirms that `extract_trace_latency_evidence()` operates entirely on raw span timing and requires neither label.  
2. **The calibration derivation must not use Stage-9 outcome data.** Source 1 (rep-1 raw traces) is completely independent of Stage-9 outcomes.  
3. **The re-designation of rep-1 from "smoke-only" to "calibration + smoke" must be formally documented** before any calibration derivation is run. The current `AGENTS.md` methodology requires that such a decision be explicitly recorded.  
4. **The calibration must be pre-specified:** parameters derived from rep-1 must be written down and locked before any Stage-10 treatment run on reps 2+3.

**Methodological concern:** The frozen baseline performance numbers (S_comb Top@1=70%, MRR=0.8011) were measured on rep-1 under oracle window mode. Those metrics do not depend on TCEC or persistence. Using rep-1 traces for calibrating propagation delay does not interfere with those frozen values. There is no contamination risk from using rep-1 trace timing to inform a persistence model, provided the model parameters are derived from spans only and not from anomaly detection or RCA outcomes.

---

## 4. Existing Code Capable of Deriving Causal Delay Without Labels

### `src/digital_detective/traces.py` — `extract_trace_latency_evidence()`

**Verified label-free:** The function accepts a `TelemetryCase` and an optional `max_timestamp`. It does not access `inject_time`, `root_cause_service`, or any Stage-9 outcome field. It operates exclusively on raw span records.

**What it can produce without labels:**

- Per-edge (`caller_service → callee_service`) duration distributions: `DistributionSummary` (min, p10, median, p90, p99, max, mean, count)
- `return_lag_dist`: the time from callee span end to parent span end — a proxy for propagation round-trip delay
- `callee_duration_dist`: raw callee execution time distribution per edge
- `relationship_count`: volume of observed interactions per edge

**Confirmed operational:** Executed on `re2ob_checkoutservice_cpu_1` with no label access and produced full edge distributions for 9 service edges across 391,997 spans, 23,969 traces.

**Candidate persistence model approach (not implemented; documented only):**

A baseline propagation delay envelope per edge could be derived as:

```
persistence_window_for_edge(u, v) = f(callee_duration_p90(u→v) + return_lag_median(u→v))
```

applied across all 30 rep-1 cases and aggregated (e.g., max or 95th percentile across cases) to produce a system-level propagation delay bound. This bound, once pre-specified, could replace the arbitrary 60-second constant in the Stage-9 memory-horizon reset.

**Critical constraint:** Duration units in the raw trace data have been observed but not formally verified in documentation as microseconds vs. milliseconds. `docs/research/rcaeval.md` records `startTime ≈ 1705353846065999` (18 digits, likely microseconds) and `duration ≈ 2301`. This implies durations are in microseconds (~2.3 ms median for `productcatalogservice` calls). The conversion factor must be verified before any persistence window is computed from raw values.

### `src/digital_detective/topology.py` — `extract_trace_dependencies()`

Provides binary edge existence at a given `max_timestamp` but no duration distributions. Cannot alone support a delay calibration. Dependent on `traces.py` outputs.

### No other code in the repository derives propagation delay distributions.

No module in `src/digital_detective/` contains SLO definitions, operational latency targets, circuit-breaker timeouts, or pre-specified delay bounds. The `configs/` directory contains only `eval_smoke_re2_ob.json` and `investigation_default.json` — neither contains timing parameters relevant to cascade propagation.

---

## 5. Leakage Concerns

| Potential source | Independence status | Leakage risk |
|---|---|---|
| Rep-1 raw trace spans (no label) | ✅ Independent of Stage-9 evaluation population | None if inject_time not used |
| Rep-1 raw trace spans (with inject_time for pre-fault windowing) | ✅ Independent (inject_time used for temporal truncation only, not for label) | None — truncation is temporal, not supervised |
| Stage-9 `corroborating_details` timestamps | ❌ Derived from Stage-9 evaluation | **Direct leakage** — must not be used |
| Stage-9 inter-event gap distribution (median=15s, p90=60s) | ❌ Derived from Stage-9 evaluation | **Direct leakage** — must not be used |
| Stage-5 TCEC confirmation latencies | ❌ Derived from Stage-9 evaluation population | **Direct leakage** — must not be used |
| Trace elevation cache | ❌ Stage-9 evaluation population only | **Direct leakage** — must not be used |
| Rep-1 anomaly detection output | ✅ Independent (no Stage-9 cases) | None, but anomaly outcomes are not needed for delay calibration |

---

## 6. Summary of Findings

| Question | Answer |
|---|---|
| Does an independent source of propagation delay information exist? | **Yes** — rep-1 raw trace `duration` and `return_lag` per edge via `extract_trace_latency_evidence()` |
| Which files/modules are involved? | `~/.cache/rcaeval_validation/re2ob_*_1/traces.parquet` (data); `src/digital_detective/traces.py` (code) |
| Is this source independent of the Stage-9 evaluation population? | **Yes** — all Stage-2–9 experiments exclusively used rep=2 and rep=3 |
| Could it defensibly support a pre-specified persistence model? | **Yes, conditionally** — see Section 3 conditions; duration units must be formally verified first |
| Is there any existing tuning-split or calibration corpus formally designated? | **No** — the manifest system supports TUNING_SPLIT but none has been generated; rep-1 is designated SMOKE_REGRESSION only |
| Can rep-1 serve as calibration? | **Yes, conditionally** — it is the only non-evaluation population available; re-designation must be formally documented and pre-specification locked before Stage-10 runs |
| Does any leakage risk exist from the identified source? | **No**, provided inject_time is used only for temporal truncation (pre-fault baseline extraction), not as a label |
| Is there any SLO or operational timing information in the repository? | **No** — no SLOs, circuit-breaker timeouts, or pre-specified delay bounds exist anywhere in the repository |

---

## 7. Recommended Next Scientific Action

**Do not implement Stage-10 yet.**

The following steps are required before Stage-10 implementation is scientifically defensible:

1. **Verify trace duration units.** Confirm that `duration` in `re2ob` traces is microseconds by cross-referencing `startTimeMillis` and `startTime` fields for a known span. Document the verified conversion factor in `docs/research/rcaeval.md`.

2. **Formally re-designate repetition-1 as calibration + smoke.** Update `docs/PROJECT_STATE.md` and `configs/eval_smoke_re2_ob.json` to record that rep-1 serves a dual role: frozen regression smoke **and** an independent calibration corpus for Stage-10 persistence derivation. This designation decision must be recorded before any calibration derivation is run.

3. **Pre-specify the persistence model.** Using only rep-1 trace data and `extract_trace_latency_evidence()` (no labels, no Stage-9 outcomes), derive and **write down** a concrete persistence window value or formula before running any Stage-10 experiment on reps 2+3. The written pre-specification constitutes the scientific commitment.

4. **Only then implement Stage-10** using the pre-specified persistence window as a fixed constant — not a tunable parameter.

---

## Files Inspected

| File | Purpose |
|---|---|
| `eval/manifests/re2_ob_all_cases.json` | Population definition (reps 2+3 only, 60 cases) |
| `eval/manifest.py` | Manifest types, TUNING_SPLIT logic, rep-1 smoke generator |
| `eval/results/stage9_re2ob_sequential_tcec_treatment_v1.json` | Stage-9 per-case evidence timestamps (used to identify leakage risk) |
| `eval/results/.cache_re2ob_trace_elevations.json` | Confirmed rep-1 absence; Stage-9 population only |
| `eval/results/stage2_re2ob_detected_meanstd_baseline_v1.json` | Confirmed reps 2+3 exclusively |
| `eval/results/stage3_oracle_vs_detected_headroom_v1.json` | Schema inspection only |
| `configs/eval_smoke_re2_ob.json` | Rep-1 formal designation (SMOKE / REGRESSION ONLY) |
| `docs/PROJECT_STATE.md` | Frozen baseline designation for rep-1 |
| `docs/research/rcaeval.md` | Trace schema facts; startTime units evidence |
| `src/digital_detective/traces.py` | `extract_trace_latency_evidence()` label-free capability |
| `src/digital_detective/topology.py` | Edge extraction (no duration capability) |
| `~/.cache/rcaeval_validation/re2ob_*_1/` | 30 rep-1 case directories (confirmed present) |
| `~/.cache/rcaeval_validation/re2ob_checkoutservice_cpu_1/traces.parquet` | Sample rep-1 trace data; edge latency distributions extracted |

## Commands Run

```powershell
# Dataset cache inventory
python -c "from pathlib import Path; import os; ..."

# Rep-1 trace data inspection
python -c "import pyarrow.parquet as pq; ..."

# extract_trace_latency_evidence() on rep-1 case (label-free)
python -c "from digital_detective.traces import extract_trace_latency_evidence; ..."

# Stage-9 inter-event gap distribution (leakage identification)
python -c "import json, statistics; ..."

# Manifest and cache cross-reference
python -c "..."
```

## Tests Run

None. This was a read-only audit. No tests were added, modified, or executed.

---

*Audit complete. Stage-10 is not implemented. Stop.*
