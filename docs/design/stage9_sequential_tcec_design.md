# Stage 9 Design Document: Sequential Topology-Coherent Episode Confirmation (Sequential TCEC)

**Experiment ID:** Stage 9  
**Target Hypothesis:** Replacing TCEC's instantaneous single-pair Boolean confirmation rule with a sequential evidence accumulation process across discrete temporal events—bounded by the detector's physical memory horizon—will eliminate premature false-early confirmations while preserving realistic detection latency and downstream RCA accuracy.  
**Control Baseline:** Frozen Stage 5 TCEC (`eval/results/stage5_tcec_treatment_v1.json`, commit `08c456584adeb105dc1fe513fb8f4f232ef37070`).  
**Status:** Revised Predeclared Treatment Specification  

---

## 1. Diagnosis: Why the Initial Stage 9 Design Was Insufficient

The initial Stage 9 implementation operated as:
$$\text{Corroborating entities count} \ge 2 \implies \text{CONFIRMED}$$
This was diagnosed as scientifically insufficient for three distinct reasons:
1. **Spatial Breadth vs. Sequential Accumulation:** Counting unique entities alone does not require sequential accumulation across time. If two connected entities experienced metric jitter at the exact same second, the count jumped from 0 to 2 simultaneously. This was merely a spatial breadth constraint ($|\text{entities}| \ge 3$), not genuine temporal evidence accumulation.
2. **Arbitrary Parameters:** It introduced two ungrounded parameters: `min_corroborating_entities = 2` and `stale_timeout_steps = 15`. Neither had an operational or physical grounding in the detector or telemetry architecture.
3. **Loss of Propagation Dynamics:** It failed to represent the physical unfolding of an incident over distinct observation epochs.

---

## 2. Fundamental Contrast: Stage 5 vs. Stage 9

### What Stage 5 Does (Instantaneous Boolean Trigger with Infinite Memory)
```text
BOCPD candidate at t_cand
    ↓
Entity A episode active at t_cand (even if duration = 1s, then dead)
    ↓
Entity B episode active at t > t_cand (even hundreds of seconds later)
    ↓
Connected in graph?
    ↓
CONFIRMED INSTANTANEOUSLY (Z = 2)
```
* **Failure mode:** Transient baseline noise on A and B separated by hundreds of seconds was permanently glued together because Stage 5 had infinite memory and zero temporal accumulation requirements. This produced an **86.67% false-early rate**.

### What Stage 9 Does (Temporal Corroboration Sequence with Detector Memory Horizon)
```text
t1: BOCPD candidate + active episode on E_cand
    → State = SUSPECT, Z = 1, S = {E_cand}
    ↓
t2 > t1: Adjacent entity E1 activates episode in causal topology G_t2
    → Evidence advances: Z = 2, S = {E_cand, E1}
    → Direction recorded (FORWARD or REVERSE)
    ↓
t3 > t2: Independent corroboration event arrives at a strictly later time
    → Evidence advances: Z = 3
    ↓
Stopping Condition: Z >= 3 AND suspect component actively anomalous
    → State = CONFIRMED, confirmed_onset = t3
    ↓
Hypothesis Reset Rule:
    If all entities in S become inactive for > W observations (W = 60s detector window),
    hypothesis resets to NORMAL (Z = 0, S = empty).
```

---

## 3. Mathematical Formulation

Let $\mathcal{T} = (t_0, t_1, \dots, t_N)$ be the discrete chronological observation timestamps.
Let $\mathcal{U}$ be the candidate entity universe ($|\mathcal{U}| = 11$ for Online Boutique).
At each observation step $t$, the causal topology $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$ is strictly bounded by the current timestamp:
$$\mathcal{E}_t = \text{extract\_trace\_dependencies}(\text{case}, \text{max\_timestamp}=t)$$

### 3.1 State Representation
$$S(t) \in \{\text{NORMAL}, \text{SUSPECT}, \text{CONFIRMED}\}$$
Initial state at $t = t_{\text{candidate}}$: $S(t) = \text{NORMAL}$.

### 3.2 Suspect Trigger ($\text{NORMAL} \to \text{SUSPECT}$)
At step $t \ge t_{\text{candidate}}$:
If an entity $E_{\text{cand}}$ satisfies `is_in_episode[t] == True` (persistence $K=3$, consensus $M=2$) and its episode start timestamp $t_{\text{cand\_start}} \ge t_{\text{candidate}}$ (or spans $t_{\text{candidate}}$):
$$S(t) = \text{SUSPECT}$$
$$\mathcal{S}_t = \{E_{\text{cand}}\}, \quad Z_t = 1, \quad t_{\text{last\_event}} = t, \quad t_{\text{last\_active}} = t$$

### 3.3 Sequential Evidence Update Equation
At each subsequent observation step $t > t_{\text{suspect}}$ while $S(t) = \text{SUSPECT}$:

Let $\mathcal{A}_t = \{e \in \mathcal{S}_{t-1} \mid \text{is\_in\_episode}_e[t] == \text{True}\}$ be the set of suspect entities currently active at step $t$.

1. **Activity Maintenance:**
   If $\mathcal{A}_t \ne \emptyset$:
   $$t_{\text{last\_active}} = t$$

2. **Detector-Horizon Reset Rule ($\text{SUSPECT} \to \text{NORMAL}$):**
   If no entity in the suspect hypothesis has been active for longer than the detector's rolling memory window $W = 60$ seconds:
   $$t - t_{\text{last\_active}} > W \implies S(t) = \text{NORMAL}, \quad \mathcal{S}_t = \emptyset, \quad Z_t = 0$$
   *Physical Meaning:* The rolling mean/std detector operates with window size $W = 60$. If 60 seconds elapse with zero anomaly observations, the entire rolling window has returned to baseline statistics. The hypothesis has completely dissolved.

3. **Temporal Corroboration Event ($Z_t = Z_{t-1} + 1$):**
   An independent corroboration event occurs at step $t$ if there exists an entity $E_{\text{new}} \in \mathcal{U} \setminus \mathcal{S}_{t-1}$ such that:
   - $E_{\text{new}}$ is currently active: `is_in_episode[t] == True`
   - $E_{\text{new}}$ is connected in $\mathcal{E}_t$ to an already-confirmed suspect entity $E_{\text{src}} \in \mathcal{S}_{t-1}$:
     $$(E_{\text{src}}, E_{\text{new}}) \in \mathcal{E}_t \quad (\text{FORWARD: caller}\to\text{callee}) \quad \text{or} \quad (E_{\text{new}}, E_{\text{src}}) \in \mathcal{E}_t \quad (\text{REVERSE: callee}\to\text{caller})$$
   - Temporal precedence: $t > t_{\text{last\_event}}$ (strictly later observation step).

   When this event occurs:
   $$\mathcal{S}_t = \mathcal{S}_{t-1} \cup \{E_{\text{new}}\}$$
   $$Z_t = Z_{t-1} + 1$$
   $$t_{\text{last\_event}} = t$$

### 3.4 Predeclared Stopping Rule ($\text{SUSPECT} \to \text{CONFIRMED}$)
Confirmation occurs at the exact first observation timestamp $t$ where the sequential stopping criterion is satisfied:
$$\tau = \inf \{ t \ge t_{\text{candidate}} : Z_t \ge 3 \quad \text{AND} \quad \mathcal{A}_t \ne \emptyset \}$$

When satisfied:
$$S(\tau) = \text{CONFIRMED}, \quad \text{confirmation\_time} = \tau, \quad \text{confirmed\_onset} = \tau$$

Evaluation terminates immediately.

> [!IMPORTANT]
> **Strict Non-Backdating Requirement:**
> $$\text{confirmed\_onset} \equiv \text{confirmation\_time} \equiv \tau$$
> The operational confirmation timestamp is the exact stopping time $\tau$ at which sequential evidence became sufficient to confirm the incident. It is NOT backdated to the earliest evidence timestamp (e.g. $t_1$). Earlier timestamps ($t_{\text{candidate}}$, $t_{\text{first\_evidence}}$) are preserved strictly as historical audit metadata.

---

## 4. Predeclared Structural Parameter $k_{\text{req}} = 3$

In this sequential evidence architecture:
* $Z = 1$: Initial suspect hypothesis is established ($E_{\text{cand}}$) following a BOCPD candidate changepoint. No propagation has been observed.
* $Z = 2$: A single adjacent entity has activated ($E_{\text{cand}} \to E_1$). In noisy distributed systems, an isolated 2-node coincidence is common during baseline operations.
* $Z = 3$: A **second sequential corroboration event** is observed at $t_3 > t_2 > t_1$.

$k_{\text{req}} = 3$ is classified as a **predeclared structural treatment parameter**. It is NOT described as statistically optimal, maximum-likelihood derived, or Bayes-optimal; rather, it is the minimal discrete sequence length that enforces genuine multi-event corroboration across time rather than a single pairwise event blip.

---

## 5. Parameter Audit

Every parameter belongs to an explicit governance category:

| Parameter | Value | Governance Category | Origin & Scientific Justification |
| :--- | :---: | :---: | :--- |
| `persistence` ($K$) | 3 | **Inherited frozen control** | Inherited from Stage 5 `EpisodeConfig`. Requires 3 consecutive anomalous evaluations per metric streak. |
| `consensus` ($M$) | 2 | **Inherited frozen control** | Inherited from Stage 5 `EpisodeConfig`. Requires 2 concurrent persistent metrics per entity. |
| `window_size` ($W$) | 60 | **Inherited frozen control** | Inherited directly from `detect_metric_anomalies` rolling window. If $W$ seconds pass with zero anomalies, the rolling statistics have completely returned to normal. |
| `anomaly_threshold` | 3.0 | **Inherited frozen control** | Inherited from Stage 2/4/5 rolling normalization. |
| `causal_max_timestamp` | $t$ | **Filtration requirement** | Bounded strictly by current observation step $t$ to ensure non-anticipative causality. |
| `stopping_threshold` ($k_{\text{req}}$) | 3 | **Predeclared structural treatment parameter** | Minimal discrete sequence length enforcing multi-event temporal corroboration ($t_1 \to t_2 \to t_3$) beyond an isolated pairwise coincidence. |

---

## 6. Causality & Stopping Time Proof

1. **Adapted Filtration $\mathcal{F}_t$:** At observation step $t$, the decision function depends solely on $\mathcal{X}_{0:t}$ (metrics up to $t$) and $\text{Traces}_{0:t}$ (spans completing $\le t$).
2. **Online Episode Status:** `is_in_episode[t]` is evaluated online using running streaks up to step $t$. Future episode terminations (`end_timestamp`) are never inspected.
3. **Causal Trace Dependencies:** Trace dependencies are extracted with `max_timestamp = curr_ts`. Future spans cannot influence topological connectivity.
4. **Stopping Time:** The event $\{\tau \le t\}$ is strictly $\mathcal{F}_t$-measurable. $\tau$ is a mathematically valid causal stopping time.
5. **Operational Confirmation:**
   $$\text{confirmed\_onset} = \tau$$
   Because $\tau$ is a valid stopping time and $\text{confirmed\_onset} = \tau$, the downstream RCA boundary $\text{analysis\_end} = \text{confirmed\_onset}$ is also strictly non-anticipative and does not use future information.

---

## 7. Downstream RCA Boundary

Stage 9 preserves the exact Stage-5 causal RCA isolation contract:
$$\text{analysis\_end} = \text{confirmed\_onset} = \tau$$
Telemetry, anomaly scores, trace graphs, and trace latency distributions are strictly truncated at $t \le \text{confirmed\_onset}$.
No information past `confirmed_onset` enters $S_{\text{comb}}$, $E_{\text{elev}}$, or fusion.
