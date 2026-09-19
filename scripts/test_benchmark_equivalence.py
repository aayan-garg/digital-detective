"""Compare old reference implementation (calling extract_trace_dependencies repeatedly inside the loop)
vs the optimized implementation (incremental cached G_t timeline) on representative real benchmark cases.

Verifies:
- status
- candidate_onset
- first_evidence_timestamp
- confirmation_time
- confirmed_onset
- tau
- state transitions
- evidence sequence
- direction labels
- corroboration events
Exact equality is strictly required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from digital_detective.bocpd import BOCPDResult
from digital_detective.rcaeval import load_rcaeval_case
from digital_detective.sequential_tcec import (
    CorroboratingEvidence,
    SequentialTCECConfig,
    SequentialTCECResult,
    confirm_sequential_tcec,
)
from digital_detective.topology import extract_trace_dependencies, build_entity_graph
from digital_detective.anomaly import detect_metric_anomalies
from digital_detective.episodes import aggregate_entity_episodes


def confirm_sequential_tcec_REFERENCE(
    case: Any,
    bocpd_result: BOCPDResult,
    config: SequentialTCECConfig | None = None,
    service_aliases: dict[str, str] | None = None,
) -> SequentialTCECResult:
    """The unoptimized reference implementation that repeatedly calls extract_trace_dependencies."""
    case_id = (
        getattr(getattr(case, "metadata", None), "case_id", None)
        or getattr(case, "case_id", "unknown")
    )
    timestamps = bocpd_result.timestamps
    cfg = config or SequentialTCECConfig()

    if bocpd_result.status != "detected" or bocpd_result.onset_ts is None:
        return SequentialTCECResult(
            case_id=case_id,
            candidate_onset=None,
            confirmed_onset=None,
            status="no_detection",
            candidate_entity=None,
            corroborating_entities=(),
            corroborating_details=(),
            confirmation_latency_sec=None,
            state_transitions=(),
            timestamps=timestamps,
            audit={"reason": "no_candidate"},
        )

    t_candidate = int(bocpd_result.onset_ts)
    det_res = detect_metric_anomalies(case)
    ts_list = list(det_res.timestamps)
    ts_to_idx = {int(ts): idx for idx, ts in enumerate(ts_list)}
    num_steps = len(ts_list)

    cand_idx = ts_to_idx.get(t_candidate, 0)

    aliases = service_aliases or {"frontendservice": "frontend"}
    base_graph = build_entity_graph(det_res.metric_names, dependencies=())
    ep_evidence = aggregate_entity_episodes(det_res, base_graph, config=cfg.episode_config)

    state = "NORMAL"
    Z = 0
    candidate_entity: str | None = None
    suspect_set: set[str] = set()
    corroborating_entities: list[str] = []
    corroborating_details: list[CorroboratingEvidence] = []
    last_event_ts = 0
    last_active_ts = 0
    state_transitions: list[dict[str, Any]] = []
    confirmed_onset: int | None = None
    first_evidence_ts: int | None = None

    for step in range(cand_idx, num_steps):
        curr_ts = int(ts_list[step])
        active_ents = {ent for ent, ev in ep_evidence.items() if ev.is_in_episode[step]}

        if state == "NORMAL":
            for ent_name in active_ents:
                ev = ep_evidence[ent_name]
                ep_start = None
                for ep in ev.episodes:
                    if ep.start_idx <= step <= ep.end_idx:
                        ep_start = int(ep.start_timestamp)
                        break

                if ep_start is not None and (ep_start >= t_candidate or ep_start <= t_candidate <= curr_ts):
                    state = "SUSPECT"
                    Z = 1
                    candidate_entity = ent_name
                    suspect_set = {ent_name}
                    corroborating_entities = []
                    corroborating_details = []
                    last_event_ts = curr_ts
                    last_active_ts = curr_ts
                    first_evidence_ts = ep_start
                    state_transitions.append({
                        "timestamp": curr_ts,
                        "step": step,
                        "from_state": "NORMAL",
                        "to_state": "SUSPECT",
                        "trigger_entity": ent_name,
                        "evidence_level": Z,
                    })
                    break

        elif state == "SUSPECT":
            active_suspect = suspect_set & active_ents
            if active_suspect:
                last_active_ts = curr_ts
            else:
                if (curr_ts - last_active_ts) > cfg.memory_window_seconds:
                    state_transitions.append({
                        "timestamp": curr_ts,
                        "step": step,
                        "from_state": "SUSPECT",
                        "to_state": "NORMAL",
                        "reason": f"detector_memory_horizon_elapsed_{curr_ts - last_active_ts}s",
                        "abandoned_suspect_set": list(suspect_set),
                    })
                    state = "NORMAL"
                    Z = 0
                    candidate_entity = None
                    suspect_set = set()
                    corroborating_entities = []
                    corroborating_details = []
                    last_event_ts = 0
                    last_active_ts = 0
                    first_evidence_ts = None
                    continue

            # Reference: repeated extraction
            causal_deps = extract_trace_dependencies(
                case,
                service_aliases=aliases,
                max_timestamp=curr_ts,
            )
            forward_edges: dict[str, set[str]] = {}
            reverse_edges: dict[str, set[str]] = {}
            for d in causal_deps:
                forward_edges.setdefault(d.source, set()).add(d.target)
                reverse_edges.setdefault(d.target, set()).add(d.source)

            for ent_name in active_ents:
                if ent_name in suspect_set:
                    continue
                if curr_ts <= last_event_ts:
                    continue

                connected_to = None
                direction = None
                for s_ent in suspect_set:
                    if ent_name in forward_edges.get(s_ent, ()):
                        connected_to = s_ent
                        direction = "FORWARD"
                        break
                    elif ent_name in reverse_edges.get(s_ent, ()):
                        connected_to = s_ent
                        direction = "REVERSE"
                        break

                if connected_to is not None and direction is not None:
                    suspect_set.add(ent_name)
                    corroborating_entities.append(ent_name)
                    Z += 1
                    last_event_ts = curr_ts
                    corroborating_details.append(
                        CorroboratingEvidence(
                            entity=ent_name,
                            connected_to=connected_to,
                            direction=direction,
                            activation_timestamp=curr_ts,
                            step_index=step,
                        )
                    )
                    state_transitions.append({
                        "timestamp": curr_ts,
                        "step": step,
                        "event": "CORROBORATION",
                        "entity": ent_name,
                        "connected_to": connected_to,
                        "direction": direction,
                        "evidence_level": Z,
                    })
                    break

            if Z >= cfg.stopping_threshold and active_suspect:
                state = "CONFIRMED"
                confirmed_onset = curr_ts
                state_transitions.append({
                    "timestamp": curr_ts,
                    "step": step,
                    "from_state": "SUSPECT",
                    "to_state": "CONFIRMED",
                    "final_evidence_level": Z,
                    "suspect_entities": list(suspect_set),
                    "corroborating_entities": list(corroborating_entities),
                })
                break

    status = "confirmed" if confirmed_onset is not None else "unconfirmed"
    confirmation_time = confirmed_onset
    latency = (confirmation_time - t_candidate) if confirmation_time is not None else None

    return SequentialTCECResult(
        case_id=case_id,
        candidate_onset=t_candidate,
        confirmed_onset=confirmed_onset,
        status=status,
        candidate_entity=candidate_entity,
        corroborating_entities=tuple(corroborating_entities),
        corroborating_details=tuple(corroborating_details),
        confirmation_latency_sec=latency,
        state_transitions=tuple(state_transitions),
        timestamps=timestamps,
        audit={},
        confirmation_time=confirmation_time,
        first_evidence_timestamp=first_evidence_ts,
    )


def main() -> None:
    dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
    repo_root = Path(__file__).resolve().parent.parent

    # Test on representative real benchmark cases from manifest
    test_cases = [
        ("re2ob_checkoutservice_cpu_2", 1705354562),
        ("re2ob_checkoutservice_cpu_3", 1705354562),
        ("re2ob_checkoutservice_delay_2", 1705367746),
    ]

    print("=" * 80)
    print("TESTING FULL SEQUENTIAL TCEC REFERENCE VS OPTIMIZED EQUIVALENCE")
    print("=" * 80)

    for cid, cand_ts in test_cases:
        print(f"\nEvaluating Case: {cid} (candidate_onset={cand_ts})")
        case = load_rcaeval_case(dataset_root, cid)
        bocpd = BOCPDResult(
            case_id=cid,
            onset_ts=cand_ts,
            status="detected",
            audit={},
            changepoints=(),
            timestamps=(),
        )

        t0 = time.perf_counter()
        ref_res = confirm_sequential_tcec_REFERENCE(case, bocpd)
        t_ref = time.perf_counter() - t0

        t1 = time.perf_counter()
        opt_res = confirm_sequential_tcec(case, bocpd)
        t_opt = time.perf_counter() - t1

        print(f"  Reference Runtime: {t_ref:.3f}s")
        print(f"  Optimized Runtime: {t_opt:.3f}s")
        print(f"  Speedup:           {t_ref / t_opt:.1f}x")

        # Rigorous Equivalence Assertions:
        assert ref_res.status == opt_res.status, f"Status mismatch: {ref_res.status} vs {opt_res.status}"
        assert ref_res.candidate_onset == opt_res.candidate_onset, f"candidate_onset mismatch"
        assert ref_res.confirmed_onset == opt_res.confirmed_onset, f"confirmed_onset mismatch: {ref_res.confirmed_onset} vs {opt_res.confirmed_onset}"
        assert ref_res.confirmation_time == opt_res.confirmation_time, f"confirmation_time mismatch"
        assert ref_res.first_evidence_timestamp == opt_res.first_evidence_timestamp, f"first_evidence_timestamp mismatch"
        assert ref_res.candidate_entity == opt_res.candidate_entity, f"candidate_entity mismatch: {ref_res.candidate_entity} vs {opt_res.candidate_entity}"
        assert ref_res.corroborating_entities == opt_res.corroborating_entities, f"corroborating_entities mismatch: {ref_res.corroborating_entities} vs {opt_res.corroborating_entities}"
        assert ref_res.confirmation_latency_sec == opt_res.confirmation_latency_sec, f"confirmation_latency_sec mismatch"
        assert len(ref_res.state_transitions) == len(opt_res.state_transitions), f"state_transitions length mismatch"

        for i, (st_ref, st_opt) in enumerate(zip(ref_res.state_transitions, opt_res.state_transitions)):
            assert st_ref == st_opt, f"Transition {i} mismatch: {st_ref} vs {st_opt}"

        assert len(ref_res.corroborating_details) == len(opt_res.corroborating_details), f"corroborating_details length mismatch"
        for i, (cd_ref, cd_opt) in enumerate(zip(ref_res.corroborating_details, opt_res.corroborating_details)):
            assert cd_ref == cd_opt, f"CorroboratingDetail {i} mismatch: {cd_ref} vs {cd_opt}"

        print("  -> EXACT EQUIVALENCE VERIFIED ON ALL FIELDS & TRANSITIONS!")


if __name__ == "__main__":
    main()
