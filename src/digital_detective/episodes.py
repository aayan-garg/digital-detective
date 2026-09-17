"""Entity Anomaly Episode construction from metric-level anomaly evidence.

Provides deterministic, causality-preserving aggregation of metric anomalies into
sustained, multi-metric Entity Anomaly Episodes.

An entity anomaly episode becomes active at observation step t if and only if:
    at least M distinct metrics belonging to the entity
    are concurrently persistently anomalous (streak >= K consecutive evaluated observations).

The episode builder operates purely downstream on immutable MetricAnomalyResult
and EntityGraph. It has no access to ground truth, injection times, or fault labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .anomaly import MetricAnomalyResult
from .topology import EntityGraph


@dataclass(frozen=True)
class EpisodeConfig:
    """Configuration defining persistence and consensus criteria for Entity Anomaly Episodes."""

    persistence: int = 3  # K consecutive observations per metric
    consensus: int = 2  # M concurrent persistent metrics on the entity

    def __post_init__(self) -> None:
        if self.persistence < 1:
            raise ValueError(f"persistence must be >= 1, got {self.persistence}")
        if self.consensus < 1:
            raise ValueError(f"consensus must be >= 1, got {self.consensus}")


@dataclass(frozen=True)
class EpisodeInterval:
    """A maximal contiguous interval where an entity was in an anomalous episode."""

    start_idx: int
    end_idx: int
    start_timestamp: Any
    end_timestamp: Any
    duration_seconds: int
    peak_active_metrics: int
    contributing_metrics: tuple[str, ...]


@dataclass(frozen=True)
class EntityEpisodeEvidence:
    """Immutable entity-level episode evidence aggregated from MetricAnomalyResult."""

    case_id: str
    entity: str
    timestamps: tuple[Any, ...]
    active_metric_counts: tuple[int, ...]
    is_in_episode: tuple[bool, ...]
    episodes: tuple[EpisodeInterval, ...]
    first_episode_start_ts: int | None
    peak_active_metrics: int
    all_contributing_metrics: tuple[str, ...]
    has_episode: bool


def aggregate_entity_episodes(
    anomaly_result: MetricAnomalyResult,
    graph: EntityGraph,
    config: EpisodeConfig | None = None,
) -> Mapping[str, EntityEpisodeEvidence]:
    """Aggregate metric-level anomalies into Entity Anomaly Episodes.

    Operates purely downstream on MetricAnomalyResult and EntityGraph without
    inspecting future observations or accessing ground-truth labels.

    Parameters
    ----------
    anomaly_result:
        Immutable result of metric anomaly detection for one case.
    graph:
        EntityGraph defining entities, owned metrics, and topology.
    config:
        EpisodeConfig defining persistence K and consensus M.
        Defaults to persistence=3, consensus=2.

    Returns
    -------
    Mapping[str, EntityEpisodeEvidence]
        Per-entity episode evidence.
    """
    if config is None:
        config = EpisodeConfig()

    eff_persistence = config.persistence
    eff_consensus = config.consensus

    timestamps = anomaly_result.timestamps
    num_steps = len(timestamps)
    available_metrics = set(anomaly_result.metric_names)

    evidence_by_entity: dict[str, EntityEpisodeEvidence] = {}

    for entity_name, entity_node in graph.entities.items():
        entity_metrics = [m for m in entity_node.metrics if m in available_metrics]

        # 1. Track causal consecutive anomaly streaks for each owned metric
        streaks: dict[str, list[int]] = {}
        for m in entity_metrics:
            statuses = anomaly_result.evaluation_statuses[m]
            m_streak = [0] * num_steps
            curr = 0
            for idx in range(num_steps):
                if statuses[idx] == "anomaly":
                    curr += 1
                else:
                    curr = 0
                m_streak[idx] = curr
            streaks[m] = m_streak

        # 2. Compute persistent active metric counts and track contributing metrics per step
        active_counts = [0] * num_steps
        step_persistent_metrics: list[list[str]] = [[] for _ in range(num_steps)]

        for m in entity_metrics:
            m_streak = streaks[m]
            for idx in range(num_steps):
                if m_streak[idx] >= eff_persistence:
                    active_counts[idx] += 1
                    step_persistent_metrics[idx].append(m)

        # 3. Episode condition: >= eff_consensus persistent anomalous metrics
        # Literal consensus: if an entity has fewer than M metrics, condition cannot be satisfied.
        is_in_ep = tuple(c >= eff_consensus for c in active_counts)

        # 4. Extract maximal contiguous episode intervals
        episodes: list[EpisodeInterval] = []
        in_ep = False
        ep_start_idx = 0

        for idx in range(num_steps):
            if is_in_ep[idx]:
                if not in_ep:
                    in_ep = True
                    ep_start_idx = idx
            else:
                if in_ep:
                    in_ep = False
                    ep_end_idx = idx - 1
                    start_ts = timestamps[ep_start_idx]
                    end_ts = timestamps[ep_end_idx]

                    contrib: set[str] = set()
                    peak_m = 0
                    for s in range(ep_start_idx, ep_end_idx + 1):
                        contrib.update(step_persistent_metrics[s])
                        if active_counts[s] > peak_m:
                            peak_m = active_counts[s]

                    dur = int(end_ts - start_ts) if isinstance(start_ts, (int, float)) and isinstance(end_ts, (int, float)) else 0
                    episodes.append(
                        EpisodeInterval(
                            start_idx=ep_start_idx,
                            end_idx=ep_end_idx,
                            start_timestamp=start_ts,
                            end_timestamp=end_ts,
                            duration_seconds=dur,
                            peak_active_metrics=peak_m,
                            contributing_metrics=tuple(sorted(contrib)),
                        )
                    )

        if in_ep:
            ep_end_idx = num_steps - 1
            start_ts = timestamps[ep_start_idx]
            end_ts = timestamps[ep_end_idx]

            contrib = set()
            peak_m = 0
            for s in range(ep_start_idx, ep_end_idx + 1):
                contrib.update(step_persistent_metrics[s])
                if active_counts[s] > peak_m:
                    peak_m = active_counts[s]

            dur = int(end_ts - start_ts) if isinstance(start_ts, (int, float)) and isinstance(end_ts, (int, float)) else 0
            episodes.append(
                EpisodeInterval(
                    start_idx=ep_start_idx,
                    end_idx=ep_end_idx,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    duration_seconds=dur,
                    peak_active_metrics=peak_m,
                    contributing_metrics=tuple(sorted(contrib)),
                )
            )

        # 5. Entity-level episode evidence summary
        has_ep = len(episodes) > 0
        first_ts = int(episodes[0].start_timestamp) if has_ep and isinstance(episodes[0].start_timestamp, (int, float)) else None
        peak_active = max(active_counts) if active_counts else 0
        all_contrib = tuple(sorted(set().union(*(ep.contributing_metrics for ep in episodes)))) if has_ep else ()

        evidence_by_entity[entity_name] = EntityEpisodeEvidence(
            case_id=anomaly_result.case_id,
            entity=entity_name,
            timestamps=timestamps,
            active_metric_counts=tuple(active_counts),
            is_in_episode=is_in_ep,
            episodes=tuple(episodes),
            first_episode_start_ts=first_ts,
            peak_active_metrics=peak_active,
            all_contributing_metrics=all_contrib,
            has_episode=has_ep,
        )

    return evidence_by_entity


def truncate_entity_episodes(
    ep_evidence: Mapping[str, EntityEpisodeEvidence],
    max_timestamp: Any,
) -> dict[str, EntityEpisodeEvidence]:
    """Causally truncate EntityEpisodeEvidence at max_timestamp.

    Preserves observations with timestamp <= max_timestamp, discarding
    any episodes or parts of episodes strictly after max_timestamp.
    """
    truncated: dict[str, EntityEpisodeEvidence] = {}
    for entity_name, ev in ep_evidence.items():
        if not ev.timestamps:
            truncated[entity_name] = ev
            continue

        cutoff_idx = -1
        for idx, ts in enumerate(ev.timestamps):
            if ts <= max_timestamp:
                cutoff_idx = idx
            else:
                break

        n = cutoff_idx + 1
        new_timestamps = ev.timestamps[:n]
        new_active_counts = ev.active_metric_counts[:n]
        new_is_in_ep = ev.is_in_episode[:n]

        new_episodes: list[EpisodeInterval] = []
        for ep in ev.episodes:
            if ep.start_timestamp > max_timestamp:
                continue
            if ep.end_timestamp <= max_timestamp:
                new_episodes.append(ep)
            else:
                clamped_end_idx = n - 1
                clamped_end_ts = new_timestamps[-1] if new_timestamps else max_timestamp
                clamped_dur = (
                    int(clamped_end_ts - ep.start_timestamp)
                    if isinstance(clamped_end_ts, (int, float)) and isinstance(ep.start_timestamp, (int, float))
                    else 0
                )
                peak_m = max(new_active_counts[ep.start_idx : clamped_end_idx + 1]) if new_active_counts else 0
                new_episodes.append(
                    EpisodeInterval(
                        start_idx=ep.start_idx,
                        end_idx=clamped_end_idx,
                        start_timestamp=ep.start_timestamp,
                        end_timestamp=clamped_end_ts,
                        duration_seconds=clamped_dur,
                        peak_active_metrics=peak_m,
                        contributing_metrics=ep.contributing_metrics,
                    )
                )

        has_ep = len(new_episodes) > 0
        first_ts = (
            int(new_episodes[0].start_timestamp)
            if has_ep and isinstance(new_episodes[0].start_timestamp, (int, float))
            else None
        )
        peak_active = max(new_active_counts) if new_active_counts else 0
        all_contrib = tuple(sorted(set().union(*(ep.contributing_metrics for ep in new_episodes)))) if has_ep else ()

        truncated[entity_name] = EntityEpisodeEvidence(
            case_id=ev.case_id,
            entity=ev.entity,
            timestamps=new_timestamps,
            active_metric_counts=new_active_counts,
            is_in_episode=new_is_in_ep,
            episodes=tuple(new_episodes),
            first_episode_start_ts=first_ts,
            peak_active_metrics=peak_active,
            all_contributing_metrics=all_contrib,
            has_episode=has_ep,
        )

    return truncated
