"""Structural causal residual feature extraction for microservice root cause analysis.

Fits regularized regressors on a healthy pre-incident baseline window using observed
topological parent signals (callers in the service call graph) to predict each metric.
Standardized residuals:
    z_i(t) = |X_i(t) - f_hat(PA_i(t))| / (sigma_hat_epsilon + eps)
are computed strictly over the post-onset window without future data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

from digital_detective.telemetry import TelemetryCase
from digital_detective.topology import EntityGraph, parse_rcaeval_metric_identifier


@dataclass(frozen=True)
class EntityCausalResiduals:
    entity: str
    mean_residual_z: float
    max_residual_z: float
    early_residual_ratio: float
    parent_count: int
    metric_residuals: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalResidualResult:
    case_id: str
    baseline_window_length: int
    incident_window_length: int
    entity_residuals: Mapping[str, EntityCausalResiduals]
    assumptions_note: str = (
        "Observational residuals quantify discrepancy from topological caller expectation; "
        "they do not formally prove unconfounded counterfactual causality."
    )

    @property
    def total_metrics(self) -> int:
        return sum(len(res.metric_residuals) for res in self.entity_residuals.values())


def extract_causal_residuals(
    case: TelemetryCase,
    graph: EntityGraph | None = None,
    candidate_universe: Sequence[str] = (),
    *,
    baseline_steps: int = 120,
    onset_timestamp: int | None = None,
    ridge_lambda: float = 1.0,
    epsilon: float = 1e-6,
) -> CausalResidualResult:
    """Compute standardized causal residuals for each candidate service.

    Parameters
    ----------
    case:
        TelemetryCase containing raw metric series in `case.metrics`.
    graph:
        EntityGraph containing directional service call dependencies (source -> target).
    candidate_universe:
        Universe of service entities to evaluate.
    baseline_steps:
        Number of initial time steps treated as healthy baseline if onset_timestamp not given.
    onset_timestamp:
        Timestamp of detected change-point / onset. Baseline is all steps prior to onset.
    ridge_lambda:
        L2 regularization factor for parent metric regression.
    epsilon:
        Numerical stability constant for standard deviation division.
    """
    case_id = case.metadata.case_id if case.metadata else "unknown"
    if case.metrics is None or case.metrics.raw_data is None:
        empty_res = {
            ent: EntityCausalResiduals(
                entity=ent,
                mean_residual_z=0.0,
                max_residual_z=0.0,
                early_residual_ratio=0.0,
                parent_count=0,
            )
            for ent in candidate_universe
        }
        return CausalResidualResult(case_id, 0, 0, empty_res)

    table = case.metrics.raw_data
    ts_col = "time" if "time" in table.column_names else ("timestamp" if "timestamp" in table.column_names else None)
    if ts_col is None:
        return CausalResidualResult(case_id, 0, 0, empty_res)
    timestamps = table[ts_col].to_numpy()
    total_len = len(timestamps)

    # Determine split boundary between baseline (pre-incident) and evaluation (post-onset)
    if onset_timestamp is not None:
        indices = np.where(timestamps < onset_timestamp)[0]
        split_idx = int(indices[-1] + 1) if len(indices) > 0 else min(baseline_steps, total_len)
    else:
        split_idx = min(baseline_steps, max(10, total_len // 2))

    split_idx = max(5, min(split_idx, total_len - 1))
    base_len = split_idx
    eval_len = total_len - split_idx

    # Group metrics by entity
    entity_metrics: dict[str, list[tuple[str, np.ndarray]]] = {}
    metric_cols = [c for c in table.column_names if c not in ("time", "timestamp")]

    for col in metric_cols:
        parsed = parse_rcaeval_metric_identifier(col)
        arr = table[col].to_numpy().astype(float)
        # Clean NaNs in-place with forward fill / mean
        mask = np.isnan(arr)
        if np.any(mask):
            mean_val = np.nanmean(arr) if not np.all(mask) else 0.0
            arr[mask] = mean_val
        entity_metrics.setdefault(parsed.entity, []).append((col, arr))

    # Identify candidate universe
    entities = list(candidate_universe) if candidate_universe else list(entity_metrics.keys())

    # Build caller map from graph: callers of entity E are sources where source -> E
    callers_map: dict[str, list[str]] = {}
    if graph is not None:
        for dep in graph.dependencies:
            # dep.source calls dep.target. Target's parent is dep.source
            callers_map.setdefault(dep.target, []).append(dep.source)

    entity_results: dict[str, EntityCausalResiduals] = {}

    for ent in entities:
        metrics = entity_metrics.get(ent, [])
        if not metrics:
            entity_results[ent] = EntityCausalResiduals(
                entity=ent,
                mean_residual_z=0.0,
                max_residual_z=0.0,
                early_residual_ratio=0.0,
                parent_count=0,
            )
            continue

        parents = callers_map.get(ent, [])
        parent_series: list[np.ndarray] = []
        for p in parents:
            if p in entity_metrics and p != ent:
                # aggregate parent's metrics as predictor signal
                p_arrs = [arr for _, arr in entity_metrics[p]]
                if p_arrs:
                    parent_series.append(np.mean(p_arrs, axis=0))

        has_parents = len(parent_series) > 0
        if has_parents:
            X_all = np.column_stack(parent_series)  # shape: (T, P)
            X_base = X_all[:split_idx, :]
            X_eval = X_all[split_idx:, :]
            # Standardize X using baseline statistics only
            x_mean = np.mean(X_base, axis=0)
            x_std = np.std(X_base, axis=0) + epsilon
            X_base_norm = (X_base - x_mean) / x_std
            X_eval_norm = (X_eval - x_mean) / x_std
            # Add bias column
            X_base_design = np.column_stack([np.ones(split_idx), X_base_norm])
            X_eval_design = np.column_stack([np.ones(eval_len), X_eval_norm])
            # Closed-form Ridge: beta = (X^T X + lambda I)^(-1) X^T y
            p_dim = X_base_design.shape[1]
            reg_matrix = X_base_design.T @ X_base_design + ridge_lambda * np.eye(p_dim)
            reg_matrix[0, 0] -= ridge_lambda  # don't regularize intercept

        metric_z_scores: dict[str, float] = {}
        all_eval_z: list[np.ndarray] = []

        for m_name, series in metrics:
            y_base = series[:split_idx]
            y_eval = series[split_idx:]

            if has_parents:
                try:
                    beta = np.linalg.solve(reg_matrix, X_base_design.T @ y_base)
                    y_base_pred = X_base_design @ beta
                    y_eval_pred = X_eval_design @ beta
                except np.linalg.LinAlgError:
                    y_base_pred = np.full_like(y_base, np.mean(y_base))
                    y_eval_pred = np.full_like(y_eval, np.mean(y_base))
            else:
                # Baseline mean model
                y_base_pred = np.full_like(y_base, np.mean(y_base))
                y_eval_pred = np.full_like(y_eval, np.mean(y_base))

            residuals_base = y_base - y_base_pred
            sigma_eps = float(np.std(residuals_base) + epsilon)

            residuals_eval = np.abs(y_eval - y_eval_pred)
            z_eval = residuals_eval / sigma_eps
            all_eval_z.append(z_eval)
            metric_z_scores[m_name] = float(np.mean(z_eval))

        if all_eval_z:
            stacked_z = np.vstack(all_eval_z)  # (M, eval_len)
            mean_z = float(np.mean(stacked_z))
            max_z = float(np.max(stacked_z))
            # Early residual: fraction of residual occurring in first 40% of incident window
            early_cutoff = max(1, int(0.4 * eval_len))
            early_sum = np.sum(stacked_z[:, :early_cutoff])
            total_sum = np.sum(stacked_z) + epsilon
            early_ratio = float(early_sum / total_sum)
        else:
            mean_z, max_z, early_ratio = 0.0, 0.0, 0.0

        entity_results[ent] = EntityCausalResiduals(
            entity=ent,
            mean_residual_z=mean_z,
            max_residual_z=max_z,
            early_residual_ratio=early_ratio,
            parent_count=len(parent_series),
            metric_residuals=metric_z_scores,
        )

    return CausalResidualResult(
        case_id=case_id,
        baseline_window_length=base_len,
        incident_window_length=eval_len,
        entity_residuals=entity_results,
    )
