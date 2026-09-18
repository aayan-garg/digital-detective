"""Causal Bayesian Online Change Point Detection (BOCPD) for incident-onset detection.

Methodological references:
- Adams & MacKay (2007): "Bayesian Online Changepoint Detection", University of Cambridge.
- Pham, Ha, Zhang (FSE 2024): "BARO: Robust Root Cause Analysis for Microservices via
  Multivariate Bayesian Online Change Point Detection".

Strict Causality Rules:
- At discrete observation step t, evaluations depend exclusively on observations
  X_0 ... X_t and never X_{t+1} ... X_end.
- Standardization and baseline prior parameters are initialized strictly from the
  initial warmup window [0, min_warmup - 1] (default 60 observations).
- No lookahead min-max normalization or global centering is performed.
- No ground-truth labels, injection timestamps, or fault types are ever accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import scipy.special as sc

from .anomaly import _extract_series
from .telemetry import TelemetryCase


@dataclass(frozen=True)
class BOCPDConfig:
    """Configuration for causal Bayesian Online Changepoint Detection."""

    hazard_lambda: float = 100.0
    min_warmup: int = 60
    threshold: float = 0.5
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.hazard_lambda <= 0:
            raise ValueError(f"hazard_lambda must be positive, got {self.hazard_lambda}")
        if self.min_warmup < 2:
            raise ValueError(f"min_warmup must be at least 2, got {self.min_warmup}")
        if not (0.0 < self.threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")


@dataclass(frozen=True)
class BOCPDResult:
    """Immutable result of BOCPD incident onset detection for one case."""

    case_id: str
    onset_ts: int | None
    status: str
    audit: Mapping[str, Any]
    changepoints: tuple[int, ...]
    timestamps: tuple[Any, ...]


class CausalBOCPD:
    """Causal Bayesian Online Changepoint Detector with conjugate Normal-Inverse-Gamma prior.

    Maintains the posterior distribution over run lengths r_t in {0, ..., t} recursively
    using exact online message-passing:
      P(r_t = r_{t-1} + 1, x_{1:t}) = P(r_{t-1}, x_{1:t-1}) * p(x_t | r_{t-1}) * (1 - H)
      P(r_t = 0, x_{1:t})           = sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1}) * p(x_t | r_{t-1}) * H
    where H = 1 / hazard_lambda is the constant hazard rate.
    """

    def __init__(
        self,
        *,
        hazard_lambda: float = 100.0,
        min_warmup: int = 60,
        threshold: float = 0.5,
        epsilon: float = 1e-6,
    ) -> None:
        if hazard_lambda <= 0:
            raise ValueError(f"hazard_lambda must be positive, got {hazard_lambda}")
        if min_warmup < 2:
            raise ValueError(f"min_warmup must be at least 2, got {min_warmup}")
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.hazard_lambda = hazard_lambda
        self.min_warmup = min_warmup
        self.threshold = threshold
        self.epsilon = epsilon
        self.hazard = 1.0 / hazard_lambda
        self.log_hazard = float(np.log(self.hazard))
        self.log_1m_hazard = float(np.log(1.0 - self.hazard))

    def fit_predict(
        self,
        timestamps: Sequence[Any],
        data_matrix: np.ndarray,
    ) -> tuple[int | None, str, dict[str, Any], tuple[int, ...]]:
        """Run online causal changepoint detection over a (T, D) observation matrix.

        Parameters
        ----------
        timestamps:
            Sequence of observation timestamps of length T.
        data_matrix:
            2D numpy array of shape (T, D) containing numeric telemetry series.

        Returns
        -------
        tuple:
            (onset_ts, status, audit_dict, changepoint_indices)
        """
        T, D = data_matrix.shape
        if T < self.min_warmup:
            audit = {
                "reason": "insufficient_history",
                "observations": T,
                "min_warmup": self.min_warmup,
                "onset_idx": None,
                "onset_ts": None,
                "changepoints_count": 0,
            }
            return None, "insufficient_history", audit, ()

        # 1. Causal baseline standardization computed exclusively from warmup observations
        warmup_slice = data_matrix[: self.min_warmup]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warmup_mean = np.nanmean(warmup_slice, axis=0)
            warmup_std = np.nanstd(warmup_slice, axis=0)

        # Numerical stabilization for zero variance or NaN columns
        warmup_std = np.where(
            (warmup_std < self.epsilon) | np.isnan(warmup_std),
            1.0,
            warmup_std,
        )
        warmup_mean = np.nan_to_num(warmup_mean, nan=0.0)

        # Standardize observations causally
        std_data = (data_matrix - warmup_mean) / warmup_std
        std_data = np.nan_to_num(std_data, nan=0.0)

        # 2. Conjugate Normal-Inverse-Gamma prior hyperparameters
        # Standardized baseline has mean 0 and variance 1
        mu0 = 0.0
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = 1.0

        # Posterior tracking log P(r_t = r | x_{1:t})
        # Initial condition: at t=0, log P(r_0 = 0) = 0.0
        log_R = np.zeros(1)

        # Sufficient statistics arrays across active hypotheses r in {0, ..., t}
        mu_arr = np.array([np.full(D, mu0, dtype=float)])
        kappa_arr = np.array([kappa0], dtype=float)
        alpha_arr = np.array([alpha0], dtype=float)
        beta_arr = np.array([np.full(D, beta0, dtype=float)])

        changepoints: list[int] = []
        onset_ts: int | None = None
        onset_idx: int | None = None
        first_p_short_run: float | None = None

        for t in range(T):
            x = std_data[t]  # (D,)

            # 3. Evaluate Student's t predictive probability under each hypothesis r
            df = 2.0 * alpha_arr  # (r+1,)
            loc = mu_arr          # (r+1, D)
            scale_sq = (beta_arr * (kappa_arr[:, None] + 1.0)) / (
                alpha_arr[:, None] * kappa_arr[:, None]
            )
            scale = np.sqrt(np.maximum(scale_sq, self.epsilon))  # (r+1, D)

            diff = (x - loc) / scale
            df_b = df[:, None]
            log_c = (
                sc.gammaln((df_b + 1.0) / 2.0)
                - sc.gammaln(df_b / 2.0)
                - 0.5 * np.log(np.pi * df_b)
                - np.log(scale)
            )
            log_p_dim = log_c - 0.5 * (df_b + 1.0) * np.log1p((diff ** 2) / df_b)
            # Joint log-likelihood summed across observed dimensions
            log_pred = np.sum(log_p_dim, axis=-1)  # (r+1,)

            # 4. Growth and changepoint probabilities
            log_growth = log_R + log_pred + self.log_1m_hazard
            log_cp = float(sc.logsumexp(log_R + log_pred + self.log_hazard))

            # 5. Form updated joint posterior over run lengths r_t in {0, ..., t+1}
            new_log_R = np.empty(len(log_R) + 1, dtype=float)
            new_log_R[0] = log_cp
            new_log_R[1:] = log_growth

            # Normalization
            log_evidence = sc.logsumexp(new_log_R)
            new_log_R -= log_evidence
            log_R = new_log_R

            # 6. Changepoint detection decision
            # After warmup, check if posterior probability of a recent changepoint is high
            p_short_run = float(np.sum(np.exp(log_R[:3])))
            map_r = int(np.argmax(log_R))

            if t >= self.min_warmup:
                if p_short_run >= self.threshold or map_r == 0:
                    changepoints.append(t)
                    if onset_ts is None:
                        onset_ts = int(timestamps[t])
                        onset_idx = t
                        first_p_short_run = p_short_run

            # 7. Update sufficient statistics for subsequent step
            new_kappa = kappa_arr + 1.0
            new_alpha = alpha_arr + 0.5
            diff_mu = x - mu_arr
            new_mu = (kappa_arr[:, None] * mu_arr + x) / new_kappa[:, None]
            new_beta = beta_arr + 0.5 * (diff_mu ** 2) * (
                kappa_arr[:, None] / new_kappa[:, None]
            )

            # Prepend prior hyperparameters for run length r=0
            kappa_arr = np.concatenate([[kappa0], new_kappa])
            alpha_arr = np.concatenate([[alpha0], new_alpha])
            mu_arr = np.vstack([np.full(D, mu0, dtype=float), new_mu])
            beta_arr = np.vstack([np.full(D, beta0, dtype=float), new_beta])

        status = "detected" if onset_ts is not None else "no_detection"
        audit = {
            "onset_idx": onset_idx,
            "onset_ts": onset_ts,
            "changepoints_count": len(changepoints),
            "first_changepoint_idx": changepoints[0] if changepoints else None,
            "first_p_short_run": first_p_short_run,
            "total_observations": T,
            "dimensions": D,
            "min_warmup": self.min_warmup,
            "hazard_lambda": self.hazard_lambda,
            "threshold": self.threshold,
        }
        return onset_ts, status, audit, tuple(changepoints)


def detect_bocpd_onset(
    case: TelemetryCase,
    *,
    config: BOCPDConfig | None = None,
    hazard_lambda: float = 100.0,
    min_warmup: int = 60,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> BOCPDResult:
    """Detect incident onset timestamp using causal Bayesian Online Changepoint Detection.

    Parameters
    ----------
    case:
        TelemetryCase containing metrics modality.
    config:
        Optional BOCPDConfig. If provided, overrides explicit parameter kwargs.
    hazard_lambda:
        Prior expected run-length timescale (1 / hazard).
    min_warmup:
        Minimum number of initial observations before changepoint evaluation begins.
    threshold:
        Posterior probability threshold for short run-length reset.
    epsilon:
        Numerical stability parameter.

    Returns
    -------
    BOCPDResult:
        Immutable result with onset timestamp, status, changepoints, and audit metadata.
    """
    if config is not None:
        hazard_lambda = config.hazard_lambda
        min_warmup = config.min_warmup
        threshold = config.threshold
        epsilon = config.epsilon

    case_id = getattr(case.metadata, "case_id", "UNKNOWN_CASE")
    if case.metrics is None:
        raise ValueError(f"TelemetryCase {case_id!r} has no metrics modality")

    ts_field = getattr(getattr(case.metrics, "provenance", None), "timestamp_field", None)
    if not ts_field:
        raise ValueError(f"Metrics modality provenance for case {case_id!r} has no timestamp_field")

    timestamps, metric_series = _extract_series(case.metrics.raw_data, ts_field)
    num_obs = len(timestamps)

    if num_obs < min_warmup:
        raise ValueError(
            f"Insufficient observations ({num_obs}) for case {case_id!r}; "
            f"requires at least min_warmup={min_warmup}"
        )

    # Sort chronologically if timestamps are not monotonically non-decreasing
    is_sorted = all(timestamps[i] <= timestamps[i + 1] for i in range(num_obs - 1))
    if not is_sorted:
        order = sorted(range(num_obs), key=lambda idx: timestamps[idx])
        timestamps = tuple(timestamps[idx] for idx in order)
        metric_series = {
            col: tuple(vals[idx] for idx in order)
            for col, vals in metric_series.items()
        }

    # Operate on the identical canonical metric universe as the control detector
    selected_cols = sorted(metric_series.keys())

    # Build 2D matrix of shape (T, D) with float dtype, replacing None with np.nan
    matrix = np.array(
        [[v if v is not None else np.nan for v in metric_series[col]] for col in selected_cols],
        dtype=float,
    ).T

    detector = CausalBOCPD(
        hazard_lambda=hazard_lambda,
        min_warmup=min_warmup,
        threshold=threshold,
        epsilon=epsilon,
    )
    onset_ts, status, audit, changepoints = detector.fit_predict(timestamps, matrix)
    audit["selected_columns"] = tuple(selected_cols)
    audit["selected_metrics"] = tuple(selected_cols)

    return BOCPDResult(
        case_id=case_id,
        onset_ts=onset_ts,
        status=status,
        audit=audit,
        changepoints=changepoints,
        timestamps=tuple(timestamps),
    )
