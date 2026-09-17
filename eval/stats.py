"""Statistical evaluation infrastructure for Digital Detective (Stage 1C).

Provides paired cluster-level randomization significance testing,
percentile cluster bootstrap confidence intervals, and Holm-Bonferroni
multiple-comparison adjustments.

Hierarchy:
- execution/case: measurement observation
- scenario_family: inferential cluster (suite, system, root_cause_service, fault)
- suite/system: benchmark environment grouping
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import statistics
from typing import Any, Mapping, Sequence

from .manifest import BenchmarkManifest, ManifestCase
from .models import CaseMetrics


@dataclass(frozen=True)
class PairedCaseObservation:
    """Aligned measurement observation for two methods on a single case."""

    case_id: str
    scenario_family: tuple[str, str, str, str]
    value_a: float
    value_b: float

    def __post_init__(self) -> None:
        validate_scenario_family(self.scenario_family, cid=self.case_id)
        if not math.isfinite(self.value_a):
            raise ValueError(f"Non-finite value_a ({self.value_a}) for case_id {self.case_id!r}.")
        if not math.isfinite(self.value_b):
            raise ValueError(f"Non-finite value_b ({self.value_b}) for case_id {self.case_id!r}.")

    @property
    def difference(self) -> float:
        """Paired difference (Method A - Method B)."""
        return self.value_a - self.value_b


def validate_scenario_family(fam: Any, cid: str | None = None) -> tuple[str, str, str, str]:
    """Validate that scenario_family is a hashable 4-component tuple of non-empty strings.

    Identity: (suite, system, root_cause_service, fault).
    """
    context = f" for case_id {cid!r}" if cid else ""
    try:
        hash(fam)
    except TypeError:
        raise TypeError(f"Scenario family{context} must be hashable, got unhashable type {type(fam).__name__}: {fam!r}")

    if not isinstance(fam, tuple):
        raise TypeError(f"Scenario family{context} must be a tuple, got {type(fam).__name__}: {fam!r}")

    if len(fam) != 4:
        raise ValueError(
            f"Scenario family{context} must be a 4-component identity (suite, system, root_cause_service, fault), "
            f"got tuple of length {len(fam)}: {fam!r}."
        )

    suite, system, service, fault = fam
    if not all(isinstance(x, str) for x in (suite, system, service, fault)):
        raise TypeError(
            f"All components of scenario family{context} must be strings, got "
            f"({type(suite).__name__}, {type(system).__name__}, {type(service).__name__}, {type(fault).__name__}): {fam!r}"
        )

    if not all(x.strip() for x in (suite, system, service, fault)):
        raise ValueError(
            f"All components of scenario family{context} must be non-empty strings: {fam!r}"
        )

    return (suite, system, service, fault)


def compute_quantile(sorted_data: Sequence[float], q: float) -> float:
    """Compute quantile q in [0, 1] using continuous linear interpolation (Type 7 / NumPy default).

    Formula:
        virtual_index = q * (n - 1)
        i = floor(virtual_index)
        fraction = virtual_index - i
        quantile = x[i] + fraction * (x[i + 1] - x[i])
    """
    if not sorted_data:
        raise ValueError("Cannot compute quantile on empty data.")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile q must be between 0.0 and 1.0, got {q}.")
    n = len(sorted_data)
    if n == 1:
        return float(sorted_data[0])
    idx = q * (n - 1)
    i = int(math.floor(idx))
    fraction = idx - i
    if i >= n - 1:
        return float(sorted_data[-1])
    return float(sorted_data[i]) + fraction * float(sorted_data[i + 1] - sorted_data[i])


@dataclass
class StatisticalComparisonResult:
    """Immutable result of a paired statistical comparison between two RCA methods."""

    method_a: str
    method_b: str
    metric: str
    observed_difference: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_executions: int
    n_scenario_families: int
    repetitions_summary: dict[str, Any]
    adjusted_p_value: float | None = None
    confidence_level: float = 0.95
    bootstrap_replicates: int = 10_000
    randomization_replicates: int = 10_000
    seed: int = 42
    test_name: str = "paired_cluster_randomization_test"
    weighting_description: str = "execution_weighted_mean_difference"
    metadata: dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        """Serialize statistical comparison result to a dictionary."""
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "metric": self.metric,
            "observed_difference": round(self.observed_difference, 6),
            "p_value": round(self.p_value, 6),
            "adjusted_p_value": round(self.adjusted_p_value, 6) if self.adjusted_p_value is not None else None,
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "confidence_level": self.confidence_level,
            "n_executions": self.n_executions,
            "n_scenario_families": self.n_scenario_families,
            "repetitions_summary": self.repetitions_summary,
            "bootstrap_replicates": self.bootstrap_replicates,
            "randomization_replicates": self.randomization_replicates,
            "seed": self.seed,
            "test_name": self.test_name,
            "weighting_description": self.weighting_description,
            "metadata": self.metadata,
        }


def extract_metric_value(metric: str, obj: Any) -> float:
    """Extract a float metric value from a CaseMetrics object, dict, or raw float/int."""
    if isinstance(obj, (int, float, bool)):
        val = float(obj)
    elif isinstance(obj, Mapping):
        norm_metric = metric.lower().replace("-", "_")
        found = False
        val = 0.0
        for k in (metric, norm_metric, norm_metric.replace("_", "")):
            if k in obj:
                val = float(obj[k])
                found = True
                break
        if not found:
            raise KeyError(f"Mapping has no key matching metric {metric!r}.")
    elif isinstance(obj, CaseMetrics):
        norm_metric = metric.lower().replace("-", "_")
        metric_map = {
            "top_1": float(obj.top1),
            "top1": float(obj.top1),
            "top_3": float(obj.top3),
            "top3": float(obj.top3),
            "top_5": float(obj.top5),
            "top5": float(obj.top5),
            "mrr": float(obj.mrr),
            "ac_1": float(obj.ac1),
            "ac1": float(obj.ac1),
            "ac_2": float(obj.ac2),
            "ac2": float(obj.ac2),
            "ac_3": float(obj.ac3),
            "ac3": float(obj.ac3),
            "ac_4": float(obj.ac4),
            "ac4": float(obj.ac4),
            "ac_5": float(obj.ac5),
            "ac5": float(obj.ac5),
            "avg_3": float(obj.avg3),
            "avg3": float(obj.avg3),
            "avg_5": float(obj.avg5),
            "avg5": float(obj.avg5),
            "target_rank": float(obj.target_rank) if obj.target_rank is not None else 0.0,
        }
        if norm_metric in metric_map:
            val = float(metric_map[norm_metric])
        elif hasattr(obj, norm_metric):
            raw_attr = getattr(obj, norm_metric)
            if isinstance(raw_attr, (int, float, bool)):
                val = float(raw_attr)
            else:
                raise TypeError(f"CaseMetrics attribute {norm_metric!r} is not numeric: {type(raw_attr).__name__}.")
        else:
            raise AttributeError(f"CaseMetrics has no attribute matching metric {metric!r}.")
    else:
        raise TypeError(f"Cannot extract metric value from type {type(obj).__name__}.")

    if not math.isfinite(val):
        raise ValueError(f"Non-finite metric value {val!r} encountered for metric {metric!r}.")
    return val


def align_paired_observations(
    cases_a: Mapping[str, Any] | Sequence[Any],
    cases_b: Mapping[str, Any] | Sequence[Any],
    manifest_or_family_map: BenchmarkManifest | Mapping[str, tuple[str, str, str, str]] | Sequence[ManifestCase],
    metric: str = "top_1",
) -> list[PairedCaseObservation]:
    """Align evaluation results for two methods by exact case ID and scenario family.

    Fails loudly on:
    - empty case inputs
    - duplicate case IDs in either method
    - mismatched case ID sets between methods
    - missing or malformed scenario family assignments
    - non-finite metric values
    """
    # 1. Parse inputs into dict[case_id, raw_val]
    def _to_map(data: Mapping[str, Any] | Sequence[Any]) -> dict[str, float]:
        res: dict[str, float] = {}
        if isinstance(data, Mapping):
            for k, v in data.items():
                cid = str(k)
                if cid in res:
                    raise ValueError(f"Duplicate case_id {cid!r} in method cases mapping.")
                res[cid] = extract_metric_value(metric, v)
        else:
            for item in data:
                if isinstance(item, Mapping):
                    cid = str(item.get("case_id") or item.get("case") or "")
                elif hasattr(item, "case_id"):
                    cid = str(getattr(item, "case_id"))
                else:
                    raise TypeError(f"Sequence item must be CaseMetrics or dict, got {type(item).__name__}.")
                if not cid:
                    raise ValueError(f"Could not extract case_id from sequence item: {item!r}")
                if cid in res:
                    raise ValueError(f"Duplicate case_id {cid!r} in method cases sequence.")
                res[cid] = extract_metric_value(metric, item)
        return res

    map_a = _to_map(cases_a)
    map_b = _to_map(cases_b)

    if not map_a:
        raise ValueError("Method A cases cannot be empty.")
    if not map_b:
        raise ValueError("Method B cases cannot be empty.")

    cids_a = set(map_a.keys())
    cids_b = set(map_b.keys())

    if cids_a != cids_b:
        missing_in_b = cids_a - cids_b
        missing_in_a = cids_b - cids_a
        raise ValueError(
            f"Case set mismatch between Method A ({len(cids_a)} cases) and Method B ({len(cids_b)} cases). "
            f"Missing in B: {sorted(missing_in_b)[:3]!r}, Missing in A: {sorted(missing_in_a)[:3]!r}."
        )

    # 2. Build family mapping (case_id -> scenario_family)
    family_map: dict[str, tuple[str, str, str, str]] = {}
    if isinstance(manifest_or_family_map, BenchmarkManifest):
        for c in manifest_or_family_map.cases:
            family_map[c.case_id] = validate_scenario_family(c.scenario_family, cid=c.case_id)
    elif isinstance(manifest_or_family_map, Sequence):
        for c in manifest_or_family_map:
            if hasattr(c, "scenario_family"):
                cid = str(getattr(c, "case_id", ""))
                family_map[cid] = validate_scenario_family(getattr(c, "scenario_family"), cid=cid)
            elif isinstance(c, tuple) and len(c) == 2:
                cid = str(c[0])
                family_map[cid] = validate_scenario_family(c[1], cid=cid)
            else:
                raise TypeError(f"Unsupported sequence item type: {type(c).__name__}.")
    elif isinstance(manifest_or_family_map, Mapping):
        for k, v in manifest_or_family_map.items():
            cid = str(k)
            family_map[cid] = validate_scenario_family(v, cid=cid)
    else:
        raise TypeError(f"Unsupported manifest_or_family_map type: {type(manifest_or_family_map).__name__}.")

    observations: list[PairedCaseObservation] = []
    for cid in sorted(cids_a):
        if cid not in family_map:
            raise ValueError(f"Missing scenario_family mapping for case_id {cid!r}.")
        fam = family_map[cid]
        observations.append(
            PairedCaseObservation(
                case_id=cid,
                scenario_family=fam,
                value_a=map_a[cid],
                value_b=map_b[cid],
            )
        )

    return observations


def paired_cluster_randomization_test(
    observations: Sequence[PairedCaseObservation],
    randomization_replicates: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Perform a paired cluster-level Monte Carlo randomization (permutation) test.

    Under the null hypothesis, method labels are exchangeable within each scenario-family cluster.
    Cluster-level sign swapping preserves the exact intra-cluster repetition dependency structure.

    Returns:
    (observed_mean_difference, monte_carlo_p_value)
    """
    if not observations:
        raise ValueError("Cannot perform randomization test on empty observations.")
    if randomization_replicates < 1:
        raise ValueError("randomization_replicates must be >= 1.")

    n_total = len(observations)
    diffs: list[float] = []
    family_clusters: dict[tuple[str, str, str, str], list[float]] = {}
    for obs in observations:
        validate_scenario_family(obs.scenario_family, cid=getattr(obs, "case_id", None))
        diff = obs.difference
        if not math.isfinite(diff):
            raise ValueError(
                f"Non-finite observation difference ({diff}) for case {getattr(obs, 'case_id', None)!r}."
            )
        diffs.append(diff)
        family_clusters.setdefault(obs.scenario_family, []).append(diff)

    t_obs = statistics.mean(diffs)
    abs_t_obs = abs(t_obs)

    cluster_diffs = list(family_clusters.values())
    rng = random.Random(seed)

    more_extreme_count = 0
    # Floating point tolerance for exact equality comparison
    tol = 1e-12

    for _ in range(randomization_replicates):
        sum_perturbed = 0.0
        for cluster in cluster_diffs:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            sum_perturbed += sign * sum(cluster)

        t_k = sum_perturbed / n_total
        if abs(t_k) >= (abs_t_obs - tol):
            more_extreme_count += 1

    # Standard Monte Carlo p-value convention avoiding zero reported p-values
    p_val = (1.0 + more_extreme_count) / (float(randomization_replicates) + 1.0)
    return t_obs, p_val


def cluster_bootstrap_ci(
    observations: Sequence[PairedCaseObservation],
    confidence_level: float = 0.95,
    bootstrap_replicates: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a two-sided percentile cluster bootstrap confidence interval.

    Resamples unique scenario families with replacement, retaining all repetitions
    belonging to each sampled family. Computes quantiles using linear interpolation.

    Returns:
    (observed_mean_difference, ci_lower, ci_upper)
    """
    if not observations:
        raise ValueError("Cannot compute cluster bootstrap CI on empty observations.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"Invalid confidence_level: {confidence_level}. Must be between 0 and 1.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be >= 1.")

    diffs: list[float] = []
    family_clusters: dict[tuple[str, str, str, str], list[float]] = {}
    for obs in observations:
        validate_scenario_family(obs.scenario_family, cid=getattr(obs, "case_id", None))
        diff = obs.difference
        if not math.isfinite(diff):
            raise ValueError(
                f"Non-finite observation difference ({diff}) for case {getattr(obs, 'case_id', None)!r}."
            )
        diffs.append(diff)
        family_clusters.setdefault(obs.scenario_family, []).append(diff)

    t_obs = statistics.mean(diffs)

    unique_families = list(family_clusters.keys())
    k_clusters = len(unique_families)

    if k_clusters == 1:
        # Edge case: single cluster sampled with replacement is identical
        return t_obs, t_obs, t_obs

    cluster_list = [family_clusters[f] for f in unique_families]
    rng = random.Random(seed)

    replicates: list[float] = []
    for _ in range(bootstrap_replicates):
        # Sample K clusters with replacement
        sampled_clusters = rng.choices(cluster_list, k=k_clusters)
        total_sum = sum(sum(c) for c in sampled_clusters)
        total_cases = sum(len(c) for c in sampled_clusters)
        replicates.append(total_sum / total_cases)

    replicates.sort()
    alpha = 1.0 - confidence_level
    q_lower = alpha / 2.0
    q_upper = 1.0 - (alpha / 2.0)

    ci_lower = compute_quantile(replicates, q_lower)
    ci_upper = compute_quantile(replicates, q_upper)

    return t_obs, ci_lower, ci_upper



def apply_holm_correction(
    results_or_pvalues: Sequence[StatisticalComparisonResult] | Sequence[float],
) -> list[StatisticalComparisonResult] | list[float]:
    """Apply Holm-Bonferroni step-down correction for multiple comparisons.

    Accepts either a sequence of raw p-values or a sequence of StatisticalComparisonResult
    instances and returns adjusted values preserving input order.
    """
    if not results_or_pvalues:
        return []

    is_results = isinstance(results_or_pvalues[0], StatisticalComparisonResult)
    if is_results:
        raw_pvals = [r.p_value for r in results_or_pvalues]  # type: ignore
    else:
        raw_pvals = [float(p) for p in results_or_pvalues]  # type: ignore

    m = len(raw_pvals)
    indexed_p = sorted(enumerate(raw_pvals), key=lambda x: x[1])

    adjusted_map: dict[int, float] = {}
    cum_max = 0.0

    for step, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - step
        adj_val = min(1.0, p_val * multiplier)
        cum_max = max(cum_max, adj_val)
        adjusted_map[orig_idx] = cum_max

    adjusted_pvals = [adjusted_map[i] for i in range(m)]

    if is_results:
        res_list: list[StatisticalComparisonResult] = []
        for r, adj_p in zip(results_or_pvalues, adjusted_pvals):  # type: ignore
            res_dict = asdict(r)
            res_dict["adjusted_p_value"] = adj_p
            res_list.append(StatisticalComparisonResult(**res_dict))
        return res_list
    return adjusted_pvals


def _calculate_summary_stats(values: Sequence[float]) -> dict[str, float]:
    """Calculate mean, median, and IQR for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "iqr": 0.0, "q25": 0.0, "q75": 0.0}
    m = statistics.mean(values)
    med = statistics.median(values)
    if len(values) >= 2:
        # Compute inclusive quartiles
        q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
        iqr = q3 - q1
    else:
        q1, q3, iqr = values[0], values[0], 0.0
    return {"mean": round(m, 6), "median": round(med, 6), "iqr": round(iqr, 6), "q25": round(q1, 6), "q75": round(q3, 6)}


def compare_methods_paired(
    method_a: str,
    method_b: str,
    metric: str,
    cases_a: Mapping[str, Any] | Sequence[Any],
    cases_b: Mapping[str, Any] | Sequence[Any],
    manifest_or_family_map: BenchmarkManifest | Mapping[str, tuple[str, str, str, str]] | Sequence[ManifestCase],
    randomization_replicates: int = 10_000,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> StatisticalComparisonResult:
    """Execute a complete paired statistical comparison between two RCA methods.

    Metrics supported:
    - Primary accuracy: 'top1' / 'top_1', 'mrr'
    - Secondary: 'top3', 'top5', 'ac1'...'ac5', 'avg3', 'avg5', 'query_count', 'runtime_sec'
    """
    observations = align_paired_observations(
        cases_a=cases_a,
        cases_b=cases_b,
        manifest_or_family_map=manifest_or_family_map,
        metric=metric,
    )

    n_executions = len(observations)
    family_counts: dict[tuple[str, str, str, str], int] = {}
    for obs in observations:
        family_counts[obs.scenario_family] = family_counts.get(obs.scenario_family, 0) + 1

    n_families = len(family_counts)
    rep_counts = list(family_counts.values())
    rep_summary = {
        "min_repetitions_per_family": min(rep_counts),
        "max_repetitions_per_family": max(rep_counts),
        "mean_repetitions_per_family": round(statistics.mean(rep_counts), 4),
    }

    t_obs, p_val = paired_cluster_randomization_test(
        observations=observations,
        randomization_replicates=randomization_replicates,
        seed=seed,
    )

    _, ci_lower, ci_upper = cluster_bootstrap_ci(
        observations=observations,
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )

    # Calculate metric-specific effect size metadata
    vals_a = [obs.value_a for obs in observations]
    vals_b = [obs.value_b for obs in observations]
    stats_a = _calculate_summary_stats(vals_a)
    stats_b = _calculate_summary_stats(vals_b)

    metadata: dict[str, Any] = {
        "method_a_summary": stats_a,
        "method_b_summary": stats_b,
    }

    norm_metric = metric.lower().replace("-", "_")
    if norm_metric.startswith("top") or norm_metric.startswith("ac"):
        n_10 = sum(1 for obs in observations if obs.value_a >= 0.5 and obs.value_b < 0.5)
        n_01 = sum(1 for obs in observations if obs.value_a < 0.5 and obs.value_b >= 0.5)
        n_11 = sum(1 for obs in observations if obs.value_a >= 0.5 and obs.value_b >= 0.5)
        n_00 = sum(1 for obs in observations if obs.value_a < 0.5 and obs.value_b < 0.5)
        metadata.update({
            "n_10_a_succeeds_b_fails": n_10,
            "n_01_a_fails_b_succeeds": n_01,
            "n_11_both_succeed": n_11,
            "n_00_both_fail": n_00,
            "absolute_success_rate_difference": round(t_obs, 6),
        })
    elif norm_metric == "mrr":
        metadata.update({
            "delta_mrr": round(t_obs, 6),
        })
    elif "query" in norm_metric:
        metadata.update({
            "delta_queries": round(t_obs, 6),
            "interpretation": "Lower query consumption is better; negative difference indicates Method A used fewer queries."
        })
    elif "runtime" in norm_metric:
        valid_pos = [(obs.value_a, obs.value_b) for obs in observations if obs.value_a > 0 and obs.value_b > 0]
        if valid_pos:
            log_diffs = [math.log(a) - math.log(b) for a, b in valid_pos]
            mean_log_diff = statistics.mean(log_diffs)
            metadata["mean_log_runtime_diff"] = round(mean_log_diff, 6)
            metadata["multiplicative_runtime_ratio"] = round(math.exp(mean_log_diff), 6)

    return StatisticalComparisonResult(
        method_a=method_a,
        method_b=method_b,
        metric=metric,
        observed_difference=t_obs,
        p_value=p_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_executions=n_executions,
        n_scenario_families=n_families,
        repetitions_summary=rep_summary,
        bootstrap_replicates=bootstrap_replicates,
        randomization_replicates=randomization_replicates,
        seed=seed,
        test_name="paired_cluster_randomization_test",
        weighting_description="execution_weighted_mean_difference",
        metadata=metadata,
    )
