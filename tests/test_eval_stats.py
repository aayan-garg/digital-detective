"""Unit tests for Stage 1C statistical evaluation infrastructure."""

import math
import unittest

from eval.manifest import BenchmarkManifest, ManifestCase
from eval.models import CaseMetrics
from eval.stats import (
    PairedCaseObservation,
    StatisticalComparisonResult,
    align_paired_observations,
    apply_holm_correction,
    cluster_bootstrap_ci,
    compare_methods_paired,
    compute_quantile,
    extract_metric_value,
    paired_cluster_randomization_test,
    validate_scenario_family,
)


class TestEvalStats(unittest.TestCase):
    """Test suite for eval/stats.py statistical functions and integrity checks."""

    def setUp(self) -> None:
        # Construct standard test manifest with 2 scenario families across 2 suites
        self.case1 = ManifestCase("c1", "RE1-OB", "ob", "cpu", "cartservice", 1, "all", suite="RE1")
        self.case2 = ManifestCase("c2", "RE1-OB", "ob", "cpu", "cartservice", 2, "all", suite="RE1")
        self.case3 = ManifestCase("c3", "RE2-OB", "ob", "cpu", "cartservice", 1, "all", suite="RE2")
        self.case4 = ManifestCase("c4", "RE2-OB", "ob", "cpu", "cartservice", 2, "all", suite="RE2")

        self.manifest = BenchmarkManifest(
            manifest_id="stats_test_manifest",
            description="Manifest for statistical unit tests",
            created_at="2026-09-17T00:00:00Z",
            is_smoke_test=False,
            manifest_type="SCIENTIFIC_BENCHMARK",
            cases=(self.case1, self.case2, self.case3, self.case4),
        )

    def test_pairing_alignment_success(self) -> None:
        """Verify alignment succeeds when case sets match perfectly."""
        cases_a = {"c1": 1.0, "c2": 0.8, "c3": 1.0, "c4": 0.5}
        cases_b = {"c1": 0.0, "c2": 0.8, "c3": 0.0, "c4": 0.5}

        obs = align_paired_observations(cases_a, cases_b, self.manifest, metric="top_1")
        self.assertEqual(len(obs), 4)
        self.assertEqual(obs[0].case_id, "c1")
        self.assertEqual(obs[0].difference, 1.0)
        self.assertEqual(obs[1].difference, 0.0)

    def test_pairing_mismatched_cases_rejected(self) -> None:
        """Verify mismatched case IDs between methods raise ValueError."""
        cases_a = {"c1": 1.0, "c2": 0.8}
        cases_b = {"c1": 1.0, "c3": 0.5}

        with self.assertRaises(ValueError) as ctx:
            align_paired_observations(cases_a, cases_b, self.manifest)
        self.assertIn("Case set mismatch", str(ctx.exception))

    def test_pairing_duplicate_cases_rejected(self) -> None:
        """Verify duplicate case IDs within a method raise ValueError."""
        # Using a sequence of case dicts with duplicate IDs
        c1 = {"case_id": "c1", "top_1": 1.0}
        c1_dup = {"case_id": "c1", "top_1": 0.5}
        cases_a = [c1, c1_dup]
        cases_b = [c1, {"case_id": "c2", "top_1": 0.0}]

        with self.assertRaises(ValueError) as ctx:
            align_paired_observations(cases_a, cases_b, self.manifest)
        self.assertIn("Duplicate case_id", str(ctx.exception))

    def test_scenario_family_clustering(self) -> None:
        """Verify cases are clustered strictly by (suite, system, root_cause_service, fault)."""
        cases_a = {"c1": 1.0, "c2": 0.8, "c3": 0.6, "c4": 0.4}
        cases_b = {"c1": 0.5, "c2": 0.4, "c3": 0.3, "c4": 0.2}

        obs = align_paired_observations(cases_a, cases_b, self.manifest)
        fam_re1 = ("RE1", "ob", "cartservice", "cpu")
        fam_re2 = ("RE2", "ob", "cartservice", "cpu")

        obs_re1 = [o for o in obs if o.scenario_family == fam_re1]
        obs_re2 = [o for o in obs if o.scenario_family == fam_re2]

        self.assertEqual(len(obs_re1), 2)
        self.assertEqual(len(obs_re2), 2)
        self.assertNotEqual(fam_re1, fam_re2, "RE1 and RE2 families must remain distinct clusters!")

    def test_randomization_test_reproducibility_and_non_zero_pvalue(self) -> None:
        """Verify paired cluster randomization is reproducible and never reports p=0."""
        obs = [
            PairedCaseObservation("c1", ("RE1", "ob", "s1", "f1"), 1.0, 0.0),
            PairedCaseObservation("c2", ("RE1", "ob", "s1", "f1"), 1.0, 0.0),
            PairedCaseObservation("c3", ("RE2", "ob", "s2", "f2"), 1.0, 0.0),
            PairedCaseObservation("c4", ("RE2", "ob", "s2", "f2"), 1.0, 0.0),
        ]
        t_obs_1, p_val_1 = paired_cluster_randomization_test(obs, randomization_replicates=1000, seed=42)
        t_obs_2, p_val_2 = paired_cluster_randomization_test(obs, randomization_replicates=1000, seed=42)

        self.assertEqual(t_obs_1, 1.0)
        self.assertEqual(t_obs_1, t_obs_2)
        self.assertEqual(p_val_1, p_val_2)
        self.assertGreater(p_val_1, 0.0, "Monte Carlo p-value must never be exactly zero!")

    def test_cluster_bootstrap_ci_reproducibility(self) -> None:
        """Verify cluster bootstrap CI is reproducible and preserves all repetitions per cluster."""
        obs = [
            PairedCaseObservation("c1", ("RE1", "ob", "s1", "f1"), 1.0, 0.0),
            PairedCaseObservation("c2", ("RE1", "ob", "s1", "f1"), 0.8, 0.0),
            PairedCaseObservation("c3", ("RE2", "ob", "s2", "f2"), 0.4, 0.2),
            PairedCaseObservation("c4", ("RE2", "ob", "s2", "f2"), 0.6, 0.2),
        ]
        t1, lower1, upper1 = cluster_bootstrap_ci(obs, confidence_level=0.95, bootstrap_replicates=1000, seed=123)
        t2, lower2, upper2 = cluster_bootstrap_ci(obs, confidence_level=0.95, bootstrap_replicates=1000, seed=123)

        self.assertEqual(t1, t2)
        self.assertEqual(lower1, lower2)
        self.assertEqual(upper1, upper2)
        self.assertLessEqual(lower1, t1)
        self.assertLessEqual(t1, upper1)

    def test_holm_correction(self) -> None:
        """Verify Holm-Bonferroni correction against known textbook step-down values."""
        raw_pvals = [0.01, 0.04, 0.03]
        adj_pvals = apply_holm_correction(raw_pvals)

        # Sorted order: 0.01 (x3 -> 0.03), 0.03 (x2 -> 0.06), 0.04 (x1 -> 0.04 -> max 0.06)
        # Expected in original order: [0.03, 0.06, 0.06]
        self.assertAlmostEqual(adj_pvals[0], 0.03, places=6)
        self.assertAlmostEqual(adj_pvals[1], 0.06, places=6)
        self.assertAlmostEqual(adj_pvals[2], 0.06, places=6)

    def test_edge_cases(self) -> None:
        """Test edge cases: single cluster, all zero differences, empty inputs."""
        # 1. Empty observations
        with self.assertRaises(ValueError):
            paired_cluster_randomization_test([], randomization_replicates=100)

        # 2. Single cluster
        obs_single = [
            PairedCaseObservation("c1", ("RE1", "ob", "s1", "f1"), 1.0, 0.0),
            PairedCaseObservation("c2", ("RE1", "ob", "s1", "f1"), 0.8, 0.0),
        ]
        t_single, lower_s, upper_s = cluster_bootstrap_ci(obs_single, bootstrap_replicates=100)
        self.assertEqual(t_single, 0.9)
        self.assertEqual(lower_s, 0.9)
        self.assertEqual(upper_s, 0.9)

        # 3. All paired differences are zero
        obs_zero = [
            PairedCaseObservation("c1", ("RE1", "ob", "s1", "f1"), 0.5, 0.5),
            PairedCaseObservation("c2", ("RE2", "ob", "s2", "f2"), 0.8, 0.8),
        ]
        t_z, p_z = paired_cluster_randomization_test(obs_zero, randomization_replicates=100)
        self.assertEqual(t_z, 0.0)
        self.assertEqual(p_z, 1.0)

    def test_compare_methods_paired_full_pipeline(self) -> None:
        """Test complete compare_methods_paired orchestration with CaseMetrics objects."""
        method_a_metrics = {
            "c1": CaseMetrics(True, True, True, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
            "c2": CaseMetrics(True, True, True, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
            "c3": CaseMetrics(False, True, True, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 2),
            "c4": CaseMetrics(True, True, True, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
        }
        method_b_metrics = {
            "c1": CaseMetrics(False, True, True, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 2),
            "c2": CaseMetrics(True, True, True, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1),
            "c3": CaseMetrics(False, False, True, 0.2, 0.0, 0.0, 0.33, 0.5, 0.5, 0.2, 0.2, 5),
            "c4": CaseMetrics(False, False, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None),
        }

        res = compare_methods_paired(
            method_a="MethodA",
            method_b="MethodB",
            metric="top1",
            cases_a=method_a_metrics,
            cases_b=method_b_metrics,
            manifest_or_family_map=self.manifest,
            randomization_replicates=500,
            bootstrap_replicates=500,
            seed=42,
        )

        self.assertIsInstance(res, StatisticalComparisonResult)
        self.assertEqual(res.method_a, "MethodA")
        self.assertEqual(res.method_b, "MethodB")
        self.assertEqual(res.metric, "top1")
        self.assertEqual(res.n_executions, 4)
        self.assertEqual(res.n_scenario_families, 2)
        self.assertGreater(res.observed_difference, 0.0)
        self.assertIn("n_10_a_succeeds_b_fails", res.metadata)

        # Test dictionary serialization round-trip
        d = res.to_dict()
        self.assertEqual(d["method_a"], "MethodA")
        self.assertIn("observed_difference", d)
        self.assertIn("ci_lower", d)
        self.assertIn("ci_upper", d)

    def test_compute_quantile_linear_interpolation(self) -> None:
        """Verify compute_quantile calculates exact Type 7 continuous linear interpolation."""
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        # Extremes
        self.assertEqual(compute_quantile(data, 0.0), 10.0)
        self.assertEqual(compute_quantile(data, 1.0), 50.0)
        # Median: idx = 0.5 * 4 = 2.0 -> data[2] = 30.0
        self.assertEqual(compute_quantile(data, 0.5), 30.0)
        # Small count where interpolation matters:
        # idx = 0.1 * 4 = 0.4 -> data[0] + 0.4 * (data[1] - data[0]) = 10 + 4 = 14.0
        self.assertAlmostEqual(compute_quantile(data, 0.1), 14.0, places=9)
        # idx = 0.75 * 4 = 3.0 -> data[3] = 40.0
        self.assertEqual(compute_quantile(data, 0.75), 40.0)
        # Single element
        self.assertEqual(compute_quantile([42.0], 0.25), 42.0)
        self.assertEqual(compute_quantile([42.0], 0.0), 42.0)

        # Invalid inputs
        with self.assertRaises(ValueError):
            compute_quantile([], 0.5)
        with self.assertRaises(ValueError):
            compute_quantile(data, -0.1)
        with self.assertRaises(ValueError):
            compute_quantile(data, 1.1)

    def test_cluster_bootstrap_ci_confidence_levels(self) -> None:
        """Verify cluster_bootstrap_ci supports confidence levels other than 0.95 and behaves monotonically."""
        obs = [
            PairedCaseObservation("c1", ("RE1", "ob", "s1", "f1"), 1.0, 0.0),
            PairedCaseObservation("c2", ("RE1", "ob", "s1", "f1"), 0.8, 0.2),
            PairedCaseObservation("c3", ("RE2", "ob", "s2", "f2"), 0.6, 0.4),
            PairedCaseObservation("c4", ("RE2", "ob", "s2", "f2"), 0.9, 0.1),
        ]
        t90, low90, up90 = cluster_bootstrap_ci(obs, confidence_level=0.90, bootstrap_replicates=1000, seed=42)
        t95, low95, up95 = cluster_bootstrap_ci(obs, confidence_level=0.95, bootstrap_replicates=1000, seed=42)
        t99, low99, up99 = cluster_bootstrap_ci(obs, confidence_level=0.99, bootstrap_replicates=1000, seed=42)

        self.assertEqual(t90, t95)
        self.assertEqual(t95, t99)

        # Higher confidence interval must be wider or equal
        self.assertLessEqual(low99, low95)
        self.assertLessEqual(low95, low90)
        self.assertLessEqual(low90, t90)
        self.assertLessEqual(t90, up90)
        self.assertLessEqual(up90, up95)
        self.assertLessEqual(up95, up99)

        # Invalid confidence levels
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(obs, confidence_level=0.0)
        with self.assertRaises(ValueError):
            cluster_bootstrap_ci(obs, confidence_level=1.0)

    def test_cluster_bootstrap_ci_small_replicates_interpolation(self) -> None:
        """Verify cluster_bootstrap_ci handles small replicate counts deterministically with linear interpolation."""
        obs = [
            PairedCaseObservation("c1", ("RE1", "ob", "s1", "f1"), 1.0, 0.0),
            PairedCaseObservation("c2", ("RE1", "ob", "s1", "f1"), 0.5, 0.0),
            PairedCaseObservation("c3", ("RE2", "ob", "s2", "f2"), 0.8, 0.2),
            PairedCaseObservation("c4", ("RE2", "ob", "s2", "f2"), 0.4, 0.2),
        ]
        t1, low1, up1 = cluster_bootstrap_ci(obs, confidence_level=0.95, bootstrap_replicates=10, seed=99)
        t2, low2, up2 = cluster_bootstrap_ci(obs, confidence_level=0.95, bootstrap_replicates=10, seed=99)

        self.assertEqual(t1, t2)
        self.assertEqual(low1, low2)
        self.assertEqual(up1, up2)
        self.assertTrue(math.isfinite(low1))
        self.assertTrue(math.isfinite(up1))
        self.assertLessEqual(low1, t1)
        self.assertLessEqual(t1, up1)

    def test_malformed_and_missing_scenario_family_rejected(self) -> None:
        """Verify scenario family validation rejects unhashable, wrong length, non-string, or empty components."""
        # 1. Direct validation checks
        # Unhashable list
        with self.assertRaises(TypeError):
            validate_scenario_family(["RE1", "ob", "s1", "f1"])  # type: ignore

        # Not a tuple
        with self.assertRaises(TypeError):
            validate_scenario_family("RE1-ob-s1-f1")

        # Wrong length (3 components)
        with self.assertRaises(ValueError) as ctx3:
            validate_scenario_family(("RE1", "ob", "s1"))
        self.assertIn("4-component identity", str(ctx3.exception))

        # Wrong length (5 components)
        with self.assertRaises(ValueError) as ctx5:
            validate_scenario_family(("RE1", "ob", "s1", "f1", "extra"))
        self.assertIn("4-component identity", str(ctx5.exception))

        # Non-string components
        with self.assertRaises(TypeError) as ctxt:
            validate_scenario_family(("RE1", 123, "s1", "f1"))  # type: ignore
        self.assertIn("must be strings", str(ctxt.exception))

        # Empty string component
        with self.assertRaises(ValueError) as ctxe:
            validate_scenario_family(("RE1", "", "s1", "f1"))
        self.assertIn("non-empty strings", str(ctxe.exception))

        # Whitespace-only string component
        with self.assertRaises(ValueError) as ctxw:
            validate_scenario_family(("RE1", "   ", "s1", "f1"))
        self.assertIn("non-empty strings", str(ctxw.exception))

        # 2. PairedCaseObservation rejects malformed family
        with self.assertRaises(TypeError):
            PairedCaseObservation("c1", ["RE1", "ob", "s1", "f1"], 1.0, 0.0)  # type: ignore

        # 3. align_paired_observations rejects missing family assignment
        cases_a = {"c1": 1.0, "c2": 0.5}
        cases_b = {"c1": 0.0, "c2": 0.5}
        incomplete_fam_map = {"c1": ("RE1", "ob", "s1", "f1")}  # missing c2
        with self.assertRaises(ValueError) as ctxm:
            align_paired_observations(cases_a, cases_b, incomplete_fam_map)
        self.assertIn("Missing scenario_family mapping", str(ctxm.exception))

        # 4. align_paired_observations rejects malformed family in mapping
        malformed_fam_map = {
            "c1": ("RE1", "ob", "s1", "f1"),
            "c2": ("RE1", "ob", "s1"),  # 3-component tuple
        }
        with self.assertRaises(ValueError):
            align_paired_observations(cases_a, cases_b, malformed_fam_map)

    def test_non_finite_numeric_inputs_rejected(self) -> None:
        """Verify NaN, +inf, -inf values are rejected with ValueError before calculations."""
        fam = ("RE1", "ob", "s1", "f1")

        # 1. extract_metric_value
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                extract_metric_value("top_1", non_finite)
            with self.assertRaises(ValueError):
                extract_metric_value("top_1", {"top_1": non_finite})

        # 2. PairedCaseObservation
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                PairedCaseObservation("c1", fam, non_finite, 0.5)
            with self.assertRaises(ValueError):
                PairedCaseObservation("c1", fam, 0.5, non_finite)

        # 3. align_paired_observations
        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                align_paired_observations(
                    {"c1": non_finite, "c2": 0.5},
                    {"c1": 0.0, "c2": 0.5},
                    {"c1": fam, "c2": fam},
                )
            with self.assertRaises(ValueError):
                align_paired_observations(
                    {"c1": 1.0, "c2": 0.5},
                    {"c1": 0.0, "c2": non_finite},
                    {"c1": fam, "c2": fam},
                )


if __name__ == "__main__":
    unittest.main()
