"""Unit and integration tests for the unified evaluation harness.
"""

import os
import unittest
from pathlib import Path
import pyarrow.parquet as pq

from eval.harness import run_benchmark, HARNESS_SCHEMA_VERSION
from eval.manifest import create_smoke_re2_ob_rep1_manifest


class TestEvalHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_root = Path(os.path.expanduser("~/.cache/rcaeval_validation"))
        cases_file = cls.dataset_root / "cases.parquet"
        if not cases_file.is_file():
            raise unittest.SkipTest(f"RCAEval cache not found at {cases_file}")
        cases_table = pq.read_table(cases_file)
        cls.case_rows = cases_table.to_pylist()

    def test_smoke_benchmark_regression(self) -> None:
        """Verify the 30-case RE2-OB repetition-1 regression values reproduce exactly."""
        manifest = create_smoke_re2_ob_rep1_manifest(self.case_rows)
        self.assertEqual(len(manifest.cases), 30)

        report = run_benchmark(
            manifest=manifest,
            dataset_root=self.dataset_root,
            methods=["random", "simple_rca", "s_comb", "trace_elevation", "fixed_equal_weight_fusion"],
            window_mode="oracle",
        )

        self.assertEqual(report.schema_version, HARNESS_SCHEMA_VERSION)
        self.assertEqual(report.total_cases, 30)

        # 1. S_comb regression verification
        s_comb = report.method_aggregates["s_comb"]
        self.assertAlmostEqual(s_comb.top1_accuracy, 21.0 / 30.0, places=4)
        self.assertAlmostEqual(s_comb.top3_accuracy, 26.0 / 30.0, places=4)
        self.assertAlmostEqual(s_comb.top5_accuracy, 28.0 / 30.0, places=4)
        self.assertAlmostEqual(s_comb.mrr, 0.8011, places=3)

        # 2. Trace Elevation regression verification
        trace_elev = report.method_aggregates["trace_elevation"]
        self.assertAlmostEqual(trace_elev.top1_accuracy, 18.0 / 30.0, places=4)
        self.assertAlmostEqual(trace_elev.top3_accuracy, 29.0 / 30.0, places=4)
        self.assertAlmostEqual(trace_elev.top5_accuracy, 30.0 / 30.0, places=4)
        self.assertAlmostEqual(trace_elev.mrr, 0.7567, places=3)

        # 3. Candidate universe closure check: every case and method ranks all 11 candidates
        for c_res in report.case_results:
            self.assertEqual(len(c_res["candidate_universe"]), 11)
            for m_name in ("random", "simple_rca", "s_comb", "trace_elevation", "fixed_equal_weight_fusion"):
                ranking = c_res["methods"][m_name]["ranking"]
                self.assertEqual(len(ranking), 11, f"Method {m_name} dropped candidates in case {c_res['case_id']}")

        # 4. Stage 1 Headroom check
        ha = report.headroom_analysis
        self.assertIsNotNone(ha)
        self.assertEqual(ha.best_fixed_method, "s_comb")
        self.assertGreater(ha.absolute_headroom_fault["mrr"], 0.0)
        self.assertGreater(ha.absolute_headroom_incident["mrr"], 0.0)
        self.assertIn("ORACLE", ha.disclaimer)


if __name__ == "__main__":
    unittest.main()
