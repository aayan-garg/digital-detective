"""Unit tests for BenchmarkManifest, scientific populations, and leak-free tuning splits.
"""

import json
import tempfile
import unittest
from pathlib import Path

from eval.manifest import (
    BenchmarkManifest,
    ManifestCase,
    create_scientific_manifest,
    create_scientific_re2_manifest,
    create_scientific_re2_ob_manifest,
    create_scientific_re2_ss_manifest,
    create_scientific_re2_tt_manifest,
    create_smoke_re2_ob_rep1_manifest,
    partition_cases_leak_free,
)


class TestEvalManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_case_rows = [
            {"case": "re2ob_cart_cpu_1", "dataset": "RE2-OB", "suite": "RE2", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 1, "n_metrics": 100, "has_logs": True, "has_traces": True},
            {"case": "re2ob_cart_cpu_2", "dataset": "RE2-OB", "suite": "RE2", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 2, "n_metrics": 100, "has_logs": True, "has_traces": True},
            {"case": "re2ob_cart_cpu_3", "dataset": "RE2-OB", "suite": "RE2", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 3, "n_metrics": 100, "has_logs": True, "has_traces": True},
            {"case": "re2ob_check_mem_1", "dataset": "RE2-OB", "suite": "RE2", "system": "ob", "fault": "mem", "root_cause_service": "checkoutservice", "repetition": 1, "n_metrics": 100, "has_logs": True, "has_traces": True},
            {"case": "re2ss_user_cpu_1", "dataset": "RE2-SS", "suite": "RE2", "system": "ss", "fault": "cpu", "root_cause_service": "user", "repetition": 1, "n_metrics": 80, "has_logs": True, "has_traces": False},
            {"case": "re2tt_order_delay_1", "dataset": "RE2-TT", "suite": "RE2", "system": "tt", "fault": "delay", "root_cause_service": "ts-order-service", "repetition": 1, "n_metrics": 120, "has_logs": False, "has_traces": True},
        ]

    def test_smoke_manifest_creation_and_type(self) -> None:
        manifest = create_smoke_re2_ob_rep1_manifest(self.mock_case_rows)
        self.assertTrue(manifest.is_smoke_test)
        self.assertEqual(manifest.manifest_type, "SMOKE_REGRESSION")
        self.assertEqual(len(manifest.cases), 2)  # rep 1 cases for ob
        self.assertEqual(manifest.cases[0].partition, "smoke")
        self.assertIn("SMOKE", manifest.description)

    def test_scientific_manifest_constructors(self) -> None:
        # 1. RE2-OB constructor
        m_ob = create_scientific_re2_ob_manifest(self.mock_case_rows)
        self.assertEqual(m_ob.manifest_type, "SCIENTIFIC_BENCHMARK")
        self.assertFalse(m_ob.is_smoke_test)
        self.assertEqual(len(m_ob.cases), 4)
        for c in m_ob.cases:
            self.assertEqual(c.partition, "all")
            self.assertEqual(c.system, "ob")
        self.assertEqual(m_ob.modality_summary["total_cases"], 4)
        self.assertEqual(m_ob.modality_summary["traces_available"], 4)
        self.assertEqual(m_ob.modality_summary["missing_traces"], 0)

        # 2. RE2-SS constructor (preserves missing traces)
        m_ss = create_scientific_re2_ss_manifest(self.mock_case_rows)
        self.assertEqual(len(m_ss.cases), 1)
        self.assertEqual(m_ss.cases[0].system, "ss")
        self.assertEqual(m_ss.modality_summary["traces_available"], 0)
        self.assertEqual(m_ss.modality_summary["missing_traces"], 1)

        # 3. RE2-TT constructor (preserves missing logs)
        m_tt = create_scientific_re2_tt_manifest(self.mock_case_rows)
        self.assertEqual(len(m_tt.cases), 1)
        self.assertEqual(m_tt.cases[0].system, "tt")
        self.assertEqual(m_tt.modality_summary["logs_available"], 0)
        self.assertEqual(m_tt.modality_summary["missing_logs"], 1)

        # 4. RE2 full population constructor
        m_re2 = create_scientific_re2_manifest(self.mock_case_rows)
        self.assertEqual(len(m_re2.cases), 6)
        self.assertEqual(m_re2.modality_summary["total_cases"], 6)
        self.assertEqual(m_re2.modality_summary["traces_available"], 5)
        self.assertEqual(m_re2.modality_summary["missing_traces"], 1)
        self.assertEqual(m_re2.modality_summary["missing_logs"], 1)

        # Deterministic ordering by case_id
        cids = [c.case_id for c in m_re2.cases]
        self.assertEqual(cids, sorted(cids))

    def test_duplicate_case_rejection(self) -> None:
        case1 = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "dev")
        case2 = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 2, "dev")
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="dup_test",
                description="Duplicate test",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=False,
                manifest_type="TUNING_SPLIT",
                cases=(case1, case2),
            )
        self.assertIn("Duplicate case_id", str(ctx.exception))

    def test_partition_integrity(self) -> None:
        case = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "invalid_partition")
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="part_test",
                description="Partition test",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=False,
                manifest_type="TUNING_SPLIT",
                cases=(case,),
            )
        self.assertIn("invalid partition", str(ctx.exception))

    def test_repetition_family_leakage_detection(self) -> None:
        # Repetition 1 in dev, repetition 2 in val => must be rejected!
        case1 = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "dev")
        case2 = ManifestCase("c2", "RE2-OB", "ob", "cpu", "cartservice", 2, "val")
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="leakage_test",
                description="Leakage test",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=False,
                manifest_type="TUNING_SPLIT",
                cases=(case1, case2),
            )
        self.assertIn("Repetition-family leakage detected", str(ctx.exception))

    def test_deterministic_manifest_hash(self) -> None:
        case1 = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "dev")
        case2 = ManifestCase("c2", "RE2-OB", "ob", "mem", "checkoutservice", 1, "val")
        manifest_a = BenchmarkManifest(
            manifest_id="hash_test",
            description="Hash test",
            created_at="2026-09-17T00:00:00Z",
            is_smoke_test=False,
            manifest_type="TUNING_SPLIT",
            cases=(case1, case2),
        )
        manifest_b = BenchmarkManifest(
            manifest_id="hash_test",
            description="Hash test",
            created_at="2026-09-17T00:00:00Z",
            is_smoke_test=False,
            manifest_type="TUNING_SPLIT",
            cases=(case2, case1),  # reverse order in tuple
        )
        self.assertEqual(manifest_a.compute_hash(), manifest_b.compute_hash())
        self.assertIsInstance(manifest_a.compute_hash(), str)
        self.assertEqual(len(manifest_a.compute_hash()), 64)

    def test_hash_sensitivity_to_case_attributes(self) -> None:
        base_case = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "dev")
        base_m = BenchmarkManifest("h_test", "desc", "2026-09-17", False, (base_case,), "TUNING_SPLIT")
        base_hash = base_m.compute_hash()

        # 1. Changing case_id changes hash
        c_id = ManifestCase("c2", "RE2-OB", "ob", "cpu", "cartservice", 1, "dev")
        m_id = BenchmarkManifest("h_test", "desc", "2026-09-17", False, (c_id,), "TUNING_SPLIT")
        self.assertNotEqual(base_hash, m_id.compute_hash())

        # 2. Changing partition changes hash
        c_part = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "val")
        m_part = BenchmarkManifest("h_test", "desc", "2026-09-17", False, (c_part,), "TUNING_SPLIT")
        self.assertNotEqual(base_hash, m_part.compute_hash())

        # 3. Changing dataset changes hash
        c_dset = ManifestCase("c1", "RE1-OB", "ob", "cpu", "cartservice", 1, "dev")
        m_dset = BenchmarkManifest("h_test", "desc", "2026-09-17", False, (c_dset,), "TUNING_SPLIT")
        self.assertNotEqual(base_hash, m_dset.compute_hash())

        # 4. Changing grouping metadata (fault) changes hash
        c_fault = ManifestCase("c1", "RE2-OB", "ob", "mem", "cartservice", 1, "dev")
        m_fault = BenchmarkManifest("h_test", "desc", "2026-09-17", False, (c_fault,), "TUNING_SPLIT")
        self.assertNotEqual(base_hash, m_fault.compute_hash())

    def test_smoke_flag_validation(self) -> None:
        # Smoke manifest with non-smoke partition
        case_dev = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "dev")
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="smoke_bad_part",
                description="SMOKE test",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=True,
                manifest_type="SMOKE_REGRESSION",
                cases=(case_dev,),
            )
        self.assertIn("All smoke cases must have partition='smoke'", str(ctx.exception))

        # Smoke manifest without smoke/regression in description
        case_smoke = ManifestCase("c1", "RE2-OB", "ob", "cpu", "cartservice", 1, "smoke")
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="smoke_bad_desc",
                description="Standard test without keyword",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=True,
                manifest_type="SMOKE_REGRESSION",
                cases=(case_smoke,),
            )
        self.assertIn("must explicitly indicate smoke/regression status", str(ctx.exception))

        # Scientific manifest with smoke partition
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="sci_with_smoke",
                description="Scientific benchmark",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=False,
                manifest_type="SCIENTIFIC_BENCHMARK",
                cases=(case_smoke,),
            )
        self.assertIn("cannot contain case", str(ctx.exception))

    def test_manifest_serialization_and_loading(self) -> None:
        case = ManifestCase(
            case_id="c_1",
            dataset="RE2-OB",
            system="ob",
            fault="cpu",
            root_cause_service="cartservice",
            repetition=1,
            partition="all",
        )
        manifest = BenchmarkManifest(
            manifest_id="test_manifest_v1",
            description="Test scientific manifest",
            created_at="2026-09-17T00:00:00Z",
            is_smoke_test=False,
            manifest_type="SCIENTIFIC_BENCHMARK",
            cases=(case,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "manifest.json"
            manifest.save(file_path)
            self.assertTrue(file_path.is_file())

            # Load and verify integrity
            loaded = BenchmarkManifest.load(file_path)
            self.assertEqual(loaded.manifest_id, manifest.manifest_id)
            self.assertEqual(loaded.manifest_type, "SCIENTIFIC_BENCHMARK")
            self.assertEqual(len(loaded.cases), 1)
            self.assertEqual(loaded.cases[0].case_id, "c_1")
            self.assertEqual(loaded.cases[0].partition, "all")
            self.assertEqual(loaded.compute_hash(), manifest.compute_hash())

            # Corrupted count check
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["total_cases"] = 999
            corrupt_path = Path(tmpdir) / "corrupt_count.json"
            with open(corrupt_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            with self.assertRaises(ValueError) as ctx:
                BenchmarkManifest.load(corrupt_path)
            self.assertIn("case count mismatch", str(ctx.exception))

            # Corrupted hash check
            data["total_cases"] = 1
            data["manifest_hash"] = "0" * 64
            corrupt_hash_path = Path(tmpdir) / "corrupt_hash.json"
            with open(corrupt_hash_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            with self.assertRaises(ValueError) as ctx:
                BenchmarkManifest.load(corrupt_hash_path)
            self.assertIn("hash mismatch", str(ctx.exception))

    def test_leak_free_partitioning_groups_repetitions(self) -> None:
        case_rows = [
            {"case_id": "c_cpu_1", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 1},
            {"case_id": "c_cpu_2", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 2},
            {"case_id": "c_cpu_3", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 3},
            {"case_id": "c_mem_1", "dataset": "RE2-OB", "system": "ob", "fault": "mem", "root_cause_service": "cartservice", "repetition": 1},
            {"case_id": "c_mem_2", "dataset": "RE2-OB", "system": "ob", "fault": "mem", "root_cause_service": "cartservice", "repetition": 2},
        ]
        manifest = partition_cases_leak_free(case_rows, manifest_id="leak_free_test", description="Test")

        # Check cartservice_cpu repetitions
        cpu_partitions = {c.partition for c in manifest.cases if c.fault == "cpu"}
        self.assertEqual(len(cpu_partitions), 1, "All repetitions of cartservice_cpu must share the same partition!")

        # Check cartservice_mem repetitions
        mem_partitions = {c.partition for c in manifest.cases if c.fault == "mem"}
        self.assertEqual(len(mem_partitions), 1, "All repetitions of cartservice_mem must share the same partition!")

        # Check partition statistics
        self.assertIn("partition_statistics", manifest.to_dict())
        stats = manifest.partition_statistics
        self.assertEqual(stats["total_groups"], 2)
        self.assertEqual(stats["total_cases"], 5)

    def test_tuning_partition_deterministic_seed(self) -> None:
        case_rows = [
            {"case_id": "c_1", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 1},
            {"case_id": "c_2", "dataset": "RE2-OB", "system": "ob", "fault": "mem", "root_cause_service": "checkoutservice", "repetition": 1},
            {"case_id": "c_3", "dataset": "RE2-OB", "system": "ob", "fault": "delay", "root_cause_service": "frontend", "repetition": 1},
            {"case_id": "c_4", "dataset": "RE2-OB", "system": "ob", "fault": "loss", "root_cause_service": "paymentservice", "repetition": 1},
        ]
        m1 = partition_cases_leak_free(case_rows, "t1", "desc", seed=123)
        m2 = partition_cases_leak_free(case_rows, "t1", "desc", seed=123)
        self.assertEqual(m1.compute_hash(), m2.compute_hash())
        for c1, c2 in zip(m1.cases, m2.cases):
            self.assertEqual(c1.partition, c2.partition)

    def test_invalid_ratios_rejected(self) -> None:
        case_rows = [
            {"case_id": "c_1", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 1},
        ]
        with self.assertRaises(ValueError):
            partition_cases_leak_free(case_rows, "t1", "desc", dev_ratio=0.8, val_ratio=0.3)

    def test_scenario_family_cross_suite_separation(self) -> None:
        """Verify (suite, system, root_cause_service, fault) separates cross-suite families."""
        # 1. Verify (RE1, ob, serviceA, cpu) and (RE2, ob, serviceA, cpu) are treated as different families
        case_re1 = ManifestCase("c1", "RE1-OB", "ob", "cpu", "serviceA", 1, "dev", suite="RE1")
        case_re2 = ManifestCase("c2", "RE2-OB", "ob", "cpu", "serviceA", 1, "val", suite="RE2")
        self.assertNotEqual(case_re1.scenario_family, case_re2.scenario_family)
        self.assertEqual(case_re1.scenario_family, ("RE1", "ob", "serviceA", "cpu"))
        self.assertEqual(case_re2.scenario_family, ("RE2", "ob", "serviceA", "cpu"))

        # In a TUNING_SPLIT manifest, they can be in different partitions because they are separate families
        manifest = BenchmarkManifest(
            manifest_id="cross_suite_tuning",
            description="Tuning split with cross-suite families",
            created_at="2026-09-17T00:00:00Z",
            is_smoke_test=False,
            manifest_type="TUNING_SPLIT",
            cases=(case_re1, case_re2),
        )
        self.assertEqual(len(manifest.cases), 2)

        # 2. Verify repetitions within the same family CANNOT cross partitions
        case_re1_rep2 = ManifestCase("c3", "RE1-OB", "ob", "cpu", "serviceA", 2, "val", suite="RE1")
        with self.assertRaises(ValueError) as ctx:
            BenchmarkManifest(
                manifest_id="leakage_within_suite",
                description="Tuning split with within-suite leakage",
                created_at="2026-09-17T00:00:00Z",
                is_smoke_test=False,
                manifest_type="TUNING_SPLIT",
                cases=(case_re1, case_re1_rep2),
            )
        self.assertIn("Repetition-family leakage detected", str(ctx.exception))

        # 3. Verify partition_cases_leak_free keeps repetitions together and remains reproducible
        rows = [
            {"case_id": "re1_1", "suite": "RE1", "dataset": "RE1-OB", "system": "ob", "fault": "cpu", "root_cause_service": "serviceA", "repetition": 1},
            {"case_id": "re1_2", "suite": "RE1", "dataset": "RE1-OB", "system": "ob", "fault": "cpu", "root_cause_service": "serviceA", "repetition": 2},
            {"case_id": "re2_1", "suite": "RE2", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "serviceA", "repetition": 1},
            {"case_id": "re2_2", "suite": "RE2", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "serviceA", "repetition": 2},
        ]
        m1 = partition_cases_leak_free(rows, manifest_id="gen_cross_suite", description="Cross suite test", seed=42)
        m2 = partition_cases_leak_free(rows, manifest_id="gen_cross_suite", description="Cross suite test", seed=42)

        # Hash and assignments identical across runs with same seed
        self.assertEqual(m1.compute_hash(), m2.compute_hash())
        self.assertEqual(m1.partition_statistics["total_groups"], 2)
        self.assertEqual(m1.partition_statistics["total_cases"], 4)

        # Repetitions within RE1 family remain together
        re1_parts = {c.partition for c in m1.cases if c.suite == "RE1"}
        self.assertEqual(len(re1_parts), 1)

        # Repetitions within RE2 family remain together
        re2_parts = {c.partition for c in m1.cases if c.suite == "RE2"}
        self.assertEqual(len(re2_parts), 1)


if __name__ == "__main__":
    unittest.main()
