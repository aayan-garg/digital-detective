"""Unit tests for BenchmarkManifest and leak-free partitioning.
"""

import json
import tempfile
import unittest
from pathlib import Path

from eval.manifest import (
    BenchmarkManifest,
    ManifestCase,
    create_smoke_re2_ob_rep1_manifest,
    partition_cases_leak_free,
)


class TestEvalManifest(unittest.TestCase):
    def test_smoke_manifest_creation(self) -> None:
        case_rows = [
            {"case": "re2ob_cartservice_cpu_1", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 1},
            {"case": "re2ob_cartservice_cpu_2", "dataset": "RE2-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 2},
            {"case": "re1ob_cartservice_cpu_1", "dataset": "RE1-OB", "system": "ob", "fault": "cpu", "root_cause_service": "cartservice", "repetition": 1},
        ]
        manifest = create_smoke_re2_ob_rep1_manifest(case_rows)
        self.assertTrue(manifest.is_smoke_test)
        self.assertEqual(len(manifest.cases), 1)
        self.assertEqual(manifest.cases[0].case_id, "re2ob_cartservice_cpu_1")
        self.assertEqual(manifest.cases[0].partition, "smoke")

    def test_manifest_serialization_and_loading(self) -> None:
        case = ManifestCase(
            case_id="c_1",
            dataset="RE2-OB",
            system="ob",
            fault="cpu",
            root_cause_service="cartservice",
            repetition=1,
            partition="dev",
        )
        manifest = BenchmarkManifest(
            manifest_id="test_manifest_v1",
            description="Test manifest",
            created_at="2026-09-17T00:00:00Z",
            is_smoke_test=False,
            cases=(case,),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "manifest.json"
            manifest.save(file_path)
            self.assertTrue(file_path.is_file())

            loaded = BenchmarkManifest.load(file_path)
            self.assertEqual(loaded.manifest_id, manifest.manifest_id)
            self.assertEqual(len(loaded.cases), 1)
            self.assertEqual(loaded.cases[0].case_id, "c_1")
            self.assertEqual(loaded.cases[0].partition, "dev")

    def test_leak_free_partitioning_groups_repetitions(self) -> None:
        # Repetitions 1, 2, 3 of the same service/fault combination must be in the SAME partition!
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


if __name__ == "__main__":
    unittest.main()
