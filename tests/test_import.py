"""Smoke tests for package setup."""

import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_imports(self) -> None:
        import digital_detective

        self.assertIsNotNone(digital_detective)
