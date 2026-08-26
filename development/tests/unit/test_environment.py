from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from blackbox.environment import audit_environment, read_pinned_requirements


class EnvironmentAuditTests(unittest.TestCase):
    def test_reads_exact_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("numpy==1.26.4\npandas==2.2.2\n", encoding="utf-8")
            self.assertEqual(
                read_pinned_requirements(requirements),
                {"numpy": "1.26.4", "pandas": "2.2.2"},
            )

    def test_rejects_unpinned_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements.txt"
            requirements.write_text("numpy>=1.26\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact == pin"):
                read_pinned_requirements(requirements)

    def test_current_environment_matches_project_pins(self) -> None:
        requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
        report = audit_environment(requirements, require_cuda=False)
        self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
