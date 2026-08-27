from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from blackbox.submission import REQUIRED_FILES
from blackbox.submission_builder import COMBINED_COLUMNS, build_submission_package


class SubmissionBuilderTests(unittest.TestCase):
    def test_builder_merges_valid_stages_and_creates_contract_zip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "submissions"
            input_dir.mkdir()
            pd.DataFrame([{"ID": "S1", "answer": "ORIGINAL"}]).to_csv(
                input_dir / "stage1_submission.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "ID": "S2",
                        "collision_frame": 10,
                        "entry_frame": 3,
                        "evasion_space": 1,
                        "entry_side": "RIGHT",
                    }
                ]
            ).to_csv(input_dir / "stage2_submission.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ID": "S3",
                        "sample_index": 0,
                        "accel_label": "CONSTANT",
                        "steer_label": "STRAIGHT",
                    }
                ]
            ).to_csv(input_dir / "stage3_submission.csv", index=False)

            model_root = root / "models"
            for relative in (
                "stage1/best.pt",
                "stage2/best.pt",
                "stage2/resnet18-f37072fd.pth",
                "stage3/best.pt",
            ):
                path = model_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test-model")
            source_root = root / "src"
            package = source_root / "blackbox"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            requirements = root / "requirements.txt"
            requirements.write_text("torch==2.8.0\n", encoding="utf-8")
            entrypoint = root / "inference.py"
            entrypoint.write_text(
                "def predict_stage1(data_dir, model_dir): pass\n"
                "def predict_stage2(data_dir, model_dir): pass\n"
                "def predict_stage3(data_dir, model_dir): pass\n",
                encoding="utf-8",
            )
            report = build_submission_package(
                input_dir=input_dir,
                model_root=model_root,
                output_dir=root / "output",
                source_root=source_root,
                requirements=requirements,
                entrypoint=entrypoint,
            )

            combined = pd.read_csv(report.combined_csv)
            self.assertEqual(combined.columns.tolist(), COMBINED_COLUMNS)
            self.assertEqual(combined["stage"].tolist(), ["stage1", "stage2", "stage3"])
            self.assertEqual(report.combined_rows, 3)
            with ZipFile(report.archive) as archive:
                names = set(archive.namelist())
            self.assertTrue(REQUIRED_FILES.issubset(names))
            self.assertIn("submission.csv", names)
            self.assertIn("blackbox/__init__.py", names)


if __name__ == "__main__":
    unittest.main()
