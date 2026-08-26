from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from blackbox.submission import (
    REQUIRED_FILES,
    SubmissionValidationError,
    validate_submission_zip,
)


INFERENCE_SOURCE = """
def predict_stage1(data_dir, model_dir):
    pass

def predict_stage2(data_dir, model_dir):
    pass

def predict_stage3(data_dir, model_dir):
    pass
"""


def _write_submission(path: Path, *, inference: str = INFERENCE_SOURCE, omit: str | None = None) -> None:
    with ZipFile(path, "w") as archive:
        for name in REQUIRED_FILES:
            if name == omit:
                continue
            archive.writestr(name, inference if name == "inference.py" else "placeholder")


class SubmissionValidationTests(unittest.TestCase):
    def test_accepts_documented_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submit.zip"
            _write_submission(path)
            report = validate_submission_zip(path)
            self.assertEqual(report.functions, ["predict_stage1", "predict_stage2", "predict_stage3"])

    def test_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submit.zip"
            _write_submission(path, omit="model/stage3/best.pt")
            with self.assertRaisesRegex(SubmissionValidationError, "missing required files"):
                validate_submission_zip(path)

    def test_rejects_missing_function(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submit.zip"
            _write_submission(path, inference="def predict_stage1(data_dir, model_dir):\n    pass\n")
            with self.assertRaisesRegex(SubmissionValidationError, "missing top-level"):
                validate_submission_zip(path)

    def test_rejects_parent_path_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submit.zip"
            _write_submission(path)
            with ZipFile(path, "a") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(SubmissionValidationError, "unsafe archive path"):
                validate_submission_zip(path)


if __name__ == "__main__":
    unittest.main()
