#!/usr/bin/env python3
"""Quick runtime smoke for Stage schemas and the submission ZIP validator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from blackbox.contracts import validate_prediction_frame
from blackbox.submission import REQUIRED_FILES, validate_submission_zip


def main() -> int:
    samples = {
        "stage1": pd.DataFrame([{"ID": "S1_001", "answer": "ORIGINAL"}]),
        "stage2": pd.DataFrame(
            [
                {
                    "ID": "S2_001",
                    "collision_frame": 31,
                    "entry_frame": 12,
                    "evasion_space": 1,
                    "entry_side": "RIGHT",
                }
            ]
        ),
        "stage3": pd.DataFrame(
            [
                {
                    "ID": "S3_001",
                    "sample_index": 0,
                    "accel_label": "CONSTANT",
                    "steer_label": "STRAIGHT",
                }
            ]
        ),
    }
    for stage, frame in samples.items():
        validate_prediction_frame(stage, frame)

    inference_source = "\n".join(
        f"def {name}(data_dir, model_dir):\n    return None\n"
        for name in ("predict_stage1", "predict_stage2", "predict_stage3")
    )
    with tempfile.TemporaryDirectory(prefix="blackbox-contract-smoke-") as temporary:
        archive_path = Path(temporary) / "submit.zip"
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for name in REQUIRED_FILES:
                archive.writestr(name, inference_source if name == "inference.py" else "placeholder")
        validate_submission_zip(archive_path)
    print("[OK] prediction contracts and submission ZIP smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
