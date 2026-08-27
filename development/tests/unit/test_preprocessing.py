from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from blackbox.preprocessing import PREPROCESS_SCHEMA, stage1_feature_path
from blackbox.stages.stage1.dataset import Stage1TrainingDataset
from blackbox.video_metadata import inspect_video


class OfflinePreprocessingTests(unittest.TestCase):
    def test_stage1_training_loader_reads_feature_without_opening_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "sample.mp4"
            video.write_bytes(b"not-read-by-loader")
            processed = root / "processed"
            target = stage1_feature_path(
                processed,
                video,
                size=8,
                frames=2,
                slot=0,
                slots=1,
                feature_mode="rgb_fft",
            )
            target.parent.mkdir(parents=True)
            expected = np.ones((6, 2, 8, 8), dtype=np.float32)
            np.save(target, expected)
            dataset = Stage1TrainingDataset(
                [(video, 0)],
                size=8,
                frames=2,
                feature_mode="rgb_fft",
                processed_root=processed,
            )
            with patch(
                "blackbox.stages.stage1.dataset.decode_uniform_clip",
                side_effect=AssertionError("training loader must not decode video"),
            ):
                feature, label = dataset[0]
        self.assertEqual(label, 0)
        self.assertTrue(torch.equal(feature, torch.from_numpy(expected)))

    def test_metadata_diagnostic_recommends_label_derived_fps(self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.reads = 0

            def isOpened(self) -> bool:
                return True

            def get(self, property: int) -> float:
                return 480.0 if property == 5 else 1200.0

            def read(self):
                self.reads += 1
                return (self.reads <= 1200, None)

            def release(self) -> None:
                return None

        labels = pd.DataFrame([
            {"ID": "OPEN_001", "sample_index": 60, "frame_index": 120, "time_seconds": 6.0}
        ])
        with patch("blackbox.video_metadata.cv2.VideoCapture", return_value=Capture()):
            report = inspect_video("OPEN_001.mp4", labels, threshold=0.10)
        self.assertTrue(report.flagged)
        self.assertIn("cap_prop_fps_vs_label_source_fps", report.reasons)
        self.assertEqual(report.recommended_fps, 20.0)
        self.assertEqual(report.label_sequence_length, 61)


if __name__ == "__main__":
    unittest.main()
