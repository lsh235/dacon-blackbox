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
    def test_stage1_training_loader_randomly_returns_16_24_or_32_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "variable.mp4"
            video.write_bytes(b"not-read-by-loader")
            processed = root / "processed"
            target = stage1_feature_path(
                processed,
                video,
                size=8,
                frames=32,
                slot=0,
                slots=1,
                jitter_frames=0,
                forensic_size=12,
            )
            target.parent.mkdir(parents=True)
            rgb = np.zeros((3, 32, 8, 8), dtype=np.uint8)
            forensic_rgb = np.zeros((3, 32, 12, 12), dtype=np.uint8)
            np.savez_compressed(target, rgb=rgb, forensic_rgb=forensic_rgb)
            dataset = Stage1TrainingDataset(
                [(video, 0)],
                size=8,
                frames=32,
                sequence_lengths=(16, 24, 32),
                feature_mode="rgb_fft",
                slots=1,
                jitter_frames=0,
                forensic_size=12,
                fft_size=4,
                row_profile_bins=2,
                random_jitter=True,
                processed_root=processed,
            )
            torch.manual_seed(20260828)

            observed = {dataset[0][0]["rgb_clip"].shape[1] for _ in range(30)}

        self.assertEqual(observed, {16, 24, 32})

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
                jitter_frames=1,
                forensic_size=12,
            )
            target.parent.mkdir(parents=True)
            rgb = np.full((3, 4, 8, 8), 127, dtype=np.uint8)
            forensic_rgb = np.full((3, 4, 12, 12), 127, dtype=np.uint8)
            np.savez_compressed(target, rgb=rgb, forensic_rgb=forensic_rgb)
            dataset = Stage1TrainingDataset(
                [(video, 0)],
                size=8,
                frames=2,
                feature_mode="rgb_fft",
                slots=1,
                jitter_frames=1,
                forensic_size=12,
                fft_size=4,
                row_profile_bins=2,
                random_jitter=False,
                processed_root=processed,
            )
            with patch(
                "blackbox.stages.stage1.dataset.decode_contiguous_views",
                side_effect=AssertionError("training loader must not decode video"),
            ):
                inputs, label, video_index = dataset[0]
        self.assertEqual(label, 0)
        self.assertEqual(video_index, 0)
        self.assertEqual(tuple(inputs["rgb_clip"].shape), (3, 2, 8, 8))
        self.assertEqual(tuple(inputs["fft_clip"].shape), (3, 2, 4, 4))
        self.assertEqual(tuple(inputs["flicker"].shape), (4, 2))

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
