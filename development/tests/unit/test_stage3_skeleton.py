from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from blackbox.stages.stage2.dataset_stage2 import IGNORE_INDEX
from blackbox.stages.stage3.dataset_stage3 import (
    Stage3Annotation,
    Stage3SequenceWindowDataset,
    Stage3VideoRecord,
)
from blackbox.stages.stage3.model_stage3 import Stage3TwoStreamBiLSTM


class Stage3SequenceDatasetTests(unittest.TestCase):
    def test_sparse_labels_keep_source_frame_and_sample_index_separate(self) -> None:
        frames = torch.zeros(3, 3, 16, 16)
        frames[1, :, 4:10, 5:11] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "OPEN_001.mp4"
            video.write_bytes(b"stage3-fixture")
            record = Stage3VideoRecord(
                "OPEN_001",
                video,
                annotations=(
                    Stage3Annotation(7, 1, 0.7, 0, 2),
                    Stage3Annotation(99, 9, 9.9, 3, 0),
                ),
            )
            with (
                patch("blackbox.stages.stage2.dataset_stage2.video_frame_count", return_value=3),
                patch(
                    "blackbox.stages.stage2.dataset_stage2.decode_stage2_window",
                    return_value=(frames, 3),
                ),
            ):
                dataset = Stage3SequenceWindowDataset(
                    [record],
                    window_frames=3,
                    stride=3,
                    size=16,
                    flow_cache_dir=Path(temporary) / "cache",
                )
                sample = dataset[0]
        self.assertEqual(sample["frame_numbers"].tolist(), [0, 1, 2])
        self.assertEqual(sample["sample_indices"].tolist(), [IGNORE_INDEX, 7, IGNORE_INDEX])
        self.assertEqual(sample["accel_targets"].tolist(), [IGNORE_INDEX, 0, IGNORE_INDEX])
        self.assertEqual(sample["steer_targets"].tolist(), [IGNORE_INDEX, 2, IGNORE_INDEX])


class Stage3TwoStreamModelTests(unittest.TestCase):
    def test_seq2seq_heads_mask_padded_steps(self) -> None:
        model = Stage3TwoStreamBiLSTM(hidden_size=8, layers=1, frame_batch_size=1).eval()
        with torch.inference_mode():
            outputs = model(
                torch.zeros(1, 3, 3, 64, 64),
                torch.zeros(1, 3, 2, 64, 64),
                torch.tensor([2]),
            )
        self.assertEqual(tuple(outputs["accel_logits"].shape), (1, 3, 4))
        self.assertEqual(tuple(outputs["steer_logits"].shape), (1, 3, 3))
        self.assertTrue(torch.isneginf(outputs["accel_logits"][0, 2]).all().item())
        self.assertTrue(torch.isneginf(outputs["steer_logits"][0, 2]).all().item())


if __name__ == "__main__":
    unittest.main()
