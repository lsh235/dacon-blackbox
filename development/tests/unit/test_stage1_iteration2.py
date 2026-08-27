from __future__ import annotations

import unittest

import torch

from blackbox.common.runtime import CheckpointError
from blackbox.stages.stage1.baseline import DEFAULT_TTA_SLOTS, resolve_tta_slots
from blackbox.stages.stage1.dataset import Stage1TrainAugmentation


class Stage1AugmentationTests(unittest.TestCase):
    def test_disabled_augmentation_keeps_clip_unchanged(self) -> None:
        clip = torch.linspace(0.0, 1.0, 3 * 4 * 12 * 12).reshape(3, 4, 12, 12)

        output = Stage1TrainAugmentation(
            color_jitter_probability=0.0,
            affine_probability=0.0,
        )(clip)

        torch.testing.assert_close(output, clip)

    def test_clip_consistent_augmentation_preserves_temporal_identity(self) -> None:
        single_frame = torch.rand(3, 24, 24)
        clip = single_frame.unsqueeze(1).repeat(1, 4, 1, 1)
        torch.manual_seed(20260827)

        output = Stage1TrainAugmentation(
            color_jitter_probability=1.0,
            affine_probability=1.0,
        )(clip)

        self.assertEqual(tuple(output.shape), tuple(clip.shape))
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertGreaterEqual(output.min().item(), 0.0)
        self.assertLessEqual(output.max().item(), 1.0)
        for frame_index in range(1, output.shape[1]):
            torch.testing.assert_close(output[:, 0], output[:, frame_index])


class Stage1TtaTests(unittest.TestCase):
    def test_legacy_checkpoint_uses_three_temporal_slots(self) -> None:
        self.assertEqual(resolve_tta_slots({}), DEFAULT_TTA_SLOTS)

    def test_checkpoint_tta_slot_validation(self) -> None:
        self.assertEqual(resolve_tta_slots({"sampling": {"inference_tta_slots": 5}}), 5)
        with self.assertRaises(CheckpointError):
            resolve_tta_slots({"sampling": {"inference_tta_slots": 0}})


if __name__ == "__main__":
    unittest.main()
