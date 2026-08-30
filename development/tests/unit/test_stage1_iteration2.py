from __future__ import annotations

import unittest

import torch

from blackbox.common.runtime import CheckpointError
from blackbox.stages.stage1.baseline import (
    DEFAULT_TTA_SLOTS,
    BestStateTracker,
    _video_level_predictions,
    resolve_tta_slots,
)
from blackbox.stages.stage1.dataset import (
    RGB_FFT_FEATURES,
    Stage1TrainAugmentation,
    prepare_stage1_inputs,
    stage1_augmentation_profile,
    spatial_log_spectrum,
)


class Stage1AugmentationTests(unittest.TestCase):
    def test_disabled_augmentation_keeps_clip_unchanged(self) -> None:
        clip = torch.linspace(0.0, 1.0, 3 * 4 * 12 * 12).reshape(3, 4, 12, 12)

        output = Stage1TrainAugmentation(
            color_jitter_probability=0.0,
            affine_probability=0.0,
            occlusion_probability=0.0,
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

    def test_same_parameters_are_applied_before_fft_is_recomputed(self) -> None:
        clip = torch.rand(3, 4, 24, 24)
        augmentation = Stage1TrainAugmentation(
            color_jitter_probability=1.0,
            affine_probability=1.0,
        )
        torch.manual_seed(20260828)

        augmented_rgb, augmented_forensic = augmentation.apply_views(clip, clip.clone())
        inputs = prepare_stage1_inputs(
            augmented_rgb,
            augmented_forensic,
            feature_mode=RGB_FFT_FEATURES,
            fft_size=12,
            row_profile_bins=4,
        )

        torch.testing.assert_close(augmented_rgb, augmented_forensic)
        torch.testing.assert_close(
            inputs["fft_clip"],
            spatial_log_spectrum(augmented_forensic, output_size=12),
        )

    def test_default_raft_style_probabilities_and_hue_are_recorded(self) -> None:
        config = Stage1TrainAugmentation().checkpoint_config()

        self.assertEqual(config["profile"], "aggressive_photometric_occlusion")
        self.assertEqual(config["color_jitter_probability"], 0.8)
        self.assertEqual(config["occlusion_probability"], 0.5)
        self.assertEqual(config["hue"], 0.1)
        self.assertTrue(config["enable_photometric"])
        self.assertTrue(config["enable_occlusion"])

    def test_photometric_and_occlusion_can_be_toggled_independently(self) -> None:
        photometric = Stage1TrainAugmentation(
            enable_photometric=True,
            enable_occlusion=False,
            color_jitter_probability=1.0,
            affine_probability=0.0,
            occlusion_probability=1.0,
        )._sample_parameters()
        occlusion = Stage1TrainAugmentation(
            enable_photometric=False,
            enable_occlusion=True,
            color_jitter_probability=1.0,
            affine_probability=0.0,
            occlusion_probability=1.0,
        )._sample_parameters()

        self.assertTrue(photometric.color_operations)
        self.assertIsNone(photometric.occlusion)
        self.assertFalse(occlusion.color_operations)
        self.assertIsNotNone(occlusion.occlusion)

    def test_mode_g_ablation_profiles_are_exact_and_independent_copies(self) -> None:
        no_aug = stage1_augmentation_profile("mode_g_no_aug")
        photo_only = stage1_augmentation_profile("mode_g_photo_only")

        self.assertEqual(
            no_aug,
            {
                "enable_photometric": False,
                "enable_occlusion": False,
                "enable_affine": False,
            },
        )
        self.assertEqual(
            photo_only,
            {
                "enable_photometric": True,
                "enable_occlusion": False,
                "enable_affine": False,
            },
        )
        no_aug["enable_photometric"] = True
        self.assertFalse(stage1_augmentation_profile("mode_g_no_aug")["enable_photometric"])

    def test_no_aug_profile_has_no_effective_transform(self) -> None:
        flags = stage1_augmentation_profile("mode_g_no_aug")
        augmentation = Stage1TrainAugmentation(
            enable_photometric=flags["enable_photometric"],
            enable_occlusion=flags["enable_occlusion"],
            affine_probability=0.35 if flags["enable_affine"] else 0.0,
        )

        self.assertFalse(augmentation.enabled)
        self.assertEqual(augmentation.profile_name, "disabled")

    def test_occlusion_is_clip_consistent_and_shared_between_views(self) -> None:
        rgb = torch.ones(3, 4, 24, 24)
        forensic = torch.ones(3, 4, 12, 12)
        augmentation = Stage1TrainAugmentation(
            color_jitter_probability=0.0,
            affine_probability=0.0,
            occlusion_probability=1.0,
            occlusion_scale=(0.2, 0.2),
            occlusion_aspect_ratio=(1.0, 1.0),
        )
        torch.manual_seed(20260828)

        augmented_rgb, augmented_forensic = augmentation.apply_views(rgb, forensic)

        self.assertGreater((augmented_rgb == 0).float().mean().item(), 0.1)
        self.assertGreater((augmented_forensic == 0).float().mean().item(), 0.1)
        for frame_index in range(1, rgb.shape[1]):
            torch.testing.assert_close(
                augmented_rgb[:, 0],
                augmented_rgb[:, frame_index],
            )
            torch.testing.assert_close(
                augmented_forensic[:, 0],
                augmented_forensic[:, frame_index],
            )


class Stage1TtaTests(unittest.TestCase):
    def test_legacy_checkpoint_uses_three_temporal_slots(self) -> None:
        self.assertEqual(resolve_tta_slots({}), DEFAULT_TTA_SLOTS)

    def test_checkpoint_tta_slot_validation(self) -> None:
        self.assertEqual(resolve_tta_slots({"sampling": {"inference_tta_slots": 5}}), 5)
        with self.assertRaises(CheckpointError):
            resolve_tta_slots({"sampling": {"inference_tta_slots": 0}})

    def test_video_level_validation_averages_three_region_probabilities(self) -> None:
        predictions = _video_level_predictions(
            [[0.2, 0.7, 0.4], [0.7, 0.8, 0.9]],
        )

        self.assertEqual(predictions, [0, 1])


class BestStateTrackerTests(unittest.TestCase):
    def test_restore_uses_best_metric_state_not_last_state(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        tracker = BestStateTracker(model, mode="max")
        tracker.consider(model, value=0.8, epoch=1)
        with torch.no_grad():
            model.weight.fill_(2.0)
        tracker.consider(model, value=0.7, epoch=2)

        tracker.restore(model)

        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
        self.assertEqual(tracker.best_epoch, 1)
        self.assertEqual(tracker.best_value, 0.8)


if __name__ == "__main__":
    unittest.main()
