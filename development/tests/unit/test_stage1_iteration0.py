from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from blackbox.stages.stage1.baseline import Stage1MViT
from blackbox.stages.stage1.dataset import (
    RGB_FEATURES,
    RGB_FFT_FEATURES,
    feature_channels,
    prepare_stage1_features,
    spatial_log_spectrum,
    uniform_frame_indices,
)
from blackbox.stages.stage1.losses import FocalLoss


class Stage1UniformSamplingTests(unittest.TestCase):
    def test_uniform_indices_cover_the_full_video(self) -> None:
        indices = uniform_frame_indices(10, 4)

        self.assertEqual(indices.tolist(), [0, 3, 6, 9])

    def test_short_video_repeats_indices_to_keep_fixed_clip_length(self) -> None:
        indices = uniform_frame_indices(3, 5)

        self.assertEqual(indices.tolist(), [0, 0, 1, 2, 2])
        self.assertEqual(len(indices), 5)
        self.assertTrue(all(left <= right for left, right in zip(indices, indices[1:])))


class Stage1FrequencyFeatureTests(unittest.TestCase):
    def test_rgb_and_rgb_fft_feature_contracts(self) -> None:
        clip = torch.linspace(0.0, 1.0, 3 * 4 * 8 * 10, dtype=torch.float32).reshape(
            3, 4, 8, 10
        )

        rgb = prepare_stage1_features(clip, RGB_FEATURES)
        rgb_fft = prepare_stage1_features(clip, RGB_FFT_FEATURES)

        self.assertEqual(feature_channels(RGB_FEATURES), 3)
        self.assertEqual(feature_channels(RGB_FFT_FEATURES), 6)
        self.assertEqual(tuple(rgb.shape), (3, 4, 8, 10))
        self.assertEqual(tuple(rgb_fft.shape), (6, 4, 8, 10))
        self.assertEqual(rgb.dtype, clip.dtype)
        self.assertEqual(rgb_fft.dtype, clip.dtype)
        self.assertTrue(torch.isfinite(rgb).all().item())
        self.assertTrue(torch.isfinite(rgb_fft).all().item())

    def test_constant_input_has_finite_frequency_features(self) -> None:
        clip = torch.ones(3, 2, 8, 8, dtype=torch.float32)

        spectrum = spatial_log_spectrum(clip)
        features = prepare_stage1_features(clip, RGB_FFT_FEATURES)

        self.assertEqual(tuple(spectrum.shape), tuple(clip.shape))
        self.assertEqual(spectrum.dtype, clip.dtype)
        self.assertTrue(torch.isfinite(spectrum).all().item())
        self.assertTrue(torch.isfinite(features).all().item())

    def test_model_projection_matches_selected_feature_channels(self) -> None:
        for feature_mode in (RGB_FEATURES, RGB_FFT_FEATURES):
            with self.subTest(feature_mode=feature_mode):
                model = Stage1MViT(feature_mode=feature_mode)
                self.assertEqual(
                    model.net.conv_proj.in_channels,
                    feature_channels(feature_mode),
                )


class FocalLossTests(unittest.TestCase):
    def test_gamma_zero_matches_cross_entropy(self) -> None:
        logits = torch.tensor(
            [[1.5, -0.25], [-0.75, 0.5], [0.1, -0.2]], dtype=torch.float64
        )
        targets = torch.tensor([0, 1, 1])

        expected = F.cross_entropy(logits, targets)
        actual = FocalLoss(gamma=0.0)(logits, targets)

        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_focal_modulation_suppresses_easy_example_more(self) -> None:
        logits = torch.tensor([[3.0, -3.0], [0.0, 0.0]], dtype=torch.float64)
        targets = torch.tensor([0, 0])
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")

        focal = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
        modulation = focal / cross_entropy

        self.assertLess(modulation[0].item(), modulation[1].item())
        self.assertLess(focal[0].item(), cross_entropy[0].item())

    def test_extreme_logits_have_finite_loss_and_gradients(self) -> None:
        logits = torch.tensor(
            [[1000.0, -1000.0], [-1000.0, 1000.0], [1000.0, -1000.0]],
            requires_grad=True,
        )
        targets = torch.tensor([0, 0, 1])

        loss = FocalLoss(gamma=2.0)(logits, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())


if __name__ == "__main__":
    unittest.main()
