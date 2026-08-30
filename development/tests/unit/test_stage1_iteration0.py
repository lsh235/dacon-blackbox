from __future__ import annotations

import gc
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from blackbox.stages.stage1.baseline import (
    AllPairsCorrelationPyramid,
    ConvGRUCell,
    MotionConsistencyEncoder,
    MultiStageTemporalRefinementHead,
    RecurrentCorrelationUpdate,
    Stage1MViT,
    _auxiliary_batch_diagnostics,
    _loss_balance_diagnostics,
    inverse_frequency_focal_alpha,
    linear_warmup_learning_rate,
    stage1_early_stopping_triggered,
)
from blackbox.stages.stage1.dataset import (
    RGB_FEATURES,
    RGB_FFT_FEATURES,
    contiguous_frame_indices,
    feature_channels,
    prepare_stage1_inputs,
    spatial_log_spectrum,
    temporal_flicker_features,
)
from blackbox.stages.stage1.losses import (
    FocalLoss,
    Stage1MultiTaskLoss,
    explainability_reconstruction_terms,
    truncated_temporal_mse,
)


class Stage1ContiguousSamplingTests(unittest.TestCase):
    def test_three_regions_return_centered_consecutive_frames(self) -> None:
        actual = [
            contiguous_frame_indices(90, 4, slot=slot, slots=3).tolist()
            for slot in range(3)
        ]

        self.assertEqual(actual, [[13, 14, 15, 16], [43, 44, 45, 46], [73, 74, 75, 76]])

    def test_jitter_context_contains_only_contiguous_training_crops(self) -> None:
        context = contiguous_frame_indices(
            90,
            4,
            slot=1,
            slots=3,
            context_jitter_frames=3,
        )

        self.assertEqual(len(context), 10)
        for offset in range(7):
            crop = context[offset : offset + 4]
            np.testing.assert_array_equal(np.diff(crop), np.ones(3, dtype=np.int64))

    def test_short_region_repeats_boundary_frames(self) -> None:
        indices = contiguous_frame_indices(6, 4, slot=0, slots=3)

        self.assertEqual(indices.tolist(), [0, 1, 1, 1])
        self.assertEqual(len(indices), 4)


class Stage1MultiStreamFeatureTests(unittest.TestCase):
    def test_rgb_fft_and_flicker_are_separate_tensors(self) -> None:
        rgb = torch.linspace(0.0, 1.0, 3 * 4 * 32 * 40).reshape(3, 4, 32, 40)
        forensic = F.interpolate(
            rgb.permute(1, 0, 2, 3),
            size=(48, 48),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)

        inputs = prepare_stage1_inputs(
            rgb,
            forensic,
            feature_mode=RGB_FFT_FEATURES,
            fft_size=16,
            row_profile_bins=4,
        )

        self.assertEqual(feature_channels(RGB_FEATURES), 3)
        self.assertEqual(feature_channels(RGB_FFT_FEATURES), 3)
        self.assertEqual(tuple(inputs["rgb_clip"].shape), (3, 4, 32, 40))
        self.assertEqual(tuple(inputs["fft_clip"].shape), (3, 4, 16, 16))
        self.assertEqual(tuple(inputs["flicker"].shape), (6, 4))
        self.assertTrue(all(torch.isfinite(value).all() for value in inputs.values()))

    def test_constant_input_has_finite_frequency_features(self) -> None:
        clip = torch.ones(3, 2, 8, 8, dtype=torch.float32)

        spectrum = spatial_log_spectrum(clip, output_size=4)

        self.assertEqual(tuple(spectrum.shape), (3, 2, 4, 4))
        self.assertTrue(torch.isfinite(spectrum).all().item())

    def test_temporal_features_include_y_delta_and_row_profiles(self) -> None:
        levels = torch.tensor([0.1, 0.2, 0.4, 0.8])
        clip = levels[None, :, None, None].repeat(3, 1, 8, 8)

        features = temporal_flicker_features(clip, row_profile_bins=4)

        self.assertEqual(tuple(features.shape), (6, 4))
        torch.testing.assert_close(features[0], levels)
        torch.testing.assert_close(features[1], torch.tensor([0.0, 0.1, 0.2, 0.4]))

    def test_mvit_keeps_three_channel_projection_and_owns_auxiliary_branches(self) -> None:
        model = Stage1MViT(feature_mode=RGB_FFT_FEATURES, row_profile_bins=4)

        self.assertEqual(model.rgb_backbone.conv_proj.in_channels, 3)
        self.assertEqual(model.spatial_branch.frame_encoder[0].in_channels, 3)
        self.assertEqual(model.temporal_branch.input_channels, 6)
        dilations = [block.layers[0].dilation[0] for block in model.temporal_branch.blocks]
        self.assertEqual(dilations, [1, 2, 4])
        self.assertEqual(
            set(model.branch_dimensions),
            {"rgb", "spatial", "temporal", "motion"},
        )
        self.assertEqual(
            model.branch_slices["temporal"].stop,
            model.branch_slices["motion"].start,
        )

    def test_only_mvit_view_is_temporally_interpolated_to_sixteen(self) -> None:
        model = Stage1MViT(feature_mode=RGB_FFT_FEATURES, row_profile_bins=4)

        for raw_frames in (16, 24, 32):
            raw = torch.randn(1, 3, raw_frames, 8, 8)
            adapted = model.interpolate_mvit_time(raw)

            self.assertEqual(tuple(adapted.shape), (1, 3, 16, 8, 8))
            self.assertEqual(raw.shape[2], raw_frames)


class Stage1LearningRateScheduleTests(unittest.TestCase):
    def test_epoch_diagnostics_measure_loss_mask_and_convgru_balance(self) -> None:
        auxiliary = _auxiliary_batch_diagnostics(
            {
                "explainability_masks": torch.full((1, 1, 2, 2, 2), 0.5),
                "flow_update_l2_magnitudes": torch.tensor(
                    [[[3.0, 2.0, 1.0], [6.0, 4.0, 2.0]]]
                ),
            }
        )
        balance = _loss_balance_diagnostics(
            {
                "clip_classification": 1.0,
                "frame_classification": 0.4,
                "smoothing": 0.2,
                "reconstruction": 0.5,
                "mask_regularization": 0.7,
                "mask_sparsity": 0.5,
            },
            frame_classification_weight=0.25,
            smoothing_weight=0.05,
            explainability_weight=0.05,
            mask_regularization_weight=0.02,
            mask_sparsity_weight=1e-3,
        )

        self.assertEqual(auxiliary["explainability_mask_mean"], 0.5)
        self.assertEqual(auxiliary["convgru_update_l2_iteration_1"], 4.5)
        self.assertAlmostEqual(
            auxiliary["convgru_last_to_first_update_l2_ratio"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(balance["classification_focal_loss"], 1.1)
        self.assertAlmostEqual(balance["weighted_smoothing_loss"], 0.01)
        self.assertLess(
            balance["weighted_smoothing_to_classification_ratio"],
            0.01,
        )

    def test_three_epoch_warmup_reaches_differential_targets(self) -> None:
        backbone = [
            linear_warmup_learning_rate(
                epoch=epoch,
                warmup_epochs=3,
                initial_learning_rate=1e-6,
                target_learning_rate=1e-5,
            )
            for epoch in (1, 2, 3)
        ]
        auxiliary = [
            linear_warmup_learning_rate(
                epoch=epoch,
                warmup_epochs=3,
                initial_learning_rate=1e-6,
                target_learning_rate=1e-4,
            )
            for epoch in (1, 2, 3)
        ]

        self.assertEqual(backbone[0], 1e-6)
        self.assertEqual(backbone[-1], 1e-5)
        self.assertEqual(auxiliary[0], 1e-6)
        self.assertEqual(auxiliary[-1], 1e-4)
        self.assertTrue(all(left < right for left, right in zip(backbone, backbone[1:])))
        self.assertTrue(all(left < right for left, right in zip(auxiliary, auxiliary[1:])))

    def test_early_stopping_is_blocked_until_epoch_ten(self) -> None:
        self.assertFalse(
            stage1_early_stopping_triggered(
                epoch=9,
                minimum_epochs=10,
                stop_requested=True,
            )
        )
        self.assertTrue(
            stage1_early_stopping_triggered(
                epoch=10,
                minimum_epochs=10,
                stop_requested=True,
            )
        )


class Stage1TemporalConsistencyLossTests(unittest.TestCase):
    def test_multitask_defaults_match_balanced_v3_formula(self) -> None:
        criterion = Stage1MultiTaskLoss()

        self.assertEqual(criterion.smoothing_weight, 0.05)
        self.assertEqual(criterion.smoothing_truncation, 4.0)
        self.assertEqual(criterion.explainability_weight, 0.05)
        self.assertEqual(criterion.mask_regularization_weight, 0.02)
        self.assertEqual(criterion.mask_sparsity_weight, 1e-3)
        self.assertAlmostEqual(
            criterion.explainability_weight * criterion.mask_regularization_weight,
            0.001,
        )

    def test_multitask_focal_alpha_is_exposed_to_clip_and_frame_losses(self) -> None:
        criterion = Stage1MultiTaskLoss(focal_alpha=(0.5, 1.5))

        torch.testing.assert_close(
            criterion.classification.alpha,
            torch.tensor([0.5, 1.5]),
        )

    def test_inverse_frequency_alpha_uses_mean_one_sample_weight(self) -> None:
        weights = inverse_frequency_focal_alpha([0, 0, 0, 1])

        self.assertAlmostEqual(weights[0], 2.0 / 3.0)
        self.assertAlmostEqual(weights[1], 2.0)
        self.assertAlmostEqual((3 * weights[0] + weights[1]) / 4, 1.0)

    def test_inverse_frequency_alpha_rejects_fold_missing_a_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "every class"):
            inverse_frequency_focal_alpha([0, 0])

    def test_identical_frame_probabilities_have_zero_smoothing_loss(self) -> None:
        logits = torch.tensor([[[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]])

        self.assertEqual(truncated_temporal_mse(logits).item(), 0.0)

    def test_abrupt_log_probability_penalty_is_truncated(self) -> None:
        logits = torch.tensor([[[100.0, -100.0], [-100.0, 100.0]]])

        self.assertAlmostEqual(
            truncated_temporal_mse(logits, truncation=1.0).item(),
            1.0,
        )

    def test_zero_explainability_mask_is_penalized_by_bce_to_one(self) -> None:
        target = torch.ones(1, 3, 2, 4, 4)
        reconstruction = torch.zeros_like(target)
        zero_mask = torch.zeros(1, 1, 2, 4, 4)
        one_mask = torch.ones_like(zero_mask)

        zero_reconstruction, zero_regularization = explainability_reconstruction_terms(
            reconstruction,
            target,
            zero_mask,
        )
        one_reconstruction, one_regularization = explainability_reconstruction_terms(
            reconstruction,
            target,
            one_mask,
        )

        self.assertEqual(zero_reconstruction.item(), 0.0)
        self.assertGreater(zero_regularization.item(), one_regularization.item())
        self.assertGreater(one_reconstruction.item(), zero_reconstruction.item())

    def test_multitask_loss_backpropagates_all_auxiliary_outputs(self) -> None:
        outputs = {
            "logits": torch.randn(2, 2, requires_grad=True),
            "frame_logits": torch.randn(2, 4, 2, requires_grad=True),
            "reconstructed_targets": torch.randn(2, 3, 3, 4, 4, requires_grad=True),
            "target_frames": torch.randn(2, 3, 3, 4, 4),
            "explainability_masks": torch.full(
                (2, 1, 3, 4, 4),
                0.5,
                requires_grad=True,
            ),
            "explainability_mask_logits": torch.zeros(
                2,
                1,
                3,
                4,
                4,
                requires_grad=True,
            ),
        }
        terms = Stage1MultiTaskLoss()(outputs, torch.tensor([0, 1]))

        terms["total"].backward()

        self.assertEqual(
            set(terms),
            {
                "total",
                "clip_classification",
                "frame_classification",
                "smoothing",
                "reconstruction",
                "mask_regularization",
                "mask_sparsity",
            },
        )
        self.assertTrue(torch.isfinite(terms["total"]).item())
        self.assertIsNotNone(outputs["frame_logits"].grad)
        self.assertIsNotNone(outputs["explainability_masks"].grad)

    def test_all_mstcn_stages_receive_deep_supervision(self) -> None:
        stage_logits = torch.randn(2, 3, 5, 2, requires_grad=True)
        outputs = {
            "logits": torch.randn(2, 2, requires_grad=True),
            "frame_logits": stage_logits[:, -1],
            "stage_frame_logits": stage_logits,
            "reconstructed_targets": torch.zeros(2, 3, 4, 2, 2),
            "target_frames": torch.ones(2, 3, 4, 2, 2),
            "explainability_masks": torch.full((2, 1, 4, 2, 2), 0.5),
            "explainability_mask_logits": torch.zeros(2, 1, 4, 2, 2),
        }

        Stage1MultiTaskLoss()(outputs, torch.tensor([0, 1]))["total"].backward()

        self.assertIsNotNone(stage_logits.grad)
        self.assertTrue((stage_logits.grad.abs().sum(dim=(0, 2, 3)) > 0).all().item())


class Stage1MultiStageTemporalRefinementTests(unittest.TestCase):
    def test_base_initialization_and_following_rng_are_mode_invariant(self) -> None:
        def snapshot(mode: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            torch.manual_seed(20260826)
            model = Stage1MViT(
                feature_mode=RGB_FFT_FEATURES,
                row_profile_bins=4,
                temporal_refinement_mode=mode,
            )
            weights = {
                "rgb": model.rgb_backbone.conv_proj.weight.detach().clone(),
                "spatial": model.spatial_branch.frame_encoder[0].weight.detach().clone(),
                "temporal": model.temporal_branch.input_projection[0].weight.detach().clone(),
                "motion": model.motion_branch.feature_encoder[0].weight.detach().clone(),
                "classifier_hidden": model.classifier[1].weight.detach().clone(),
                "classifier_output": model.classifier[4].weight.detach().clone(),
            }
            following_random_values = torch.rand(8)
            del model
            gc.collect()
            return weights, following_random_values

        single_weights, single_random_values = snapshot("single_stage")
        gated_weights, gated_random_values = snapshot("gated_mstcn")

        self.assertEqual(single_weights.keys(), gated_weights.keys())
        for name in single_weights:
            torch.testing.assert_close(
                single_weights[name],
                gated_weights[name],
                rtol=0.0,
                atol=0.0,
                msg=f"base initialization differs for {name}",
            )
        torch.testing.assert_close(
            single_random_values,
            gated_random_values,
            rtol=0.0,
            atol=0.0,
        )

    def test_refinement_stages_receive_probabilities_only(self) -> None:
        head = MultiStageTemporalRefinementHead(
            6,
            stages=3,
            hidden_channels=8,
        )
        observed: list[torch.Tensor] = []
        handles = [
            stage.register_forward_pre_hook(
                lambda _module, inputs: observed.append(inputs[0].detach())
            )
            for stage in head.refinement_stages
        ]

        logits = head(torch.randn(2, 7, 6))
        for handle in handles:
            handle.remove()

        self.assertEqual(tuple(logits.shape), (2, 3, 7, 2))
        self.assertEqual(len(observed), 2)
        for probabilities in observed:
            self.assertEqual(probabilities.shape[-1], 2)
            self.assertGreaterEqual(probabilities.min().item(), 0.0)
            self.assertLessEqual(probabilities.max().item(), 1.0)
            torch.testing.assert_close(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities[..., 0]),
            )

    def test_single_stage_ablation_does_not_modify_clip_logits(self) -> None:
        model = Stage1MViT(
            feature_mode=RGB_FFT_FEATURES,
            row_profile_bins=4,
            temporal_refinement_mode="single_stage",
        )
        base_logits = torch.tensor([[0.2, -0.3]])
        frame_logits = torch.randn(1, 5, 2)

        combined = model.combine_clip_logits(base_logits, frame_logits)

        torch.testing.assert_close(combined, base_logits)
        self.assertEqual(model.mstcn_alpha.item(), 0.0)

    def test_gated_mstcn_starts_as_small_learnable_residual(self) -> None:
        model = Stage1MViT(
            feature_mode=RGB_FFT_FEATURES,
            row_profile_bins=4,
            temporal_refinement_mode="gated_mstcn",
            mstcn_gate_initial=0.1,
        )
        base_logits = torch.zeros(1, 2)
        frame_logits = torch.ones(1, 5, 2)

        combined = model.combine_clip_logits(base_logits, frame_logits)
        combined.sum().backward()

        torch.testing.assert_close(combined, torch.full_like(combined, 0.1))
        self.assertIsNotNone(model.mstcn_gate_logit.grad)

    def test_zero_gate_keeps_mstcn_deep_supervision_but_not_clip_logit_gradient(self) -> None:
        model = Stage1MViT(
            feature_mode=RGB_FFT_FEATURES,
            row_profile_bins=4,
            temporal_refinement_mode="gated_mstcn",
            zero_gate=True,
        )
        self.assertIsNotNone(model.temporal_refinement_head)
        self.assertIsNone(model.mstcn_gate_logit)
        self.assertEqual(model.mstcn_alpha.item(), 0.0)

        frame_feature_channels = (
            model.temporal_branch.hidden_channels
            + model.motion_branch.feature_channels
        )
        stage_logits = model.temporal_refinement_head(
            torch.randn(2, 5, frame_feature_channels)
        )
        stage_logits.retain_grad()
        base_logits = torch.randn(2, 2, requires_grad=True)
        combined = model.combine_clip_logits(base_logits, stage_logits[:, -1])
        torch.testing.assert_close(combined, base_logits)

        outputs = {
            "logits": combined,
            "frame_logits": stage_logits[:, -1],
            "stage_frame_logits": stage_logits,
            "reconstructed_targets": torch.zeros(2, 3, 4, 2, 2),
            "target_frames": torch.ones(2, 3, 4, 2, 2),
            "explainability_masks": torch.full((2, 1, 4, 2, 2), 0.5),
            "explainability_mask_logits": torch.zeros(2, 1, 4, 2, 2),
        }
        Stage1MultiTaskLoss()(outputs, torch.tensor([0, 1]))["total"].backward()

        self.assertIsNotNone(stage_logits.grad)
        self.assertTrue((stage_logits.grad.abs().sum(dim=(0, 2, 3)) > 0).all().item())
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                for parameter in model.temporal_refinement_head.parameters()
            )
        )

    def test_detached_mstcn_learns_head_and_gate_without_feature_gradient(self) -> None:
        model = Stage1MViT(
            feature_mode=RGB_FFT_FEATURES,
            row_profile_bins=4,
            temporal_refinement_mode="gated_mstcn",
            mstcn_gate_initial=0.1,
            detach_mstcn_input=True,
        )
        frame_feature_channels = (
            model.temporal_branch.hidden_channels
            + model.motion_branch.feature_channels
        )
        frame_features = torch.randn(
            2,
            5,
            frame_feature_channels,
            requires_grad=True,
        )
        stage_logits = model.refine_frame_features(frame_features)
        base_logits = torch.randn(2, 2, requires_grad=True)
        outputs = {
            "logits": model.combine_clip_logits(base_logits, stage_logits[:, -1]),
            "frame_logits": stage_logits[:, -1],
            "stage_frame_logits": stage_logits,
            "reconstructed_targets": torch.zeros(2, 3, 4, 2, 2),
            "target_frames": torch.ones(2, 3, 4, 2, 2),
            "explainability_masks": torch.full((2, 1, 4, 2, 2), 0.5),
            "explainability_mask_logits": torch.zeros(2, 1, 4, 2, 2),
        }

        Stage1MultiTaskLoss()(outputs, torch.tensor([0, 1]))["total"].backward()

        self.assertIsNone(frame_features.grad)
        self.assertIsNotNone(base_logits.grad)
        self.assertIsNotNone(model.mstcn_gate_logit)
        self.assertIsNotNone(model.mstcn_gate_logit.grad)
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                for parameter in model.temporal_refinement_head.parameters()
            )
        )


class Stage1CorrelationGruTests(unittest.TestCase):
    def test_all_pairs_volume_and_multiscale_pooling_shapes(self) -> None:
        target = torch.randn(1, 8, 8, 8)
        source = torch.randn_like(target)
        correlation = AllPairsCorrelationPyramid(scales=(1, 2, 4, 8))

        pyramid = correlation(target, source)

        self.assertEqual(
            [tuple(level.shape) for level in pyramid],
            [
                (1, 8, 8, 8, 8),
                (1, 8, 8, 4, 4),
                (1, 8, 8, 2, 2),
                (1, 8, 8, 1, 1),
            ],
        )
        y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        coordinates = torch.stack((x, y)).float().unsqueeze(0)
        lookup = correlation.lookup(pyramid, coordinates, radius=1)
        self.assertEqual(tuple(lookup.shape), (1, 4 * 9, 8, 8))

    def test_conv_gru_state_is_bounded(self) -> None:
        cell = ConvGRUCell(input_channels=5, hidden_channels=8)

        updated = cell(torch.zeros(2, 8, 6, 6), torch.randn(2, 5, 6, 6))

        self.assertLessEqual(updated.abs().max().item(), 1.0)

    def test_recurrent_update_reuses_one_gru_and_bounds_l2_step(self) -> None:
        updater = RecurrentCorrelationUpdate(
            feature_channels=8,
            hidden_channels=8,
            radius=1,
            iterations=3,
            max_delta=0.5,
        )

        hidden, displacement, magnitudes = updater(
            torch.randn(1, 8, 8, 8),
            torch.randn(1, 8, 8, 8),
        )

        self.assertEqual(tuple(hidden.shape), (1, 8, 8, 8))
        self.assertEqual(tuple(displacement.shape), (1, 2, 8, 8))
        self.assertEqual(tuple(magnitudes.shape), (1, 3))
        for iteration, magnitude in enumerate(magnitudes[0], start=1):
            maximum_l2 = 0.5 / iteration * displacement[0].numel() ** 0.5
            self.assertLessEqual(magnitude.item(), maximum_l2)
        self.assertIsInstance(updater.gru, ConvGRUCell)

    def test_motion_branch_emits_masks_reconstructions_and_iteration_trace(self) -> None:
        branch = MotionConsistencyEncoder(iterations=2, radius=1)

        outputs = branch(torch.randn(1, 3, 3, 128, 128))

        self.assertEqual(tuple(outputs["clip_features"].shape), (1, 128))
        self.assertEqual(tuple(outputs["frame_features"].shape), (1, 3, 64))
        self.assertEqual(tuple(outputs["explainability_masks"].shape), (1, 1, 2, 8, 8))
        self.assertEqual(tuple(outputs["reconstructed_targets"].shape), (1, 3, 2, 8, 8))
        self.assertEqual(tuple(outputs["flow_update_magnitudes"].shape), (1, 2, 2))
        self.assertGreaterEqual(outputs["explainability_masks"].min().item(), 0.0)
        self.assertLessEqual(outputs["explainability_masks"].max().item(), 1.0)


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
