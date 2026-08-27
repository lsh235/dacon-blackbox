from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from blackbox.stages.stage2.dataset_stage2 import IGNORE_INDEX
from blackbox.stages.stage3.dataset_stage3 import (
    Stage3Annotation,
    Stage3SequenceWindowDataset,
    Stage3TimeAxis,
    Stage3VideoRecord,
    read_stage3_time_axis,
)
from blackbox.stages.stage3.model_stage3 import Stage3TwoStreamBiLSTM
from blackbox.stages.stage3.train_stage3 import stage3_sequence_loss
from blackbox.stages.two_stream import flow_roi_statistics, mask_flow_roi, spatial_grid_pool
from blackbox.preprocessing import PREPROCESS_SCHEMA


class Stage3SequenceDatasetTests(unittest.TestCase):
    def test_sparse_labels_map_to_source_grounded_0_1_second_chunks(self) -> None:
        frames = torch.zeros(6, 3, 16, 16)
        frames[4, :, 4:10, 5:11] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "OPEN_001.mp4"
            video.write_bytes(b"stage3-fixture")
            processed_root = Path(temporary) / "processed"
            window_root = processed_root / "stage3" / "windows"
            (window_root / "rgb").mkdir(parents=True)
            (window_root / "flow").mkdir(parents=True)
            import numpy as np

            np.save(window_root / "rgb" / "fixture.npy", frames.numpy())
            np.save(window_root / "flow" / "fixture.npy", torch.zeros(6, 2, 16, 16).numpy())
            (window_root / "manifest.json").write_text(json.dumps({
                "schema": PREPROCESS_SCHEMA,
                "stage": "stage3",
                "entries": [{
                    "id": "OPEN_001",
                    "source": str(video.resolve()),
                    "start_frame": 0,
                    "end_frame": 6,
                    "valid_length": 6,
                    "window_frames": 6,
                    "stride": 6,
                    "size": 16,
                    "farneback": {
                        "pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3,
                        "poly_n": 5, "poly_sigma": 1.2, "flags": 0, "flow_clip": 20.0,
                    },
                    "rgb": "rgb/fixture.npy",
                    "flow": "flow/fixture.npy",
                }],
            }))
            record = Stage3VideoRecord(
                "OPEN_001",
                video,
                annotations=(
                    Stage3Annotation(7, 1, 0.7, 0, 2),
                    Stage3Annotation(99, 9, 9.9, 3, 0),
                ),
            )
            with (
                patch(
                    "blackbox.stages.stage2.dataset_stage2.decode_stage2_window",
                    side_effect=AssertionError("loader must not decode raw videos"),
                ),
                patch(
                    "blackbox.stages.stage3.dataset_stage3.read_stage3_time_axis",
                    return_value=Stage3TimeAxis(source_fps=30.0, frames_per_sample=3),
                ),
            ):
                dataset = Stage3SequenceWindowDataset(
                    [record],
                    window_frames=6,
                    stride=6,
                    size=16,
                    processed_root=processed_root,
                )
                sample = dataset[0]
        self.assertEqual(sample["frame_numbers"].tolist(), [0, 3])
        self.assertEqual(sample["sample_indices"].tolist(), [7, IGNORE_INDEX])
        self.assertEqual(sample["accel_targets"].tolist(), [0, IGNORE_INDEX])
        self.assertEqual(sample["steer_targets"].tolist(), [2, IGNORE_INDEX])
        self.assertEqual(int(sample["frames_per_sample"]), 3)

    def test_sparse_public_labels_define_training_steps_and_report_metadata_conflict(self) -> None:
        class Capture:
            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                self.property_id = property_id
                return 59.94

            def release(self) -> None:
                self.released = True

        annotations = (Stage3Annotation(5, 10, 0.5, 0, 1),)
        with patch("blackbox.stages.stage3.dataset_stage3.cv2.VideoCapture", return_value=Capture()):
            axis = read_stage3_time_axis("metadata-only.mp4", annotations=annotations)
        self.assertEqual(axis.frames_per_sample, 2)
        self.assertAlmostEqual(axis.source_fps, 59.94)
        self.assertAlmostEqual(axis.label_frames_per_sample or 0.0, 2.0)
        self.assertEqual(axis.metadata_frames_per_sample, 6)
        self.assertEqual(axis.mode, "sparse_public_label_mapping")
        self.assertTrue(axis.has_label_conflict)

    def test_official_evaluation_keeps_each_decoded_10hz_frame(self) -> None:
        class Capture:
            def isOpened(self) -> bool:
                return True

            def get(self, property_id: int) -> float:
                self.property_id = property_id
                return 479.78

            def release(self) -> None:
                self.released = True

        with patch("blackbox.stages.stage3.dataset_stage3.cv2.VideoCapture", return_value=Capture()):
            axis = read_stage3_time_axis("evaluation.mp4")

        self.assertEqual(axis.frames_per_sample, 1)
        self.assertEqual(axis.metadata_frames_per_sample, 48)
        self.assertEqual(axis.mode, "official_evaluation_10hz")
        self.assertFalse(axis.has_label_conflict)


class Stage3TwoStreamModelTests(unittest.TestCase):
    def test_bottom_roi_mask_removes_top_noise_without_changing_contract(self) -> None:
        clean = torch.zeros(1, 2, 2, 6, 4, dtype=torch.float32)
        clean[..., 3:, :] = 0.25
        noisy = clean.clone()
        noisy[..., :3, :] = 100.0

        masked = mask_flow_roi(noisy, top_ratio=0.5)

        self.assertEqual(masked.shape, noisy.shape)
        self.assertEqual(masked.dtype, noisy.dtype)
        self.assertTrue(torch.equal(masked[..., :3, :], torch.zeros_like(masked[..., :3, :])))
        self.assertTrue(torch.equal(masked[..., 3:, :], clean[..., 3:, :]))
        self.assertTrue(
            torch.equal(
                flow_roi_statistics(noisy, top_ratio=0.5),
                flow_roi_statistics(clean, top_ratio=0.5),
            )
        )

    def test_roi_physics_statistics_match_constant_and_variable_flow(self) -> None:
        flow = torch.zeros(1, 2, 2, 4, 2)
        flow[0, 0, 0, 2:, :] = 0.25
        flow[0, 0, 1, 2:, :] = -0.5
        flow[0, 1, 0, 2:, :] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        flow[0, 1, 1, 2:, :] = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

        statistics = flow_roi_statistics(flow, top_ratio=0.5)

        self.assertTrue(torch.allclose(statistics[0, 0], torch.tensor([0.25, -0.5, 0.0, 0.0])))
        self.assertTrue(torch.allclose(statistics[0, 1], torch.tensor([0.5, 0.0, 0.5, 1.0])))

    def test_grid_pooling_preserves_row_major_cell_order(self) -> None:
        left = torch.zeros(1, 1, 6, 6)
        right = torch.zeros_like(left)
        left[..., :2, :2] = 1.0
        right[..., :2, 4:] = 1.0

        left_embedding_input = spatial_grid_pool(left, grid_size=3)
        right_embedding_input = spatial_grid_pool(right, grid_size=3)

        self.assertEqual(tuple(left_embedding_input.shape), (1, 9))
        self.assertEqual(left_embedding_input[0].argmax().item(), 0)
        self.assertEqual(right_embedding_input[0].argmax().item(), 2)
        self.assertFalse(torch.equal(left_embedding_input, right_embedding_input))

    def test_iteration8_model_records_reproducible_architecture(self) -> None:
        model = Stage3TwoStreamBiLSTM(
            hidden_size=8,
            layers=1,
            flow_grid_size=3,
            flow_roi_top_ratio=0.5,
            flow_projection_dim=24,
        )

        self.assertEqual(
            model.architecture_metadata(),
            {
                "version": 2,
                "flow_roi_mode": "mask",
                "flow_roi_top_ratio": 0.5,
                "flow_grid_size": 3,
                "flow_projection_dim": 24,
                "use_physics_vector": False,
                "physics_features": ["mean_dx", "mean_dy", "std_dx", "std_dy"],
                "physics_projection_dim": 32,
            },
        )
        self.assertEqual(model.temporal.input_size, 512 + 24)

    def test_physics_projection_zeros_padding_after_normalization(self) -> None:
        model = Stage3TwoStreamBiLSTM(
            hidden_size=8,
            layers=1,
            use_physics_vector=True,
            physics_projection_dim=8,
        ).eval()
        flow = torch.ones(1, 3, 2, 4, 4)

        with torch.inference_mode():
            embedded = model.encode_physics_vector(
                flow,
                torch.tensor([[True, True, False]]),
            )

        self.assertEqual(tuple(embedded.shape), (1, 3, 8))
        self.assertTrue(torch.equal(embedded[0, 2], torch.zeros(8)))

    def test_sparse_sequence_loss_ignores_unlabelled_steps(self) -> None:
        accel = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [float("-inf")] * 4]], requires_grad=True)
        steer = torch.tensor([[[0.0, 0.0, 2.0], [float("-inf")] * 3]], requires_grad=True)
        loss = stage3_sequence_loss(
            {"accel_logits": accel, "steer_logits": steer},
            {
                "accel_targets": torch.tensor([[0, IGNORE_INDEX]]),
                "steer_targets": torch.tensor([[2, IGNORE_INDEX]]),
            },
        )
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

    def test_seq2seq_heads_mask_padded_steps(self) -> None:
        model = Stage3TwoStreamBiLSTM(
            hidden_size=8,
            layers=1,
            frame_batch_size=1,
            flow_grid_size=3,
            flow_roi_top_ratio=0.5,
            flow_projection_dim=16,
        ).eval()
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
