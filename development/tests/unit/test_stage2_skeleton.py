from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from blackbox.stages.stage2.dataset_stage2 import (
    IGNORE_INDEX,
    FarnebackConfig,
    Stage2SlidingWindowDataset,
    Stage2VideoRecord,
    farneback_optical_flow,
    local_event_target,
    sliding_window_starts,
)
from blackbox.preprocessing import PREPROCESS_SCHEMA
from blackbox.stages.stage2.model_stage2 import Stage2CnnBiLSTM, Stage2TwoStreamBiLSTM
from blackbox.stages.stage2.inference_stage2 import predict_two_stream_event_frames
from blackbox.stages.stage2.train_stage2 import (
    TargetMappingConfig,
    aggregate_overlapping_window_scores,
    build_window_event_target,
    map_local_peak_to_original_frame,
    select_aggregated_event_frame,
)


class Stage2SlidingWindowTests(unittest.TestCase):
    def test_tail_window_is_retained(self) -> None:
        self.assertEqual(sliding_window_starts(100, 64, 32), [0, 32, 36])
        self.assertEqual(sliding_window_starts(10, 64, 32), [0])

    def test_local_target_never_relabels_an_event_outside_its_window(self) -> None:
        self.assertEqual(local_event_target(35, start_frame=32, valid_length=16), 3)
        self.assertEqual(
            local_event_target(31, start_frame=32, valid_length=16),
            IGNORE_INDEX,
        )
        self.assertEqual(
            local_event_target(-1, start_frame=32, valid_length=16),
            IGNORE_INDEX,
        )

    def test_farneback_flow_is_dx_dy_normalized_and_padding_is_zero(self) -> None:
        frames = torch.zeros(4, 3, 32, 32)
        frames[1, :, 8:20, 10:22] = 1.0
        frames[2, :, 8:20, 12:24] = 1.0
        frames[3] = frames[2]
        flow = farneback_optical_flow(
            frames,
            valid_length=3,
            config=FarnebackConfig(flow_clip=4.0),
        )
        self.assertEqual(tuple(flow.shape), (4, 2, 32, 32))
        self.assertEqual(flow.dtype, torch.float32)
        self.assertTrue(torch.equal(flow[0], torch.zeros_like(flow[0])))
        self.assertTrue(torch.equal(flow[3], torch.zeros_like(flow[3])))
        self.assertLessEqual(float(flow.abs().max()), 1.0)
        self.assertGreater(float(flow[2, 0, 8:20, 12:24].mean()), 0.0)

    def test_dataset_reads_preprocessed_rgb_and_flow_without_online_compute(self) -> None:
        frames = torch.zeros(3, 3, 16, 16)
        frames[1, :, 4:10, 5:11] = 1.0
        frames[2, :, 4:10, 6:12] = 1.0
        flow = farneback_optical_flow(frames, valid_length=3)
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "window.mp4"
            video.write_bytes(b"content-hash-fixture")
            processed_root = Path(temporary) / "processed"
            window_root = processed_root / "stage2" / "windows"
            (window_root / "rgb").mkdir(parents=True)
            (window_root / "flow").mkdir(parents=True)
            import numpy as np

            np.save(window_root / "rgb" / "fixture.npy", frames.numpy())
            np.save(window_root / "flow" / "fixture.npy", flow.numpy())
            (window_root / "manifest.json").write_text(json.dumps({
                "schema": PREPROCESS_SCHEMA,
                "stage": "stage2",
                "entries": [{
                    "id": "fixture",
                    "source": str(video.resolve()),
                    "start_frame": 0,
                    "end_frame": 3,
                    "valid_length": 3,
                    "window_frames": 3,
                    "stride": 3,
                    "size": 16,
                    "farneback": {
                        "pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3,
                        "poly_n": 5, "poly_sigma": 1.2, "flags": 0, "flow_clip": 20.0,
                    },
                    "rgb": "rgb/fixture.npy",
                    "flow": "flow/fixture.npy",
                }],
            }))
            record = Stage2VideoRecord("fixture", video, collision_frame=1)
            with patch(
                "blackbox.stages.stage2.dataset_stage2.decode_stage2_window",
                side_effect=AssertionError("loader must not decode raw videos"),
            ):
                dataset = Stage2SlidingWindowDataset(
                    [record],
                    window_frames=3,
                    stride=3,
                    size=16,
                    processed_root=processed_root,
                )
                first = dataset[0]
                second = dataset[0]
        self.assertTrue(bool(first["flow_cache_hit"]))
        self.assertTrue(bool(second["flow_cache_hit"]))
        self.assertTrue(torch.equal(first["flow"], second["flow"]))
        self.assertEqual(dataset.flow_cache_misses, 0)
        self.assertEqual(dataset.flow_cache_hits, 2)
        self.assertTrue(torch.equal(first["frames"], frames))
        self.assertEqual(int(first["evasion_target"]), IGNORE_INDEX)
        self.assertEqual(int(first["entry_side_target"]), IGNORE_INDEX)


class Stage2SequenceModelTests(unittest.TestCase):
    def test_default_two_stream_state_dict_strictly_reloads_legacy_shapes(self) -> None:
        model = Stage2TwoStreamBiLSTM(hidden_size=8, layers=1, frame_batch_size=1)
        state = model.state_dict()
        reloaded = Stage2TwoStreamBiLSTM(hidden_size=8, layers=1, frame_batch_size=1)

        incompatible = reloaded.load_state_dict(state, strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(model.temporal.input_size, 1024)
        self.assertFalse(any(key.startswith("flow_projection") for key in state))
        self.assertFalse(any(key.startswith("physics_projection") for key in state))

    def test_cnn_bilstm_returns_window_local_contract(self) -> None:
        model = Stage2CnnBiLSTM(hidden_size=8, layers=1, frame_batch_size=1).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 2, 3, 64, 64), torch.tensor([1]))

        self.assertEqual(tuple(output["collision_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["entry_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["evasion_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["entry_side_logits"].shape), (1, 2))
        self.assertTrue(torch.isneginf(output["collision_logits"][0, 1]).item())

    def test_two_stream_bilstm_fuses_matching_rgb_and_flow_windows(self) -> None:
        model = Stage2TwoStreamBiLSTM(hidden_size=8, layers=1, frame_batch_size=1).eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(1, 2, 3, 64, 64),
                torch.zeros(1, 2, 2, 64, 64),
                torch.tensor([1]),
            )

        self.assertEqual(tuple(output["collision_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["entry_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["evasion_logits"].shape), (1, 2))
        self.assertTrue(torch.isneginf(output["entry_logits"][0, 1]).item())


class Stage2TargetMappingTests(unittest.TestCase):
    def test_gaussian_target_has_local_peak_and_ignores_outside_events(self) -> None:
        config = TargetMappingConfig(mode="gaussian", gaussian_sigma=1.0)
        target = build_window_event_target(
            12,
            start_frame=10,
            valid_length=4,
            window_frames=6,
            config=config,
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(float(target[2]), 1.0)
        self.assertGreater(float(target[2]), float(target[1]))
        self.assertGreater(float(target[1]), float(target[0]))
        self.assertTrue(torch.equal(target[4:], torch.zeros(2)))
        self.assertIsNone(
            build_window_event_target(
                -1,
                start_frame=10,
                valid_length=4,
                window_frames=6,
                config=config,
            )
        )

    def test_original_frame_mapping_masks_padding_and_aggregates_overlap(self) -> None:
        local = map_local_peak_to_original_frame(
            torch.tensor([0.1, 0.9, float("-inf")]),
            torch.tensor([20, 21, 21]),
            valid_length=2,
        )
        self.assertEqual(local, 21)
        scores = torch.tensor([[0.2, 0.7, 0.1], [0.9, 0.3, 0.2]])
        frame_numbers = torch.tensor([[10, 11, 12], [11, 12, 13]])
        valid_lengths = torch.tensor([3, 3])
        self.assertAlmostEqual(
            aggregate_overlapping_window_scores(scores, frame_numbers, valid_lengths)[11],
            0.8,
        )
        self.assertEqual(
            select_aggregated_event_frame(scores, frame_numbers, valid_lengths, policy="mean"),
            11,
        )

    def test_two_stream_inference_returns_overlap_aggregated_original_frame(self) -> None:
        class FixedScores(nn.Module):
            def forward(self, frames, flow, valid_lengths):
                return {"collision_logits": torch.log(torch.tensor([[0.2, 0.7, 0.1], [0.9, 0.3, 0.2]]))}

        batch = {
            "id": ["S2_fixture", "S2_fixture"],
            "frames": torch.zeros(2, 3, 3, 8, 8),
            "flow": torch.zeros(2, 3, 2, 8, 8),
            "valid_length": torch.tensor([3, 3]),
            "frame_numbers": torch.tensor([[10, 11, 12], [11, 12, 13]]),
        }
        self.assertEqual(
            predict_two_stream_event_frames(
                FixedScores(),  # type: ignore[arg-type]
                [batch],
                device=torch.device("cpu"),
            ),
            {"S2_fixture": 11},
        )


if __name__ == "__main__":
    unittest.main()
