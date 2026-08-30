from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from blackbox.common.runtime import CheckpointError
from blackbox.stages.stage1.dataset import RGB_FEATURES, Stage1TestDataset
from blackbox.stages.stage1.inference_stage1 import (
    CLASS_NAMES,
    discover_fold_checkpoints,
    format_stage1_submission,
    mean_fold_probabilities,
    validate_mode_g_checkpoint,
)


def _mode_g_checkpoint() -> dict[str, object]:
    return {
        "architecture": "stage1_rgb_fft_flicker_corr_gru_gated_mstcn_ablation_v5",
        "model": {"mstcn_gate_logit": torch.tensor(-2.2)},
        "size": 224,
        "frames": 16,
        "feature_mode": "rgb_fft",
        "model_config": {
            "forensic_size": 320,
            "fft_size": 112,
            "row_profile_bins": 16,
            "motion_iterations": 3,
            "correlation_radius": 2,
            "temporal_refinement": "gated_mstcn",
            "temporal_refinement_stages": 3,
            "mstcn_gate_initial": 0.1,
            "mstcn_zero_gate": False,
            "mstcn_input_detached": True,
            "base_initialization_seed": 42,
            "fusion": "base_clip_logits_plus_gated_refined_frame_logits",
        },
        "sampling": {
            "name": "centered_contiguous_regions",
            "frames_per_region": 16,
            "inference_tta_slots": 3,
        },
    }


class Stage1TestDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_video(path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (32, 24),
        )
        if not writer.isOpened():
            raise RuntimeError("test video writer could not be opened")
        try:
            for frame_index in range(18):
                frame = np.full((24, 32, 3), frame_index * 10, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def test_test_dataset_is_deterministic_and_has_exactly_three_regions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_root = Path(directory) / "videos"
            video_root.mkdir()
            self._write_video(video_root / "TEST_001.avi")
            dataset = Stage1TestDataset(
                directory,
                slots=3,
                size=16,
                frames=2,
                feature_mode=RGB_FEATURES,
                forensic_size=16,
                fft_size=8,
                row_profile_bins=2,
            )

            first, first_video, first_valid = dataset[0]
            repeated, repeated_video, repeated_valid = dataset[0]

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.video_ids, ["TEST_001"])
            self.assertFalse(hasattr(dataset, "augmentation"))
            self.assertEqual((first_video, first_valid), (0, 1))
            self.assertEqual((repeated_video, repeated_valid), (0, 1))
            self.assertEqual(tuple(first["rgb_clip"].shape), (3, 2, 16, 16))
            for name in first:
                torch.testing.assert_close(first[name], repeated[name])

    def test_test_dataset_rejects_non_validation_region_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "three-region"):
                Stage1TestDataset(
                    directory,
                    slots=5,
                    size=16,
                    frames=2,
                    feature_mode=RGB_FEATURES,
                    forensic_size=16,
                    fft_size=8,
                    row_profile_bins=2,
                )


class Stage1CheckpointDiscoveryTests(unittest.TestCase):
    def test_discovers_one_best_checkpoint_for_each_fold_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold_index in (4, 2, 0, 3, 1):
                checkpoint = root / "experiment" / f"fold_{fold_index}" / "model" / "best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()

            paths = discover_fold_checkpoints(root)

            self.assertEqual([path.parts[-3] for path in paths], [f"fold_{i}" for i in range(5)])

    def test_missing_fold_fails_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold_index in range(4):
                checkpoint = root / f"fold_{fold_index}" / "model" / "best.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()

            with self.assertRaisesRegex(CheckpointError, r"missing=\[4\]"):
                discover_fold_checkpoints(root)

    def test_mode_g_contract_requires_detachment_and_learnable_gate(self) -> None:
        checkpoint = _mode_g_checkpoint()

        geometry = validate_mode_g_checkpoint(checkpoint, "best.pt")

        self.assertEqual(geometry.slots, 3)
        self.assertEqual(geometry.feature_mode, "rgb_fft")
        checkpoint["model_config"]["mstcn_input_detached"] = False
        with self.assertRaisesRegex(CheckpointError, "not exact Mode G"):
            validate_mode_g_checkpoint(checkpoint, "best.pt")


class Stage1SoftVotingAndSubmissionTests(unittest.TestCase):
    def test_soft_vote_averages_both_class_probabilities(self) -> None:
        folds = [
            np.asarray([[0.8, 0.2], [0.1, 0.9]]),
            np.asarray([[0.6, 0.4], [0.3, 0.7]]),
            np.asarray([[0.7, 0.3], [0.2, 0.8]]),
        ]

        averaged = mean_fold_probabilities(folds)

        np.testing.assert_allclose(averaged, [[0.7, 0.3], [0.2, 0.8]])
        np.testing.assert_allclose(averaged.sum(axis=1), 1.0)

    def test_sample_order_is_preserved_and_threshold_distribution_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "sample_submission.csv"
            pd.DataFrame(
                {"ID": ["VIDEO_B", "VIDEO_A"], "answer": [None, None]}
            ).to_csv(sample_path, index=False)
            videos = [Path("VIDEO_A.mp4"), Path("VIDEO_B.mp4")]
            probabilities = np.asarray([[0.51, 0.49], [0.2, 0.8]])

            frame, distribution = format_stage1_submission(
                sample_path,
                videos,
                probabilities,
                threshold=0.5,
            )

            self.assertEqual(frame["ID"].tolist(), ["VIDEO_B", "VIDEO_A"])
            self.assertEqual(frame["answer"].tolist(), ["RERECORDED", "ORIGINAL"])
            self.assertEqual(distribution, {CLASS_NAMES[0]: 1, CLASS_NAMES[1]: 1})

    def test_probability_mode_writes_rerecorded_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "sample_submission.csv"
            pd.DataFrame({"ID": ["VIDEO_A"], "answer": [None]}).to_csv(
                sample_path,
                index=False,
            )

            frame, _ = format_stage1_submission(
                sample_path,
                [Path("VIDEO_A.mp4")],
                np.asarray([[0.25, 0.75]]),
                output_mode="probabilities",
            )

            self.assertAlmostEqual(float(frame.loc[0, "answer"]), 0.75)


if __name__ == "__main__":
    unittest.main()
