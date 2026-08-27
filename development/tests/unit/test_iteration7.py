from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from blackbox.common.runtime import make_grad_scaler
from blackbox.ensemble_inference import (
    predict_stage1_ensemble,
    predict_stage3_ensemble,
    smooth_temporal_probabilities,
)
from blackbox.training_control import TrainingControlConfig


class AmpControlTests(unittest.TestCase):
    def test_amp_flag_is_preserved_but_scaler_disables_itself_on_cpu(self) -> None:
        config = TrainingControlConfig(use_amp=True)
        scaler = make_grad_scaler(torch.device("cpu"), enabled=config.use_amp)
        self.assertTrue(config.use_amp)
        self.assertFalse(scaler.is_enabled())


class EnsembleInferenceTests(unittest.TestCase):
    def test_stage1_soft_voting_averages_probabilities_before_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            videos = Path(temporary) / "videos"
            videos.mkdir()
            (videos / "S1_A.mp4").write_bytes(b"fixture")
            with patch(
                "blackbox.ensemble_inference.score_stage1_checkpoint",
                side_effect=[[0.9], [0.3]],
            ) as scorer:
                prediction = predict_stage1_ensemble(
                    Path(temporary),
                    ["fold1.pt", "fold2.pt"],
                )
        self.assertEqual(scorer.call_count, 2)
        self.assertEqual(prediction.to_dict("records"), [{"ID": "S1_A", "answer": "RERECORDED"}])

    def test_stage3_votes_then_smooths_the_projected_10hz_probabilities(self) -> None:
        accel_fold1 = np.asarray(
            [[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]],
            dtype=np.float32,
        )
        accel_fold1 = np.pad(accel_fold1, ((0, 0), (0, 2)))
        accel_fold2 = accel_fold1.copy()
        steer = np.tile(np.asarray([[0.1, 0.8, 0.1]], dtype=np.float32), (5, 1))
        fold = {"S3_A": (accel_fold1, steer)}
        with tempfile.TemporaryDirectory() as temporary:
            videos = Path(temporary) / "videos"
            videos.mkdir()
            (videos / "S3_A.mp4").write_bytes(b"fixture")
            with patch(
                "blackbox.ensemble_inference.score_stage3_checkpoint",
                side_effect=[fold, {"S3_A": (accel_fold2, steer.copy())}],
            ):
                prediction, axes = predict_stage3_ensemble(
                    Path(temporary),
                    ["fold1.pt", "fold2.pt"],
                    smoothing_window=3,
                    frames_per_sample=1,
                )
        self.assertEqual(prediction["accel_label"].tolist(), ["ACCELERATING"] * 5)
        self.assertEqual(prediction["steer_label"].tolist(), ["STRAIGHT"] * 5)
        self.assertEqual(axes["S3_A"]["frames_per_sample"], 1)

    def test_smoothing_requires_an_odd_window(self) -> None:
        probabilities = np.asarray([[0.5, 0.5]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "positive odd"):
            smooth_temporal_probabilities(probabilities, window=2)


if __name__ == "__main__":
    unittest.main()
