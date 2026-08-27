from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from blackbox.submission_pipeline import (
    generate_submission_bundle,
    project_stage3_source_frames_by_video_fps,
)
from blackbox.stages.stage3.dataset_stage3 import Stage3TimeAxis


class SubmissionPipelineTests(unittest.TestCase):
    def test_stage3_projection_uses_each_videos_fps_derived_stride(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "ID": "S3_30FPS",
                    "sample_index": index,
                    "accel_label": "CONSTANT",
                    "steer_label": "STRAIGHT",
                }
                for index in range(6)
            ]
            + [
                {
                    "ID": "S3_60FPS",
                    "sample_index": index,
                    "accel_label": "CONSTANT",
                    "steer_label": "STRAIGHT",
                }
                for index in range(12)
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            videos = Path(temporary)
            for video_id in ("S3_30FPS", "S3_60FPS"):
                (videos / f"{video_id}.mp4").write_bytes(b"fixture")
            with patch(
                "blackbox.submission_pipeline.read_stage3_time_axis",
                side_effect=[
                    Stage3TimeAxis(30.0, 3),
                    Stage3TimeAxis(60.0, 6),
                ],
            ):
                projected, axes = project_stage3_source_frames_by_video_fps(
                    source,
                    video_dir=videos,
                )
        self.assertEqual(projected.groupby("ID")["sample_index"].count().to_dict(), {
            "S3_30FPS": 2,
            "S3_60FPS": 2,
        })
        self.assertEqual(axes["S3_30FPS"]["frames_per_sample"], 3)
        self.assertEqual(axes["S3_60FPS"]["frames_per_sample"], 6)

    def test_sequential_pipeline_projects_stage3_and_falls_back_per_failed_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            model_root = Path(temporary) / "models"
            output = Path(temporary) / "output"
            for path in (
                root / "stage1/videos/S1_A.mp4",
                root / "stage2/images/S2_A/000007.jpg",
                root / "stage3/videos/S3_A.mp4",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            predictors = {
                1: lambda data, model: pd.DataFrame([{"ID": "S1_A", "answer": "ORIGINAL"}]),
                2: lambda data, model: (_ for _ in ()).throw(RuntimeError("synthetic Stage 2 failure")),
                3: lambda data, model: pd.DataFrame(
                    [
                        {"ID": "S3_A", "sample_index": index, "accel_label": "CONSTANT", "steer_label": "STRAIGHT"}
                        for index in range(4)
                    ]
                ),
            }
            stage3_sample = Path(temporary) / "stage3_sample_submission.csv"
            pd.DataFrame(
                [
                    {"ID": "S3_A", "sample_index": 0, "accel_label": "CONSTANT", "steer_label": "STRAIGHT"},
                    {"ID": "S3_A", "sample_index": 1, "accel_label": "CONSTANT", "steer_label": "STRAIGHT"},
                ]
            ).to_csv(stage3_sample, index=False)
            with patch("blackbox.submission_pipeline.video_frame_count", return_value=4):
                summary = generate_submission_bundle(
                    root,
                    model_root,
                    output,
                    stage3_frames_per_sample=2,
                    sample_submissions={3: stage3_sample},
                    predictors=predictors,
                )

            stage2 = pd.read_csv(output / "stage2_submission.csv")
            stage3 = pd.read_csv(output / "stage3_submission.csv")
        self.assertEqual(summary["fallback_stages"], ["stage2"])
        self.assertEqual(stage2.to_dict("records"), [{
            "ID": "S2_A",
            "collision_frame": 7,
            "entry_frame": 7,
            "evasion_space": 0,
            "entry_side": "LEFT",
        }])
        self.assertEqual(stage3["sample_index"].tolist(), [0, 1])
        self.assertEqual(stage3["accel_label"].tolist(), ["CONSTANT", "CONSTANT"])


if __name__ == "__main__":
    unittest.main()
