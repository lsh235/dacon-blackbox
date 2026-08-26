from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch

from blackbox import inference
from blackbox.stages.stage2.baseline import Stage2Temporal, frame_number


class BaselineApiTests(unittest.TestCase):
    def test_required_function_signatures(self) -> None:
        for function in (
            inference.predict_stage1,
            inference.predict_stage2,
            inference.predict_stage3,
        ):
            self.assertEqual(list(inspect.signature(function).parameters), ["data_dir", "model_dir"])

    def test_stage2_temporal_shapes(self) -> None:
        model = Stage2Temporal().eval()
        with torch.inference_mode():
            collision, entry, scene = model(torch.zeros(1, 5, 512))
        self.assertEqual(tuple(collision.shape), (1,))
        self.assertEqual(tuple(entry.shape), (1,))
        self.assertEqual(tuple(scene.shape), (1, 4))

    def test_frame_number_uses_filename_suffix(self) -> None:
        self.assertEqual(frame_number(Path("frame_000127.jpg")), 127)
        self.assertEqual(frame_number(Path("unlabeled.jpg")), 0)


if __name__ == "__main__":
    unittest.main()
