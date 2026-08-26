from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from blackbox.common.runtime import CheckpointError, load_checkpoint, video_paths


class RuntimeGuardTests(unittest.TestCase):
    def test_rejects_missing_video_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "video directory not found"):
                video_paths(missing)

    def test_rejects_empty_video_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "no supported videos"):
                video_paths(temporary)

    def test_rejects_corrupt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "bad.pt"
            checkpoint.write_bytes(b"not-a-checkpoint")
            with self.assertRaisesRegex(CheckpointError, "cannot load checkpoint"):
                load_checkpoint(checkpoint)

    def test_rejects_missing_checkpoint_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "model.pt"
            torch.save({"other": {}}, checkpoint)
            with self.assertRaisesRegex(CheckpointError, "missing keys"):
                load_checkpoint(checkpoint, required_keys=("model",))


if __name__ == "__main__":
    unittest.main()
