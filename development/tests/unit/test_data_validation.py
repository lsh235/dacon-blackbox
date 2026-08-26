from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from blackbox.data_validation import DataValidationError, validate_public_example


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_video_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video-placeholder")


class PublicExampleValidationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        _write_csv(
            root / "stage1/labels.csv",
            ["ID", "path", "label"],
            [["S1", "original/a.mp4", "ORIGINAL"]],
        )
        _write_video_placeholder(root / "stage1/original/a.mp4")
        _write_csv(
            root / "stage2/labels.csv",
            ["ID", "path", "t_collision", "t_entry", "evasion_space", "entry_side"],
            [["S2", "videos/b.mp4", 4, -1, -1, -1]],
        )
        _write_video_placeholder(root / "stage2/videos/b.mp4")
        _write_csv(
            root / "stage3/labels.csv",
            ["ID", "sample_index", "frame_index", "time_seconds", "accel_label", "steer_label"],
            [["S3", 0, 0, 0.0, "CONSTANT", "STRAIGHT"]],
        )
        _write_video_placeholder(root / "stage3/videos/S3.mp4")

    def test_accepts_documented_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            self.assertEqual(
                validate_public_example(root),
                {"stage1_rows": 1, "stage2_rows": 1, "stage3_rows": 1, "stage3_videos": 1},
            )

    def test_rejects_missing_referenced_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            (root / "stage2/videos/b.mp4").unlink()
            with self.assertRaisesRegex(DataValidationError, "missing or empty video"):
                validate_public_example(root)

    def test_rejects_undecodable_video_when_decode_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            with self.assertRaisesRegex(DataValidationError, "cannot decode first frame"):
                validate_public_example(root, decode=True)


if __name__ == "__main__":
    unittest.main()
