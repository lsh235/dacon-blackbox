from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from blackbox.inventory import build_inventory, sha256_file, write_inventory


class InventoryTests(unittest.TestCase):
    def test_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.bin"
            path.write_bytes(b"same-content")
            self.assertEqual(sha256_file(path), sha256_file(path))

    def test_records_duplicate_content_even_if_video_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.mp4").write_bytes(b"same-content")
            (root / "nested").mkdir()
            (root / "nested/b.mp4").write_bytes(b"same-content")
            records, summary = build_inventory(root)
            self.assertEqual(len(records), 2)
            self.assertEqual(summary["undecodable_count"], 2)
            self.assertEqual(
                summary["duplicate_content_groups"],
                [["a.mp4", "nested/b.mp4"]],
            )

    def test_writes_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir()
            (root / "sample.mp4").write_bytes(b"placeholder")
            csv_path = Path(temporary) / "inventory.csv"
            summary_path = Path(temporary) / "summary.json"
            write_inventory(root, csv_path, summary_path)
            self.assertIn("relative_path", csv_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(summary_path.read_text())["video_count"], 1)


if __name__ == "__main__":
    unittest.main()
