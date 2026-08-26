from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from blackbox.manifest import create_manifest


class ManifestTests(unittest.TestCase):
    def test_manifest_tracks_artifact_hash_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "prediction.csv"
            artifact.write_text("ID,answer\nA,ORIGINAL\n", encoding="utf-8")
            manifest = create_manifest(
                root,
                [artifact],
                command="python run.py",
                note="structure smoke",
            )
            self.assertEqual(manifest["command"], "python run.py")
            self.assertEqual(manifest["artifacts"][0]["path"], "prediction.csv")
            self.assertEqual(len(manifest["artifacts"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
