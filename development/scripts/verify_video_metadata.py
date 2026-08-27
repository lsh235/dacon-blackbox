#!/usr/bin/env python3
"""Report OpenCV/decode/label time-axis conflicts before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blackbox.video_metadata import scan_video_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="One stage directory containing videos/ and labels.csv")
    parser.add_argument("--threshold", type=float, default=0.10, help="Relative mismatch threshold; 0.10 means 10%")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    report = scan_video_metadata(args.data_dir, threshold=args.threshold)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
