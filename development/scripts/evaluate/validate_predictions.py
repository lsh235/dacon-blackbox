#!/usr/bin/env python3
"""Validate a Stage prediction CSV against the documented contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from blackbox.contracts import validate_prediction_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["1", "2", "3", "stage1", "stage2", "stage3"])
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    validate_prediction_frame(args.stage, frame)
    print(f"[OK] {args.csv}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
