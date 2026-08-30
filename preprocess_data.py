#!/usr/bin/env python3
"""Extract training features offline into ``development/data/processed``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = ROOT / "development" / ".venv" / "bin" / "python"
if PROJECT_PYTHON.is_file() and Path(sys.executable).absolute() != PROJECT_PYTHON:
    # A direct ``./preprocess_data.py`` invocation commonly resolves to the
    # OS Python, which intentionally has no project dependencies installed.
    os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0, str(ROOT / "development" / "src"))

from blackbox.preprocessing import (  # noqa: E402
    DEFAULT_PROCESSED_ROOT,
    preprocess_stage1,
    preprocess_stage2,
    preprocess_stage3,
)
from blackbox.stages.stage2.dataset_stage2 import FarnebackConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--stages", nargs="+", choices=("stage1", "stage2", "stage3"), default=("stage1", "stage2", "stage3"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument(
        "--stage1-frames",
        type=int,
        default=32,
        help="Cached Stage 1 central-region length; v3 training requires 32.",
    )
    parser.add_argument("--stage1-slots", type=int, default=3)
    parser.add_argument("--stage1-jitter-frames", type=int, default=4)
    parser.add_argument("--stage1-forensic-size", type=int, default=320)
    parser.add_argument("--stage1-feature-mode", choices=("rgb", "rgb_fft"), default="rgb_fft")
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max-windows-per-video", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(
        args.size,
        args.stage1_frames,
        args.stage1_slots,
        args.stage1_forensic_size,
        args.window_frames,
        args.stride,
    ) < 1:
        parser.error("size, frame, slot, window, and stride values must be >= 1")
    if args.stage1_jitter_frames < 0:
        parser.error("--stage1-jitter-frames must be >= 0")
    if args.max_windows_per_video is not None and args.max_windows_per_video < 1:
        parser.error("--max-windows-per-video must be >= 1")

    root = args.data_root.resolve()
    output: dict[str, object] = {
        "data_root": str(root),
        "processed_root": str(args.processed_root.resolve()),
        "farneback": asdict(FarnebackConfig()),
        "stages": {},
    }
    if "stage1" in args.stages:
        output["stages"]["stage1"] = preprocess_stage1(
            root / "stage1",
            args.processed_root,
            size=args.size,
            frames=args.stage1_frames,
            slots=args.stage1_slots,
            jitter_frames=args.stage1_jitter_frames,
            forensic_size=args.stage1_forensic_size,
            feature_mode=args.stage1_feature_mode,
            overwrite=args.overwrite,
        )
    for stage, function in (("stage2", preprocess_stage2), ("stage3", preprocess_stage3)):
        if stage in args.stages:
            output["stages"][stage] = function(
                root / stage,
                args.processed_root,
                window_frames=args.window_frames,
                stride=args.stride,
                size=args.size,
                farneback_config=FarnebackConfig(),
                overwrite=args.overwrite,
                max_windows_per_video=args.max_windows_per_video,
            )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
