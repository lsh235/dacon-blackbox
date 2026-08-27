#!/usr/bin/env python3
"""Generate Stage 1/2/3 contract CSVs with sequential model memory release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blackbox.submission_pipeline import generate_submission_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stage3-frames-per-sample",
        type=int,
        required=True,
        help="Explicit source-frame count per official 0.1-second Stage 3 sample.",
    )
    parser.add_argument("--stage1-sample-submission", type=Path)
    parser.add_argument("--stage2-sample-submission", type=Path)
    parser.add_argument("--stage3-sample-submission", type=Path)
    args = parser.parse_args()
    sample_submissions = {
        stage: path
        for stage, path in (
            (1, args.stage1_sample_submission),
            (2, args.stage2_sample_submission),
            (3, args.stage3_sample_submission),
        )
        if path is not None
    }
    summary = generate_submission_bundle(
        args.data_root,
        args.model_root,
        args.output_root,
        stage3_frames_per_sample=args.stage3_frames_per_sample,
        sample_submissions=sample_submissions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
