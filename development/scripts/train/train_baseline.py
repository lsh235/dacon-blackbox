#!/usr/bin/env python3
"""Train one or more supplied Stage baselines outside DOC/."""

from __future__ import annotations

import argparse
from pathlib import Path

from blackbox.data_validation import validate_public_example
from blackbox.training import train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained-stage2", action="store_true")
    parser.add_argument(
        "--stage1-feature-mode",
        choices=("rgb", "rgb_fft"),
        default="rgb_fft",
        help="Use RGB alone or concatenate per-channel 2-D FFT log spectra.",
    )
    parser.add_argument(
        "--stage1-focal-gamma",
        type=float,
        default=2.0,
        help="Focal modulation strength; 0 is exactly cross entropy.",
    )
    parser.add_argument(
        "--stage1-no-augmentation",
        action="store_true",
        help="Disable weak, clip-consistent ColorJitter and RandomAffine for Stage 1 training.",
    )
    parser.add_argument(
        "--stage1-tta-slots",
        type=int,
        default=3,
        help="Number of early/middle/late temporal samples averaged for Stage 1 inference.",
    )
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be >= 0")
    if args.stage1_focal_gamma < 0:
        parser.error("--stage1-focal-gamma must be >= 0")
    if args.stage1_tta_slots < 1:
        parser.error("--stage1-tta-slots must be >= 1")

    validate_public_example(args.data_root)
    artifacts = train_baseline(
        args.data_root,
        args.model_root,
        stages=tuple(dict.fromkeys(args.stages)),
        epochs=args.epochs,
        pretrained_stage2=args.pretrained_stage2,
        stage1_feature_mode=args.stage1_feature_mode,
        stage1_focal_gamma=args.stage1_focal_gamma,
        stage1_augmentation=not args.stage1_no_augmentation,
        stage1_tta_slots=args.stage1_tta_slots,
    )
    for artifact in artifacts:
        print(f"[OK] {artifact} ({artifact.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
