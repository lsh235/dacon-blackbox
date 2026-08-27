#!/usr/bin/env python3
"""Train one or more supplied Stage baselines outside DOC/."""

from __future__ import annotations

import argparse
from pathlib import Path

from blackbox.data_validation import validate_public_example
from blackbox.training import train_baseline
from blackbox.training_control import TrainingControlConfig
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
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
    parser.add_argument("--stage1-size", type=int, default=224)
    parser.add_argument("--stage1-frames", type=int, default=16)
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
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
        stage1_size=args.stage1_size,
        stage1_frames=args.stage1_frames,
        stage1_batch_size=args.stage1_batch_size,
        training_control=TrainingControlConfig(
            min_learning_rate=args.min_learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            validation_fraction=args.validation_fraction,
            log_dir=args.log_dir,
            use_amp=args.use_amp,
        ),
        processed_root=args.processed_root,
    )
    for artifact in artifacts:
        print(f"[OK] {artifact} ({artifact.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
