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
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for Stage 2/3.")
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage1-minimum-epochs", type=int, default=10)
    parser.add_argument("--stage1-early-stopping-patience", type=int, default=7)
    parser.add_argument("--stage1-warmup-epochs", type=int, default=3)
    parser.add_argument("--stage1-backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--stage1-auxiliary-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage1-warmup-initial-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--stage1-pretrained-backbone-checkpoint",
        type=Path,
        help="Optional local MViTv2-S state_dict. Omit for random initialization.",
    )
    parser.add_argument("--pretrained-stage2", action="store_true")
    parser.add_argument(
        "--stage1-feature-mode",
        choices=("rgb", "rgb_fft"),
        default="rgb_fft",
        help="Use RGB+flicker or enable the separate spatial FFT branch as well.",
    )
    parser.add_argument(
        "--stage1-focal-gamma",
        type=float,
        default=2.0,
        help="Focal modulation strength; 0 is exactly cross entropy.",
    )
    parser.add_argument("--stage1-frame-classification-weight", type=float, default=0.25)
    parser.add_argument("--stage1-smoothing-weight", type=float, default=0.05)
    parser.add_argument("--stage1-smoothing-truncation", type=float, default=4.0)
    parser.add_argument("--stage1-explainability-weight", type=float, default=0.05)
    parser.add_argument("--stage1-mask-regularization-weight", type=float, default=0.02)
    parser.add_argument("--stage1-mask-sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--stage1-motion-iterations", type=int, default=3)
    parser.add_argument("--stage1-correlation-radius", type=int, default=2)
    parser.add_argument(
        "--stage1-no-augmentation",
        action="store_true",
        help=(
            "Disable clip-consistent photometric, affine, and occlusion "
            "augmentation for Stage 1 training."
        ),
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
    parser.add_argument("--stage1-train-slots", type=int, default=3)
    parser.add_argument("--stage1-jitter-frames", type=int, default=4)
    parser.add_argument("--stage1-no-temporal-jitter", action="store_true")
    parser.add_argument("--stage1-forensic-size", type=int, default=320)
    parser.add_argument("--stage1-fft-size", type=int, default=112)
    parser.add_argument("--stage1-row-profile-bins", type=int, default=16)
    parser.add_argument("--stage1-num-workers", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be >= 0")
    if not 0 <= args.stage1_epochs <= 30:
        parser.error("--stage1-epochs must be in [0, 30]")
    if min(
        args.stage1_minimum_epochs,
        args.stage1_early_stopping_patience,
        args.stage1_warmup_epochs,
    ) < 1:
        parser.error("Stage 1 minimum epochs, patience, and warm-up epochs must be >= 1")
    if min(
        args.stage1_focal_gamma,
        args.stage1_frame_classification_weight,
        args.stage1_smoothing_weight,
        args.stage1_explainability_weight,
        args.stage1_mask_regularization_weight,
        args.stage1_mask_sparsity_weight,
    ) < 0:
        parser.error("Stage 1 focal gamma and auxiliary loss weights must be >= 0")
    if args.stage1_smoothing_truncation <= 0:
        parser.error("--stage1-smoothing-truncation must be > 0")
    if args.stage1_motion_iterations < 1 or args.stage1_correlation_radius < 0:
        parser.error("Stage 1 motion iterations must be >= 1 and radius must be >= 0")
    if args.stage1_tta_slots < 1:
        parser.error("--stage1-tta-slots must be >= 1")
    if args.stage1_train_slots != args.stage1_tta_slots:
        parser.error("--stage1-train-slots and --stage1-tta-slots must match")
    if args.stage1_jitter_frames < 0 or args.stage1_num_workers < 0:
        parser.error("Stage 1 jitter frames and workers must be >= 0")

    validate_public_example(args.data_root)
    artifacts = train_baseline(
        args.data_root,
        args.model_root,
        stages=tuple(dict.fromkeys(args.stages)),
        epochs=args.epochs,
        stage1_epochs=args.stage1_epochs,
        stage1_minimum_epochs=args.stage1_minimum_epochs,
        stage1_early_stopping_patience=args.stage1_early_stopping_patience,
        stage1_warmup_epochs=args.stage1_warmup_epochs,
        stage1_backbone_learning_rate=args.stage1_backbone_learning_rate,
        stage1_auxiliary_learning_rate=args.stage1_auxiliary_learning_rate,
        stage1_warmup_initial_learning_rate=args.stage1_warmup_initial_learning_rate,
        stage1_pretrained_backbone_checkpoint=args.stage1_pretrained_backbone_checkpoint,
        pretrained_stage2=args.pretrained_stage2,
        stage1_feature_mode=args.stage1_feature_mode,
        stage1_focal_gamma=args.stage1_focal_gamma,
        stage1_frame_classification_weight=args.stage1_frame_classification_weight,
        stage1_smoothing_weight=args.stage1_smoothing_weight,
        stage1_smoothing_truncation=args.stage1_smoothing_truncation,
        stage1_explainability_weight=args.stage1_explainability_weight,
        stage1_mask_regularization_weight=args.stage1_mask_regularization_weight,
        stage1_mask_sparsity_weight=args.stage1_mask_sparsity_weight,
        stage1_motion_iterations=args.stage1_motion_iterations,
        stage1_correlation_radius=args.stage1_correlation_radius,
        stage1_augmentation=not args.stage1_no_augmentation,
        stage1_tta_slots=args.stage1_tta_slots,
        stage1_size=args.stage1_size,
        stage1_frames=args.stage1_frames,
        stage1_batch_size=args.stage1_batch_size,
        stage1_train_slots=args.stage1_train_slots,
        stage1_jitter_frames=args.stage1_jitter_frames,
        stage1_random_temporal_jitter=not args.stage1_no_temporal_jitter,
        stage1_forensic_size=args.stage1_forensic_size,
        stage1_fft_size=args.stage1_fft_size,
        stage1_row_profile_bins=args.stage1_row_profile_bins,
        stage1_num_workers=args.stage1_num_workers,
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
