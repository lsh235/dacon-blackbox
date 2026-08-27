"""CLI for Stage 1 training with cosine scheduling and durable epoch logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from blackbox.stages.stage1.baseline import fit_stage1
from blackbox.training_control import TrainingControlConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    checkpoint = fit_stage1(
        args.data_dir,
        args.model_dir,
        epochs=args.epochs,
        training_control=TrainingControlConfig(
            min_learning_rate=args.min_learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            validation_fraction=args.validation_fraction,
            log_dir=args.log_dir,
        ),
    )
    print(f"[OK] checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
