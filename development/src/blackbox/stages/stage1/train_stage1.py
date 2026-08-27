"""CLI for Stage 1 training with cosine scheduling and durable epoch logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from blackbox.experiment_config import config_path_value, load_experiment_config, section, stage_paths
from blackbox.stages.stage1.baseline import fit_stage1
from blackbox.training_control import TrainingControlConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML experiment configuration")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    config: dict[str, object] = {}
    config_path: Path | None = None
    if args.config is not None:
        config, config_path = load_experiment_config(args.config)
    stage = section(config, "stage1")
    training = section(config, "training")
    if config_path is not None:
        configured_data, configured_model, configured_processed = stage_paths(config, config_path, "stage1")
    else:
        configured_data = configured_model = configured_processed = None
    data_dir = args.data_dir or configured_data
    model_dir = args.model_dir or configured_model
    processed_root = args.processed_root or configured_processed
    if data_dir is None or model_dir is None or processed_root is None:
        parser.error("--config or --data-dir, --model-dir, and --processed-root are required")
    log_dir = args.log_dir or (
        config_path_value(config_path, training.get("log_dir"), field="training.log_dir")
        if config_path is not None and training.get("log_dir") is not None
        else None
    )
    checkpoint = fit_stage1(
        data_dir,
        model_dir,
        epochs=args.epochs if args.epochs is not None else int(config.get("run", {}).get("epochs", 1)),
        feature_mode=str(stage.get("feature_mode", "rgb_fft")),
        focal_gamma=float(stage.get("focal_gamma", 2.0)),
        size=int(stage.get("size", 224)),
        frames=int(stage.get("frames", 16)),
        batch_size=int(stage.get("batch_size", 1)),
        enable_augmentation=bool(stage.get("augmentation", True)),
        inference_tta_slots=int(stage.get("inference_tta_slots", 3)),
        processed_root=processed_root,
        training_control=TrainingControlConfig(
            min_learning_rate=(
                args.min_learning_rate if args.min_learning_rate is not None else float(training.get("min_learning_rate", 1e-6))
            ),
            early_stopping_patience=(
                args.early_stopping_patience if args.early_stopping_patience is not None else int(training.get("early_stopping_patience", 5))
            ),
            early_stopping_min_delta=(
                args.early_stopping_min_delta if args.early_stopping_min_delta is not None else float(training.get("early_stopping_min_delta", 0.0))
            ),
            validation_fraction=(
                args.validation_fraction if args.validation_fraction is not None else float(training.get("validation_fraction", 0.2))
            ),
            log_dir=log_dir,
        ),
    )
    print(f"[OK] checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
