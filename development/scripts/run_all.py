#!/usr/bin/env python3
"""YAML-driven baseline training followed by sequential submission inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blackbox.experiment_config import config_path_list, config_path_value, load_experiment_config, section
from blackbox.submission_pipeline import generate_submission_bundle
from blackbox.training import train_baseline
from blackbox.training_control import TrainingControlConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage1-pretrained-backbone-checkpoint",
        type=Path,
        help="Optional explicit local MViTv2-S state_dict path.",
    )
    args = parser.parse_args()
    config, config_path = load_experiment_config(args.config)
    data = section(config, "data")
    run = section(config, "run")
    training = section(config, "training")
    stage1 = section(config, "stage1")
    stage2 = section(config, "stage2")
    stage3 = section(config, "stage3")
    inference = section(config, "inference")
    unsupported = [
        name
        for name, stage in (("stage2", stage2), ("stage3", stage3))
        if stage.get("architecture", "baseline") != "baseline"
    ]
    if unsupported:
        raise ValueError(
            "run_all.py produces submission-baseline checkpoints only; run the "
            f"Two-Stream experiment CLI separately for {unsupported}"
        )
    data_root = config_path_value(config_path, data.get("root"), field="data.root")
    inference_root = config_path_value(config_path, data.get("inference_root"), field="data.inference_root")
    processed_root = config_path_value(config_path, data.get("processed_root"), field="data.processed_root")
    model_root = config_path_value(config_path, run.get("model_root"), field="run.model_root")
    output_root = config_path_value(config_path, run.get("output_root"), field="run.output_root")
    stages = tuple(int(stage) for stage in run.get("stages", (1, 2, 3)))
    if set(stages) - {1, 2, 3}:
        raise ValueError("run.stages must contain only 1, 2, and 3")
    control = TrainingControlConfig(
        min_learning_rate=float(training.get("min_learning_rate", 1e-6)),
        early_stopping_patience=int(training.get("early_stopping_patience", 5)),
        early_stopping_min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        validation_fraction=float(training.get("validation_fraction", 0.2)),
        log_dir=config_path_value(config_path, training.get("log_dir"), field="training.log_dir"),
        use_amp=bool(training.get("use_amp", False)),
    )
    for stage in stages:
        train_baseline(
            data_root,
            model_root,
            stages=(stage,),
            epochs=int(run.get("epochs", 1)),
            stage1_epochs=int(stage1.get("epochs", 30)),
            stage1_minimum_epochs=int(stage1.get("minimum_epochs", 10)),
            stage1_early_stopping_patience=int(
                stage1.get("early_stopping_patience", 7)
            ),
            stage1_warmup_epochs=int(stage1.get("warmup_epochs", 3)),
            stage1_backbone_learning_rate=float(
                stage1.get("backbone_learning_rate", 1e-5)
            ),
            stage1_auxiliary_learning_rate=float(
                stage1.get("auxiliary_learning_rate", 1e-4)
            ),
            stage1_warmup_initial_learning_rate=float(
                stage1.get("warmup_initial_learning_rate", 1e-6)
            ),
            stage1_pretrained_backbone_checkpoint=(
                args.stage1_pretrained_backbone_checkpoint
            ),
            pretrained_stage2=bool(stage2.get("pretrained_backbone", False)),
            stage1_feature_mode=str(stage1.get("feature_mode", "rgb_fft")),
            stage1_focal_gamma=float(stage1.get("focal_gamma", 2.0)),
            stage1_frame_classification_weight=float(
                stage1.get("frame_classification_weight", 0.25)
            ),
            stage1_smoothing_weight=float(stage1.get("smoothing_weight", 0.05)),
            stage1_smoothing_truncation=float(stage1.get("smoothing_truncation", 4.0)),
            stage1_explainability_weight=float(
                stage1.get("explainability_weight", 0.05)
            ),
            stage1_mask_regularization_weight=float(
                stage1.get("mask_regularization_weight", 0.02)
            ),
            stage1_mask_sparsity_weight=float(
                stage1.get("mask_sparsity_weight", 1e-3)
            ),
            stage1_motion_iterations=int(stage1.get("motion_iterations", 3)),
            stage1_correlation_radius=int(stage1.get("correlation_radius", 2)),
            stage1_augmentation=bool(stage1.get("augmentation", True)),
            stage1_tta_slots=int(stage1.get("inference_tta_slots", 3)),
            stage1_size=int(stage1.get("size", 224)),
            stage1_frames=int(stage1.get("frames", 16)),
            stage1_batch_size=int(stage1.get("batch_size", 1)),
            stage1_train_slots=int(stage1.get("train_slots", 3)),
            stage1_jitter_frames=int(stage1.get("jitter_frames", 4)),
            stage1_random_temporal_jitter=bool(stage1.get("random_temporal_jitter", True)),
            stage1_forensic_size=int(stage1.get("forensic_size", 320)),
            stage1_fft_size=int(stage1.get("fft_size", 112)),
            stage1_row_profile_bins=int(stage1.get("row_profile_bins", 16)),
            stage1_num_workers=int(stage1.get("num_workers", 0)),
            training_control=control,
            processed_root=processed_root,
        )
    frames_per_sample = stage3.get("frames_per_sample")
    configured_checkpoints = inference.get("checkpoints", {})
    if not isinstance(configured_checkpoints, dict):
        raise ValueError("inference.checkpoints must be a mapping")
    checkpoint_paths = {
        stage: config_path_list(
            config_path,
            configured_checkpoints.get(f"stage{stage}", []),
            field=f"inference.checkpoints.stage{stage}",
        )
        for stage in (1, 2, 3)
    }
    summary = generate_submission_bundle(
        inference_root,
        model_root,
        output_root,
        stage3_frames_per_sample=None if frames_per_sample is None else int(frames_per_sample),
        smoothing_window=int(inference.get("smoothing_window", 1)),
        checkpoint_paths={stage: paths for stage, paths in checkpoint_paths.items() if paths},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
