"""Orchestrate baseline training for selected stages."""

from __future__ import annotations

from pathlib import Path

from blackbox.stages.stage1 import fit_stage1
from blackbox.stages.stage2 import fit_stage2
from blackbox.stages.stage3 import fit_stage3
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT
from blackbox.training_control import TrainingControlConfig


def train_baseline(
    data_root: str | Path,
    model_root: str | Path,
    *,
    stages: tuple[int, ...] = (1, 2, 3),
    epochs: int = 1,
    stage1_epochs: int = 30,
    stage1_minimum_epochs: int = 10,
    stage1_early_stopping_patience: int = 7,
    stage1_warmup_epochs: int = 3,
    stage1_backbone_learning_rate: float = 1e-5,
    stage1_auxiliary_learning_rate: float = 1e-4,
    stage1_warmup_initial_learning_rate: float = 1e-6,
    stage1_pretrained_backbone_checkpoint: str | Path | None = None,
    pretrained_stage2: bool = False,
    stage1_feature_mode: str = "rgb_fft",
    stage1_focal_gamma: float = 2.0,
    stage1_frame_classification_weight: float = 0.25,
    stage1_smoothing_weight: float = 0.05,
    stage1_smoothing_truncation: float = 4.0,
    stage1_explainability_weight: float = 0.05,
    stage1_mask_regularization_weight: float = 0.02,
    stage1_mask_sparsity_weight: float = 1e-3,
    stage1_motion_iterations: int = 3,
    stage1_correlation_radius: int = 2,
    stage1_augmentation: bool = True,
    stage1_tta_slots: int = 3,
    stage1_size: int = 224,
    stage1_frames: int = 16,
    stage1_batch_size: int = 1,
    stage1_train_slots: int = 3,
    stage1_jitter_frames: int = 4,
    stage1_random_temporal_jitter: bool = True,
    stage1_forensic_size: int = 320,
    stage1_fft_size: int = 112,
    stage1_row_profile_bins: int = 16,
    stage1_num_workers: int = 0,
    training_control: TrainingControlConfig = TrainingControlConfig(),
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> list[Path]:
    data = Path(data_root)
    model = Path(model_root)
    artifacts: list[Path] = []
    if 1 in stages:
        artifacts.append(
            fit_stage1(
                data / "stage1",
                model / "stage1",
                epochs=stage1_epochs,
                minimum_epochs=stage1_minimum_epochs,
                early_stopping_patience=stage1_early_stopping_patience,
                warmup_epochs=stage1_warmup_epochs,
                backbone_learning_rate=stage1_backbone_learning_rate,
                auxiliary_learning_rate=stage1_auxiliary_learning_rate,
                warmup_initial_learning_rate=stage1_warmup_initial_learning_rate,
                pretrained_backbone_checkpoint=stage1_pretrained_backbone_checkpoint,
                feature_mode=stage1_feature_mode,
                focal_gamma=stage1_focal_gamma,
                frame_classification_weight=stage1_frame_classification_weight,
                smoothing_weight=stage1_smoothing_weight,
                smoothing_truncation=stage1_smoothing_truncation,
                explainability_weight=stage1_explainability_weight,
                mask_regularization_weight=stage1_mask_regularization_weight,
                mask_sparsity_weight=stage1_mask_sparsity_weight,
                motion_iterations=stage1_motion_iterations,
                correlation_radius=stage1_correlation_radius,
                size=stage1_size,
                frames=stage1_frames,
                batch_size=stage1_batch_size,
                train_slots=stage1_train_slots,
                jitter_frames=stage1_jitter_frames,
                random_temporal_jitter=stage1_random_temporal_jitter,
                forensic_size=stage1_forensic_size,
                fft_size=stage1_fft_size,
                row_profile_bins=stage1_row_profile_bins,
                num_workers=stage1_num_workers,
                enable_augmentation=stage1_augmentation,
                inference_tta_slots=stage1_tta_slots,
                training_control=training_control,
                processed_root=processed_root,
            )
        )
    if 2 in stages:
        checkpoint, backbone = fit_stage2(
            data / "stage2",
            model / "stage2",
            epochs=epochs,
            pretrained_backbone=pretrained_stage2,
            training_control=training_control,
        )
        artifacts.extend([checkpoint, backbone])
    if 3 in stages:
        artifacts.append(
            fit_stage3(
                data / "stage3",
                model / "stage3",
                epochs=epochs,
                training_control=training_control,
            )
        )
    return artifacts
