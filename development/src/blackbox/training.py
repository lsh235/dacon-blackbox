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
    pretrained_stage2: bool = False,
    stage1_feature_mode: str = "rgb_fft",
    stage1_focal_gamma: float = 2.0,
    stage1_augmentation: bool = True,
    stage1_tta_slots: int = 3,
    stage1_size: int = 224,
    stage1_frames: int = 16,
    stage1_batch_size: int = 1,
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
                epochs=epochs,
                feature_mode=stage1_feature_mode,
                focal_gamma=stage1_focal_gamma,
                size=stage1_size,
                frames=stage1_frames,
                batch_size=stage1_batch_size,
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
