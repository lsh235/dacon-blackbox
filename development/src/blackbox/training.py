"""Orchestrate baseline training for selected stages."""

from __future__ import annotations

from pathlib import Path

from blackbox.stages.stage1 import fit_stage1
from blackbox.stages.stage2 import fit_stage2
from blackbox.stages.stage3 import fit_stage3


def train_baseline(
    data_root: str | Path,
    model_root: str | Path,
    *,
    stages: tuple[int, ...] = (1, 2, 3),
    epochs: int = 1,
    pretrained_stage2: bool = False,
) -> list[Path]:
    data = Path(data_root)
    model = Path(model_root)
    artifacts: list[Path] = []
    if 1 in stages:
        artifacts.append(
            fit_stage1(data / "stage1", model / "stage1", epochs=epochs)
        )
    if 2 in stages:
        checkpoint, backbone = fit_stage2(
            data / "stage2",
            model / "stage2",
            epochs=epochs,
            pretrained_backbone=pretrained_stage2,
        )
        artifacts.extend([checkpoint, backbone])
    if 3 in stages:
        artifacts.append(
            fit_stage3(data / "stage3", model / "stage3", epochs=epochs)
        )
    return artifacts
