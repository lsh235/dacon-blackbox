"""Memory-bounded soft-voting inference for Stage 1, 2, and 3."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from blackbox.common.runtime import video_paths
from blackbox.contracts import validate_prediction_frame
from blackbox.stages.stage1.baseline import score_stage1_checkpoint
from blackbox.stages.stage2.baseline import score_stage2_checkpoint, stage2_scores_to_frame
from blackbox.stages.stage3.baseline import score_stage3_checkpoint, stage3_scores_to_frame
from blackbox.stages.stage3.dataset_stage3 import read_stage3_time_axis


def _checkpoints(paths: Sequence[str | Path], *, stage: int) -> list[Path]:
    checkpoints = [Path(path) for path in paths]
    if not checkpoints:
        raise ValueError(f"Stage {stage} ensemble requires at least one checkpoint")
    return checkpoints


def smooth_temporal_probabilities(probabilities: np.ndarray, *, window: int) -> np.ndarray:
    """Centered moving average with edge replication on a 10 Hz sequence."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [time, classes] with at least two classes")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    smoothed = np.stack(
        [np.convolve(padded[:, class_index], kernel, mode="valid") for class_index in range(values.shape[1])],
        axis=1,
    ).astype(np.float32, copy=False)
    return smoothed / smoothed.sum(axis=1, keepdims=True).clip(min=np.finfo(np.float32).eps)


def predict_stage1_ensemble(
    data_dir: str | Path,
    checkpoint_paths: Sequence[str | Path],
) -> pd.DataFrame:
    """Average the RERECORDED probability from sequentially loaded folds."""

    checkpoints = _checkpoints(checkpoint_paths, stage=1)
    videos = video_paths(Path(data_dir) / "videos")
    probability_sum = np.zeros(len(videos), dtype=np.float64)
    for checkpoint in checkpoints:
        fold = np.asarray(score_stage1_checkpoint(videos, checkpoint), dtype=np.float64)
        if fold.shape != probability_sum.shape or not np.isfinite(fold).all():
            raise ValueError(f"Stage 1 checkpoint produced incompatible probabilities: {checkpoint}")
        probability_sum += fold
    probabilities = probability_sum / len(checkpoints)
    frame = pd.DataFrame(
        [
            {
                "ID": path.stem,
                "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL",
            }
            for path, probability in zip(videos, probabilities, strict=True)
        ],
        columns=["ID", "answer"],
    )
    return validate_prediction_frame("stage1", frame)


_STAGE2_PROBABILITY_KEYS = (
    "collision_probabilities",
    "entry_probabilities",
    "evasion_probabilities",
    "entry_side_probabilities",
)


def predict_stage2_ensemble(
    data_dir: str | Path,
    checkpoint_paths: Sequence[str | Path],
) -> pd.DataFrame:
    """Average temporal and scene probabilities from sequential folds."""

    checkpoints = _checkpoints(checkpoint_paths, stage=2)
    accumulated: dict[str, dict[str, object]] = {}
    expected_ids: set[str] | None = None
    for checkpoint in checkpoints:
        fold_scores = score_stage2_checkpoint(data_dir, checkpoint)
        fold = {str(item["ID"]): item for item in fold_scores}
        if len(fold) != len(fold_scores):
            raise ValueError(f"Stage 2 checkpoint produced duplicate IDs: {checkpoint}")
        if expected_ids is None:
            expected_ids = set(fold)
        elif set(fold) != expected_ids:
            raise ValueError(f"Stage 2 checkpoint produced incompatible IDs: {checkpoint}")
        for video_id, score in fold.items():
            frames = np.asarray(score["frame_numbers"], dtype=np.int64)
            fold_probabilities = {
                key: np.asarray(score[key], dtype=np.float64)
                for key in _STAGE2_PROBABILITY_KEYS
            }
            if any(not np.isfinite(values).all() for values in fold_probabilities.values()):
                raise ValueError(f"Stage 2 checkpoint produced non-finite probabilities: {checkpoint}")
            if video_id not in accumulated:
                accumulated[video_id] = {
                    "ID": video_id,
                    "frame_numbers": frames.copy(),
                    **{key: values.copy() for key, values in fold_probabilities.items()},
                }
                continue
            target = accumulated[video_id]
            if not np.array_equal(np.asarray(target["frame_numbers"]), frames):
                raise ValueError(f"Stage 2 fold frame mapping differs for {video_id}: {checkpoint}")
            for key in _STAGE2_PROBABILITY_KEYS:
                values = fold_probabilities[key]
                total = np.asarray(target[key])
                if values.shape != total.shape:
                    raise ValueError(f"Stage 2 fold {key} differs for {video_id}: {checkpoint}")
                target[key] = total + values
    averaged: list[dict[str, object]] = []
    for video_id in sorted(accumulated):
        score = accumulated[video_id]
        averaged.append(
            {
                **score,
                **{key: np.asarray(score[key]) / len(checkpoints) for key in _STAGE2_PROBABILITY_KEYS},
            }
        )
    return stage2_scores_to_frame(averaged)


def _stage3_stride(
    path: Path,
    *,
    frames_per_sample: int | None,
) -> tuple[int, dict[str, object]]:
    if frames_per_sample is not None:
        if frames_per_sample < 1:
            raise ValueError("frames_per_sample must be >= 1")
        return frames_per_sample, {"mode": "explicit_override", "frames_per_sample": frames_per_sample}
    axis = read_stage3_time_axis(path)
    return axis.frames_per_sample, {
        "mode": "cap_prop_fps",
        "source_fps": axis.source_fps,
        "frames_per_sample": axis.frames_per_sample,
        "label_conflict": axis.has_label_conflict,
    }


def predict_stage3_ensemble(
    data_dir: str | Path,
    checkpoint_paths: Sequence[str | Path],
    *,
    smoothing_window: int,
    frames_per_sample: int | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Soft-vote folds, project to 10 Hz, smooth probabilities, then argmax."""

    checkpoints = _checkpoints(checkpoint_paths, stage=3)
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    accumulated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_ids: set[str] | None = None
    for checkpoint in checkpoints:
        fold = score_stage3_checkpoint(data_dir, checkpoint)
        if expected_ids is None:
            expected_ids = set(fold)
        elif set(fold) != expected_ids:
            raise ValueError(f"Stage 3 checkpoint produced incompatible IDs: {checkpoint}")
        for video_id, (accel, steer) in fold.items():
            accel_values = np.asarray(accel, dtype=np.float64)
            steer_values = np.asarray(steer, dtype=np.float64)
            if not np.isfinite(accel_values).all() or not np.isfinite(steer_values).all():
                raise ValueError(f"Stage 3 checkpoint produced non-finite probabilities: {checkpoint}")
            if video_id not in accumulated:
                accumulated[video_id] = (accel_values.copy(), steer_values.copy())
                continue
            accel_sum, steer_sum = accumulated[video_id]
            if accel_values.shape != accel_sum.shape or steer_values.shape != steer_sum.shape:
                raise ValueError(f"Stage 3 fold sequence shape differs for {video_id}: {checkpoint}")
            accumulated[video_id] = (accel_sum + accel_values, steer_sum + steer_values)

    projected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    time_axes: dict[str, dict[str, object]] = {}
    video_root = Path(data_dir) / "videos"
    for video_id in sorted(accumulated):
        stride, metadata = _stage3_stride(
            video_root / f"{video_id}.mp4",
            frames_per_sample=frames_per_sample,
        )
        accel_sum, steer_sum = accumulated[video_id]
        accel = accel_sum[::stride] / len(checkpoints)
        steer = steer_sum[::stride] / len(checkpoints)
        projected[video_id] = (
            smooth_temporal_probabilities(accel, window=smoothing_window),
            smooth_temporal_probabilities(steer, window=smoothing_window),
        )
        time_axes[video_id] = metadata
    return stage3_scores_to_frame(projected), time_axes
