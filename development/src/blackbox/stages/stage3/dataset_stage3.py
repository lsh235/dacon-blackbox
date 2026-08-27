"""Sparse-label 10 Hz windows for the Stage 3 Seq2Seq research path.

RGB frames and dense flow remain spatial tensors until the shared two-stream
CNN.  Official evaluation videos are already 10 Hz, so one decoded frame is one
sample.  Sparse public training labels can describe a different source-frame
spacing; those windows are pooled by the label-derived frame/sample ratio while
the unreliable OpenCV FPS remains diagnostic metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from blackbox.stages.stage2.dataset_stage2 import (
    DEFAULT_OPTICAL_FLOW_CACHE_DIR,
    IGNORE_INDEX,
    FarnebackConfig,
    Stage2SlidingWindowDataset,
    Stage2VideoRecord,
)
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT


ACCEL_LABEL_TO_INDEX = {
    "ACCELERATING": 0,
    "DECELERATING": 1,
    "CONSTANT": 2,
    "STOPPED": 3,
}
STEER_LABEL_TO_INDEX = {"LEFT": 0, "STRAIGHT": 1, "RIGHT": 2}
DEFAULT_STAGE3_FLOW_CACHE_DIR = (
    DEFAULT_OPTICAL_FLOW_CACHE_DIR.parent.parent / "stage3" / "optical_flow"
)
STAGE3_SAMPLES_PER_SECOND = 10.0


@dataclass(frozen=True)
class Stage3TimeAxis:
    """One video's official or sparse-training conversion to the 10 Hz grid."""

    source_fps: float | None
    frames_per_sample: int
    label_frames_per_sample: float | None = None
    metadata_frames_per_sample: int | None = None
    mode: str = "unspecified"

    @property
    def has_label_conflict(self) -> bool:
        """Whether sparse public labels disagree with container FPS metadata."""

        return (
            self.label_frames_per_sample is not None
            and self.metadata_frames_per_sample is not None
            and not math.isclose(
                float(self.metadata_frames_per_sample),
                self.label_frames_per_sample,
                rel_tol=0.05,
                abs_tol=0.5,
            )
        )


def infer_label_frames_per_sample(annotations: Sequence["Stage3Annotation"]) -> float | None:
    """Estimate frame/sample spacing from labels only for discrepancy reporting."""

    ratios = [
        annotation.frame_index / annotation.sample_index
        for annotation in annotations
        if annotation.sample_index > 0
    ]
    return float(np.median(ratios)) if ratios else None


def read_stage3_time_axis(
    video_path: str | Path,
    *,
    annotations: Sequence["Stage3Annotation"] = (),
) -> Stage3TimeAxis:
    """Resolve the official 10 Hz evaluation or sparse public training axis.

    DACON confirmed that private evaluation videos are already 10 Hz and have
    one decoded frame per ``sample_index``.  Public example labels are sparse
    training annotations and can use a different source-frame spacing, so that
    spacing is inferred from ``frame_index / sample_index`` when annotations
    are present.  ``CAP_PROP_FPS`` is retained only to expose broken metadata.
    """

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else math.nan
    finally:
        capture.release()
    source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0.0 else None
    metadata_frames_per_sample = (
        max(1, round(source_fps / STAGE3_SAMPLES_PER_SECOND))
        if source_fps is not None
        else None
    )
    label_frames_per_sample = infer_label_frames_per_sample(annotations)
    if label_frames_per_sample is not None:
        frames_per_sample = max(1, round(label_frames_per_sample))
        mode = "sparse_public_label_mapping"
    else:
        frames_per_sample = 1
        mode = "official_evaluation_10hz"
    return Stage3TimeAxis(
        source_fps=source_fps,
        frames_per_sample=frames_per_sample,
        label_frames_per_sample=label_frames_per_sample,
        metadata_frames_per_sample=metadata_frames_per_sample,
        mode=mode,
    )


@dataclass(frozen=True)
class Stage3Annotation:
    """One sparse 0.1-second label tied to an original source-frame index."""

    sample_index: int
    frame_index: int
    time_seconds: float
    accel_target: int
    steer_target: int


@dataclass(frozen=True)
class Stage3VideoRecord:
    video_id: str
    video_path: Path
    annotations: tuple[Stage3Annotation, ...]


def _required_nonnegative_integer(value: object, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stage 3 {field} must be a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"Stage 3 {field} must be non-negative, got {parsed}")
    return parsed


def read_stage3_records(data_dir: str | Path) -> list[Stage3VideoRecord]:
    """Read Stage 3 labels without assuming a frame/FPS-to-time conversion."""

    root = Path(data_dir)
    labels = pd.read_csv(root / "labels.csv")
    required = {"ID", "sample_index", "frame_index", "time_seconds", "accel_label", "steer_label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Stage 3 labels are missing columns: {missing}")

    records: list[Stage3VideoRecord] = []
    for video_id, group in labels.groupby("ID", sort=False):
        normalized_id = str(video_id).strip()
        if not normalized_id:
            raise ValueError("Stage 3 ID must not be empty")
        video_path = root / "videos" / f"{normalized_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Stage 3 training video not found: {video_path}")
        annotations: list[Stage3Annotation] = []
        seen_sample_indices: set[int] = set()
        seen_frame_indices: set[int] = set()
        for row in group.to_dict("records"):
            sample_index = _required_nonnegative_integer(row["sample_index"], field="sample_index")
            frame_index = _required_nonnegative_integer(row["frame_index"], field="frame_index")
            if sample_index in seen_sample_indices or frame_index in seen_frame_indices:
                raise ValueError(f"Stage 3 ID={normalized_id!r} has duplicate sample/frame annotations")
            accel_label = str(row["accel_label"]).strip().upper()
            steer_label = str(row["steer_label"]).strip().upper()
            if accel_label not in ACCEL_LABEL_TO_INDEX or steer_label not in STEER_LABEL_TO_INDEX:
                raise ValueError(f"Stage 3 ID={normalized_id!r} has unknown motion label")
            try:
                time_seconds = float(row["time_seconds"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Stage 3 time_seconds must be numeric, got {row['time_seconds']!r}") from exc
            if time_seconds < 0.0:
                raise ValueError("Stage 3 time_seconds must be non-negative")
            annotations.append(
                Stage3Annotation(
                    sample_index=sample_index,
                    frame_index=frame_index,
                    time_seconds=time_seconds,
                    accel_target=ACCEL_LABEL_TO_INDEX[accel_label],
                    steer_target=STEER_LABEL_TO_INDEX[steer_label],
                )
            )
            seen_sample_indices.add(sample_index)
            seen_frame_indices.add(frame_index)
        records.append(
            Stage3VideoRecord(
                video_id=normalized_id,
                video_path=video_path,
                annotations=tuple(sorted(annotations, key=lambda item: item.frame_index)),
            )
        )
    if not records:
        raise ValueError("Stage 3 labels must contain at least one video")
    return records


class Stage3SequenceWindowDataset(Dataset):
    """Convert cached raw RGB+flow windows into source-grounded 10 Hz steps."""

    def __init__(
        self,
        records: Sequence[Stage3VideoRecord],
        *,
        window_frames: int = 64,
        stride: int = 32,
        size: int = 224,
        farneback_config: FarnebackConfig = FarnebackConfig(),
        processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    ) -> None:
        if not records:
            raise ValueError("Stage 3 records must not be empty")
        self._annotations = {record.video_id: record.annotations for record in records}
        self._time_axes = {
            record.video_id: read_stage3_time_axis(record.video_path, annotations=record.annotations)
            for record in records
        }
        self._windows = Stage2SlidingWindowDataset(
            [Stage2VideoRecord(record.video_id, record.video_path) for record in records],
            window_frames=window_frames,
            stride=stride,
            size=size,
            include_flow=True,
            farneback_config=farneback_config,
            processed_root=processed_root,
            processed_stage="stage3",
        )

    @property
    def flow_cache_hits(self) -> int:
        return self._windows.flow_cache_hits

    @property
    def flow_cache_misses(self) -> int:
        return self._windows.flow_cache_misses

    def __len__(self) -> int:
        return len(self._windows)

    @staticmethod
    def _pool_to_time_steps(
        frames: torch.Tensor,
        flow: torch.Tensor,
        frame_numbers: torch.Tensor,
        *,
        valid_length: int,
        frames_per_sample: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        """Pool each source-frame chunk while preserving its spatial layout."""

        if frames.ndim != 4 or flow.ndim != 4:
            raise ValueError("raw Stage 3 frames and flow must be [time, channels, height, width]")
        if frames.shape[0] != flow.shape[0] or frame_numbers.numel() != frames.shape[0]:
            raise ValueError("raw Stage 3 tensors must share the same time dimension")
        if not 1 <= valid_length <= frames.shape[0]:
            raise ValueError("valid_length must be in [1, raw time]")
        if frames_per_sample < 1:
            raise ValueError("frames_per_sample must be >= 1")

        # The window can start at an arbitrary source frame.  Drop only its
        # prefix until the next global 0.1-second boundary; overlapping windows
        # cover the same prefix in the preceding chunk.
        first_source_frame = int(frame_numbers[0].item())
        local_start = (-first_source_frame) % frames_per_sample
        pooled_frames: list[torch.Tensor] = []
        pooled_flow: list[torch.Tensor] = []
        pooled_numbers: list[int] = []
        source_ranges: list[tuple[int, int]] = []
        while local_start < valid_length:
            local_end = min(local_start + frames_per_sample, valid_length)
            pooled_frames.append(frames[local_start:local_end].mean(dim=0))
            # Mean pooling retains an HxW flow map for the temporal CNN; it
            # never collapses the road-ground motion to a scalar statistic.
            pooled_flow.append(flow[local_start:local_end].mean(dim=0))
            start = int(frame_numbers[local_start].item())
            end = int(frame_numbers[local_end - 1].item()) + 1
            pooled_numbers.append(start)
            source_ranges.append((start, end))
            local_start = local_end
        if not pooled_frames:
            # A sub-step tail can occur only for an unaligned one-frame window.
            # Retain it rather than emitting an empty sequence to the LSTM.
            pooled_frames = [frames[:valid_length].mean(dim=0)]
            pooled_flow = [flow[:valid_length].mean(dim=0)]
            start = int(frame_numbers[0].item())
            pooled_numbers = [start]
            source_ranges = [(start, int(frame_numbers[valid_length - 1].item()) + 1)]
        return (
            torch.stack(pooled_frames),
            torch.stack(pooled_flow),
            torch.tensor(pooled_numbers, dtype=torch.long),
            source_ranges,
        )

    def __getitem__(self, index: int) -> dict[str, object]:
        source = self._windows[index]
        video_id = str(source["id"])
        frame_numbers = source["frame_numbers"]
        valid_length = int(source["valid_length"])
        if not isinstance(frame_numbers, torch.Tensor):
            raise TypeError("Stage 2 window frame_numbers must be a tensor")
        frames = source["frames"]
        flow = source["flow"]
        if not isinstance(frames, torch.Tensor) or not isinstance(flow, torch.Tensor):
            raise TypeError("Stage 2 window frames and flow must be tensors")
        time_axis = self._time_axes[video_id]
        frames, flow, frame_numbers, source_ranges = self._pool_to_time_steps(
            frames,
            flow,
            frame_numbers,
            valid_length=valid_length,
            frames_per_sample=time_axis.frames_per_sample,
        )
        accel_targets = torch.full_like(frame_numbers, IGNORE_INDEX)
        steer_targets = torch.full_like(frame_numbers, IGNORE_INDEX)
        sample_indices = torch.full_like(frame_numbers, IGNORE_INDEX)
        for annotation in self._annotations[video_id]:
            for local_index, (start_frame, end_frame) in enumerate(source_ranges):
                if start_frame <= annotation.frame_index < end_frame:
                    accel_targets[local_index] = annotation.accel_target
                    steer_targets[local_index] = annotation.steer_target
                    sample_indices[local_index] = annotation.sample_index
                    break
        return {
            "id": video_id,
            "frames": frames,
            "flow": flow,
            "flow_cache_hit": source["flow_cache_hit"],
            "valid_length": torch.tensor(frames.shape[0], dtype=torch.long),
            "frame_numbers": frame_numbers,
            "sample_indices": sample_indices,
            "accel_targets": accel_targets,
            "steer_targets": steer_targets,
            "source_fps": torch.tensor(
                time_axis.source_fps if time_axis.source_fps is not None else float("nan"),
                dtype=torch.float32,
            ),
            "frames_per_sample": torch.tensor(time_axis.frames_per_sample, dtype=torch.long),
            "time_axis_conflict": torch.tensor(time_axis.has_label_conflict, dtype=torch.bool),
        }


def collate_stage3_windows(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Collate Stage 3 windows while keeping original frame/sample mappings."""

    if not samples:
        raise ValueError("cannot collate an empty Stage 3 batch")
    time = max(int(sample["valid_length"]) for sample in samples)

    def pad_time(tensor: torch.Tensor, *, value: float | int = 0) -> torch.Tensor:
        if tensor.shape[0] == time:
            return tensor
        padding = torch.full(
            (time - tensor.shape[0], *tensor.shape[1:]),
            value,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, padding], dim=0)

    time_tensor_keys = ("frames", "flow", "frame_numbers", "sample_indices", "accel_targets", "steer_targets")
    padding_values = {
        "frames": 0.0,
        "flow": 0.0,
        "frame_numbers": 0,
        "sample_indices": IGNORE_INDEX,
        "accel_targets": IGNORE_INDEX,
        "steer_targets": IGNORE_INDEX,
    }
    scalar_tensor_keys = ("flow_cache_hit", "valid_length", "source_fps", "frames_per_sample", "time_axis_conflict")
    return {
        "id": [str(sample["id"]) for sample in samples],
        **{
            key: torch.stack([pad_time(sample[key], value=padding_values[key]) for sample in samples])
            for key in time_tensor_keys
        },
        **{key: torch.stack([sample[key] for sample in samples]) for key in scalar_tensor_keys},
    }
