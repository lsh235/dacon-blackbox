"""Sparse-label, sliding-window inputs for the Stage 3 Seq2Seq research path.

Source-frame positions and official 0.1-second sample indices are both kept in
the sample contract.  This module never invents an FPS conversion: callers can
train on the supplied ``frame_index`` labels while a future official time-axis
configuration selects inference rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    """Reuse the cached Stage 2 RGB+flow window decoder for Stage 3 labels."""

    def __init__(
        self,
        records: Sequence[Stage3VideoRecord],
        *,
        window_frames: int = 64,
        stride: int = 32,
        size: int = 224,
        farneback_config: FarnebackConfig = FarnebackConfig(),
        flow_cache_dir: str | Path | None = DEFAULT_STAGE3_FLOW_CACHE_DIR,
    ) -> None:
        if not records:
            raise ValueError("Stage 3 records must not be empty")
        self._annotations = {record.video_id: record.annotations for record in records}
        self._windows = Stage2SlidingWindowDataset(
            [Stage2VideoRecord(record.video_id, record.video_path) for record in records],
            window_frames=window_frames,
            stride=stride,
            size=size,
            include_flow=True,
            farneback_config=farneback_config,
            flow_cache_dir=flow_cache_dir,
        )

    @property
    def flow_cache_hits(self) -> int:
        return self._windows.flow_cache_hits

    @property
    def flow_cache_misses(self) -> int:
        return self._windows.flow_cache_misses

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        source = self._windows[index]
        video_id = str(source["id"])
        frame_numbers = source["frame_numbers"]
        valid_length = int(source["valid_length"])
        if not isinstance(frame_numbers, torch.Tensor):
            raise TypeError("Stage 2 window frame_numbers must be a tensor")
        accel_targets = torch.full_like(frame_numbers, IGNORE_INDEX)
        steer_targets = torch.full_like(frame_numbers, IGNORE_INDEX)
        sample_indices = torch.full_like(frame_numbers, IGNORE_INDEX)
        start_frame = int(frame_numbers[0])
        for annotation in self._annotations[video_id]:
            local_index = annotation.frame_index - start_frame
            if 0 <= local_index < valid_length:
                accel_targets[local_index] = annotation.accel_target
                steer_targets[local_index] = annotation.steer_target
                sample_indices[local_index] = annotation.sample_index
        return {
            "id": video_id,
            "frames": source["frames"],
            "flow": source["flow"],
            "flow_cache_hit": source["flow_cache_hit"],
            "valid_length": source["valid_length"],
            "frame_numbers": frame_numbers,
            "sample_indices": sample_indices,
            "accel_targets": accel_targets,
            "steer_targets": steer_targets,
        }


def collate_stage3_windows(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Collate Stage 3 windows while keeping original frame/sample mappings."""

    if not samples:
        raise ValueError("cannot collate an empty Stage 3 batch")
    tensor_keys = (
        "frames",
        "flow",
        "flow_cache_hit",
        "valid_length",
        "frame_numbers",
        "sample_indices",
        "accel_targets",
        "steer_targets",
    )
    return {
        "id": [str(sample["id"]) for sample in samples],
        **{key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys},
    }
