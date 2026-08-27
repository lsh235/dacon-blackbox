"""Memory-bounded sliding-window inputs for the Stage 2 research track.

The official Stage 2 output is per video: ``ID``, ``collision_frame``,
``entry_frame``, ``evasion_space``, and ``entry_side``.  The two frame values
must remain *original video frame numbers*, not a newly assigned window index.
This module therefore returns both local targets and their source-frame map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100
ENTRY_SIDE_TO_INDEX = {"LEFT": 0, "RIGHT": 1}


@dataclass(frozen=True)
class Stage2VideoRecord:
    """One Stage 2 video and its optional official supervision values.

    ``-1`` means that the public example did not provide a target.  It is not
    converted to a class: the training loss must ignore it.
    """

    video_id: str
    video_path: Path
    collision_frame: int = -1
    entry_frame: int = -1
    evasion_space: int = -1
    entry_side: int = -1


@dataclass(frozen=True)
class Stage2Window:
    """A fixed-length window whose frame range is [start_frame, end_frame)."""

    record: Stage2VideoRecord
    start_frame: int
    end_frame: int


def _optional_integer(value: object, *, field: str, allowed: set[int] | None = None) -> int:
    if value is None or pd.isna(value) or str(value).strip() in {"", "-1"}:
        return -1
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stage 2 {field} must be an integer or -1, got {value!r}") from exc
    if parsed < 0:
        return -1
    if allowed is not None and parsed not in allowed:
        raise ValueError(f"Stage 2 {field} must be one of {sorted(allowed)} or -1, got {parsed}")
    return parsed


def _optional_entry_side(value: object) -> int:
    if value is None or pd.isna(value):
        return -1
    normalized = str(value).strip().upper()
    if normalized in {"", "-1", "NAN"}:
        return -1
    try:
        return ENTRY_SIDE_TO_INDEX[normalized]
    except KeyError as exc:
        raise ValueError(
            "Stage 2 entry_side must be LEFT, RIGHT, or -1; "
            f"got {value!r}"
        ) from exc


def read_stage2_records(
    data_dir: str | Path,
    *,
    labels_csv: str | Path | None = None,
) -> list[Stage2VideoRecord]:
    """Load Stage 2 metadata without decoding video pixels into memory."""

    root = Path(data_dir)
    table = pd.read_csv(root / "labels.csv" if labels_csv is None else labels_csv)
    required = {"ID", "path"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Stage 2 labels are missing columns: {missing}")

    records: list[Stage2VideoRecord] = []
    for row in table.to_dict("records"):
        video_id = str(row["ID"]).strip()
        if not video_id:
            raise ValueError("Stage 2 ID must not be empty")
        video_path = root / str(row["path"])
        if not video_path.is_file():
            raise FileNotFoundError(f"Stage 2 training video not found: {video_path}")
        records.append(
            Stage2VideoRecord(
                video_id=video_id,
                video_path=video_path,
                collision_frame=_optional_integer(row.get("t_collision", -1), field="t_collision"),
                entry_frame=_optional_integer(row.get("t_entry", -1), field="t_entry"),
                evasion_space=_optional_integer(
                    row.get("evasion_space", -1), field="evasion_space", allowed={0, 1}
                ),
                entry_side=_optional_entry_side(row.get("entry_side", -1)),
            )
        )
    if not records:
        raise ValueError("Stage 2 labels must contain at least one video")
    if len({record.video_id for record in records}) != len(records):
        raise ValueError("Stage 2 labels contain duplicate IDs")
    return records


def video_frame_count(path: str | Path) -> int:
    """Read only video metadata; no decoded frames are retained."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open Stage 2 video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count < 1:
        raise ValueError(f"Stage 2 video has no frames: {path}")
    return frame_count


def sliding_window_starts(total_frames: int, window_frames: int, stride: int) -> list[int]:
    """Return deterministic starts and retain a tail window when needed."""

    if total_frames < 1 or window_frames < 1 or stride < 1:
        raise ValueError("total_frames, window_frames, and stride must be >= 1")
    final_start = max(0, total_frames - window_frames)
    starts = list(range(0, final_start + 1, stride)) or [0]
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _resize_center_crop(rgb: np.ndarray, size: int) -> torch.Tensor:
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    y = (resized_height - size) // 2
    x = (resized_width - size) // 2
    crop = resized[y : y + size, x : x + size].copy()
    return torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0


def decode_stage2_window(
    path: str | Path,
    *,
    start_frame: int,
    window_frames: int,
    size: int,
) -> tuple[torch.Tensor, int]:
    """Decode one window only and pad its tail by repeating the final frame.

    This is the OOM boundary for Stage 2: each worker owns at most
    ``window_frames`` decoded tensors instead of an entire video.
    """

    if start_frame < 0 or window_frames < 1 or size < 1:
        raise ValueError("start_frame must be >= 0 and window_frames/size must be >= 1")
    capture = cv2.VideoCapture(str(path))
    frames: list[torch.Tensor] = []
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open Stage 2 video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(window_frames):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                break
            frames.append(_resize_center_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"cannot decode Stage 2 window at frame {start_frame}: {path}")
    valid_length = len(frames)
    while len(frames) < window_frames:
        frames.append(frames[-1])
    return torch.stack(frames), valid_length


def local_event_target(event_frame: int, *, start_frame: int, valid_length: int) -> int:
    """Map an original frame number to a local target or ``IGNORE_INDEX``."""

    if event_frame < start_frame or event_frame >= start_frame + valid_length:
        return IGNORE_INDEX
    return event_frame - start_frame


class Stage2SlidingWindowDataset(Dataset):
    """Lazy window dataset for CNN+sequence models.

    ``__init__`` touches frame-count metadata only.  ``__getitem__`` opens one
    video, decodes one chunk, and immediately returns a fixed-size tensor.
    The original-frame map lets inference convert a local argmax back to the
    official ``collision_frame``/``entry_frame`` values.
    """

    def __init__(
        self,
        records: Sequence[Stage2VideoRecord],
        *,
        window_frames: int = 64,
        stride: int = 32,
        size: int = 224,
    ) -> None:
        if not records:
            raise ValueError("Stage 2 records must not be empty")
        if window_frames < 1 or stride < 1 or size < 1:
            raise ValueError("window_frames, stride, and size must be >= 1")
        self.window_frames = window_frames
        self.size = size
        self.windows: list[Stage2Window] = []
        for record in records:
            total_frames = video_frame_count(record.video_path)
            for start_frame in sliding_window_starts(total_frames, window_frames, stride):
                self.windows.append(
                    Stage2Window(
                        record=record,
                        start_frame=start_frame,
                        end_frame=min(start_frame + window_frames, total_frames),
                    )
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        frames, valid_length = decode_stage2_window(
            window.record.video_path,
            start_frame=window.start_frame,
            window_frames=self.window_frames,
            size=self.size,
        )
        frame_numbers = torch.arange(window.start_frame, window.start_frame + self.window_frames)
        frame_numbers[valid_length:] = window.start_frame + valid_length - 1
        return {
            "id": window.record.video_id,
            "frames": frames,
            "valid_length": torch.tensor(valid_length, dtype=torch.long),
            "frame_numbers": frame_numbers,
            "collision_target": torch.tensor(
                local_event_target(
                    window.record.collision_frame,
                    start_frame=window.start_frame,
                    valid_length=valid_length,
                ),
                dtype=torch.long,
            ),
            "entry_target": torch.tensor(
                local_event_target(
                    window.record.entry_frame,
                    start_frame=window.start_frame,
                    valid_length=valid_length,
                ),
                dtype=torch.long,
            ),
            "evasion_target": torch.tensor(window.record.evasion_space, dtype=torch.long),
            "entry_side_target": torch.tensor(window.record.entry_side, dtype=torch.long),
        }


def collate_stage2_windows(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    """Collate fixed-size windows while retaining their source-video IDs."""

    if not samples:
        raise ValueError("cannot collate an empty Stage 2 batch")
    tensor_keys = (
        "frames",
        "valid_length",
        "frame_numbers",
        "collision_target",
        "entry_target",
        "evasion_target",
        "entry_side_target",
    )
    return {
        "id": [str(sample["id"]) for sample in samples],
        **{key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys},
    }
