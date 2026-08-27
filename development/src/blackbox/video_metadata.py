"""Video metadata diagnostics for variable-frame-rate and label conflicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import pandas as pd

from blackbox.common.runtime import VIDEO_EXTENSIONS


@dataclass(frozen=True)
class VideoMetadataReport:
    video_id: str
    path: str
    cap_prop_fps: float | None
    cap_prop_frame_count: int | None
    decoded_frame_count: int | None
    label_sequence_length: int | None
    label_source_fps: float | None
    label_sample_hz: float | None
    recommended_fps: float | None
    fps_relative_error: float | None
    frame_count_relative_error: float | None
    flagged: bool
    reasons: list[str]


def _relative_error(observed: float | int | None, expected: float | int | None) -> float | None:
    if observed is None or expected is None or expected == 0:
        return None
    return abs(float(observed) - float(expected)) / abs(float(expected))


def _label_info(labels: pd.DataFrame, video_id: str) -> tuple[int | None, float | None, float | None]:
    if "ID" not in labels.columns:
        return None, None, None
    group = labels[labels["ID"].astype(str) == video_id]
    if group.empty or "sample_index" not in group.columns:
        return None, None, None
    sequence_length = int(pd.to_numeric(group["sample_index"], errors="coerce").max()) + 1
    source_fps: float | None = None
    sample_hz: float | None = None
    if {"frame_index", "time_seconds"}.issubset(group.columns):
        frames = pd.to_numeric(group["frame_index"], errors="coerce")
        seconds = pd.to_numeric(group["time_seconds"], errors="coerce")
        source_ratios = [float(frame / second) for frame, second in zip(frames, seconds) if second > 0]
        source_fps = float(median(source_ratios)) if source_ratios else None
        sample_indices = pd.to_numeric(group["sample_index"], errors="coerce")
        sample_ratios = [float(index / second) for index, second in zip(sample_indices, seconds) if second > 0]
        sample_hz = float(median(sample_ratios)) if sample_ratios else None
    return sequence_length, source_fps, sample_hz


def inspect_video(path: str | Path, labels: pd.DataFrame | None = None, *, threshold: float = 0.10) -> VideoMetadataReport:
    """Cross-check OpenCV metadata, an actual decode, and optional labels."""

    video = Path(path)
    capture = cv2.VideoCapture(str(video))
    cap_prop_fps: float | None = None
    cap_prop_frame_count: int | None = None
    decoded_frame_count: int | None = None
    reasons: list[str] = []
    try:
        if not capture.isOpened():
            reasons.append("cannot_open")
        else:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            cap_prop_fps = fps if math.isfinite(fps) and fps > 0 else None
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_prop_frame_count = count if count >= 0 else None
            decoded = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded += 1
            decoded_frame_count = decoded
    finally:
        capture.release()
    label_length, label_fps, label_hz = _label_info(labels, video.stem) if labels is not None else (None, None, None)
    fps_error = _relative_error(cap_prop_fps, label_fps)
    count_error = _relative_error(cap_prop_frame_count, decoded_frame_count)
    if fps_error is not None and fps_error > threshold:
        reasons.append("cap_prop_fps_vs_label_source_fps")
    if count_error is not None and count_error > threshold:
        reasons.append("cap_prop_frame_count_vs_decoded_count")
    if label_hz is not None and not math.isclose(label_hz, 10.0, rel_tol=threshold, abs_tol=0.1):
        reasons.append("label_sample_hz_not_10")
    return VideoMetadataReport(
        video_id=video.stem,
        path=str(video),
        cap_prop_fps=cap_prop_fps,
        cap_prop_frame_count=cap_prop_frame_count,
        decoded_frame_count=decoded_frame_count,
        label_sequence_length=label_length,
        label_source_fps=label_fps,
        label_sample_hz=label_hz,
        recommended_fps=label_fps if "cap_prop_fps_vs_label_source_fps" in reasons else cap_prop_fps,
        fps_relative_error=fps_error,
        frame_count_relative_error=count_error,
        flagged=bool(reasons),
        reasons=reasons,
    )


def scan_video_metadata(data_dir: str | Path, *, threshold: float = 0.10) -> dict[str, Any]:
    """Scan one Stage directory containing ``videos/`` and optional labels.csv."""

    root = Path(data_dir)
    labels_path = root / "labels.csv"
    labels = pd.read_csv(labels_path) if labels_path.is_file() else None
    video_root = root / "videos"
    if not video_root.is_dir():
        raise FileNotFoundError(f"video directory not found: {video_root}")
    videos = sorted(path for path in video_root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise ValueError(f"no videos found: {video_root}")
    reports = [asdict(inspect_video(path, labels, threshold=threshold)) for path in videos]
    return {
        "data_dir": str(root),
        "threshold": threshold,
        "videos_scanned": len(reports),
        "flagged_count": sum(bool(report["flagged"]) for report in reports),
        "reports": reports,
    }
