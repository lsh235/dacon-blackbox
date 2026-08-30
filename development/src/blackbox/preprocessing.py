"""Offline, reproducible feature extraction for training-only datasets.

The preprocessor owns every expensive OpenCV decode, FFT, and Farneback call.
Training loaders consume only the ``data/processed`` manifest and ``.npy``
arrays.  Submission inference intentionally remains separate because an
evaluation video cannot be preprocessed ahead of time.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import numpy as np
import torch


PREPROCESS_SCHEMA = "blackbox-offline-features-v1"
STAGE1_PREPROCESS_SCHEMA = "blackbox-stage1-multistream-v2"
DEFAULT_PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npy(path: Path, value: torch.Tensor | np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.asarray(array, dtype=np.float32)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **values: torch.Tensor | np.ndarray) -> None:
    """Atomically store named Stage 1 uint8 views in one compressed archive."""

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for name, value in values.items()
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_name(path: str | Path) -> str:
    return str(Path(path).resolve())


def stage1_feature_key(
    video_path: str | Path,
    *,
    size: int,
    frames: int,
    slot: int,
    slots: int,
    jitter_frames: int,
    forensic_size: int,
) -> str:
    """Name a raw-view cache without reading the source at load time."""

    return _digest(
        {
            "schema": STAGE1_PREPROCESS_SCHEMA,
            "stage": "stage1",
            "source": _source_name(video_path),
            "size": size,
            "frames": frames,
            "slot": slot,
            "slots": slots,
            "jitter_frames": jitter_frames,
            "forensic_size": forensic_size,
            "sampling": "centered_contiguous_region_context",
        }
    )


def stage1_feature_path(
    processed_root: str | Path,
    video_path: str | Path,
    *,
    size: int,
    frames: int,
    slot: int,
    slots: int,
    jitter_frames: int,
    forensic_size: int,
) -> Path:
    key = stage1_feature_key(
        video_path,
        size=size,
        frames=frames,
        slot=slot,
        slots=slots,
        jitter_frames=jitter_frames,
        forensic_size=forensic_size,
    )
    return Path(processed_root) / "stage1" / "clips" / f"{key}.npz"


def load_stage1_clip(
    processed_root: str | Path,
    video_path: str | Path,
    *,
    size: int,
    frames: int,
    slot: int,
    slots: int,
    jitter_frames: int,
    forensic_size: int,
) -> dict[str, torch.Tensor]:
    """Load pre-augmentation RGB views and fail closed when absent/stale."""

    path = stage1_feature_path(
        processed_root,
        video_path,
        size=size,
        frames=frames,
        slot=slot,
        slots=slots,
        jitter_frames=jitter_frames,
        forensic_size=forensic_size,
    )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"rgb", "forensic_rgb"}:
                raise ValueError(f"invalid Stage 1 cache keys: {archive.files}")
            rgb = archive["rgb"].copy()
            forensic_rgb = archive["forensic_rgb"].copy()
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"missing or incompatible offline Stage 1 clip: {path}; "
            "run preprocess_data.py before training"
        ) from exc
    context_frames = frames + 2 * jitter_frames
    expected_rgb = (3, context_frames, size, size)
    expected_forensic = (3, context_frames, forensic_size, forensic_size)
    if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
        raise ValueError(f"invalid Stage 1 RGB cache {path}: expected uint8 {expected_rgb}")
    if forensic_rgb.shape != expected_forensic or forensic_rgb.dtype != np.uint8:
        raise ValueError(
            f"invalid Stage 1 forensic cache {path}: expected uint8 {expected_forensic}"
        )
    return {
        "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
        "forensic_rgb": torch.from_numpy(np.ascontiguousarray(forensic_rgb)),
    }


def window_manifest_path(processed_root: str | Path, stage: str) -> Path:
    return Path(processed_root) / stage / "windows" / "manifest.json"


def load_processed_window_entries(
    processed_root: str | Path,
    *,
    stage: str,
    records: Iterable[tuple[str, str | Path]],
    window_frames: int,
    stride: int,
    size: int,
    farneback: dict[str, object],
) -> list[dict[str, Any]]:
    """Read only matching precomputed window entries; never open a video."""

    manifest_path = window_manifest_path(processed_root, stage)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileNotFoundError(
            f"missing offline {stage} manifest: {manifest_path}; run preprocess_data.py before training"
        ) from exc
    if manifest.get("schema") != PREPROCESS_SCHEMA or manifest.get("stage") != stage:
        raise ValueError(f"incompatible processed manifest: {manifest_path}")
    expected_sources = {(str(video_id), _source_name(path)) for video_id, path in records}
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"processed manifest entries must be a list: {manifest_path}")
    selected = [
        entry
        for entry in entries
        if (str(entry.get("id")), str(entry.get("source"))) in expected_sources
        and entry.get("window_frames") == window_frames
        and entry.get("stride") == stride
        and entry.get("size") == size
        and entry.get("farneback") == farneback
    ]
    found_sources = {(str(entry.get("id")), str(entry.get("source"))) for entry in selected}
    missing = expected_sources - found_sources
    if missing:
        raise FileNotFoundError(
            f"processed {stage} windows are missing for {sorted(missing)}; run preprocess_data.py"
        )
    selected.sort(key=lambda entry: (str(entry["id"]), int(entry["start_frame"])))
    return selected


def _existing_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    entries = payload.get("entries", [])
    return entries if isinstance(entries, list) else []


def preprocess_stage1(
    data_dir: str | Path,
    processed_root: str | Path,
    *,
    size: int,
    frames: int,
    slots: int,
    jitter_frames: int,
    forensic_size: int,
    feature_mode: str,
    overwrite: bool = False,
) -> dict[str, int]:
    """Cache contiguous RGB views; model features are computed after augmentation."""

    import pandas as pd

    from blackbox.stages.stage1.dataset import decode_contiguous_views, feature_channels

    root = Path(data_dir)
    labels = pd.read_csv(root / "labels.csv")
    if not {"path", "label"}.issubset(labels.columns):
        raise ValueError("Stage 1 labels need path and label columns")
    if min(size, frames, slots, forensic_size) < 1 or jitter_frames < 0:
        raise ValueError("Stage 1 cache geometry is invalid")
    feature_channels(feature_mode)
    videos = sorted({root / str(path) for path in labels["path"].tolist()})
    created = reused = 0
    for video in videos:
        if not video.is_file():
            raise FileNotFoundError(f"Stage 1 video not found: {video}")
        for slot in range(slots):
            target = stage1_feature_path(
                processed_root,
                video,
                size=size,
                frames=frames,
                slot=slot,
                slots=slots,
                jitter_frames=jitter_frames,
                forensic_size=forensic_size,
            )
            if target.is_file() and not overwrite:
                reused += 1
                continue
            views = decode_contiguous_views(
                video,
                size=size,
                forensic_size=forensic_size,
                frames=frames,
                slot=slot,
                slots=slots,
                context_jitter_frames=jitter_frames,
            )
            _atomic_npz(target, **views)
            created += 1
    _atomic_json(
        Path(processed_root) / "stage1" / "manifest.json",
        {
            "schema": STAGE1_PREPROCESS_SCHEMA,
            "stage": "stage1",
            "size": size,
            "frames": frames,
            "slots": slots,
            "jitter_frames": jitter_frames,
            "forensic_size": forensic_size,
            "cache_dtype": "uint8",
            "sampling": "centered_contiguous_region_context",
            "supported_feature_modes": ["rgb", "rgb_fft"],
            "sources": [_source_name(video) for video in videos],
        },
    )
    return {"created": created, "reused": reused, "videos": len(videos)}


def _preprocess_windows(
    *,
    stage: str,
    records: Sequence[tuple[str, Path]],
    processed_root: str | Path,
    window_frames: int,
    stride: int,
    size: int,
    farneback_config,
    overwrite: bool,
    max_windows_per_video: int | None,
) -> dict[str, int]:
    """Decode RGB windows and Farneback flow exactly once, outside DataLoader."""

    from blackbox.stages.stage2.dataset_stage2 import (
        decode_stage2_window,
        farneback_optical_flow,
        sliding_window_starts,
        video_frame_count,
    )

    manifest_path = window_manifest_path(processed_root, stage)
    existing = _existing_entries(manifest_path)
    retained = {
        (str(entry.get("id")), str(entry.get("source")), int(entry.get("start_frame", -1))): entry
        for entry in existing
    }
    entries: list[dict[str, Any]] = []
    created = reused = 0
    for video_id, video_path in records:
        total = video_frame_count(video_path)
        starts = sliding_window_starts(total, window_frames, stride)
        if max_windows_per_video is not None:
            starts = starts[:max_windows_per_video]
        for start in starts:
            source = _source_name(video_path)
            entry_key = (video_id, source, start)
            key = _digest(
                {
                    "schema": PREPROCESS_SCHEMA,
                    "stage": stage,
                    "source": source,
                    "start_frame": start,
                    "window_frames": window_frames,
                    "stride": stride,
                    "size": size,
                    "farneback": asdict(farneback_config),
                }
            )
            relative_rgb = Path("rgb") / f"{key}.npy"
            relative_flow = Path("flow") / f"{key}.npy"
            rgb_path = manifest_path.parent / relative_rgb
            flow_path = manifest_path.parent / relative_flow
            previous = retained.get(entry_key)
            if (
                previous is not None
                and rgb_path.is_file()
                and flow_path.is_file()
                and not overwrite
            ):
                entries.append({**previous, "stride": stride})
                reused += 1
                continue
            frames, valid_length = decode_stage2_window(
                video_path,
                start_frame=start,
                window_frames=window_frames,
                size=size,
            )
            flow = farneback_optical_flow(frames, valid_length=valid_length, config=farneback_config)
            _atomic_npy(rgb_path, frames)
            _atomic_npy(flow_path, flow)
            entries.append(
                {
                    "id": video_id,
                    "source": source,
                    "start_frame": start,
                    "end_frame": min(start + window_frames, total),
                    "valid_length": valid_length,
                    "window_frames": window_frames,
                    "size": size,
                    "farneback": asdict(farneback_config),
                    "rgb": str(relative_rgb),
                    "flow": str(relative_flow),
                }
            )
            created += 1
    entries.sort(key=lambda entry: (str(entry["id"]), int(entry["start_frame"])))
    _atomic_json(
        manifest_path,
        {"schema": PREPROCESS_SCHEMA, "stage": stage, "entries": entries},
    )
    return {"created": created, "reused": reused, "videos": len(records), "windows": len(entries)}


def preprocess_stage2(
    data_dir: str | Path,
    processed_root: str | Path,
    **kwargs: Any,
) -> dict[str, int]:
    """Precompute Stage 2 RGB and dense-flow windows without label content."""

    from blackbox.stages.stage2.dataset_stage2 import read_stage2_records

    records = read_stage2_records(data_dir)
    return _preprocess_windows(
        stage="stage2",
        records=[(record.video_id, record.video_path) for record in records],
        processed_root=processed_root,
        **kwargs,
    )


def preprocess_stage3(
    data_dir: str | Path,
    processed_root: str | Path,
    **kwargs: Any,
) -> dict[str, int]:
    """Precompute Stage 3 RGB and dense-flow windows without motion labels."""

    from blackbox.stages.stage3.dataset_stage3 import read_stage3_records

    records = read_stage3_records(data_dir)
    return _preprocess_windows(
        stage="stage3",
        records=[(record.video_id, record.video_path) for record in records],
        processed_root=processed_root,
        **kwargs,
    )
