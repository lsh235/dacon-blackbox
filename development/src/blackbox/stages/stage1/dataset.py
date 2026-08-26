"""Memory-bounded video sampling and frequency features for Stage 1."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from blackbox.common.runtime import S1_MEAN, S1_STD


RGB_FEATURES = "rgb"
RGB_FFT_FEATURES = "rgb_fft"
DEFAULT_FEATURE_MODE = RGB_FFT_FEATURES
_FEATURE_CHANNELS = {RGB_FEATURES: 3, RGB_FFT_FEATURES: 6}


def feature_channels(feature_mode: str) -> int:
    """Return the MViT input-channel count for a Stage 1 feature mode."""

    try:
        return _FEATURE_CHANNELS[feature_mode]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Stage 1 feature mode: {feature_mode!r}; "
            f"expected one of {sorted(_FEATURE_CHANNELS)}"
        ) from exc


def uniform_frame_indices(
    total_frames: int,
    frames: int,
    *,
    slot: int = 0,
    slots: int = 1,
) -> np.ndarray:
    """Sample a fixed number of frame IDs uniformly inside one temporal slot."""

    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if slots < 1:
        raise ValueError("slots must be >= 1")
    if slot < 0 or slot >= slots:
        raise ValueError(f"slot must satisfy 0 <= slot < slots: slot={slot}, slots={slots}")

    boundaries = np.linspace(0, total_frames, slots + 1)
    start = int(np.floor(boundaries[slot]))
    stop = max(start, int(np.ceil(boundaries[slot + 1])) - 1)
    return np.linspace(start, stop, frames).round().astype(np.int64)


def spatial_log_spectrum(rgb_clip: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Build a standardized per-channel 2-D FFT magnitude for every frame.

    Periodic display-camera interference creates concentrated peaks in the
    spatial frequency plane. Removing the DC term and applying ``log1p`` makes
    those weaker peaks visible without letting image brightness dominate them.
    """

    if rgb_clip.ndim != 4 or rgb_clip.shape[0] != 3:
        raise ValueError(
            "rgb_clip must have shape [3, time, height, width], "
            f"got {tuple(rgb_clip.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be > 0")

    centered = rgb_clip - rgb_clip.mean(dim=(-2, -1), keepdim=True)
    coefficients = torch.fft.fft2(centered, dim=(-2, -1), norm="ortho")
    coefficients = torch.fft.fftshift(coefficients, dim=(-2, -1))
    spectrum = torch.log1p(coefficients.abs())
    mean = spectrum.mean(dim=(-2, -1), keepdim=True)
    std = spectrum.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (spectrum - mean) / std.clamp_min(eps)


def prepare_stage1_features(rgb_clip: torch.Tensor, feature_mode: str) -> torch.Tensor:
    """Apply the exact Stage 1 normalization used by both train and inference."""

    feature_channels(feature_mode)
    if rgb_clip.ndim != 4 or rgb_clip.shape[0] != 3:
        raise ValueError(
            "rgb_clip must have shape [3, time, height, width], "
            f"got {tuple(rgb_clip.shape)}"
        )
    rgb_clip = rgb_clip.float()
    mean = S1_MEAN.to(device=rgb_clip.device, dtype=rgb_clip.dtype)
    std = S1_STD.to(device=rgb_clip.device, dtype=rgb_clip.dtype)
    normalized_rgb = (rgb_clip - mean) / std
    if feature_mode == RGB_FEATURES:
        return normalized_rgb
    return torch.cat((normalized_rgb, spatial_log_spectrum(rgb_clip)), dim=0)


def _crop_rgb(rgb: np.ndarray, size: int) -> torch.Tensor:
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    y = (resized_height - size) // 2
    x = (resized_width - size) // 2
    cropped = resized[y : y + size, x : x + size].copy()
    return torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0


def decode_uniform_clip(
    path: str | Path,
    *,
    size: int,
    frames: int,
    slot: int = 0,
    slots: int = 1,
) -> torch.Tensor:
    """Decode only a fixed-size sample instead of retaining every video frame."""

    video = Path(path)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")

        total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        wanted = uniform_frame_indices(total, frames, slot=slot, slots=slots).tolist()
        output: list[torch.Tensor] = []
        capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        position = wanted[0]
        cached_index: int | None = None
        cached_frame: torch.Tensor | None = None

        for index in wanted:
            if index == cached_index and cached_frame is not None:
                output.append(cached_frame)
                continue

            ok = False
            bgr = None
            while position <= index:
                ok, bgr = capture.read()
                position += 1
                if not ok:
                    break
            if not ok or bgr is None:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cached_frame = _crop_rgb(rgb, size)
            cached_index = index
            output.append(cached_frame)
    finally:
        capture.release()
    if not output:
        raise ValueError(f"cannot decode video: {video.name}")
    while len(output) < frames:
        output.append(output[-1])
    return torch.stack(output, dim=1)


class Stage1TrainingDataset(Dataset):
    """Labeled, fixed-frame Stage 1 clips with bounded host-memory use."""

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        *,
        size: int,
        frames: int,
        feature_mode: str,
        slots: int = 1,
    ) -> None:
        if not samples:
            raise ValueError("Stage 1 training samples must not be empty")
        if slots < 1:
            raise ValueError("slots must be >= 1")
        feature_channels(feature_mode)
        invalid_labels = sorted({label for _, label in samples if label not in (0, 1)})
        if invalid_labels:
            raise ValueError(f"Stage 1 labels must be 0 or 1: {invalid_labels}")
        self.samples = list(samples)
        self.size = size
        self.frames = frames
        self.feature_mode = feature_mode
        self.slots = slots

    def __len__(self) -> int:
        return len(self.samples) * self.slots

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        video_index, slot = divmod(index, self.slots)
        path, label = self.samples[video_index]
        rgb_clip = decode_uniform_clip(
            path,
            size=self.size,
            frames=self.frames,
            slot=slot,
            slots=self.slots,
        )
        return prepare_stage1_features(rgb_clip, self.feature_mode), label


class Stage1InferenceDataset(Dataset):
    """Unlabeled multi-slot clips with a conservative decode-failure marker."""

    def __init__(
        self,
        videos: Sequence[Path],
        *,
        slots: int,
        size: int,
        frames: int,
        feature_mode: str,
    ) -> None:
        if slots < 1:
            raise ValueError("slots must be >= 1")
        feature_channels(feature_mode)
        self.videos = list(videos)
        self.slots = slots
        self.size = size
        self.frames = frames
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.videos) * self.slots

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        video_index, slot = divmod(index, self.slots)
        path = self.videos[video_index]
        try:
            rgb_clip = decode_uniform_clip(
                path,
                size=self.size,
                frames=self.frames,
                slot=slot,
                slots=self.slots,
            )
            clip = prepare_stage1_features(rgb_clip, self.feature_mode)
            valid = 1
        except (OSError, ValueError, cv2.error):
            clip = torch.zeros(
                feature_channels(self.feature_mode),
                self.frames,
                self.size,
                self.size,
            )
            valid = 0
        return clip, video_index, valid
