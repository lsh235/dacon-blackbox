"""Runtime primitives shared by baseline training and inference."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
S1_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None, None]
S1_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None, None]
S3_MEAN = torch.tensor([0.45, 0.45, 0.45])[:, None, None]
S3_STD = torch.tensor([0.225, 0.225, 0.225])[:, None, None]
DEFAULT_SEED = 20260825

cv2.setNumThreads(1)


class CheckpointError(RuntimeError):
    """Raised when a model checkpoint cannot be loaded or is incomplete."""


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(*, require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda and os.getenv("BLACKBOX_ALLOW_CPU") != "1":
        raise RuntimeError(
            "CUDA GPU is required for submission inference. "
            "Set BLACKBOX_ALLOW_CPU=1 only for local structural checks."
        )
    return torch.device("cpu")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def release_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_checkpoint(
    path: str | Path,
    *,
    required_keys: tuple[str, ...] = (),
    weights_only: bool = False,
) -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=weights_only,
        )
    except Exception as exc:
        raise CheckpointError(f"cannot load checkpoint: {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise CheckpointError(
            f"checkpoint must contain a mapping: {checkpoint_path}"
        )
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise CheckpointError(
            f"checkpoint is missing keys {missing}: {checkpoint_path}"
        )
    return checkpoint


def video_paths(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"video directory not found: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no supported videos found: {directory}")
    return paths


def decode_video_frames(path: str | Path) -> list[np.ndarray]:
    video = Path(path)
    capture = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {video}")
    return frames


def crop_tensor(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    y = (resized_height - size) // 2
    x = (resized_width - size) // 2
    cropped = resized[y : y + size, x : x + size].copy()
    return torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0


def center_clip(
    path: str | Path,
    *,
    frames: int = 16,
    center: int | None = None,
    size: int = 224,
) -> tuple[torch.Tensor, int]:
    decoded = decode_video_frames(path)
    total = len(decoded)
    if center is None:
        indices = np.linspace(0, total - 1, frames).round().astype(int)
    else:
        indices = np.clip(center - frames // 2 + np.arange(frames), 0, total - 1)
    clip = torch.stack([crop_tensor(decoded[int(index)], size) for index in indices], dim=1)
    return clip, total
