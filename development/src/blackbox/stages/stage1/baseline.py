"""Stage 1 MViTv2-S baseline for original versus re-recorded videos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import mvit_v2_s

from blackbox.common.runtime import (
    DEFAULT_SEED,
    S1_MEAN,
    S1_STD,
    autocast_context,
    center_clip,
    choose_device,
    load_checkpoint,
    release_device_cache,
    seed_everything,
    video_paths,
)
from blackbox.contracts import validate_prediction_frame


class Stage1MViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = mvit_v2_s(weights=None)
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def fit_stage1(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
) -> Path:
    seed_everything(seed)
    data_root = Path(data_dir)
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data_root / "labels.csv")
    device = choose_device()
    model = Stage1MViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(max(0, epochs)):
        model.train()
        for row in labels.sample(frac=1, random_state=seed).itertuples():
            clip, _ = center_clip(data_root / row.path, frames=16)
            clip = (clip - S1_MEAN) / S1_STD
            target = torch.tensor(
                [0 if row.label == "ORIGINAL" else 1], dtype=torch.long, device=device
            )
            loss = nn.functional.cross_entropy(model(clip[None].to(device)), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    checkpoint = output / "best.pt"
    torch.save({"model": model.net.state_dict(), "size": 224, "frames": 16}, checkpoint)
    del optimizer, model
    release_device_cache(device)
    return checkpoint


def _clip_ids(path: Path, frame_count: int, slot: int, slots: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    center = (slot + 0.5) * total / slots
    start = max(0, min(total - frame_count, round(center - frame_count / 2)))
    return np.linspace(
        start,
        min(total - 1, start + frame_count - 1),
        frame_count,
    ).round().astype(int)


def _decode_clip(path: Path, size: int, frame_ids: np.ndarray) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    output: list[np.ndarray] = []
    wanted = [int(index) for index in frame_ids]
    capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    position = wanted[0]
    for index in wanted:
        ok = False
        bgr = None
        while position <= index:
            ok, bgr = capture.read()
            position += 1
            if not ok:
                break
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = size / min(height, width)
        resized_height = max(size, round(height * scale))
        resized_width = max(size, round(width * scale))
        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        y = (resized_height - size) // 2
        x = (resized_width - size) // 2
        output.append(rgb[y : y + size, x : x + size])
    capture.release()
    if not output:
        raise ValueError(f"cannot decode video: {path.name}")
    while len(output) < len(wanted):
        output.append(output[-1])
    clip = torch.from_numpy(np.stack(output)).permute(3, 0, 1, 2).float() / 255.0
    return (clip - S1_MEAN) / S1_STD


class _Stage1Clips(Dataset):
    def __init__(self, videos: list[Path], slots: int, size: int, frames: int) -> None:
        self.videos = videos
        self.slots = slots
        self.size = size
        self.frames = frames

    def __len__(self) -> int:
        return len(self.videos) * self.slots

    def __getitem__(self, index: int):
        video_index, slot = index // self.slots, index % self.slots
        path = self.videos[video_index]
        try:
            clip = _decode_clip(path, self.size, _clip_ids(path, self.frames, slot, self.slots))
            valid = 1
        except (OSError, ValueError, cv2.error):
            clip = torch.zeros(3, self.frames, self.size, self.size)
            valid = 0
        return clip, video_index, valid


def predict_stage1(data_dir, model_dir):
    device = choose_device(require_cuda=True)
    checkpoint = load_checkpoint(
        Path(model_dir) / "best.pt",
        required_keys=("model", "size", "frames"),
    )
    size = int(checkpoint["size"])
    frame_count = int(checkpoint["frames"])
    model = mvit_v2_s(weights=None)
    model.head[1] = nn.Linear(model.head[1].in_features, 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    videos = video_paths(Path(data_dir) / "videos")
    slots = 3
    dataset = _Stage1Clips(videos, slots, size, frame_count)
    workers = min(4, max(0, len(videos)))
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    scores: list[list[float]] = [[] for _ in videos]
    with torch.inference_mode():
        for clips, video_indices, valid in loader:
            with autocast_context(device):
                probabilities = torch.softmax(
                    model(clips.to(device, non_blocking=True)), dim=1
                )[:, 1]
            for index, value, ok in zip(
                video_indices.tolist(), probabilities.float().cpu().tolist(), valid.tolist()
            ):
                if ok:
                    scores[index].append(float(value))

    rows = []
    for path, values in zip(videos, scores):
        probability = float(np.mean(values)) if values else 1.0
        rows.append(
            {
                "ID": path.stem,
                "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL",
            }
        )
    del model
    release_device_cache(device)
    frame = pd.DataFrame(rows, columns=["ID", "answer"])
    return validate_prediction_frame("stage1", frame)
