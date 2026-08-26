"""Stage 3 MViTv2-S multi-head baseline for vehicle-motion labels."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.models.video import mvit_v2_s

from blackbox.common.runtime import (
    DEFAULT_SEED,
    S3_MEAN,
    S3_STD,
    autocast_context,
    center_clip,
    choose_device,
    load_checkpoint,
    release_device_cache,
    seed_everything,
    video_paths,
)
from blackbox.contracts import validate_prediction_frame


ACCEL_LABELS = ["ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"]
STEER_LABELS = ["LEFT", "STRAIGHT", "RIGHT"]


class Stage3MViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = mvit_v2_s(weights=None)
        dimension = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.accel = nn.Linear(dimension, 4)
        self.steer = nn.Linear(dimension, 3)

    def forward(self, inputs: torch.Tensor):
        features = self.backbone(inputs)
        return self.accel(features), self.steer(features)


def fit_stage3(
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
    accel_map = {label: index for index, label in enumerate(ACCEL_LABELS)}
    steer_map = {label: index for index, label in enumerate(STEER_LABELS)}
    device = choose_device()
    model = Stage3MViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(max(0, epochs)):
        model.train()
        for row in labels.itertuples():
            clip, _ = center_clip(
                data_root / "videos" / f"{row.ID}.mp4",
                frames=16,
                center=int(row.frame_index),
            )
            clip = (clip - S3_MEAN[:, None, :, :]) / S3_STD[:, None, :, :]
            accel, steer = model(clip[None].to(device))
            loss = nn.functional.cross_entropy(
                accel,
                torch.tensor([accel_map[row.accel_label]], dtype=torch.long, device=device),
            )
            loss += nn.functional.cross_entropy(
                steer,
                torch.tensor([steer_map[row.steer_label]], dtype=torch.long, device=device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    checkpoint = output / "best.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    del optimizer, model
    release_device_cache(device)
    return checkpoint


def _decode_frames(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        scale = 256 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)))
        width, height = image.size
        x = (width - 224) // 2
        y = (height - 224) // 2
        image = image.crop((x, y, x + 224, y + 224))
        frames.append(
            torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8)
        )
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {path.name}")
    return torch.stack(frames)


def predict_stage3(data_dir, model_dir):
    device = choose_device(require_cuda=True)
    checkpoint = load_checkpoint(
        Path(model_dir) / "best.pt",
        required_keys=("model",),
    )
    model = Stage3MViT()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    videos = video_paths(Path(data_dir) / "videos")
    batch_size = max(1, int(os.getenv("BLACKBOX_STAGE3_BATCH_SIZE", "8")))
    rows = []
    with torch.inference_mode():
        for path in videos:
            frames = _decode_frames(path)
            frame_count = len(frames)
            centers = np.arange(frame_count)
            accel_predictions: list[int] = []
            steer_predictions: list[int] = []
            for start in range(0, frame_count, batch_size):
                center = centers[start : start + batch_size]
                indices = np.clip(
                    center[:, None] - 8 + np.arange(16)[None, :],
                    0,
                    frame_count - 1,
                )
                clips = (
                    frames[torch.from_numpy(indices)].permute(0, 2, 1, 3, 4).float()
                    / 255.0
                )
                clips = (
                    clips - S3_MEAN[None, :, None, :, :]
                ) / S3_STD[None, :, None, :, :]
                with autocast_context(device):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(dim=1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(dim=1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(
                zip(accel_predictions, steer_predictions)
            ):
                rows.append(
                    {
                        "ID": path.stem,
                        "sample_index": sample_index,
                        "accel_label": ACCEL_LABELS[accel],
                        "steer_label": STEER_LABELS[steer],
                    }
                )
    del model
    release_device_cache(device)
    frame = pd.DataFrame(
        rows,
        columns=["ID", "sample_index", "accel_label", "steer_label"],
    )
    return validate_prediction_frame("stage3", frame)
