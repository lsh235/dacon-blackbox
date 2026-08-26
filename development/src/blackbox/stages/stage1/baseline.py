"""Stage 1 MViTv2-S experiment for original versus re-recorded videos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s

from blackbox.common.runtime import (
    CheckpointError,
    DEFAULT_SEED,
    autocast_context,
    choose_device,
    load_checkpoint,
    release_device_cache,
    seed_everything,
    video_paths,
)
from blackbox.contracts import validate_prediction_frame
from blackbox.stages.stage1.dataset import (
    DEFAULT_FEATURE_MODE,
    RGB_FEATURES,
    Stage1InferenceDataset,
    Stage1TrainingDataset,
    feature_channels,
)
from blackbox.stages.stage1.losses import FocalLoss


LABEL_TO_INDEX = {"ORIGINAL": 0, "RERECORDED": 1}


class Stage1MViT(nn.Module):
    def __init__(self, *, feature_mode: str = DEFAULT_FEATURE_MODE) -> None:
        super().__init__()
        input_channels = feature_channels(feature_mode)
        self.net = mvit_v2_s(weights=None)
        if input_channels != self.net.conv_proj.in_channels:
            projection = self.net.conv_proj
            self.net.conv_proj = nn.Conv3d(
                input_channels,
                projection.out_channels,
                kernel_size=projection.kernel_size,
                stride=projection.stride,
                padding=projection.padding,
                dilation=projection.dilation,
                groups=projection.groups,
                bias=projection.bias is not None,
                padding_mode=projection.padding_mode,
            )
        self.net.head[1] = nn.Linear(self.net.head[1].in_features, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def fit_stage1(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    focal_gamma: float = 2.0,
    size: int = 224,
    frames: int = 16,
    batch_size: int = 1,
    label_frame: pd.DataFrame | None = None,
) -> Path:
    if epochs < 0:
        raise ValueError("epochs must be >= 0")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be >= 0")
    if size < 1 or frames < 1 or batch_size < 1:
        raise ValueError("size, frames, and batch_size must be >= 1")
    input_channels = feature_channels(feature_mode)
    seed_everything(seed)
    data_root = Path(data_dir)
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data_root / "labels.csv") if label_frame is None else label_frame.copy()
    required_columns = {"path", "label"}
    missing_columns = sorted(required_columns - set(labels.columns))
    if missing_columns:
        raise ValueError(f"Stage 1 labels are missing columns: {missing_columns}")
    labels["label"] = labels["label"].astype(str)
    unknown_labels = sorted(set(labels["label"]) - set(LABEL_TO_INDEX))
    if unknown_labels:
        raise ValueError(f"unsupported Stage 1 labels: {unknown_labels}")
    samples = [
        (data_root / str(row.path), LABEL_TO_INDEX[str(row.label)])
        for row in labels.itertuples(index=False)
    ]
    missing = [str(path) for path, _ in samples if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 1 training videos not found: {missing}")

    device = choose_device()
    dataset = Stage1TrainingDataset(
        samples,
        size=size,
        frames=frames,
        feature_mode=feature_mode,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model = Stage1MViT(feature_mode=feature_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FocalLoss(gamma=focal_gamma)
    for _ in range(max(0, epochs)):
        model.train()
        for clips, targets in loader:
            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(clips), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    checkpoint = output / "best.pt"
    torch.save(
        {
            "model": model.net.state_dict(),
            "size": size,
            "frames": frames,
            "feature_mode": feature_mode,
            "input_channels": input_channels,
            "loss": {"name": "focal", "gamma": float(focal_gamma), "alpha": None},
            "sampling": {"name": "uniform", "train_slots": 1},
        },
        checkpoint,
    )
    del criterion, dataset, loader, optimizer, model
    release_device_cache(device)
    return checkpoint


def score_stage1_videos(
    videos: Sequence[str | Path],
    model_dir: str | Path,
) -> list[float]:
    """Return RERECORDED probabilities in input order for local evaluation."""

    paths = [Path(video) for video in videos]
    if not paths:
        raise ValueError("Stage 1 scoring requires at least one video")
    device = choose_device(require_cuda=True)
    checkpoint = load_checkpoint(
        Path(model_dir) / "best.pt",
        required_keys=("model", "size", "frames"),
    )
    size = int(checkpoint["size"])
    frame_count = int(checkpoint["frames"])
    feature_mode = str(checkpoint.get("feature_mode", RGB_FEATURES))
    expected_channels = feature_channels(feature_mode)
    input_channels = int(checkpoint.get("input_channels", expected_channels))
    if input_channels != expected_channels:
        raise CheckpointError(
            "Stage 1 checkpoint preprocessing mismatch: "
            f"feature_mode={feature_mode!r} requires {expected_channels} channels, "
            f"checkpoint declares {input_channels}"
        )
    model = Stage1MViT(feature_mode=feature_mode).net
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    slots = 3
    dataset = Stage1InferenceDataset(
        paths,
        slots=slots,
        size=size,
        frames=frame_count,
        feature_mode=feature_mode,
    )
    workers = min(4, len(paths))
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    scores: list[list[float]] = [[] for _ in paths]
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

    del model
    release_device_cache(device)
    return [float(np.mean(values)) if values else 1.0 for values in scores]


def predict_stage1(data_dir, model_dir):
    videos = video_paths(Path(data_dir) / "videos")
    probabilities = score_stage1_videos(videos, model_dir)
    rows = [
        {
            "ID": path.stem,
            "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL",
        }
        for path, probability in zip(videos, probabilities)
    ]
    frame = pd.DataFrame(rows, columns=["ID", "answer"])
    return validate_prediction_frame("stage1", frame)
