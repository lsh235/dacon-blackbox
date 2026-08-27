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
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT
from blackbox.training_control import (
    EarlyStopping,
    JsonlTrainingLogger,
    TrainingControlConfig,
    cosine_scheduler,
    group_holdout_indices,
    macro_f1_score,
)
from blackbox.stages.stage1.dataset import (
    DEFAULT_FEATURE_MODE,
    RGB_FEATURES,
    Stage1InferenceDataset,
    Stage1TrainAugmentation,
    Stage1TrainingDataset,
    feature_channels,
)
from blackbox.stages.stage1.losses import FocalLoss


LABEL_TO_INDEX = {"ORIGINAL": 0, "RERECORDED": 1}
DEFAULT_TTA_SLOTS = 3


def resolve_tta_slots(checkpoint: dict[str, object]) -> int:
    """Read temporal TTA slots, while keeping legacy checkpoints usable."""

    sampling = checkpoint.get("sampling", {})
    if not isinstance(sampling, dict):
        raise CheckpointError("Stage 1 checkpoint sampling metadata must be a mapping")
    value = sampling.get("inference_tta_slots", DEFAULT_TTA_SLOTS)
    try:
        slots = int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"Stage 1 inference_tta_slots must be an integer, got {value!r}") from exc
    if slots < 1:
        raise CheckpointError(f"Stage 1 inference_tta_slots must be >= 1, got {slots}")
    return slots


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
    enable_augmentation: bool = True,
    inference_tta_slots: int = DEFAULT_TTA_SLOTS,
    label_frame: pd.DataFrame | None = None,
    training_control: TrainingControlConfig = TrainingControlConfig(),
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> Path:
    if epochs < 0:
        raise ValueError("epochs must be >= 0")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be >= 0")
    if size < 1 or frames < 1 or batch_size < 1:
        raise ValueError("size, frames, and batch_size must be >= 1")
    if inference_tta_slots < 1:
        raise ValueError("inference_tta_slots must be >= 1")
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

    train_indices, valid_indices = group_holdout_indices(
        [path.stem for path, _ in samples],
        validation_fraction=training_control.validation_fraction,
    )
    train_samples = [sample for index, sample in enumerate(samples) if index in train_indices]
    valid_samples = [sample for index, sample in enumerate(samples) if index in valid_indices]
    device = choose_device()
    augmentation = Stage1TrainAugmentation() if enable_augmentation else None
    dataset = Stage1TrainingDataset(
        train_samples,
        size=size,
        frames=frames,
        feature_mode=feature_mode,
        augmentation=augmentation,
        processed_root=processed_root,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            Stage1TrainingDataset(
                valid_samples,
                size=size,
                frames=frames,
                feature_mode=feature_mode,
                augmentation=None,
                processed_root=processed_root,
            ),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        if valid_samples
        else None
    )
    model = Stage1MViT(feature_mode=feature_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = FocalLoss(gamma=focal_gamma)
    scheduler = cosine_scheduler(
        optimizer,
        epochs=epochs,
        minimum_learning_rate=training_control.min_learning_rate,
    )
    logger = JsonlTrainingLogger("stage1", training_control.log_dir)
    early_stopping = EarlyStopping(
        mode="max" if validation_loader is not None else "min",
        patience=training_control.early_stopping_patience,
        min_delta=training_control.early_stopping_min_delta,
    )
    for epoch in range(max(0, epochs)):
        model.train()
        losses: list[float] = []
        for clips, targets in loader:
            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(clips), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        average_loss = sum(losses) / max(1, len(losses))
        valid_metric: float | None = None
        if validation_loader is not None:
            model.eval()
            valid_targets: list[int] = []
            valid_predictions: list[int] = []
            with torch.inference_mode():
                for clips, targets in validation_loader:
                    logits = model(clips.to(device, non_blocking=True))
                    valid_targets.extend(targets.tolist())
                    valid_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            valid_metric = macro_f1_score(valid_targets, valid_predictions, labels=range(2))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        monitor_name = "valid_macro_f1_group_holdout" if valid_metric is not None else "train_loss_proxy_no_validation"
        monitor_value = valid_metric if valid_metric is not None else average_loss
        logger.log(
            epoch=epoch + 1,
            train_loss=average_loss,
            learning_rate=learning_rate,
            valid_metric=valid_metric,
            monitor_name=monitor_name,
            monitor_value=monitor_value,
        )
        print(
            f"[Stage1][epoch {epoch + 1}/{epochs}] loss={average_loss:.6f} "
            f"lr={learning_rate:.3e} valid_macro_f1="
            f"{'unavailable' if valid_metric is None else f'{valid_metric:.6f}'}"
        )
        scheduler.step()
        if early_stopping.step(monitor_value):
            print(f"[Stage1] early stopping at epoch {epoch + 1}")
            break
    checkpoint = output / "best.pt"
    torch.save(
        {
            "model": model.net.state_dict(),
            "size": size,
            "frames": frames,
            "feature_mode": feature_mode,
            "input_channels": input_channels,
            "loss": {"name": "focal", "gamma": float(focal_gamma), "alpha": None},
            "sampling": {
                "name": "uniform",
                "train_slots": 1,
                "inference_tta_slots": int(inference_tta_slots),
                "aggregation": "mean_rerecorded_probability",
            },
            "augmentation": (
                {"enabled": True, **augmentation.checkpoint_config()}
                if augmentation is not None
                else {"enabled": False}
            ),
        },
        checkpoint,
    )
    del criterion, dataset, loader, validation_loader, scheduler, optimizer, model
    release_device_cache(device)
    return checkpoint


def score_stage1_videos(
    videos: Sequence[str | Path],
    model_dir: str | Path,
    *,
    tta_slots: int | None = None,
) -> list[float]:
    """Average RERECORDED probabilities over early/middle/late video slots."""

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

    slots = resolve_tta_slots(checkpoint) if tta_slots is None else int(tta_slots)
    if slots < 1:
        raise ValueError("tta_slots must be >= 1")
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
