"""Stage 2 ResNet18 plus bidirectional GRU baseline."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from blackbox.common.runtime import (
    DEFAULT_SEED,
    autocast_context,
    choose_device,
    decode_video_frames,
    load_checkpoint,
    release_device_cache,
    seed_everything,
)
from blackbox.contracts import validate_prediction_frame
from blackbox.training_control import (
    EarlyStopping,
    JsonlTrainingLogger,
    TrainingControlConfig,
    cosine_scheduler,
    group_holdout_indices,
)


class Stage2Temporal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r = nn.GRU(
            512,
            192,
            2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.tc = nn.Linear(384, 1)
        self.te = nn.Linear(384, 1)
        self.scene = nn.Sequential(
            nn.Linear(768, 192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, 4),
        )

    def logits(self, inputs: torch.Tensor):
        hidden, _ = self.r(inputs)
        return (
            self.tc(hidden).squeeze(-1),
            self.te(hidden).squeeze(-1),
            hidden,
        )

    def forward(self, inputs: torch.Tensor):
        collision, entry, hidden = self.logits(inputs)
        collision_index = collision.argmax(dim=1)
        entry_index = entry.argmax(dim=1)
        batch = torch.arange(len(hidden), device=hidden.device)
        scene_input = torch.cat(
            [hidden[batch, collision_index], hidden[batch, entry_index]], dim=1
        )
        return collision_index, entry_index, self.scene(scene_input)


def _resnet_backbone(*, pretrained: bool) -> nn.Module:
    if pretrained:
        try:
            return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            print(f"warning: pretrained ResNet18 unavailable, using random weights: {exc}")
    return resnet18(weights=None)


def fit_stage2(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
    pretrained_backbone: bool = False,
    training_control: TrainingControlConfig = TrainingControlConfig(),
) -> tuple[Path, Path]:
    seed_everything(seed)
    data_root = Path(data_dir)
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data_root / "labels.csv")
    device = choose_device()
    backbone = _resnet_backbone(pretrained=pretrained_backbone)
    backbone_path = output / "resnet18-f37072fd.pth"
    torch.save(backbone.state_dict(), backbone_path)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    sequences: list[tuple[str, torch.Tensor, int]] = []
    with torch.inference_mode():
        for row in labels.itertuples():
            frames = decode_video_frames(data_root / row.path)
            batches = []
            for start in range(0, len(frames), 64):
                images = torch.stack(
                    [transform(Image.fromarray(frame)) for frame in frames[start : start + 64]]
                ).to(device)
                batches.append(backbone(images).float().cpu())
            sequences.append((str(row.ID), torch.cat(batches), min(int(row.t_collision), len(frames) - 1)))

    train_indices, valid_indices = group_holdout_indices(
        [video_id for video_id, _, _ in sequences],
        validation_fraction=training_control.validation_fraction,
    )
    train_sequences = [sequence for index, sequence in enumerate(sequences) if index in train_indices]
    valid_sequences = [sequence for index, sequence in enumerate(sequences) if index in valid_indices]

    temporal = Stage2Temporal().to(device)
    optimizer = torch.optim.AdamW(temporal.parameters(), lr=2e-4)
    scheduler = cosine_scheduler(
        optimizer,
        epochs=epochs,
        minimum_learning_rate=training_control.min_learning_rate,
    )
    logger = JsonlTrainingLogger("stage2", training_control.log_dir)
    early_stopping = EarlyStopping(
        mode="max" if valid_sequences else "min",
        patience=training_control.early_stopping_patience,
        min_delta=training_control.early_stopping_min_delta,
    )
    for epoch in range(max(0, epochs)):
        temporal.train()
        losses: list[float] = []
        for _, sequence, target in train_sequences:
            collision, _, _ = temporal.logits(sequence[None].to(device))
            loss = nn.functional.cross_entropy(
                collision,
                torch.tensor([target], dtype=torch.long, device=device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        average_loss = sum(losses) / max(1, len(losses))
        valid_metric: float | None = None
        if valid_sequences:
            temporal.eval()
            correct = 0
            with torch.inference_mode():
                for _, sequence, target in valid_sequences:
                    collision, _, _ = temporal.logits(sequence[None].to(device))
                    correct += int(int(collision.argmax(dim=1).item()) == target)
            valid_metric = correct / len(valid_sequences)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        monitor_name = "valid_collision_top1_group_holdout" if valid_metric is not None else "train_loss_proxy_no_validation"
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
            f"[Stage2][epoch {epoch + 1}/{epochs}] loss={average_loss:.6f} "
            f"lr={learning_rate:.3e} valid_collision_top1="
            f"{'unavailable' if valid_metric is None else f'{valid_metric:.6f}'}"
        )
        scheduler.step()
        if early_stopping.step(monitor_value):
            print(f"[Stage2] early stopping at epoch {epoch + 1}")
            break
    checkpoint = output / "best.pt"
    torch.save({"model": temporal.state_dict()}, checkpoint)
    del backbone, temporal, scheduler, optimizer, sequences, train_sequences, valid_sequences
    release_device_cache(device)
    return checkpoint, backbone_path


class _Stage2Frames(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def predict_stage2(data_dir, model_dir):
    device = choose_device(require_cuda=True)
    model_root = Path(model_dir)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone = resnet18(weights=None)
    backbone.load_state_dict(
        load_checkpoint(
            model_root / "resnet18-f37072fd.pth",
            weights_only=True,
        )
    )
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    temporal = Stage2Temporal()
    temporal_checkpoint = load_checkpoint(
        model_root / "best.pt",
        required_keys=("model",),
    )
    temporal.load_state_dict(temporal_checkpoint["model"])
    temporal.to(device).eval()

    image_root = Path(data_dir) / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Stage 2 image directory not found: {image_root}")
    folders = sorted(path for path in image_root.iterdir() if path.is_dir())
    if not folders:
        raise ValueError(f"no Stage 2 image folders found: {image_root}")
    rows = []
    with torch.inference_mode():
        for folder in folders:
            paths = sorted(
                (
                    path
                    for path in folder.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ),
                key=frame_number,
            )
            if not paths:
                raise ValueError(f"no Stage 2 frames found: {folder}")
            loader = DataLoader(
                _Stage2Frames(paths, transform),
                batch_size=256,
                num_workers=min(6, len(paths)),
                pin_memory=device.type == "cuda",
            )
            features = []
            for images in loader:
                with autocast_context(device):
                    features.append(
                        backbone(images.to(device, non_blocking=True)).float().cpu()
                    )
            sequence = torch.cat(features)[None].to(device)
            collision_index, entry_index, scene = temporal(sequence)
            frame_numbers = [frame_number(path) for path in paths]
            rows.append(
                {
                    "ID": folder.name,
                    "collision_frame": frame_numbers[int(collision_index.item())],
                    "entry_frame": frame_numbers[int(entry_index.item())],
                    "evasion_space": int(scene[:, :2].argmax(dim=1).item()),
                    "entry_side": "RIGHT"
                    if int(scene[:, 2:].argmax(dim=1).item())
                    else "LEFT",
                }
            )
    del backbone, temporal
    release_device_cache(device)
    frame = pd.DataFrame(
        rows,
        columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
    )
    return validate_prediction_frame("stage2", frame)
