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

    sequences: list[tuple[torch.Tensor, int]] = []
    with torch.inference_mode():
        for row in labels.itertuples():
            frames = decode_video_frames(data_root / row.path)
            batches = []
            for start in range(0, len(frames), 64):
                images = torch.stack(
                    [transform(Image.fromarray(frame)) for frame in frames[start : start + 64]]
                ).to(device)
                batches.append(backbone(images).float().cpu())
            sequences.append(
                (torch.cat(batches), min(int(row.t_collision), len(frames) - 1))
            )

    temporal = Stage2Temporal().to(device)
    optimizer = torch.optim.AdamW(temporal.parameters(), lr=2e-4)
    for _ in range(max(0, epochs)):
        temporal.train()
        for sequence, target in sequences:
            collision, _, _ = temporal.logits(sequence[None].to(device))
            loss = nn.functional.cross_entropy(
                collision,
                torch.tensor([target], dtype=torch.long, device=device),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    checkpoint = output / "best.pt"
    torch.save({"model": temporal.state_dict()}, checkpoint)
    del backbone, temporal, optimizer, sequences
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
