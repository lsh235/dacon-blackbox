"""Runnable-but-experimental trainer for the Stage 2 sliding-window skeleton.

This is intentionally not wired into the submission trainer yet.  Public
examples only supervise collision frames; the official semantics and full
labels for entry/evasion/direction must be confirmed before selecting it as a
submission model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from blackbox.common.runtime import DEFAULT_SEED, choose_device, release_device_cache, seed_everything
from blackbox.stages.stage2.dataset_stage2 import (
    IGNORE_INDEX,
    Stage2SlidingWindowDataset,
    collate_stage2_windows,
    read_stage2_records,
)
from blackbox.stages.stage2.model_stage2 import Stage2CnnBiLSTM


@dataclass(frozen=True)
class Stage2WindowTrainingConfig:
    window_frames: int = 64
    stride: int = 32
    size: int = 224
    batch_size: int = 1
    frame_batch_size: int = 8
    learning_rate: float = 2e-4


def stage2_window_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, object],
) -> torch.Tensor:
    """Average only losses whose target is available in this window.

    A target frame outside the current chunk is ``IGNORE_INDEX`` rather than a
    false negative.  Full-video event aggregation remains a separate inference
    task; this skeleton establishes safe chunk-level supervision first.
    """

    terms: list[torch.Tensor] = []
    task_specs = (
        ("collision_logits", "collision_target"),
        ("entry_logits", "entry_target"),
        ("evasion_logits", "evasion_target"),
        ("entry_side_logits", "entry_side_target"),
    )
    for logits_key, target_key in task_specs:
        targets = batch[target_key]
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"{target_key} must be a tensor")
        available = targets != IGNORE_INDEX
        if bool(available.any()):
            terms.append(functional.cross_entropy(predictions[logits_key][available], targets[available]))
    if not terms:
        # Time logits include -inf in padded positions, so use a finite head
        # when constructing a differentiable zero.
        return predictions["evasion_logits"].sum() * 0.0
    return torch.stack(terms).mean()


def train_stage2_window_epoch(
    model: Stage2CnnBiLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> float:
    """Train one epoch without ever loading an entire source video."""

    model.train()
    losses: list[float] = []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        valid_lengths = batch["valid_length"].to(device, non_blocking=True)
        tensor_batch = {
            key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        loss = stage2_window_loss(model(frames, valid_lengths), tensor_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def fit_stage2_window_skeleton(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
    config: Stage2WindowTrainingConfig = Stage2WindowTrainingConfig(),
) -> Path:
    """Train the experimental chunk model and save an explicitly non-final checkpoint."""

    if epochs < 0:
        raise ValueError("epochs must be >= 0")
    seed_everything(seed)
    records = read_stage2_records(data_dir)
    dataset = Stage2SlidingWindowDataset(
        records,
        window_frames=config.window_frames,
        stride=config.stride,
        size=config.size,
    )
    device = choose_device()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_stage2_windows,
        pin_memory=device.type == "cuda",
    )
    model = Stage2CnnBiLSTM(frame_batch_size=config.frame_batch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for _ in range(epochs):
        train_stage2_window_epoch(model, loader, optimizer, device=device)

    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "stage2_window_skeleton.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "format": "experimental_stage2_window_skeleton",
            "config": asdict(config),
            "official_output": [
                "ID",
                "collision_frame",
                "entry_frame",
                "evasion_space",
                "entry_side",
            ],
        },
        checkpoint,
    )
    del dataset, loader, model, optimizer
    release_device_cache(device)
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    args = parser.parse_args()
    checkpoint = fit_stage2_window_skeleton(
        args.data_dir,
        args.model_dir,
        epochs=args.epochs,
        config=Stage2WindowTrainingConfig(
            window_frames=args.window_frames,
            stride=args.stride,
            frame_batch_size=args.frame_batch_size,
        ),
    )
    print(f"[OK] experimental checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
