"""Experimental Stage 2 Two-Stream training and event-frame mapping helpers.

This research path is separate from the submission baseline. It uses RGB plus
cached Farneback windows and preserves unavailable public labels as ignored.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from blackbox.common.runtime import DEFAULT_SEED, choose_device, release_device_cache, seed_everything
from blackbox.stages.stage2.dataset_stage2 import (
    DEFAULT_OPTICAL_FLOW_CACHE_DIR,
    IGNORE_INDEX,
    FarnebackConfig,
    Stage2SlidingWindowDataset,
    collate_stage2_windows,
    local_event_target,
    read_stage2_records,
)
from blackbox.stages.stage2.model_stage2 import Stage2TwoStreamBiLSTM
from blackbox.training_control import (
    EarlyStopping,
    JsonlTrainingLogger,
    TrainingControlConfig,
    cosine_scheduler,
)


@dataclass(frozen=True)
class TargetMappingConfig:
    """Local-event target and overlapping-window aggregation policy."""

    mode: Literal["binary_mask", "gaussian"] = "gaussian"
    gaussian_sigma: float = 2.0
    binary_tolerance_radius: int = 0
    aggregation_policy: Literal["mean", "max"] = "mean"

    def __post_init__(self) -> None:
        if self.mode not in {"binary_mask", "gaussian"}:
            raise ValueError("target mode must be 'binary_mask' or 'gaussian'")
        if self.gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be > 0")
        if self.binary_tolerance_radius < 0:
            raise ValueError("binary_tolerance_radius must be >= 0")
        if self.aggregation_policy not in {"mean", "max"}:
            raise ValueError("aggregation_policy must be 'mean' or 'max'")


@dataclass(frozen=True)
class Stage2WindowTrainingConfig:
    window_frames: int = 64
    stride: int = 32
    size: int = 224
    batch_size: int = 1
    frame_batch_size: int = 8
    learning_rate: float = 2e-4
    num_workers: int = 0
    flow_cache_dir: str | None = str(DEFAULT_OPTICAL_FLOW_CACHE_DIR)
    farneback: FarnebackConfig = FarnebackConfig()
    target_mapping: TargetMappingConfig = TargetMappingConfig()
    training_control: TrainingControlConfig = TrainingControlConfig()

    def __post_init__(self) -> None:
        if min(self.window_frames, self.stride, self.size, self.batch_size, self.frame_batch_size) < 1:
            raise ValueError("window, stride, size, batch size, and frame batch size must be >= 1")
        if self.learning_rate <= 0.0 or self.num_workers < 0:
            raise ValueError("learning_rate must be > 0 and num_workers must be >= 0")


def build_window_event_target(
    event_frame: int,
    *,
    start_frame: int,
    valid_length: int,
    window_frames: int,
    config: TargetMappingConfig = TargetMappingConfig(),
) -> torch.Tensor | None:
    """Return a local target, or ``None`` when an event is not in the window.

    The Gaussian peak remains 1.0 for inspection. ``masked_soft_target_loss``
    normalizes it on valid positions before computing soft cross entropy.
    """

    if window_frames < valid_length or valid_length < 1:
        raise ValueError("window_frames must be >= valid_length >= 1")
    local_target = local_event_target(event_frame, start_frame=start_frame, valid_length=valid_length)
    if local_target == IGNORE_INDEX:
        return None
    positions = torch.arange(window_frames, dtype=torch.float32)
    distance = positions - float(local_target)
    if config.mode == "gaussian":
        target = torch.exp(-(distance.square()) / (2.0 * config.gaussian_sigma**2))
    else:
        target = (distance.abs() <= config.binary_tolerance_radius).to(dtype=torch.float32)
    target[valid_length:] = 0.0
    return target


def build_batch_event_targets(
    local_targets: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    time: int,
    config: TargetMappingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create masked target rows while never turning unavailable labels negative."""

    if local_targets.ndim != 1 or valid_lengths.shape != local_targets.shape:
        raise ValueError("local_targets and valid_lengths must be one-dimensional and aligned")
    if time < 1:
        raise ValueError("time must be >= 1")
    lengths = valid_lengths.to(dtype=torch.long).clamp(max=time)
    if bool((lengths < 1).any()):
        raise ValueError("valid_lengths must be >= 1")
    available = local_targets != IGNORE_INDEX
    positions = torch.arange(time, device=local_targets.device)[None, :]
    valid_mask = positions < lengths[:, None]
    distance = (positions - local_targets.clamp(min=0)[:, None]).to(dtype=torch.float32)
    if config.mode == "gaussian":
        targets = torch.exp(-(distance.square()) / (2.0 * config.gaussian_sigma**2))
    else:
        targets = (distance.abs() <= config.binary_tolerance_radius).to(dtype=torch.float32)
    return targets * valid_mask * available[:, None], available, valid_mask


def masked_soft_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    available: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Compute cross entropy against normalized binary/Gaussian target rows."""

    if logits.ndim != 2 or targets.shape != logits.shape or valid_mask.shape != logits.shape:
        raise ValueError("logits, targets, and valid_mask must share [batch, time] shape")
    if available.ndim != 1 or available.shape[0] != logits.shape[0]:
        raise ValueError("available must contain one value per batch item")
    if not bool(available.any()):
        return None
    target_sums = targets.sum(dim=1, keepdim=True)
    if bool((target_sums[available] <= 0.0).any()):
        raise ValueError("each available event target needs a valid position")
    distribution = targets / target_sums.clamp_min(torch.finfo(targets.dtype).eps)
    log_probabilities = functional.log_softmax(logits, dim=1).masked_fill(~valid_mask, 0.0)
    per_sample = -(distribution * log_probabilities).sum(dim=1)
    return per_sample[available].mean()


def stage2_window_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, object],
    *,
    target_config: TargetMappingConfig = TargetMappingConfig(),
) -> torch.Tensor:
    """Average losses only for labels that are available in each window."""

    valid_lengths = batch["valid_length"]
    if not isinstance(valid_lengths, torch.Tensor):
        raise TypeError("valid_length must be a tensor")
    terms: list[torch.Tensor] = []
    for logits_key, target_key in (("collision_logits", "collision_target"), ("entry_logits", "entry_target")):
        local_targets = batch[target_key]
        logits = predictions[logits_key]
        if not isinstance(local_targets, torch.Tensor):
            raise TypeError(f"{target_key} must be a tensor")
        targets, available, valid_mask = build_batch_event_targets(
            local_targets, valid_lengths, time=logits.shape[1], config=target_config
        )
        event_loss = masked_soft_target_loss(
            logits,
            targets.to(dtype=logits.dtype),
            available=available,
            valid_mask=valid_mask,
        )
        if event_loss is not None:
            terms.append(event_loss)
    for logits_key, target_key in (("evasion_logits", "evasion_target"), ("entry_side_logits", "entry_side_target")):
        targets = batch[target_key]
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"{target_key} must be a tensor")
        available = targets != IGNORE_INDEX
        if bool(available.any()):
            terms.append(functional.cross_entropy(predictions[logits_key][available], targets[available]))
    return torch.stack(terms).mean() if terms else predictions["evasion_logits"].sum() * 0.0


def map_local_peak_to_original_frame(
    scores: torch.Tensor,
    frame_numbers: torch.Tensor,
    *,
    valid_length: int,
) -> int:
    """Find a masked local argmax and return its original video frame number."""

    if scores.ndim != 1 or frame_numbers.ndim != 1 or scores.shape != frame_numbers.shape:
        raise ValueError("scores and frame_numbers must be aligned one-dimensional tensors")
    if not 1 <= valid_length <= scores.numel():
        raise ValueError("valid_length must be in [1, number of scores]")
    valid_scores = scores[:valid_length]
    if not bool(torch.isfinite(valid_scores).all()):
        raise ValueError("valid event scores must be finite")
    return int(frame_numbers[int(valid_scores.argmax().item())].item())


def aggregate_overlapping_window_scores(
    scores: torch.Tensor,
    frame_numbers: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    policy: Literal["mean", "max"] = "mean",
) -> dict[int, float]:
    """Aggregate matching original-frame probabilities from overlapping chunks."""

    if scores.ndim != 2 or frame_numbers.shape != scores.shape:
        raise ValueError("scores and frame_numbers must have matching [window, time] shape")
    if valid_lengths.ndim != 1 or valid_lengths.numel() != scores.shape[0]:
        raise ValueError("valid_lengths must have one entry per window")
    if policy not in {"mean", "max"}:
        raise ValueError("policy must be 'mean' or 'max'")
    grouped: dict[int, list[float]] = {}
    for window_index, length in enumerate(valid_lengths.detach().cpu().tolist()):
        if not 1 <= int(length) <= scores.shape[1]:
            raise ValueError("valid_lengths must be in [1, time]")
        for local_index in range(int(length)):
            value = scores[window_index, local_index]
            if not bool(torch.isfinite(value)):
                raise ValueError("valid event scores must be finite")
            original_frame = int(frame_numbers[window_index, local_index].item())
            grouped.setdefault(original_frame, []).append(float(value.item()))
    if policy == "mean":
        return {frame: sum(values) / len(values) for frame, values in grouped.items()}
    return {frame: max(values) for frame, values in grouped.items()}


def select_aggregated_event_frame(
    scores: torch.Tensor,
    frame_numbers: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    policy: Literal["mean", "max"] = "mean",
) -> int:
    """Select one original frame after overlap-aware aggregation."""

    aggregated = aggregate_overlapping_window_scores(scores, frame_numbers, valid_lengths, policy=policy)
    if not aggregated:
        raise ValueError("cannot select an event frame from no valid window scores")
    return max(aggregated, key=aggregated.__getitem__)


def train_stage2_window_epoch(
    model: Stage2TwoStreamBiLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    target_config: TargetMappingConfig = TargetMappingConfig(),
) -> float:
    """Train one RGB+Flow epoch without retaining whole videos in memory."""

    model.train()
    losses: list[float] = []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        flow = batch["flow"].to(device, non_blocking=True)
        valid_lengths = batch["valid_length"].to(device, non_blocking=True)
        tensor_batch = {
            key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        loss = stage2_window_loss(
            model(frames, flow, valid_lengths), tensor_batch, target_config=target_config
        )
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
    """Save an explicitly non-submission RGB+Flow experimental checkpoint."""

    if epochs < 0:
        raise ValueError("epochs must be >= 0")
    seed_everything(seed)
    dataset = Stage2SlidingWindowDataset(
        read_stage2_records(data_dir),
        window_frames=config.window_frames,
        stride=config.stride,
        size=config.size,
        include_flow=True,
        farneback_config=config.farneback,
        flow_cache_dir=config.flow_cache_dir,
    )
    device = choose_device()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_stage2_windows,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = Stage2TwoStreamBiLSTM(frame_batch_size=config.frame_batch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = cosine_scheduler(
        optimizer,
        epochs=epochs,
        minimum_learning_rate=config.training_control.min_learning_rate,
    )
    logger = JsonlTrainingLogger("stage2_two_stream", config.training_control.log_dir)
    early_stopping = EarlyStopping(
        mode="min",
        patience=config.training_control.early_stopping_patience,
        min_delta=config.training_control.early_stopping_min_delta,
    )
    for epoch in range(epochs):
        hits_before, misses_before = dataset.flow_cache_hits, dataset.flow_cache_misses
        average_loss = train_stage2_window_epoch(
            model, loader, optimizer, device=device, target_config=config.target_mapping
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        logger.log(
            epoch=epoch + 1,
            train_loss=average_loss,
            learning_rate=learning_rate,
            valid_metric=None,
            monitor_name="train_loss_proxy_no_validation",
            monitor_value=average_loss,
        )
        print(
            f"[Stage2][epoch {epoch + 1}/{epochs}] loss={average_loss:.6f} "
            f"lr={learning_rate:.3e} valid_metric=unavailable "
            f"flow_cache(hit={dataset.flow_cache_hits - hits_before}, "
            f"miss={dataset.flow_cache_misses - misses_before})"
        )
        scheduler.step()
        if early_stopping.step(average_loss):
            print(f"[Stage2] early stopping at epoch {epoch + 1}")
            break
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "stage2_two_stream_experimental.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "format": "experimental_stage2_two_stream_farneback_bilstm",
            "config": asdict(config),
            "farneback": asdict(config.farneback),
            "target_mapping": asdict(config.target_mapping),
            "official_output": ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
        },
        checkpoint,
    )
    del dataset, loader, model, scheduler, optimizer
    release_device_cache(device)
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--flow-cache-dir", type=Path, default=DEFAULT_OPTICAL_FLOW_CACHE_DIR)
    parser.add_argument("--target-mode", choices=("binary_mask", "gaussian"), default="gaussian")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--aggregation-policy", choices=("mean", "max"), default="mean")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    checkpoint = fit_stage2_window_skeleton(
        args.data_dir,
        args.model_dir,
        epochs=args.epochs,
        config=Stage2WindowTrainingConfig(
            window_frames=args.window_frames,
            stride=args.stride,
            size=args.size,
            batch_size=args.batch_size,
            frame_batch_size=args.frame_batch_size,
            num_workers=args.num_workers,
            flow_cache_dir=str(args.flow_cache_dir),
            target_mapping=TargetMappingConfig(
                mode=args.target_mode,
                gaussian_sigma=args.gaussian_sigma,
                aggregation_policy=args.aggregation_policy,
            ),
            training_control=TrainingControlConfig(
                min_learning_rate=args.min_learning_rate,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                log_dir=args.log_dir,
            ),
        ),
    )
    print(f"[OK] experimental checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
