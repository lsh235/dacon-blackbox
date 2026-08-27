"""Stage 3 baseline and 10 Hz Two-Stream training entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from blackbox.common.runtime import DEFAULT_SEED, choose_device, release_device_cache, seed_everything
from blackbox.stages.stage3.baseline import fit_stage3
from blackbox.stages.stage3.dataset_stage3 import (
    IGNORE_INDEX,
    FarnebackConfig,
    Stage3SequenceWindowDataset,
    collate_stage3_windows,
    read_stage3_records,
)
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT
from blackbox.experiment_config import config_path_value, load_experiment_config, section, stage_paths
from blackbox.stages.stage3.model_stage3 import Stage3TwoStreamBiLSTM
from blackbox.training_control import (
    EarlyStopping,
    JsonlTrainingLogger,
    TrainingControlConfig,
    cosine_scheduler,
    group_holdout_indices,
)


@dataclass(frozen=True)
class Stage3TwoStreamTrainingConfig:
    """Memory-bounded 10 Hz RGB + dense-flow sequence training settings."""

    window_frames: int = 96
    stride: int = 48
    size: int = 224
    batch_size: int = 1
    frame_batch_size: int = 8
    learning_rate: float = 2e-4
    num_workers: int = 0
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT
    farneback: FarnebackConfig = FarnebackConfig()
    training_control: TrainingControlConfig = TrainingControlConfig()

    def __post_init__(self) -> None:
        if min(self.window_frames, self.stride, self.size, self.batch_size, self.frame_batch_size) < 1:
            raise ValueError("window, stride, size, batch size, and frame batch size must be >= 1")
        if self.learning_rate <= 0.0 or self.num_workers < 0:
            raise ValueError("learning_rate must be > 0 and num_workers must be >= 0")


def stage3_sequence_loss(predictions: dict[str, torch.Tensor], batch: dict[str, object]) -> torch.Tensor | None:
    """Cross entropy only at sparse 0.1-second positions that have labels."""

    terms: list[torch.Tensor] = []
    for logits_key, targets_key in (("accel_logits", "accel_targets"), ("steer_logits", "steer_targets")):
        targets = batch[targets_key]
        logits = predictions[logits_key]
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f"{targets_key} must be a tensor")
        available = targets != IGNORE_INDEX
        if bool(available.any()):
            terms.append(functional.cross_entropy(logits[available], targets[available]))
    return torch.stack(terms).mean() if terms else None


def _run_epoch(
    model: Stage3TwoStreamBiLSTM,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float | None:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            predictions = model(
                tensor_batch["frames"],
                tensor_batch["flow"],
                tensor_batch["valid_length"],
            )
            loss = stage3_sequence_loss(predictions, tensor_batch)
            if loss is None:
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else None


def fit_stage3_two_stream_skeleton(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = 1,
    seed: int = DEFAULT_SEED,
    config: Stage3TwoStreamTrainingConfig = Stage3TwoStreamTrainingConfig(),
) -> Path:
    """Train the metadata-derived 10 Hz research model and save its checkpoint."""

    if epochs < 0:
        raise ValueError("epochs must be >= 0")
    seed_everything(seed)
    records = read_stage3_records(data_dir)
    train_indices, valid_indices = group_holdout_indices(
        [record.video_id for record in records],
        validation_fraction=config.training_control.validation_fraction,
    )
    train_records = [record for index, record in enumerate(records) if index in train_indices]
    valid_records = [record for index, record in enumerate(records) if index in valid_indices]
    device = choose_device()
    train_dataset = Stage3SequenceWindowDataset(
        train_records,
        window_frames=config.window_frames,
        stride=config.stride,
        size=config.size,
        farneback_config=config.farneback,
        processed_root=config.processed_root,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_stage3_windows,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_dataset = (
        Stage3SequenceWindowDataset(
            valid_records,
            window_frames=config.window_frames,
            stride=config.stride,
            size=config.size,
            farneback_config=config.farneback,
            processed_root=config.processed_root,
        )
        if valid_records
        else None
    )
    valid_loader = (
        DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_stage3_windows,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        if valid_dataset is not None
        else None
    )
    model = Stage3TwoStreamBiLSTM(frame_batch_size=config.frame_batch_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = cosine_scheduler(
        optimizer,
        epochs=epochs,
        minimum_learning_rate=config.training_control.min_learning_rate,
    )
    logger = JsonlTrainingLogger("stage3_two_stream", config.training_control.log_dir)
    early_stopping = EarlyStopping(
        mode="min",
        patience=config.training_control.early_stopping_patience,
        min_delta=config.training_control.early_stopping_min_delta,
    )
    for epoch in range(epochs):
        average_loss = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        if average_loss is None:
            raise ValueError("Stage 3 Two-Stream training found no sparse targets in its windows")
        valid_metric = _run_epoch(model, valid_loader, device=device, optimizer=None) if valid_loader else None
        learning_rate = float(optimizer.param_groups[0]["lr"])
        monitor_name = "valid_sparse_ce_group_holdout" if valid_metric is not None else "train_loss_proxy_no_validation"
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
            f"[Stage3 Two-Stream][epoch {epoch + 1}/{epochs}] loss={average_loss:.6f} "
            f"lr={learning_rate:.3e} valid_sparse_ce="
            f"{'unavailable' if valid_metric is None else f'{valid_metric:.6f}'}"
        )
        scheduler.step()
        if early_stopping.step(monitor_value):
            print(f"[Stage3 Two-Stream] early stopping at epoch {epoch + 1}")
            break
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "stage3_two_stream_experimental.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "format": "experimental_stage3_two_stream_farneback_bilstm_10hz",
            "config": asdict(config),
            "time_axis": "round(cv2.CAP_PROP_FPS / 10) per video",
        },
        checkpoint,
    )
    del valid_loader, valid_dataset, train_loader, train_dataset, scheduler, optimizer, model
    release_device_cache(device)
    return checkpoint
from blackbox.training_control import TrainingControlConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML experiment configuration")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--architecture", choices=("baseline", "two-stream"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--frame-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--processed-root", type=Path)
    args = parser.parse_args()
    config: dict[str, object] = {}
    config_path: Path | None = None
    if args.config is not None:
        config, config_path = load_experiment_config(args.config)
    stage = section(config, "stage3")
    training = section(config, "training")
    if config_path is not None:
        configured_data, configured_model, configured_processed = stage_paths(config, config_path, "stage3")
    else:
        configured_data = configured_model = configured_processed = None
    data_dir = args.data_dir or configured_data
    model_dir = args.model_dir or configured_model
    processed_root = args.processed_root or configured_processed
    if data_dir is None or model_dir is None or processed_root is None:
        parser.error("--config or --data-dir, --model-dir, and --processed-root are required")
    log_dir = args.log_dir or (
        config_path_value(config_path, training.get("log_dir"), field="training.log_dir")
        if config_path is not None and training.get("log_dir") is not None
        else None
    )
    control = TrainingControlConfig(
        min_learning_rate=args.min_learning_rate if args.min_learning_rate is not None else float(training.get("min_learning_rate", 1e-6)),
        early_stopping_patience=args.early_stopping_patience if args.early_stopping_patience is not None else int(training.get("early_stopping_patience", 5)),
        early_stopping_min_delta=args.early_stopping_min_delta if args.early_stopping_min_delta is not None else float(training.get("early_stopping_min_delta", 0.0)),
        validation_fraction=args.validation_fraction if args.validation_fraction is not None else float(training.get("validation_fraction", 0.2)),
        log_dir=log_dir,
    )
    architecture = args.architecture or str(stage.get("architecture", "baseline"))
    epochs = args.epochs if args.epochs is not None else int(config.get("run", {}).get("epochs", 1))
    if architecture == "baseline":
        checkpoint = fit_stage3(
            data_dir,
            model_dir,
            epochs=epochs,
            training_control=control,
        )
    else:
        checkpoint = fit_stage3_two_stream_skeleton(
            data_dir,
            model_dir,
            epochs=epochs,
            config=Stage3TwoStreamTrainingConfig(
                window_frames=args.window_frames if args.window_frames is not None else int(stage.get("window_frames", 96)),
                stride=args.stride if args.stride is not None else int(stage.get("stride", 48)),
                size=args.size if args.size is not None else int(stage.get("size", 224)),
                batch_size=args.batch_size if args.batch_size is not None else int(stage.get("batch_size", 1)),
                frame_batch_size=args.frame_batch_size if args.frame_batch_size is not None else int(stage.get("frame_batch_size", 8)),
                learning_rate=float(stage.get("learning_rate", 2e-4)),
                num_workers=args.num_workers if args.num_workers is not None else int(stage.get("num_workers", 0)),
                processed_root=processed_root,
                training_control=control,
            ),
        )
    print(f"[OK] checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
