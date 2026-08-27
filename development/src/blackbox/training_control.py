"""Small, dependency-free controls shared by every local training entrypoint.

When two or more source groups are available, callers use a deterministic
group holdout and log a real validation metric.  A one-group experiment logs
``valid_metric=None`` and explicitly falls back to training loss for stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Hashable, Iterable, Literal

import torch


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


@dataclass(frozen=True)
class TrainingControlConfig:
    """Scheduler, stopping, AMP, and local JSONL logging settings."""

    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    validation_fraction: float = 0.2
    log_dir: str | Path | None = None
    use_amp: bool = False

    def __post_init__(self) -> None:
        if self.min_learning_rate < 0.0:
            raise ValueError("min_learning_rate must be >= 0")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not isinstance(self.use_amp, bool):
            raise TypeError("use_amp must be a boolean")


class JsonlTrainingLogger:
    """Append one durable, machine-readable record per completed epoch."""

    def __init__(self, stage: str, log_dir: str | Path | None = None) -> None:
        directory = DEFAULT_LOG_DIR if log_dir is None else Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{stage}.jsonl"
        self.stage = stage

    def log(
        self,
        *,
        epoch: int,
        train_loss: float,
        learning_rate: float,
        valid_metric: float | None,
        monitor_name: str,
        monitor_value: float,
    ) -> None:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "stage": self.stage,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "valid_metric": None if valid_metric is None else float(valid_metric),
            "learning_rate": float(learning_rate),
            "monitor_name": monitor_name,
            "monitor_value": float(monitor_value),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class EarlyStopping:
    """Stop after a monitored scalar has not improved for ``patience`` epochs."""

    def __init__(self, *, mode: Literal["min", "max"], patience: int, min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0.0:
            raise ValueError("min_delta must be >= 0")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Create a finite-horizon cosine schedule that also supports one epoch."""

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=minimum_learning_rate,
    )


def group_holdout_indices(groups: Iterable[Hashable], *, validation_fraction: float) -> tuple[set[int], set[int]]:
    """Split examples by group so paired/source-linked videos stay together."""

    ordered_groups = list(groups)
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    unique_groups = sorted(set(ordered_groups), key=str)
    if validation_fraction == 0.0 or len(unique_groups) < 2:
        return set(range(len(ordered_groups))), set()
    validation_count = min(
        len(unique_groups) - 1,
        max(1, round(len(unique_groups) * validation_fraction)),
    )
    validation_groups = set(unique_groups[-validation_count:])
    validation = {index for index, group in enumerate(ordered_groups) if group in validation_groups}
    training = set(range(len(ordered_groups))) - validation
    return training, validation


def macro_f1_score(targets: Iterable[int], predictions: Iterable[int], *, labels: Iterable[int]) -> float:
    """Dependency-free macro F1 for small held-out classification sets."""

    expected = list(targets)
    actual = list(predictions)
    if len(expected) != len(actual):
        raise ValueError("targets and predictions must have equal length")
    if not expected:
        raise ValueError("macro F1 requires at least one target")
    scores: list[float] = []
    for label in labels:
        true_positive = sum(1 for target, prediction in zip(expected, actual) if target == label and prediction == label)
        false_positive = sum(1 for target, prediction in zip(expected, actual) if target != label and prediction == label)
        false_negative = sum(1 for target, prediction in zip(expected, actual) if target == label and prediction != label)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0
