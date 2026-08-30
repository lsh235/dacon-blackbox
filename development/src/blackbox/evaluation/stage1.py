"""Leakage-auditable local metrics for Stage 1 classification experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


STAGE1_LABELS = ("ORIGINAL", "RERECORDED")


class Stage1EvaluationError(ValueError):
    """Raised when local Stage 1 labels or predictions are incomplete."""


def _validated_labels(values: Sequence[str], *, name: str) -> list[str]:
    labels = [str(value) for value in values]
    unknown = sorted(set(labels) - set(STAGE1_LABELS))
    if unknown:
        raise Stage1EvaluationError(
            f"{name} has unsupported values {unknown}; expected {list(STAGE1_LABELS)}"
        )
    return labels


def evaluate_stage1_classification(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, object]:
    """Calculate Macro F1, accuracy, confusion matrix, and class diagnostics.

    Macro F1 gives equal weight to ORIGINAL and RERECORDED, so a majority-class
    prediction cannot appear strong merely because the real training set is
    imbalanced. Zero-denominator precision/recall/F1 values are reported as 0.
    """

    actual = _validated_labels(y_true, name="y_true")
    predicted = _validated_labels(y_pred, name="y_pred")
    if not actual:
        raise Stage1EvaluationError("y_true and y_pred must not be empty")
    if len(actual) != len(predicted):
        raise Stage1EvaluationError(
            f"y_true and y_pred length mismatch: {len(actual)} != {len(predicted)}"
        )

    label_to_index = {label: index for index, label in enumerate(STAGE1_LABELS)}
    matrix = np.zeros((len(STAGE1_LABELS), len(STAGE1_LABELS)), dtype=np.int64)
    for target, prediction in zip(actual, predicted):
        matrix[label_to_index[target], label_to_index[prediction]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, label in enumerate(STAGE1_LABELS):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    return {
        "metric": "macro_f1",
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(np.trace(matrix) / len(actual)),
        "labels": list(STAGE1_LABELS),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "samples": len(actual),
    }


def format_stage1_evaluation_report(metrics: dict[str, object], *, title: str) -> str:
    """Format metric data as a human-readable Markdown report."""

    labels = list(metrics["labels"])
    matrix = metrics["confusion_matrix"]
    per_class = metrics["per_class"]
    lines = [
        f"# {title}",
        "",
        f"- Macro F1: {float(metrics['macro_f1']):.6f}",
        f"- Accuracy: {float(metrics['accuracy']):.6f}",
        f"- Samples: {int(metrics['samples'])}",
        "",
        "## Confusion matrix (row=true, column=predicted)",
        "",
        f"| true \\ predicted | {labels[0]} | {labels[1]} |",
        "| --- | ---: | ---: |",
    ]
    for label, row in zip(labels, matrix):
        lines.append(f"| {label} | {int(row[0])} | {int(row[1])} |")
    lines.extend(
        [
            "",
            "## Per-class precision / recall / F1",
            "",
            "| class | precision | recall | F1 | support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in labels:
        scores = per_class[label]
        lines.append(
            "| {label} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {support} |".format(
                label=label,
                precision=float(scores["precision"]),
                recall=float(scores["recall"]),
                f1=float(scores["f1"]),
                support=int(scores["support"]),
            )
        )
    return "\n".join(lines) + "\n"


def save_stage1_evaluation(
    output_dir: str | Path,
    metrics: dict[str, object],
    *,
    title: str,
) -> None:
    """Persist JSON plus a Markdown report for a reproducible local run."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "metrics.md").write_text(
        format_stage1_evaluation_report(metrics, title=title),
        encoding="utf-8",
    )


def validated_training_history(
    history: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate and order Stage 1 epoch history embedded in a checkpoint."""

    records = [dict(record) for record in history]
    if not records:
        return []
    required = {"epoch", "train_loss"}
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise Stage1EvaluationError(
                f"training history row {index} is missing fields: {missing}"
            )
        record["epoch"] = int(record["epoch"])
        record["train_loss"] = float(record["train_loss"])
        for name in (
            "valid_loss",
            "valid_macro_f1",
            "valid_prediction_change_rate",
            "valid_probability_mean_abs_delta",
            "learning_rate",
        ):
            if record.get(name) is not None:
                record[name] = float(record[name])
    records.sort(key=lambda record: int(record["epoch"]))
    epochs = [int(record["epoch"]) for record in records]
    if len(set(epochs)) != len(epochs) or any(epoch < 1 for epoch in epochs):
        raise Stage1EvaluationError("training history epochs must be unique positive integers")
    return records


def summarize_training_convergence(
    history: Sequence[Mapping[str, object]],
    *,
    fixed_point_window: int = 3,
    loss_relative_tolerance: float = 0.02,
    probability_delta_tolerance: float = 0.01,
) -> dict[str, object]:
    """Assess loss stabilization and epoch-to-epoch prediction fixed points."""

    if fixed_point_window < 2:
        raise ValueError("fixed_point_window must be >= 2")
    if loss_relative_tolerance < 0.0 or probability_delta_tolerance < 0.0:
        raise ValueError("convergence tolerances must be >= 0")
    records = validated_training_history(history)
    if not records:
        return {
            "status": "unavailable",
            "reason": "checkpoint does not contain epoch history",
            "epochs": 0,
            "loss_converged": None,
            "prediction_fixed_point": None,
            "stable": None,
        }

    tail = records[-fixed_point_window:]
    loss_name = (
        "valid_loss"
        if all(record.get("valid_loss") is not None for record in tail)
        else "train_loss"
    )
    losses = np.asarray([float(record[loss_name]) for record in tail], dtype=np.float64)
    loss_relative_range = float(
        (losses.max() - losses.min()) / max(abs(float(losses.mean())), 1e-12)
    )
    enough_loss_epochs = len(records) >= fixed_point_window
    loss_converged = enough_loss_epochs and loss_relative_range <= loss_relative_tolerance

    comparable = [
        record
        for record in records
        if record.get("valid_prediction_change_rate") is not None
        and record.get("valid_probability_mean_abs_delta") is not None
    ]
    comparison_tail = comparable[-max(1, fixed_point_window - 1):]
    enough_prediction_epochs = len(comparable) >= fixed_point_window - 1
    max_change_rate = (
        max(float(record["valid_prediction_change_rate"]) for record in comparison_tail)
        if comparison_tail
        else None
    )
    max_probability_delta = (
        max(float(record["valid_probability_mean_abs_delta"]) for record in comparison_tail)
        if comparison_tail
        else None
    )
    prediction_fixed_point = (
        enough_prediction_epochs
        and max_change_rate == 0.0
        and max_probability_delta is not None
        and max_probability_delta <= probability_delta_tolerance
    )
    status = "measured" if enough_loss_epochs and enough_prediction_epochs else "insufficient_epochs"
    return {
        "status": status,
        "epochs": len(records),
        "window": fixed_point_window,
        "loss_source": loss_name,
        "loss_relative_range": loss_relative_range,
        "loss_relative_tolerance": loss_relative_tolerance,
        "loss_converged": loss_converged if enough_loss_epochs else None,
        "max_prediction_change_rate": max_change_rate,
        "max_probability_mean_abs_delta": max_probability_delta,
        "probability_delta_tolerance": probability_delta_tolerance,
        "prediction_fixed_point": (
            prediction_fixed_point if enough_prediction_epochs else None
        ),
        "stable": (
            bool(loss_converged and prediction_fixed_point)
            if status == "measured"
            else None
        ),
    }


def save_loss_curve_svg(
    path: str | Path,
    history: Sequence[Mapping[str, object]],
) -> None:
    """Write a dependency-free SVG of train and validation loss curves."""

    records = validated_training_history(history)
    if not records:
        raise Stage1EvaluationError("cannot plot an empty training history")
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 45, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    epochs = np.asarray([int(record["epoch"]) for record in records], dtype=np.float64)
    train = np.asarray([float(record["train_loss"]) for record in records], dtype=np.float64)
    valid_values = [record.get("valid_loss") for record in records]
    valid = np.asarray(
        [np.nan if value is None else float(value) for value in valid_values],
        dtype=np.float64,
    )
    finite_losses = np.concatenate((train[np.isfinite(train)], valid[np.isfinite(valid)]))
    if finite_losses.size == 0:
        raise Stage1EvaluationError("training history does not contain finite losses")
    y_min = float(finite_losses.min())
    y_max = float(finite_losses.max())
    if y_max <= y_min:
        padding = max(abs(y_min) * 0.05, 0.05)
        y_min -= padding
        y_max += padding

    def x_coordinate(epoch: float) -> float:
        span = max(float(epochs.max() - epochs.min()), 1.0)
        return left + (epoch - float(epochs.min())) / span * plot_width

    def y_coordinate(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * plot_height

    def polyline(values: np.ndarray, color: str) -> str:
        finite_points = [
            (x_coordinate(epoch), y_coordinate(float(value)))
            for epoch, value in zip(epochs, values)
            if np.isfinite(value)
        ]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in finite_points)
        markers = "".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" />'
            for x, y in finite_points
        )
        return (
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'stroke-linejoin="round" points="{points}" />{markers}'
        )

    grid: list[str] = []
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        grid.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dddddd" />',
                f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-size="13">{value:.4f}</text>',
            ]
        )
    epoch_labels = "".join(
        f'<text x="{x_coordinate(epoch):.2f}" y="{height-bottom+28}" text-anchor="middle" font-size="13">{int(epoch)}</text>'
        for epoch in epochs
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width/2}" y="27" text-anchor="middle" font-size="20" font-family="sans-serif">Stage 1 loss convergence</text>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222222" />
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222222" />
{polyline(train, '#2563eb')}
{polyline(valid, '#dc2626')}
{epoch_labels}
<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14">Epoch</text>
<text x="18" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 18 {height/2})">Loss</text>
<line x1="{width-245}" y1="52" x2="{width-215}" y2="52" stroke="#2563eb" stroke-width="3" /><text x="{width-207}" y="57" font-size="13">Train loss</text>
<line x1="{width-125}" y1="52" x2="{width-95}" y2="52" stroke="#dc2626" stroke-width="3" /><text x="{width-87}" y="57" font-size="13">Validation loss</text>
</svg>
'''
    Path(path).write_text(svg, encoding="utf-8")


def probe_video_duration(path: str | Path) -> dict[str, float | int]:
    """Read frame count/FPS without decoding a full video."""

    video = Path(path)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise Stage1EvaluationError(f"cannot open video metadata: {video}")
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count < 1 or not np.isfinite(fps) or fps <= 0.0:
        raise Stage1EvaluationError(
            f"invalid video duration metadata: {video} frames={frame_count} fps={fps}"
        )
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps,
    }


def assign_duration_groups(durations: Sequence[float]) -> list[str]:
    """Split videos at duration tertiles without separating equal durations."""

    values = np.asarray(list(durations), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise Stage1EvaluationError("durations must be non-empty, finite, and positive")
    lower, upper = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if np.isclose(lower, upper):
        return ["medium"] * values.size
    return [
        "short" if value <= lower else "medium" if value <= upper else "long"
        for value in values
    ]


def evaluate_duration_groups(
    rows: pd.DataFrame,
    *,
    target_column: str = "label",
    prediction_column: str = "answer",
    duration_column: str = "duration_seconds",
) -> dict[str, object]:
    """Compare classification quality across duration-ranked thirds."""

    required = {target_column, prediction_column, duration_column, "duration_group"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise Stage1EvaluationError(f"duration evaluation is missing columns: {missing}")
    output: dict[str, object] = {}
    for label in ("short", "medium", "long"):
        selected = rows.loc[rows["duration_group"] == label]
        if selected.empty:
            output[label] = {"samples": 0, "macro_f1": None}
            continue
        metrics = evaluate_stage1_classification(
            selected[target_column].tolist(),
            selected[prediction_column].tolist(),
        )
        output[label] = {
            "duration_min_seconds": float(selected[duration_column].min()),
            "duration_max_seconds": float(selected[duration_column].max()),
            **metrics,
        }
    return output


def temporal_probability_diagnostics(
    probabilities: Sequence[float],
    *,
    jump_threshold: float = 0.25,
) -> dict[str, float | int]:
    """Measure ordered clip-probability smoothness and threshold switches."""

    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        raise Stage1EvaluationError("temporal diagnostics require at least two finite values")
    if (values < 0.0).any() or (values > 1.0).any():
        raise Stage1EvaluationError("probabilities must be in [0, 1]")
    if not 0.0 <= jump_threshold <= 1.0:
        raise ValueError("jump_threshold must be in [0, 1]")
    steps = np.abs(np.diff(values))
    switches = np.diff(values >= 0.5) != 0
    return {
        "probability_mean": float(values.mean()),
        "probability_std": float(values.std()),
        "mean_absolute_step": float(steps.mean()),
        "max_absolute_step": float(steps.max()),
        "total_variation": float(steps.sum()),
        "smoothness_score": float(np.clip(1.0 - steps.mean(), 0.0, 1.0)),
        "label_switches": int(switches.sum()),
        "large_jump_events": int((steps >= jump_threshold).sum()),
        "transitions": int(steps.size),
    }


def summarize_fold_generalization(
    fold_metrics: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Report GroupKFold Macro F1 dispersion on held-out domains."""

    if not fold_metrics:
        raise Stage1EvaluationError("fold metrics must not be empty")
    scores = np.asarray(
        [float(metrics["macro_f1"]) for _, metrics in sorted(fold_metrics.items())],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise Stage1EvaluationError("fold Macro F1 values must be finite")
    return {
        "folds": int(scores.size),
        "fold_macro_f1": scores.tolist(),
        "macro_f1_mean": float(scores.mean()),
        "macro_f1_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
        "macro_f1_min": float(scores.min()),
        "macro_f1_max": float(scores.max()),
        "macro_f1_range": float(scores.max() - scores.min()),
    }
