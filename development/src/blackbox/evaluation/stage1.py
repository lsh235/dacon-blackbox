"""Leakage-auditable local metrics for Stage 1 classification experiments."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

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
