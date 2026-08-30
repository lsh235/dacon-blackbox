"""Leakage-resistant group-aware folds for Stage 1 local experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from blackbox.evaluation.stage1 import STAGE1_LABELS


GROUP_COLUMN_CANDIDATES = (
    "session_id",
    "capture_session_id",
    "device_id",
    "capture_device_id",
    "camera_id",
    "source_content_id",
    "scene_id",
    "original_video_id",
    "group_id",
)


class Stage1SplitError(ValueError):
    """Raised when metadata cannot prove a leakage-safe split."""


@dataclass(frozen=True)
class Stage1FoldPlan:
    """Fold assignment plus the metadata source used to define a group."""

    assignments: pd.DataFrame
    group_source: str


def resolve_stage1_groups(
    metadata: pd.DataFrame,
    *,
    group_column: str | None = None,
) -> tuple[pd.Series, str]:
    """Resolve a user-supplied group column or a conservative path-stem fallback.

    The fallback is intentionally restricted to the video filename. It groups
    the supplied ``original/000001.mp4`` and ``rerecorded/000001.mp4`` pair,
    but production data should pass an explicit source/scene column whenever
    filename stems are not the original-content identifiers.
    """

    if group_column is not None:
        if group_column not in metadata.columns:
            raise Stage1SplitError(
                f"group column {group_column!r} is absent; available={list(metadata.columns)}"
            )
        values = metadata[group_column]
        source = group_column
    else:
        source = next((name for name in GROUP_COLUMN_CANDIDATES if name in metadata.columns), None)
        if source is not None:
            values = metadata[source]
        elif "path" in metadata.columns:
            values = metadata["path"].map(lambda value: Path(str(value)).stem)
            source = "path_stem"
        else:
            raise Stage1SplitError(
                "no group metadata found: pass --group-column mapped to source_content_id "
                "or scene_id; a path column is required for the documented fallback"
            )

    values = values.astype("string")
    if values.isna().any() or (values.str.strip() == "").any():
        raise Stage1SplitError(f"group source {source!r} contains missing or empty values")
    return values.astype(str), source


def make_stratified_group_folds(
    metadata: pd.DataFrame,
    *,
    n_splits: int,
    group_column: str | None = None,
    label_column: str = "label",
    seed: int = 20260825,
) -> Stage1FoldPlan:
    """Assign whole source/scene groups to folds while balancing both labels.

    This is a deterministic greedy approximation of StratifiedGroupKFold. Each
    group is assigned exactly once, so a source group cannot appear in both
    train and validation for any fold. The allocation minimizes deviation from
    the expected class counts and total sample count per fold.
    """

    if n_splits < 2:
        raise Stage1SplitError("n_splits must be >= 2")
    if label_column not in metadata.columns:
        raise Stage1SplitError(f"label column {label_column!r} is absent")
    if metadata.empty:
        raise Stage1SplitError("cannot split an empty metadata frame")

    labels = metadata[label_column].astype(str)
    unknown = sorted(set(labels) - set(STAGE1_LABELS))
    if unknown:
        raise Stage1SplitError(
            f"label column {label_column!r} has unsupported values {unknown}; "
            f"expected {list(STAGE1_LABELS)}"
        )
    groups, group_source = resolve_stage1_groups(metadata, group_column=group_column)
    group_table = pd.crosstab(groups, labels).reindex(columns=STAGE1_LABELS, fill_value=0)
    if len(group_table) < n_splits:
        raise Stage1SplitError(
            f"n_splits={n_splits} requires at least {n_splits} groups, found {len(group_table)}"
        )

    group_counts = group_table.to_numpy(dtype=np.float64)
    group_sizes = group_counts.sum(axis=1)
    total_counts = group_counts.sum(axis=0)
    expected_counts = total_counts / n_splits
    expected_size = float(group_sizes.sum() / n_splits)
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(len(group_table))
    order = sorted(
        range(len(group_table)),
        key=lambda index: (
            -group_sizes[index],
            -float(group_counts[index].max()),
            float(tie_breaker[index]),
        ),
    )

    fold_counts = np.zeros((n_splits, len(STAGE1_LABELS)), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    group_folds: dict[str, int] = {}
    for group_index in order:
        counts = group_counts[group_index]
        size = group_sizes[group_index]
        scores: list[float] = []
        for fold in range(n_splits):
            candidate_counts = fold_counts.copy()
            candidate_sizes = fold_sizes.copy()
            candidate_counts[fold] += counts
            candidate_sizes[fold] += size
            class_deviation = np.square(
                (candidate_counts - expected_counts) / np.maximum(expected_counts, 1.0)
            ).sum()
            size_deviation = np.square(
                (candidate_sizes - expected_size) / max(expected_size, 1.0)
            ).sum()
            scores.append(float(class_deviation + 0.05 * size_deviation))
        best_fold = min(range(n_splits), key=lambda fold: (scores[fold], fold_sizes[fold], fold))
        fold_counts[best_fold] += counts
        fold_sizes[best_fold] += size
        group_folds[str(group_table.index[group_index])] = best_fold

    assignments = metadata.copy()
    assignments["group_value"] = groups.to_numpy()
    assignments["fold"] = assignments["group_value"].map(group_folds).astype(int)
    return Stage1FoldPlan(assignments=assignments, group_source=group_source)
