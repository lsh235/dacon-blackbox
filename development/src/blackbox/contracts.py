"""Prediction DataFrame contracts derived from the supplied competition docs."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class ContractError(ValueError):
    """Raised when a Stage prediction violates the submission contract."""


STAGE_COLUMNS = {
    "stage1": ["ID", "answer"],
    "stage2": ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"],
    "stage3": ["ID", "sample_index", "accel_label", "steer_label"],
}

ALLOWED_VALUES = {
    "stage1": {"answer": {"ORIGINAL", "RERECORDED"}},
    "stage2": {"evasion_space": {0, 1}, "entry_side": {"LEFT", "RIGHT"}},
    "stage3": {
        "accel_label": {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"},
        "steer_label": {"LEFT", "STRAIGHT", "RIGHT"},
    },
}


def normalize_stage(stage: str | int) -> str:
    """Return a canonical stage name such as ``stage1``."""

    value = str(stage).strip().lower().replace("_", "").replace("-", "")
    if value in {"1", "stage1"}:
        return "stage1"
    if value in {"2", "stage2"}:
        return "stage2"
    if value in {"3", "stage3"}:
        return "stage3"
    raise ContractError(f"unknown stage: {stage!r}")


def _invalid_values(series: pd.Series, allowed: Iterable[object]) -> list[object]:
    allowed_set = set(allowed)
    return sorted(set(series.dropna().tolist()) - allowed_set, key=repr)


def _validate_nonnegative_integers(frame: pd.DataFrame, columns: Iterable[str], errors: list[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values) | (values < 0) | (values % 1 != 0)
        if bool(invalid.any()):
            rows = frame.index[invalid].tolist()[:5]
            errors.append(f"{column} must contain non-negative integers; invalid rows={rows}")


def validate_prediction_frame(
    stage: str | int,
    frame: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Validate a Stage prediction and return it unchanged on success."""

    stage_name = normalize_stage(stage)
    if not isinstance(frame, pd.DataFrame):
        raise ContractError(f"prediction must be a pandas.DataFrame, got {type(frame).__name__}")

    expected_columns = STAGE_COLUMNS[stage_name]
    actual_columns = frame.columns.tolist()
    if actual_columns != expected_columns:
        raise ContractError(
            f"{stage_name} columns must be {expected_columns}, got {actual_columns}"
        )
    if frame.empty and not allow_empty:
        raise ContractError(f"{stage_name} prediction must not be empty")

    errors: list[str] = []
    if frame["ID"].isna().any() or not frame["ID"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        errors.append("ID must contain non-empty strings")

    if stage_name in {"stage1", "stage2"} and frame["ID"].duplicated().any():
        errors.append("ID must be unique for each video")

    for column, allowed in ALLOWED_VALUES[stage_name].items():
        invalid = _invalid_values(frame[column], allowed)
        if frame[column].isna().any() or invalid:
            errors.append(f"{column} contains values outside {sorted(allowed, key=repr)}: {invalid}")

    if stage_name == "stage2":
        _validate_nonnegative_integers(frame, ("collision_frame", "entry_frame"), errors)
    elif stage_name == "stage3":
        _validate_nonnegative_integers(frame, ("sample_index",), errors)
        if frame[["ID", "sample_index"]].duplicated().any():
            errors.append("(ID, sample_index) must be unique")
        for video_id, group in frame.groupby("ID", sort=False):
            numeric = pd.to_numeric(group["sample_index"], errors="coerce")
            if numeric.isna().any():
                continue
            actual = sorted(numeric.astype(int).tolist())
            expected = list(range(len(group)))
            if actual != expected:
                errors.append(
                    f"sample_index for ID={video_id!r} must be contiguous from 0; "
                    f"expected={expected[:5]} actual={actual[:5]}"
                )

    if errors:
        raise ContractError("; ".join(errors))
    return frame
