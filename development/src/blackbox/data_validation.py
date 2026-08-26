"""Validation for a copied public-example dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


class DataValidationError(ValueError):
    """Raised when example data does not match the documented layout."""


LABEL_COLUMNS = {
    "stage1": ["ID", "path", "label"],
    "stage2": ["ID", "path", "t_collision", "t_entry", "evasion_space", "entry_side"],
    "stage3": ["ID", "sample_index", "frame_index", "time_seconds", "accel_label", "steer_label"],
}

STAGE1_LABELS = {"ORIGINAL", "RERECORDED"}
ACCEL_LABELS = {"ACCELERATING", "DECELERATING", "CONSTANT", "STOPPED"}
STEER_LABELS = {"LEFT", "STRAIGHT", "RIGHT"}


def _read_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataValidationError(f"missing label file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise DataValidationError(
                f"invalid columns in {path}: expected={expected_columns}, actual={reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise DataValidationError(f"empty label file: {path}")
    return rows


def _require_integer(value: str, *, name: str, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be an integer: {value!r}") from exc
    if minimum is not None and parsed < minimum:
        raise DataValidationError(f"{name} must be >= {minimum}: {parsed}")
    return parsed


def _require_video(path: Path, *, decode: bool) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise DataValidationError(f"missing or empty video: {path}")
    if not decode:
        return
    import cv2

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise DataValidationError(f"cannot decode first frame: {path}")


def validate_public_example(root: str | Path, *, decode: bool = False) -> dict[str, int]:
    """Validate the documented Stage 1/2/3 public-example layout."""

    data_root = Path(root).resolve()
    if not data_root.is_dir():
        raise DataValidationError(f"data root is not a directory: {data_root}")

    stage1 = data_root / "stage1"
    stage1_rows = _read_rows(stage1 / "labels.csv", LABEL_COLUMNS["stage1"])
    stage1_ids: set[str] = set()
    for row in stage1_rows:
        if not row["ID"] or row["ID"] in stage1_ids:
            raise DataValidationError(f"duplicate or empty Stage 1 ID: {row['ID']!r}")
        stage1_ids.add(row["ID"])
        if row["label"] not in STAGE1_LABELS:
            raise DataValidationError(f"invalid Stage 1 label: {row['label']!r}")
        _require_video(stage1 / row["path"], decode=decode)

    stage2 = data_root / "stage2"
    stage2_rows = _read_rows(stage2 / "labels.csv", LABEL_COLUMNS["stage2"])
    stage2_ids: set[str] = set()
    for row in stage2_rows:
        if not row["ID"] or row["ID"] in stage2_ids:
            raise DataValidationError(f"duplicate or empty Stage 2 ID: {row['ID']!r}")
        stage2_ids.add(row["ID"])
        _require_integer(row["t_collision"], name="t_collision", minimum=0)
        for column in ("t_entry", "evasion_space", "entry_side"):
            _require_integer(row[column], name=column, minimum=-1)
        _require_video(stage2 / row["path"], decode=decode)

    stage3 = data_root / "stage3"
    stage3_rows = _read_rows(stage3 / "labels.csv", LABEL_COLUMNS["stage3"])
    stage3_pairs: set[tuple[str, int]] = set()
    stage3_videos: set[Path] = set()
    for row in stage3_rows:
        sample_index = _require_integer(row["sample_index"], name="sample_index", minimum=0)
        _require_integer(row["frame_index"], name="frame_index", minimum=0)
        try:
            if float(row["time_seconds"]) < 0:
                raise ValueError
        except ValueError as exc:
            raise DataValidationError(
                f"time_seconds must be a non-negative number: {row['time_seconds']!r}"
            ) from exc
        if row["accel_label"] not in ACCEL_LABELS:
            raise DataValidationError(f"invalid accel_label: {row['accel_label']!r}")
        if row["steer_label"] not in STEER_LABELS:
            raise DataValidationError(f"invalid steer_label: {row['steer_label']!r}")
        key = (row["ID"], sample_index)
        if not row["ID"] or key in stage3_pairs:
            raise DataValidationError(f"duplicate or empty Stage 3 key: {key!r}")
        stage3_pairs.add(key)
        stage3_videos.add(stage3 / "videos" / f"{row['ID']}.mp4")
    for video in sorted(stage3_videos):
        _require_video(video, decode=decode)

    return {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(stage2_rows),
        "stage3_rows": len(stage3_rows),
        "stage3_videos": len(stage3_videos),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate copied public-example data.")
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--decode", action="store_true", help="Decode the first frame of every video.")
    args = parser.parse_args()
    print(json.dumps(validate_public_example(args.data_root, decode=args.decode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
