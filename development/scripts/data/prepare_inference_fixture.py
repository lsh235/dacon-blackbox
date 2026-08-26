#!/usr/bin/env python3
"""Convert copied public training examples into documented evaluation input folders."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import cv2

from blackbox.data_validation import validate_public_example


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_frames(video: Path, destination: Path, *, limit: int | None = None) -> int:
    destination.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output = destination / f"{index:06d}.jpg"
        if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"failed to write frame: {output}")
        index += 1
        if limit is not None and index >= limit:
            break
    capture.release()
    if index == 0:
        raise RuntimeError(f"cannot decode Stage 2 video: {video}")
    return index


def _copy_or_trim_video(source: Path, destination: Path, *, limit: int | None) -> None:
    if limit is None:
        shutil.copy2(source, destination)
        return
    capture = cv2.VideoCapture(str(source))
    ok, first = capture.read()
    if not ok or first is None:
        capture.release()
        raise RuntimeError(f"cannot decode Stage 3 video: {source}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create trimmed Stage 3 video: {destination}")
    written = 0
    frame = first
    while written < limit:
        writer.write(frame)
        written += 1
        ok, frame = capture.read()
        if not ok or frame is None:
            break
    capture.release()
    writer.release()
    if written == 0:
        raise RuntimeError(f"no frames written for Stage 3 video: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--video-limit", type=int, default=None)
    parser.add_argument("--stage2-frame-limit", type=int, default=None)
    parser.add_argument("--stage3-frame-limit", type=int, default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    for name in ("video_limit", "stage2_frame_limit", "stage3_frame_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    validate_public_example(source, decode=True)
    if destination.exists():
        raise SystemExit(f"destination already exists; refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"source": str(source)}
    with tempfile.TemporaryDirectory(
        prefix="blackbox-eval-fixture-", dir=destination.parent
    ) as temporary:
        temporary_root = Path(temporary) / "fixture"

        stage1_output = temporary_root / "stage1/videos"
        stage1_output.mkdir(parents=True)
        stage1_rows = _rows(source / "stage1/labels.csv")[: args.video_limit]
        for row in stage1_rows:
            original = source / "stage1" / row["path"]
            shutil.copy2(original, stage1_output / f"{row['ID']}{original.suffix.lower()}")
        summary["stage1_videos"] = len(stage1_rows)

        stage2_output = temporary_root / "stage2/images"
        stage2_output.mkdir(parents=True)
        stage2_frames = 0
        stage2_rows = _rows(source / "stage2/labels.csv")[: args.video_limit]
        for row in stage2_rows:
            stage2_frames += _extract_frames(
                source / "stage2" / row["path"],
                stage2_output / row["ID"],
                limit=args.stage2_frame_limit,
            )
        summary["stage2_videos"] = len(stage2_rows)
        summary["stage2_frames"] = stage2_frames

        stage3_output = temporary_root / "stage3/videos"
        stage3_output.mkdir(parents=True)
        stage3_ids = sorted({row["ID"] for row in _rows(source / "stage3/labels.csv")})[
            : args.video_limit
        ]
        for video_id in stage3_ids:
            _copy_or_trim_video(
                source / "stage3/videos" / f"{video_id}.mp4",
                stage3_output / f"{video_id}.mp4",
                limit=args.stage3_frame_limit,
            )
        summary["stage3_videos"] = len(stage3_ids)

        (temporary_root / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(temporary_root), destination)

    print(json.dumps({"destination": str(destination), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
