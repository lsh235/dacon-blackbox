"""Build reproducible video inventories with hashes and container metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from blackbox.common.runtime import VIDEO_EXTENSIONS


@dataclass(frozen=True)
class VideoRecord:
    relative_path: str
    bytes: int
    sha256: str
    decodable: bool
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float | None


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path, root: Path) -> VideoRecord:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    ok, frame = capture.read() if opened else (False, None)
    capture.release()
    decodable = bool(opened and ok and frame is not None)
    duration = frame_count / fps if decodable and fps > 0 else None
    return VideoRecord(
        relative_path=path.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        decodable=decodable,
        width=width,
        height=height,
        fps=round(fps, 6),
        frame_count=frame_count,
        duration_seconds=round(duration, 6) if duration is not None else None,
    )


def build_inventory(root: str | Path) -> tuple[list[VideoRecord], dict[str, object]]:
    data_root = Path(root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"inventory root not found: {data_root}")
    paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no video files found under: {data_root}")
    records = [inspect_video(path, data_root) for path in paths]
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record.relative_path)
    duplicate_groups = [sorted(group) for group in by_hash.values() if len(group) > 1]
    summary: dict[str, object] = {
        "root": str(data_root),
        "video_count": len(records),
        "total_bytes": sum(record.bytes for record in records),
        "decodable_count": sum(record.decodable for record in records),
        "undecodable_count": sum(not record.decodable for record in records),
        "duplicate_content_groups": sorted(duplicate_groups),
    }
    return records, summary


def write_inventory(
    root: str | Path,
    csv_path: str | Path,
    summary_path: str | Path,
) -> dict[str, object]:
    records, summary = build_inventory(root)
    csv_output = Path(csv_path)
    json_output = Path(summary_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = write_inventory(args.data_root, args.csv, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
