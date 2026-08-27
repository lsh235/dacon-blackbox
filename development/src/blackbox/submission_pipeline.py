"""Sequential, safe local generation of the three documented Stage CSV files."""

from __future__ import annotations

import gc
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from blackbox.common.runtime import video_paths
from blackbox.contracts import STAGE_COLUMNS, validate_prediction_frame
from blackbox.inference import predict_stage1, predict_stage2, predict_stage3
from blackbox.stages.stage2.dataset_stage2 import video_frame_count


StagePredictor = Callable[[str | Path, str | Path], pd.DataFrame]
DEFAULT_PREDICTORS: Mapping[int, StagePredictor] = {
    1: predict_stage1,
    2: predict_stage2,
    3: predict_stage3,
}
OUTPUT_FILENAMES = {1: "stage1_submission.csv", 2: "stage2_submission.csv", 3: "stage3_submission.csv"}
_FRAME_SUFFIX = re.compile(r"(\d+)$")


def release_pipeline_vram() -> None:
    """Release references between stages so the three models never coexist."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stage1_fallback(data_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ID": path.stem, "answer": "ORIGINAL"} for path in video_paths(data_dir / "videos")],
        columns=STAGE_COLUMNS["stage1"],
    )


def _frame_number(path: Path) -> int:
    match = _FRAME_SUFFIX.search(path.stem)
    return int(match.group(1)) if match else 0


def _stage2_fallback(data_dir: Path) -> pd.DataFrame:
    image_root = data_dir / "images"
    folders = sorted(path for path in image_root.iterdir() if path.is_dir()) if image_root.is_dir() else []
    rows = []
    for folder in folders:
        numbers = [_frame_number(path) for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        first_frame = min(numbers) if numbers else 0
        rows.append(
            {
                "ID": folder.name,
                "collision_frame": first_frame,
                "entry_frame": first_frame,
                "evasion_space": 0,
                "entry_side": "LEFT",
            }
        )
    return pd.DataFrame(rows, columns=STAGE_COLUMNS["stage2"])


def _stage3_fallback(data_dir: Path, *, frames_per_sample: int) -> pd.DataFrame:
    rows = []
    for path in video_paths(data_dir / "videos"):
        try:
            frame_count = video_frame_count(path)
            sample_count = max(1, (frame_count + frames_per_sample - 1) // frames_per_sample)
        except ValueError:
            # A damaged video still needs one contract-valid row if its ID is
            # discoverable; the caller records this fallback in the manifest.
            sample_count = 1
        rows.extend(
            {
                "ID": path.stem,
                "sample_index": sample_index,
                "accel_label": "CONSTANT",
                "steer_label": "STRAIGHT",
            }
            for sample_index in range(sample_count)
        )
    return pd.DataFrame(rows, columns=STAGE_COLUMNS["stage3"])


def project_stage3_source_frames(prediction: pd.DataFrame, *, frames_per_sample: int) -> pd.DataFrame:
    """Select source-frame predictions at a caller-specified 0.1-second stride.

    The public material does not reconcile Stage 3 container FPS with its 10Hz
    index.  ``frames_per_sample`` is therefore an explicit invocation setting,
    not an implicit inference from video metadata.
    """

    if frames_per_sample < 1:
        raise ValueError("frames_per_sample must be >= 1")
    source = validate_prediction_frame("stage3", prediction).copy()
    rows: list[pd.DataFrame] = []
    for _, group in source.groupby("ID", sort=False):
        ordered = group.sort_values("sample_index", kind="stable")
        selected = ordered.iloc[::frames_per_sample].copy()
        selected["sample_index"] = range(len(selected))
        rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=STAGE_COLUMNS["stage3"])
    return pd.concat(rows, ignore_index=True)[STAGE_COLUMNS["stage3"]]


def _expected_ids(stage: int, data_dir: Path) -> set[str]:
    if stage in {1, 3}:
        return {path.stem for path in video_paths(data_dir / "videos")}
    image_root = data_dir / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Stage 2 image directory not found: {image_root}")
    return {path.name for path in image_root.iterdir() if path.is_dir()}


def _fallback_for_stage(stage: int, data_dir: Path, *, frames_per_sample: int) -> pd.DataFrame:
    if stage == 1:
        return _stage1_fallback(data_dir)
    if stage == 2:
        return _stage2_fallback(data_dir)
    return _stage3_fallback(data_dir, frames_per_sample=frames_per_sample)


def _append_missing_video_fallbacks(
    stage: int,
    prediction: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    expected_ids: set[str],
) -> pd.DataFrame:
    actual_ids = set(prediction["ID"].tolist())
    unexpected = sorted(actual_ids - expected_ids)
    if unexpected:
        raise ValueError(f"Stage {stage} produced IDs absent from its input: {unexpected}")
    missing = expected_ids - actual_ids
    if missing:
        prediction = pd.concat(
            [prediction, fallback[fallback["ID"].isin(missing)]],
            ignore_index=True,
        )
    return prediction


def _validate_sample_format(prediction: pd.DataFrame, sample_path: Path, *, stage: int) -> None:
    """Require a supplied DACON sample CSV to match this stage exactly."""

    sample = pd.read_csv(sample_path)
    if sample.columns.tolist() != prediction.columns.tolist():
        raise ValueError(
            f"Stage {stage} sample columns do not match generated columns: "
            f"sample={sample.columns.tolist()} generated={prediction.columns.tolist()}"
        )
    for column in prediction.columns:
        if pd.api.types.is_integer_dtype(sample[column].dtype) and not pd.api.types.is_integer_dtype(prediction[column].dtype):
            raise ValueError(
                f"Stage {stage} {column!r} must retain sample integer dtype, got {prediction[column].dtype}"
            )
    if not sample.empty:
        if stage == 3:
            sample_keys = set(zip(sample["ID"], sample["sample_index"], strict=True))
            prediction_keys = set(zip(prediction["ID"], prediction["sample_index"], strict=True))
            if sample_keys != prediction_keys:
                raise ValueError("Stage 3 generated (ID, sample_index) keys do not match the supplied sample")
        elif set(sample["ID"]) != set(prediction["ID"]):
            raise ValueError(f"Stage {stage} generated IDs do not match the supplied sample")


def generate_submission_bundle(
    data_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    *,
    stage3_frames_per_sample: int,
    sample_submissions: Mapping[int, str | Path] | None = None,
    predictors: Mapping[int, StagePredictor] = DEFAULT_PREDICTORS,
) -> dict[str, object]:
    """Run Stage 1→2→3 sequentially and write contract-valid CSV outputs.

    Every predictor owns its model only for its call.  After each call this
    runner explicitly collects Python objects and empties CUDA cache before the
    next Stage begins.  A failed whole-stage inference, or a missing video in a
    partial result, receives documented contract-safe fallback rows.
    """

    if stage3_frames_per_sample < 1:
        raise ValueError("stage3_frames_per_sample must be >= 1")
    root = Path(data_root)
    models = Path(model_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    samples = sample_submissions or {}
    summary: dict[str, object] = {"fallback_stages": [], "stages": {}}

    for stage in (1, 2, 3):
        stage_name = f"stage{stage}"
        stage_data = root / stage_name
        fallback = _fallback_for_stage(stage, stage_data, frames_per_sample=stage3_frames_per_sample)
        expected_ids = _expected_ids(stage, stage_data)
        started = perf_counter()
        error: str | None = None
        try:
            prediction = predictors[stage](stage_data, models / stage_name)
            prediction = validate_prediction_frame(stage_name, prediction)
            if stage == 3:
                prediction = project_stage3_source_frames(
                    prediction,
                    frames_per_sample=stage3_frames_per_sample,
                )
        except Exception as exc:  # Safe output is preferable to an absent CSV.
            prediction = fallback
            error = f"{type(exc).__name__}: {exc}"
            summary["fallback_stages"].append(stage_name)
        finally:
            release_pipeline_vram()

        prediction = _append_missing_video_fallbacks(
            stage,
            prediction,
            fallback,
            expected_ids=expected_ids,
        )
        prediction = validate_prediction_frame(stage_name, prediction)
        if stage in samples:
            _validate_sample_format(prediction, Path(samples[stage]), stage=stage)
        path = output / OUTPUT_FILENAMES[stage]
        prediction.to_csv(path, index=False, encoding="utf-8-sig")
        stage_summary = {
            "rows": len(prediction),
            "output": str(path),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "fallback": error,
        }
        summary["stages"][stage_name] = stage_summary

    manifest = output / "submission_manifest.json"
    manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["manifest"] = str(manifest)
    return summary
