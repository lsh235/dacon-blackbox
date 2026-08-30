"""Five-fold Mode G inference and sample-aligned Stage 1 CSV generation."""

from __future__ import annotations

import argparse
import gc
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from blackbox.common.runtime import (
    CheckpointError,
    autocast_context,
    choose_device,
    load_checkpoint,
    release_device_cache,
)
from blackbox.contracts import validate_prediction_frame
from blackbox.stages.stage1.baseline import (
    DEFAULT_BASE_INITIALIZATION_SEED,
    DEFAULT_CORRELATION_RADIUS,
    DEFAULT_MOTION_ITERATIONS,
    DEFAULT_MSTCN_GATE_INITIAL,
    DEFAULT_MSTCN_STAGES,
    STAGE1_ARCHITECTURE,
    Stage1MViT,
)
from blackbox.stages.stage1.dataset import (
    DEFAULT_FFT_SIZE,
    DEFAULT_FORENSIC_SIZE,
    DEFAULT_ROW_PROFILE_BINS,
    DEFAULT_TEMPORAL_SLOTS,
    RGB_FFT_FEATURES,
    Stage1TestDataset,
)


LOGGER = logging.getLogger("stage1-inference")
EXPECTED_FOLD_INDICES = tuple(range(5))
CLASS_NAMES = ("ORIGINAL", "RERECORDED")
_FOLD_DIRECTORY = re.compile(r"^fold[_-]?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Stage1InferenceGeometry:
    """Checkpoint fields that must agree across all ensemble members."""

    size: int
    frames: int
    slots: int
    feature_mode: str
    forensic_size: int
    fft_size: int
    row_profile_bins: int
    motion_iterations: int
    correlation_radius: int
    temporal_refinement_stages: int
    mstcn_gate_initial: float
    base_initialization_seed: int


@dataclass(frozen=True)
class Stage1EnsembleResult:
    """Raw per-fold and soft-voted probabilities in ORIGINAL/RERECORDED order."""

    videos: tuple[Path, ...]
    checkpoint_paths: tuple[Path, ...]
    fold_probabilities: tuple[np.ndarray, ...]
    ensemble_probabilities: np.ndarray


def _fold_index(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = _FOLD_DIRECTORY.fullmatch(part)
        if match is not None:
            return int(match.group(1))
    return None


def discover_fold_checkpoints(
    checkpoint_dir: str | Path,
    *,
    expected_fold_indices: Sequence[int] = EXPECTED_FOLD_INDICES,
) -> tuple[Path, ...]:
    """Find exactly one ``best.pt`` for every requested GroupKFold split."""

    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 1 checkpoint directory not found: {root}")
    expected = tuple(int(index) for index in expected_fold_indices)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_fold_indices must contain unique fold indices")

    by_fold: dict[int, list[Path]] = {index: [] for index in expected}
    for path in sorted(root.rglob("best.pt")):
        fold_index = _fold_index(path.relative_to(root))
        if fold_index in by_fold:
            by_fold[fold_index].append(path)

    missing = [index for index, paths in by_fold.items() if not paths]
    duplicates = {
        index: [str(path) for path in paths]
        for index, paths in by_fold.items()
        if len(paths) > 1
    }
    if missing or duplicates:
        raise CheckpointError(
            "Stage 1 ensemble requires exactly one best.pt under each fold_0..fold_4; "
            f"missing={missing}, duplicates={duplicates}"
        )
    return tuple(by_fold[index][0] for index in expected)


def _required_mapping(
    container: Mapping[str, object],
    key: str,
    *,
    checkpoint_path: Path,
) -> Mapping[str, object]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointError(
            f"Stage 1 checkpoint {key!r} must be a mapping: {checkpoint_path}"
        )
    return value


def validate_mode_g_checkpoint(
    checkpoint: Mapping[str, object],
    checkpoint_path: str | Path,
) -> Stage1InferenceGeometry:
    """Reject checkpoints that are not detached, learnably gated Mode G."""

    path = Path(checkpoint_path)
    if checkpoint.get("architecture") != STAGE1_ARCHITECTURE:
        raise CheckpointError(
            "Stage 1 checkpoint architecture mismatch for Mode G: "
            f"{path}"
        )
    config = _required_mapping(checkpoint, "model_config", checkpoint_path=path)
    sampling = _required_mapping(checkpoint, "sampling", checkpoint_path=path)
    state = _required_mapping(checkpoint, "model", checkpoint_path=path)

    mismatches: dict[str, object] = {}
    expected_values = {
        "feature_mode": (checkpoint.get("feature_mode"), RGB_FFT_FEATURES),
        "temporal_refinement": (config.get("temporal_refinement"), "gated_mstcn"),
        "mstcn_input_detached": (config.get("mstcn_input_detached"), True),
        "mstcn_zero_gate": (config.get("mstcn_zero_gate"), False),
        "fusion": (
            config.get("fusion"),
            "base_clip_logits_plus_gated_refined_frame_logits",
        ),
        "sampling.name": (sampling.get("name"), "centered_contiguous_regions"),
        "sampling.inference_tta_slots": (
            sampling.get("inference_tta_slots"),
            DEFAULT_TEMPORAL_SLOTS,
        ),
    }
    for name, (actual, expected) in expected_values.items():
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise CheckpointError(f"checkpoint is not exact Mode G: {path}: {mismatches}")

    gate = state.get("mstcn_gate_logit")
    if (
        not isinstance(gate, torch.Tensor)
        or gate.numel() != 1
        or not bool(torch.isfinite(gate).all())
    ):
        raise CheckpointError(
            "Mode G requires a finite learnable mstcn_gate_logit parameter: "
            f"{path}"
        )

    try:
        geometry = Stage1InferenceGeometry(
            size=int(checkpoint["size"]),
            frames=int(checkpoint["frames"]),
            slots=int(sampling["inference_tta_slots"]),
            feature_mode=str(checkpoint["feature_mode"]),
            forensic_size=int(config["forensic_size"]),
            fft_size=int(config["fft_size"]),
            row_profile_bins=int(config["row_profile_bins"]),
            motion_iterations=int(config["motion_iterations"]),
            correlation_radius=int(config["correlation_radius"]),
            temporal_refinement_stages=int(config["temporal_refinement_stages"]),
            mstcn_gate_initial=float(config["mstcn_gate_initial"]),
            base_initialization_seed=int(
                config.get("base_initialization_seed", DEFAULT_BASE_INITIALIZATION_SEED)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(
            f"Mode G checkpoint has incomplete geometry metadata: {path}: {exc}"
        ) from exc
    if min(
        geometry.size,
        geometry.frames,
        geometry.slots,
        geometry.forensic_size,
        geometry.fft_size,
        geometry.row_profile_bins,
        geometry.motion_iterations,
        geometry.correlation_radius,
        geometry.temporal_refinement_stages,
    ) < 1:
        raise CheckpointError(f"Mode G checkpoint geometry must be positive: {path}")
    if sampling.get("frames_per_region") != geometry.frames:
        raise CheckpointError(
            "Mode G frames_per_region differs from checkpoint frames: "
            f"{path}"
        )
    if not 0.0 < geometry.mstcn_gate_initial < 1.0:
        raise CheckpointError(f"Mode G gate initialization must be in (0, 1): {path}")
    return geometry


def load_mode_g_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[Stage1MViT, Stage1InferenceGeometry, float]:
    """Instantiate one strict Mode G fold and report its learned gate value."""

    path = Path(checkpoint_path)
    checkpoint = load_checkpoint(
        path,
        required_keys=(
            "architecture",
            "model",
            "size",
            "frames",
            "feature_mode",
            "model_config",
            "sampling",
        ),
    )
    geometry = validate_mode_g_checkpoint(checkpoint, path)
    model = Stage1MViT(
        feature_mode=geometry.feature_mode,
        row_profile_bins=geometry.row_profile_bins,
        motion_iterations=geometry.motion_iterations,
        correlation_radius=geometry.correlation_radius,
        temporal_refinement_stages=geometry.temporal_refinement_stages,
        temporal_refinement_mode="gated_mstcn",
        mstcn_gate_initial=geometry.mstcn_gate_initial,
        zero_gate=False,
        detach_mstcn_input=True,
        base_initialization_seed=geometry.base_initialization_seed,
    )
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise CheckpointError(f"cannot strictly load Mode G checkpoint {path}: {exc}") from exc
    if (
        model.temporal_refinement_mode != "gated_mstcn"
        or not model.detach_mstcn_input
        or model.zero_gate
        or not isinstance(model.mstcn_gate_logit, nn.Parameter)
    ):
        raise CheckpointError(f"instantiated model is not learnably gated detached Mode G: {path}")
    learned_gate = float(model.mstcn_alpha.detach().cpu())
    del checkpoint
    return model.to(device).eval(), geometry, learned_gate


def _move_inputs(
    inputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in inputs.items()
    }


def score_mode_g_fold(
    model: Stage1MViT,
    loader: DataLoader,
    *,
    video_count: int,
    expected_regions: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Average raw two-class Softmax probabilities over three regions."""

    probability_sum = np.zeros((video_count, len(CLASS_NAMES)), dtype=np.float64)
    region_counts = np.zeros(video_count, dtype=np.int64)
    with torch.inference_mode():
        for inputs, video_indices, valid in loader:
            with autocast_context(device, enabled=use_amp):
                logits = model(**_move_inputs(inputs, device))
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise RuntimeError("Mode G model must return [batch, classes] logits")
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            indices = video_indices.numpy().astype(np.int64, copy=False)
            valid_mask = valid.numpy().astype(bool, copy=False)
            np.add.at(probability_sum, indices[valid_mask], probabilities[valid_mask])
            np.add.at(region_counts, indices[valid_mask], 1)

    failed = np.flatnonzero(region_counts != expected_regions)
    if failed.size:
        raise RuntimeError(
            "Stage 1 online decoding did not produce all three regions for video "
            f"indices={failed.tolist()}, region_counts={region_counts[failed].tolist()}"
        )
    fold_probabilities = probability_sum / region_counts[:, None]
    if (
        not np.isfinite(fold_probabilities).all()
        or np.any(fold_probabilities < 0.0)
        or np.any(fold_probabilities > 1.0)
        or not np.allclose(fold_probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise RuntimeError("Mode G fold produced invalid Softmax probabilities")
    return fold_probabilities


def mean_fold_probabilities(folds: Sequence[np.ndarray]) -> np.ndarray:
    """Validate and average per-fold ORIGINAL/RERECORDED probabilities."""

    values = [np.asarray(fold, dtype=np.float64) for fold in folds]
    if not values:
        raise ValueError("at least one fold probability matrix is required")
    expected_shape = values[0].shape
    if len(expected_shape) != 2 or expected_shape[1] != len(CLASS_NAMES):
        raise ValueError("fold probabilities must have shape [videos, 2]")
    for index, fold in enumerate(values):
        if fold.shape != expected_shape:
            raise ValueError(
                f"fold {index} probability shape mismatch: {fold.shape} != {expected_shape}"
            )
        if not np.isfinite(fold).all() or not np.allclose(
            fold.sum(axis=1), 1.0, atol=1e-5
        ):
            raise ValueError(f"fold {index} does not contain valid Softmax probabilities")
    averaged = np.mean(np.stack(values, axis=0), axis=0)
    return averaged / averaged.sum(axis=1, keepdims=True)


def predict_mode_g_ensemble(
    test_data_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    batch_size: int = 1,
    num_workers: int = 2,
    require_cuda: bool = True,
    use_amp: bool = True,
) -> Stage1EnsembleResult:
    """Run all five folds sequentially and soft-vote their probabilities."""

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be >= 1 and num_workers must be >= 0")
    checkpoint_paths = discover_fold_checkpoints(checkpoint_dir)
    device = choose_device(require_cuda=require_cuda)
    dataset: Stage1TestDataset | None = None
    loader: DataLoader | None = None
    reference_geometry: Stage1InferenceGeometry | None = None
    fold_probabilities: list[np.ndarray] = []

    LOGGER.info(
        "Discovered %d folds; device=%s, AMP=%s",
        len(checkpoint_paths),
        device,
        bool(use_amp and device.type == "cuda"),
    )
    for fold_index, checkpoint_path in zip(EXPECTED_FOLD_INDICES, checkpoint_paths, strict=True):
        model, geometry, learned_gate = load_mode_g_model(
            checkpoint_path,
            device=device,
        )
        try:
            if reference_geometry is None:
                reference_geometry = geometry
                dataset = Stage1TestDataset(
                    test_data_dir,
                    size=geometry.size,
                    frames=geometry.frames,
                    slots=geometry.slots,
                    feature_mode=geometry.feature_mode,
                    forensic_size=geometry.forensic_size,
                    fft_size=geometry.fft_size,
                    row_profile_bins=geometry.row_profile_bins,
                )
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=num_workers > 0,
                )
                LOGGER.info(
                    "Test videos=%d, regions/video=%d, frames/region=%d",
                    len(dataset.videos),
                    geometry.slots,
                    geometry.frames,
                )
            elif geometry != reference_geometry:
                raise CheckpointError(
                    "Stage 1 fold preprocessing/model geometry mismatch: "
                    f"fold={fold_index}, expected={reference_geometry}, actual={geometry}"
                )
            if dataset is None or loader is None:
                raise RuntimeError("Stage 1 test loader was not initialized")

            LOGGER.info(
                "Scoring fold_%d with learned residual gate alpha=%.6f: %s",
                fold_index,
                learned_gate,
                checkpoint_path,
            )
            probabilities = score_mode_g_fold(
                model,
                loader,
                video_count=len(dataset.videos),
                expected_regions=geometry.slots,
                device=device,
                use_amp=use_amp,
            )
            fold_probabilities.append(probabilities)
            LOGGER.info(
                "fold_%d P(RERECORDED): min=%.6f mean=%.6f max=%.6f",
                fold_index,
                float(probabilities[:, 1].min()),
                float(probabilities[:, 1].mean()),
                float(probabilities[:, 1].max()),
            )
        finally:
            del model
            gc.collect()
            release_device_cache(device)

    if dataset is None:
        raise RuntimeError("Stage 1 ensemble did not initialize a test dataset")
    ensemble = mean_fold_probabilities(fold_probabilities)
    return Stage1EnsembleResult(
        videos=tuple(dataset.videos),
        checkpoint_paths=checkpoint_paths,
        fold_probabilities=tuple(fold_probabilities),
        ensemble_probabilities=ensemble,
    )


def predict_stage1(data_dir, model_dir):
    """Competition entrypoint for the five-fold detached Mode G ensemble."""

    result = predict_mode_g_ensemble(
        data_dir,
        model_dir,
        batch_size=1,
        num_workers=2,
        require_cuda=True,
        use_amp=True,
    )
    frame = pd.DataFrame(
        {
            "ID": [path.stem for path in result.videos],
            "answer": np.where(
                result.ensemble_probabilities[:, 1] >= 0.5,
                CLASS_NAMES[1],
                CLASS_NAMES[0],
            ),
        },
        columns=["ID", "answer"],
    )
    return validate_prediction_frame("stage1", frame)


def format_stage1_submission(
    sample_submission_csv: str | Path,
    videos: Sequence[str | Path],
    probabilities: np.ndarray,
    *,
    output_mode: Literal["labels", "probabilities"] = "labels",
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Align predictions to the sample CSV and retain its exact row order."""

    if output_mode not in {"labels", "probabilities"}:
        raise ValueError("output_mode must be 'labels' or 'probabilities'")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    sample_path = Path(sample_submission_csv)
    if not sample_path.is_file():
        raise FileNotFoundError(f"sample submission CSV not found: {sample_path}")
    sample = pd.read_csv(sample_path, dtype={"ID": str})
    if len(sample.columns) != 2 or set(sample.columns) != {"ID", "answer"}:
        raise ValueError(
            "Stage 1 sample submission must contain exactly ID and answer columns, "
            f"got {sample.columns.tolist()}"
        )
    if (
        sample["ID"].isna().any()
        or not sample["ID"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
        or sample["ID"].duplicated().any()
    ):
        raise ValueError("sample submission IDs must be non-empty and unique")

    paths = [Path(video) for video in videos]
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(paths), len(CLASS_NAMES)):
        raise ValueError(
            "ensemble probabilities must have shape "
            f"({len(paths)}, {len(CLASS_NAMES)}), got {values.shape}"
        )
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("ensemble probabilities are not valid two-class probabilities")

    video_ids = [path.stem for path in paths]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("test video stems must be unique")
    probability_by_id = {
        video_id: float(value[1])
        for video_id, value in zip(video_ids, values, strict=True)
    }
    sample_ids = sample["ID"].astype(str).tolist()
    missing = sorted(set(sample_ids) - set(probability_by_id))
    extra = sorted(set(probability_by_id) - set(sample_ids))
    if missing or extra:
        raise ValueError(
            "test videos do not match sample submission IDs: "
            f"missing_videos={missing[:10]}, extra_videos={extra[:10]}"
        )

    rerecorded = np.asarray([probability_by_id[video_id] for video_id in sample_ids])
    labels = np.where(rerecorded >= threshold, CLASS_NAMES[1], CLASS_NAMES[0])
    output = sample.copy()
    output["answer"] = rerecorded if output_mode == "probabilities" else labels
    distribution = {
        label: int(np.count_nonzero(labels == label))
        for label in CLASS_NAMES
    }
    return output, distribution


def write_stage1_submission(
    output_csv: str | Path,
    sample_submission_csv: str | Path,
    result: Stage1EnsembleResult,
    *,
    output_mode: Literal["labels", "probabilities"] = "labels",
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Format and persist a UTF-8-BOM CSV accepted by DACON tooling."""

    frame, distribution = format_stage1_submission(
        sample_submission_csv,
        result.videos,
        result.ensemble_probabilities,
        output_mode=output_mode,
        threshold=threshold,
    )
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame, distribution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sample-submission-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-mode",
        choices=("labels", "probabilities"),
        default="labels",
        help=(
            "Use labels for the official Stage 1 Macro-F1 contract. "
            "Use probabilities only if a different leaderboard explicitly scores AUC/LogLoss."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for local smoke checks; competition inference requires CUDA.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA float16 autocast (enabled by default).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    result = predict_mode_g_ensemble(
        args.test_data_dir,
        args.checkpoint_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        require_cuda=not args.allow_cpu,
        use_amp=args.amp,
    )
    frame, distribution = write_stage1_submission(
        args.output_csv,
        args.sample_submission_csv,
        result,
        output_mode=args.output_mode,
        threshold=args.threshold,
    )
    rerecorded = result.ensemble_probabilities[:, 1]
    LOGGER.info(
        "Ensemble P(RERECORDED): min=%.6f mean=%.6f max=%.6f",
        float(rerecorded.min()),
        float(rerecorded.mean()),
        float(rerecorded.max()),
    )
    LOGGER.info("Thresholded class distribution: %s", distribution)
    if 0 in distribution.values():
        LOGGER.warning("Predictions collapsed to a single class; review before leaderboard use")
    LOGGER.info(
        "Saved Stage 1 submission: rows=%d mode=%s path=%s",
        len(frame),
        args.output_mode,
        args.output_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
