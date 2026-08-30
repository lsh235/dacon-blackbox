#!/usr/bin/env python3
"""Run full leakage-resistant Stage 1 Mode G GroupKFold training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from blackbox.common.runtime import load_checkpoint
from blackbox.evaluation.stage1 import (
    evaluate_stage1_classification,
    save_stage1_evaluation,
    summarize_fold_generalization,
)
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT
from blackbox.stages.stage1.baseline import (
    LABEL_TO_INDEX,
    fit_stage1,
    inverse_frequency_focal_alpha,
    score_stage1_videos,
)
from blackbox.stages.stage1.dataset import stage1_augmentation_profile
from blackbox.stages.stage1.splits import make_stratified_group_folds
from blackbox.training_control import TrainingControlConfig


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_mode: str
    focal_gamma: float
    temporal_refinement_mode: str = "gated_mstcn"
    enable_photometric: bool = True
    enable_occlusion: bool = True
    enable_affine: bool = True
    mstcn_gate_initial: float = 0.1
    zero_gate: bool = False
    detach_mstcn_input: bool = False


FULL_MODE_G_EXPERIMENT = Experiment(
    "stage1_v6_mode_g_full_augmented",
    "rgb_fft",
    2.0,
    temporal_refinement_mode="gated_mstcn",
    **stage1_augmentation_profile("mode_g_full"),
    mstcn_gate_initial=0.1,
    zero_gate=False,
    detach_mstcn_input=True,
)
RGB_ABLATION_EXPERIMENT = Experiment("ablation_rgb_cross_entropy", "rgb", 0.0)
ABLATION_EXPERIMENTS = {
    "photometric_only": Experiment(
        "mode_a_baseline_v3_photometric_only",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="single_stage",
        enable_photometric=True,
        enable_occlusion=False,
        enable_affine=False,
    ),
    "occlusion_only": Experiment(
        "mode_b_baseline_v3_occlusion_only",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="single_stage",
        enable_photometric=False,
        enable_occlusion=True,
        enable_affine=False,
    ),
    "gated_mstcn": Experiment(
        "mode_c_gated_mstcn_no_advanced_augmentation",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        enable_photometric=False,
        enable_occlusion=False,
        enable_affine=False,
        mstcn_gate_initial=0.1,
    ),
    "mstcn_loss_only": Experiment(
        "mode_d_baseline_v3_mstcn_loss_only",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        enable_photometric=False,
        enable_occlusion=False,
        enable_affine=False,
        mstcn_gate_initial=0.1,
        zero_gate=True,
    ),
    "detached_mstcn": Experiment(
        "mode_e_baseline_v3_detached_mstcn",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        enable_photometric=False,
        enable_occlusion=False,
        enable_affine=False,
        mstcn_gate_initial=0.1,
        zero_gate=False,
        detach_mstcn_input=True,
    ),
    "true_control": Experiment(
        "mode_f_baseline_v3_true_control",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        enable_photometric=False,
        enable_occlusion=False,
        enable_affine=False,
        mstcn_gate_initial=0.1,
        zero_gate=True,
        detach_mstcn_input=True,
    ),
    "forward_only": Experiment(
        "mode_g_baseline_v3_forward_only",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        enable_photometric=False,
        enable_occlusion=False,
        enable_affine=False,
        mstcn_gate_initial=0.1,
        zero_gate=False,
        detach_mstcn_input=True,
    ),
    "backward_only": Experiment(
        "mode_h_baseline_v3_backward_only",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        enable_photometric=False,
        enable_occlusion=False,
        enable_affine=False,
        mstcn_gate_initial=0.1,
        zero_gate=True,
        detach_mstcn_input=False,
    ),
    "mode_g_no_aug": Experiment(
        "mode_i_mode_g_no_aug",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        **stage1_augmentation_profile("mode_g_no_aug"),
        mstcn_gate_initial=0.1,
        zero_gate=False,
        detach_mstcn_input=True,
    ),
    "mode_g_photo_only": Experiment(
        "mode_j_mode_g_photo_only",
        "rgb_fft",
        2.0,
        temporal_refinement_mode="gated_mstcn",
        **stage1_augmentation_profile("mode_g_photo_only"),
        mstcn_gate_initial=0.1,
        zero_gate=False,
        detach_mstcn_input=True,
    ),
}


def _resolve_ablation_experiments(mode: str) -> tuple[Experiment, ...]:
    aliases = {
        "mode_a": "photometric_only",
        "mode_b": "occlusion_only",
        "mode_c": "gated_mstcn",
        "mode_d": "mstcn_loss_only",
        "mode_e": "detached_mstcn",
        "mode_f": "true_control",
        "mode_g": "forward_only",
        "mode_h": "backward_only",
        "mode_i": "mode_g_no_aug",
        "mode_j": "mode_g_photo_only",
    }
    selected = aliases.get(mode, mode)
    if selected == "all":
        return tuple(ABLATION_EXPERIMENTS.values())
    if selected == "fgh":
        return tuple(
            ABLATION_EXPERIMENTS[name]
            for name in ("true_control", "forward_only", "backward_only")
        )
    return (ABLATION_EXPERIMENTS[selected],)


def resolve_dataset_root(
    data_dir: Path | None,
    full_data_dir: Path | None,
    *,
    environment: dict[str, str] | None = None,
) -> tuple[Path, str]:
    """Resolve an explicit local/custom root or the full tournament root."""

    if data_dir is not None and full_data_dir is not None:
        raise ValueError("--data-dir and --full-data-dir are mutually exclusive")
    if full_data_dir is not None:
        return full_data_dir.expanduser().resolve(), "full_tournament_cli"
    if data_dir is not None:
        return data_dir.expanduser().resolve(), "explicit_data_dir"
    variables = os.environ if environment is None else environment
    environment_path = variables.get("BLACKBOX_FULL_DATA_DIR")
    if environment_path:
        return Path(environment_path).expanduser().resolve(), "full_tournament_environment"
    raise ValueError(
        "provide --data-dir, --full-data-dir, or set BLACKBOX_FULL_DATA_DIR"
    )


def _video_path(data_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_dir / path


def _load_labels(path: Path, data_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"ID", "path", "label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels CSV is missing columns: {missing}")
    if labels["ID"].isna().any() or labels["ID"].duplicated().any():
        raise ValueError("labels CSV requires non-empty, unique ID values")
    missing_videos = [
        str(video)
        for video in (_video_path(data_dir, value) for value in labels["path"])
        if not video.is_file()
    ]
    if missing_videos:
        raise FileNotFoundError("labeled Stage 1 videos not found: " + str(missing_videos))
    return labels


def _verify_fold_schedule(checkpoint_path: Path) -> dict[str, object]:
    """Fail closed when a fold deviates from the required v3 schedule."""

    checkpoint = load_checkpoint(
        checkpoint_path,
        required_keys=("training_schedule", "training_history", "selection"),
    )
    schedule = checkpoint["training_schedule"]
    if not isinstance(schedule, dict):
        raise ValueError("Stage 1 training_schedule must be a mapping")
    expected = {
        "configured_epochs": 30,
        "maximum_epochs": 30,
        "minimum_epochs_before_early_stopping": 10,
        "early_stopping_patience": 7,
    }
    mismatches = {
        name: {"expected": value, "actual": schedule.get(name)}
        for name, value in expected.items()
        if schedule.get(name) != value
    }
    warmup = schedule.get("warmup")
    after_warmup = schedule.get("after_warmup")
    if not isinstance(warmup, dict) or not isinstance(after_warmup, dict):
        raise ValueError("Stage 1 schedule is missing warm-up or cosine metadata")
    expected_warmup = {
        "name": "linear",
        "epochs": 3,
        "initial_learning_rate": 1e-6,
        "backbone_target_learning_rate": 1e-5,
        "head_auxiliary_target_learning_rate": 1e-4,
    }
    for name, value in expected_warmup.items():
        actual = warmup.get(name)
        if actual != value:
            mismatches[f"warmup.{name}"] = {"expected": value, "actual": actual}
    if after_warmup.get("name") != "CosineAnnealingLR":
        mismatches["after_warmup.name"] = {
            "expected": "CosineAnnealingLR",
            "actual": after_warmup.get("name"),
        }
    if after_warmup.get("minimum_learning_rate") != 1e-6:
        mismatches["after_warmup.minimum_learning_rate"] = {
            "expected": 1e-6,
            "actual": after_warmup.get("minimum_learning_rate"),
        }
    if mismatches:
        raise ValueError(f"Stage 1 fold schedule mismatch: {mismatches}")

    history = checkpoint["training_history"]
    if not isinstance(history, list) or not history:
        raise ValueError("Stage 1 fold checkpoint has no training history")
    if not 10 <= len(history) <= 30:
        raise ValueError(
            "Stage 1 fold must complete at least 10 and at most 30 epochs: "
            f"actual={len(history)}"
        )
    expected_backbone = (1e-6, 5.5e-6, 1e-5)
    expected_auxiliary = (1e-6, 5.05e-5, 1e-4)
    for index, (backbone_lr, auxiliary_lr) in enumerate(
        zip(expected_backbone, expected_auxiliary)
    ):
        record = history[index]
        if abs(float(record["backbone_learning_rate"]) - backbone_lr) > 1e-12:
            raise ValueError(f"fold warm-up backbone LR mismatch at epoch {index + 1}")
        if abs(float(record["auxiliary_learning_rate"]) - auxiliary_lr) > 1e-12:
            raise ValueError(f"fold warm-up auxiliary LR mismatch at epoch {index + 1}")
    selection = checkpoint["selection"]
    if not isinstance(selection, dict) or selection.get("monitor") != (
        "valid_macro_f1_three_region_mean_probability"
    ):
        raise ValueError("Stage 1 fold selection must monitor Validation Macro F1")
    return {
        "verified": True,
        "epochs_completed": len(history),
        "best_epoch": int(selection["best_epoch"]),
        "best_macro_f1": float(selection["best_value"]),
        "early_stopping_blocked_through_epoch": 9,
    }


def _write_comparison(
    output_root: Path,
    *,
    dataset_source: str,
    data_dir: Path,
    group_source: str,
    folds: int,
    fold_indices: Sequence[int],
    epochs: int,
    results: dict[str, dict[str, object]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(
        json.dumps(
            {
                "dataset_source": dataset_source,
                "data_dir": str(data_dir),
                "group_source": group_source,
                "folds": folds,
                "evaluated_fold_indices": list(fold_indices),
                "epochs": epochs,
                "experiments": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage 1 GroupKFold evaluation",
        "",
        f"- Dataset source: `{dataset_source}`",
        f"- Data root: `{data_dir}`",
        f"- Group source: `{group_source}`",
        f"- GroupKFold splits: {folds}",
        f"- Evaluated fold indices: {list(fold_indices)}",
        f"- Epochs per fold: {epochs}",
        (
            "- Full tournament dataset override selected; duration diagnostics are "
            "computed from the resulting OOF rows."
            if dataset_source.startswith("full_tournament")
            else "- Explicit data root selected; verify its provenance before treating OOF as generalization evidence."
        ),
        "",
        "| experiment | temporal head | detached input | zero gate | photometric | occlusion | affine | OOF Macro F1 | samples |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            "| {name} | {temporal_head} | {detached_input} | {zero_gate} | {photometric} | {occlusion} | {affine} | {macro_f1:.6f} | {samples} |".format(
                name=name,
                temporal_head=result["temporal_refinement_mode"],
                detached_input=result["detach_mstcn_input"],
                zero_gate=result["zero_gate"],
                photometric=result["enable_photometric"],
                occlusion=result["enable_occlusion"],
                affine=result["enable_affine"],
                macro_f1=float(metrics["macro_f1"]),
                samples=int(metrics["samples"]),
            )
        )
    (output_root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_experiment(
    experiment: Experiment,
    *,
    assignments: pd.DataFrame,
    data_dir: Path,
    output_root: Path,
    folds: int,
    fold_indices: Sequence[int],
    epochs: int,
    seed: int,
    processed_root: Path,
    train_slots: int,
    jitter_frames: int,
    random_temporal_jitter: bool,
    size: int,
    frames: int,
    forensic_size: int,
    fft_size: int,
    row_profile_bins: int,
    num_workers: int,
    frame_classification_weight: float,
    smoothing_weight: float,
    smoothing_truncation: float,
    explainability_weight: float,
    mask_regularization_weight: float,
    mask_sparsity_weight: float,
    motion_iterations: int,
    correlation_radius: int,
    pretrained_backbone_checkpoint: Path | None,
    use_amp: bool,
    trace_windows: int,
    diagnostic_batch_size: int,
    focal_alpha_mode: str,
) -> dict[str, object]:
    experiment_root = output_root / experiment.name
    oof_parts: list[pd.DataFrame] = []
    fold_metrics: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for fold in fold_indices:
        train_rows = assignments.loc[assignments["fold"] != fold].copy()
        valid_rows = assignments.loc[assignments["fold"] == fold].copy()
        if train_rows.empty or valid_rows.empty:
            raise ValueError(f"fold {fold} has an empty train or validation split")

        train_class_indices = [
            LABEL_TO_INDEX[str(label)] for label in train_rows["label"].tolist()
        ]
        fold_focal_alpha = (
            inverse_frequency_focal_alpha(train_class_indices)
            if focal_alpha_mode == "inverse_frequency"
            else None
        )

        fold_root = experiment_root / f"fold_{fold}"
        checkpoint = fit_stage1(
            data_dir,
            fold_root / "model",
            epochs=epochs,
            minimum_epochs=10,
            early_stopping_patience=7,
            warmup_epochs=3,
            backbone_learning_rate=1e-5,
            auxiliary_learning_rate=1e-4,
            warmup_initial_learning_rate=1e-6,
            pretrained_backbone_checkpoint=pretrained_backbone_checkpoint,
            seed=seed + fold,
            feature_mode=experiment.feature_mode,
            focal_gamma=experiment.focal_gamma,
            focal_alpha=fold_focal_alpha,
            frame_classification_weight=frame_classification_weight,
            smoothing_weight=smoothing_weight,
            smoothing_truncation=smoothing_truncation,
            explainability_weight=explainability_weight,
            mask_regularization_weight=mask_regularization_weight,
            mask_sparsity_weight=mask_sparsity_weight,
            motion_iterations=motion_iterations,
            correlation_radius=correlation_radius,
            temporal_refinement_mode=experiment.temporal_refinement_mode,
            mstcn_gate_initial=experiment.mstcn_gate_initial,
            zero_gate=experiment.zero_gate,
            detach_mstcn_input=experiment.detach_mstcn_input,
            size=size,
            frames=frames,
            train_slots=train_slots,
            jitter_frames=jitter_frames,
            random_temporal_jitter=random_temporal_jitter,
            forensic_size=forensic_size,
            fft_size=fft_size,
            row_profile_bins=row_profile_bins,
            num_workers=num_workers,
            enable_augmentation=True,
            enable_photometric_augmentation=experiment.enable_photometric,
            enable_occlusion_augmentation=experiment.enable_occlusion,
            enable_affine_augmentation=experiment.enable_affine,
            inference_tta_slots=train_slots,
            processed_root=processed_root,
            label_frame=train_rows,
            validation_label_frame=valid_rows,
            training_control=TrainingControlConfig(
                min_learning_rate=1e-6,
                early_stopping_patience=7,
                validation_fraction=0.0,
                log_dir=fold_root / "logs",
                use_amp=use_amp,
            ),
        )
        schedule_audit = _verify_fold_schedule(checkpoint)
        schedule_audit["focal_alpha"] = {
            "mode": focal_alpha_mode,
            "class_order": ["ORIGINAL", "RERECORDED"],
            "training_class_counts": {
                label: int((train_rows["label"] == label).sum())
                for label in ("ORIGINAL", "RERECORDED")
            },
            "weights": None if fold_focal_alpha is None else list(fold_focal_alpha),
        }
        (fold_root / "schedule_audit.json").write_text(
            json.dumps(schedule_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        probabilities = score_stage1_videos(
            [_video_path(data_dir, value) for value in valid_rows["path"]],
            checkpoint.parent,
        )
        valid_rows["rerecorded_probability"] = probabilities
        valid_rows["answer"] = [
            "RERECORDED" if probability >= 0.5 else "ORIGINAL"
            for probability in probabilities
        ]
        metrics = evaluate_stage1_classification(
            valid_rows["label"].tolist(),
            valid_rows["answer"].tolist(),
        )
        save_stage1_evaluation(
            fold_root,
            metrics,
            title=f"{experiment.name} fold {fold}",
        )
        valid_rows.to_csv(fold_root / "validation_predictions.csv", index=False)
        fold_metrics[str(fold)] = metrics
        oof_parts.append(valid_rows)

    oof = pd.concat(oof_parts, ignore_index=False).sort_index()
    metrics = evaluate_stage1_classification(oof["label"].tolist(), oof["answer"].tolist())
    save_stage1_evaluation(experiment_root, metrics, title=f"{experiment.name} out-of-fold")
    oof.to_csv(experiment_root / "oof_predictions.csv", index=False)
    (experiment_root / "fold_metrics.json").write_text(
        json.dumps(fold_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fold_generalization = summarize_fold_generalization(fold_metrics)
    (experiment_root / "fold_generalization.json").write_text(
        json.dumps(fold_generalization, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    from evaluate_stage1_model import evaluate_cv_experiment

    evaluation = evaluate_cv_experiment(
        experiment_root,
        data_dir=data_dir,
        trace_windows=trace_windows,
        batch_size=diagnostic_batch_size,
        num_workers=num_workers,
    )
    duration_groups = evaluation.get("duration_groups")
    if not isinstance(duration_groups, dict) or set(duration_groups) != {
        "short",
        "medium",
        "long",
    }:
        raise ValueError(
            "Stage 1 OOF evaluation must emit short/medium/long duration groups"
        )
    return {
        **asdict(experiment),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metrics": metrics,
        "fold_generalization": fold_generalization,
        "evaluation": evaluation,
        "focal_alpha_mode": focal_alpha_mode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Explicit Stage 1 root (for example the local public fixture).",
    )
    parser.add_argument(
        "--full-data-dir",
        type=Path,
        help=(
            "Full tournament Stage 1 root. If neither data flag is supplied, "
            "BLACKBOX_FULL_DATA_DIR is used."
        ),
    )
    parser.add_argument("--labels-csv", type=Path, help="Defaults to DATA_DIR/labels.csv")
    parser.add_argument(
        "--group-column",
        help="session/device/source/scene column; auto-detected when omitted",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold-indices",
        type=int,
        nargs="+",
        help="Diagnostic ablations only; full Mode G always evaluates folds 0-4.",
    )
    parser.add_argument(
        "--ablation-mode",
        choices=(
            "all",
            "fgh",
            "mode_a",
            "mode_b",
            "mode_c",
            "mode_d",
            "mode_e",
            "mode_f",
            "mode_g",
            "mode_h",
            "mode_i",
            "mode_j",
            "photometric_only",
            "occlusion_only",
            "gated_mstcn",
            "mstcn_loss_only",
            "detached_mstcn",
            "true_control",
            "forward_only",
            "backward_only",
            "mode_g_no_aug",
            "mode_g_photo_only",
        ),
        help=(
            "Run Stage 1 diagnostic ablations on zero-based Fold 1 only; "
            "'fgh' runs the synchronized controls and 'all' runs modes A through J."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--include-rgb-ablation",
        action="store_true",
        help="Also run the legacy RGB/CE ablation after the full Mode G experiment.",
    )
    parser.add_argument(
        "--pretrained-backbone-checkpoint",
        type=Path,
        help="Optional local MViTv2-S state_dict; omitted means random initialization.",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--train-slots", type=int, default=3)
    parser.add_argument("--jitter-frames", type=int, default=4)
    parser.add_argument("--no-random-temporal-jitter", action="store_true")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--forensic-size", type=int, default=320)
    parser.add_argument("--fft-size", type=int, default=112)
    parser.add_argument("--row-profile-bins", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trace-windows", type=int, default=9)
    parser.add_argument("--diagnostic-batch-size", type=int, default=4)
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--frame-classification-weight", type=float, default=0.25)
    parser.add_argument(
        "--focal-alpha-mode",
        choices=("inverse_frequency", "none"),
        default="inverse_frequency",
        help=(
            "Per-fold Focal alpha policy. inverse_frequency uses only the training "
            "partition and normalizes weights to mean sample weight 1."
        ),
    )
    parser.add_argument("--smoothing-weight", type=float, default=0.05)
    parser.add_argument("--smoothing-truncation", type=float, default=4.0)
    parser.add_argument("--explainability-weight", type=float, default=0.05)
    parser.add_argument("--mask-regularization-weight", type=float, default=0.02)
    parser.add_argument("--mask-sparsity-weight", type=float, default=1e-3)
    parser.add_argument("--motion-iterations", type=int, default=3)
    parser.add_argument("--correlation-radius", type=int, default=2)
    args = parser.parse_args()
    if args.epochs != 30:
        parser.error("Stage 1 v3 GroupKFold strictly requires --epochs 30")
    if min(
        args.train_slots,
        args.size,
        args.frames,
        args.forensic_size,
        args.fft_size,
        args.row_profile_bins,
    ) < 1 or min(args.jitter_frames, args.num_workers) < 0:
        parser.error("Stage 1 sampling, feature, and worker values are invalid")
    if min(
        args.frame_classification_weight,
        args.smoothing_weight,
        args.explainability_weight,
        args.mask_regularization_weight,
        args.mask_sparsity_weight,
    ) < 0 or args.smoothing_truncation <= 0:
        parser.error("Stage 1 auxiliary loss weights/truncation are invalid")
    if args.motion_iterations < 1 or args.correlation_radius < 0:
        parser.error("Stage 1 motion iterations/radius are invalid")
    if args.trace_windows < 2 or args.diagnostic_batch_size < 1:
        parser.error("diagnostic trace windows must be >= 2 and batch size >= 1")
    if args.ablation_mode is not None and args.include_rgb_ablation:
        parser.error("--ablation-mode cannot be combined with --include-rgb-ablation")

    try:
        data_dir, dataset_source = resolve_dataset_root(args.data_dir, args.full_data_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if not data_dir.is_dir():
        parser.error(f"Stage 1 data directory does not exist: {data_dir}")
    labels_csv = args.labels_csv or data_dir / "labels.csv"
    labels = _load_labels(labels_csv, data_dir)
    plan = make_stratified_group_folds(
        labels,
        n_splits=args.folds,
        group_column=args.group_column,
        seed=args.seed,
    )
    if args.ablation_mode is not None:
        if args.folds <= 1:
            parser.error("--ablation-mode requires at least two GroupKFold splits")
        if args.fold_indices is not None and tuple(args.fold_indices) != (1,):
            parser.error("--ablation-mode is strictly limited to --fold-indices 1")
        fold_indices = (1,)
    else:
        if args.folds != 5:
            parser.error("full Mode G execution strictly requires --folds 5")
        if args.fold_indices is not None:
            parser.error("full Mode G execution must evaluate all five folds")
        fold_indices = tuple(range(5))
    invalid_fold_indices = [
        fold for fold in fold_indices if fold < 0 or fold >= args.folds
    ]
    if not fold_indices or invalid_fold_indices:
        parser.error(
            "--fold-indices must contain unique values in "
            f"[0, {args.folds - 1}], got {list(fold_indices)}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan.assignments.to_csv(args.output_root / "split_assignments.csv", index=False)
    split_audit = {
        "dataset_source": dataset_source,
        "data_dir": str(data_dir),
        "labels_csv": str(labels_csv.resolve()),
        "group_source": plan.group_source,
        "groups": int(plan.assignments["group_value"].nunique()),
        "samples": int(len(plan.assignments)),
        "folds": {
            str(fold): {
                "validation_samples": int((plan.assignments["fold"] == fold).sum()),
                "validation_groups": int(
                    plan.assignments.loc[
                        plan.assignments["fold"] == fold,
                        "group_value",
                    ].nunique()
                ),
                "validation_label_counts": {
                    str(label): int(count)
                    for label, count in plan.assignments.loc[
                        plan.assignments["fold"] == fold,
                        "label",
                    ].value_counts().items()
                },
            }
            for fold in range(args.folds)
        },
    }
    (args.output_root / "split_audit.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    experiments = (
        _resolve_ablation_experiments(args.ablation_mode)
        if args.ablation_mode is not None
        else (
            (FULL_MODE_G_EXPERIMENT, RGB_ABLATION_EXPERIMENT)
            if args.include_rgb_ablation
            else (FULL_MODE_G_EXPERIMENT,)
        )
    )
    results = {
        experiment.name: _run_experiment(
            experiment,
            assignments=plan.assignments,
            data_dir=data_dir,
            output_root=args.output_root,
            folds=args.folds,
            fold_indices=fold_indices,
            epochs=args.epochs,
            seed=args.seed,
            processed_root=args.processed_root,
            train_slots=args.train_slots,
            jitter_frames=args.jitter_frames,
            random_temporal_jitter=not args.no_random_temporal_jitter,
            size=args.size,
            frames=args.frames,
            forensic_size=args.forensic_size,
            fft_size=args.fft_size,
            row_profile_bins=args.row_profile_bins,
            num_workers=args.num_workers,
            frame_classification_weight=args.frame_classification_weight,
            smoothing_weight=args.smoothing_weight,
            smoothing_truncation=args.smoothing_truncation,
            explainability_weight=args.explainability_weight,
            mask_regularization_weight=args.mask_regularization_weight,
            mask_sparsity_weight=args.mask_sparsity_weight,
            motion_iterations=args.motion_iterations,
            correlation_radius=args.correlation_radius,
            pretrained_backbone_checkpoint=args.pretrained_backbone_checkpoint,
            use_amp=args.use_amp,
            trace_windows=args.trace_windows,
            diagnostic_batch_size=args.diagnostic_batch_size,
            focal_alpha_mode=args.focal_alpha_mode,
        )
        for experiment in experiments
    }
    _write_comparison(
        args.output_root,
        dataset_source=dataset_source,
        data_dir=data_dir,
        group_source=plan.group_source,
        folds=args.folds,
        fold_indices=fold_indices,
        epochs=args.epochs,
        results=results,
    )
    print((args.output_root / "comparison.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
