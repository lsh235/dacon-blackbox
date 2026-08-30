#!/usr/bin/env python3
"""Evaluate a refactored Stage 1 checkpoint with convergence and branch diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from blackbox.common.runtime import load_checkpoint
from blackbox.evaluation.stage1 import (
    STAGE1_LABELS,
    Stage1EvaluationError,
    assign_duration_groups,
    evaluate_duration_groups,
    evaluate_stage1_classification,
    probe_video_duration,
    save_loss_curve_svg,
    summarize_fold_generalization,
    summarize_training_convergence,
    temporal_probability_diagnostics,
    validated_training_history,
)
from blackbox.stages.stage1.baseline import (
    resolve_tta_slots,
    score_stage1_checkpoint,
    score_stage1_checkpoint_diagnostics,
)


BRANCHES = ("rgb", "spatial", "temporal", "motion")


def _video_path(data_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_dir / path


def _load_labels(path: Path, data_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"path", "label"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise Stage1EvaluationError(f"labels CSV is missing columns: {missing}")
    labels = labels.copy()
    if "ID" not in labels.columns:
        labels["ID"] = labels["path"].map(lambda value: Path(str(value)).stem)
    if labels["ID"].isna().any() or labels["ID"].duplicated().any():
        raise Stage1EvaluationError("labels CSV requires non-empty, unique ID values")
    unknown = sorted(set(labels["label"].astype(str)) - set(STAGE1_LABELS))
    if unknown:
        raise Stage1EvaluationError(f"unsupported Stage 1 labels: {unknown}")
    labels["video_path"] = [str(_video_path(data_dir, value)) for value in labels["path"]]
    missing_videos = [path for path in labels["video_path"] if not Path(path).is_file()]
    if missing_videos:
        raise FileNotFoundError(f"labeled Stage 1 videos not found: {missing_videos}")
    return labels


def _load_history(
    checkpoint: Mapping[str, object],
    history_path: Path | None,
) -> list[dict[str, object]]:
    if history_path is None:
        raw = checkpoint.get("training_history", [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise Stage1EvaluationError("checkpoint training_history must be a sequence")
        return validated_training_history(raw)
    if not history_path.is_file():
        raise FileNotFoundError(f"training history not found: {history_path}")
    if history_path.suffix.lower() == ".jsonl":
        raw = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise Stage1EvaluationError("training history file must contain a JSON list or JSONL")
    normalized: list[dict[str, object]] = []
    for record in raw:
        if not isinstance(record, dict):
            raise Stage1EvaluationError("each training history row must be a mapping")
        copied = dict(record)
        if "valid_macro_f1" not in copied and "valid_metric" in copied:
            copied["valid_macro_f1"] = copied["valid_metric"]
        normalized.append(copied)
    return validated_training_history(normalized)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _branch_summary(
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    videos = diagnostics["videos"]
    if not isinstance(videos, list):
        raise Stage1EvaluationError("checkpoint diagnostics videos must be a list")
    summary: dict[str, object] = {
        "classifier_weight_rms": diagnostics["classifier_branch_weight_rms"],
        "branches": {},
    }
    for branch in BRANCHES:
        activations: list[float] = []
        weighted: list[float] = []
        ablations: list[float] = []
        for video in videos:
            for window in video["windows"]:
                activations.append(float(window["activation_rms"][branch]))
                weighted.append(float(window["weighted_activation_rms"][branch]))
                ablations.append(abs(float(window["probability_delta_without_branch"][branch])))
        summary["branches"][branch] = {
            "activation_rms_mean": _mean_or_none(activations),
            "activation_rms_std": float(np.std(activations)) if activations else None,
            "weighted_activation_rms_mean": _mean_or_none(weighted),
            "weighted_activation_rms_std": float(np.std(weighted)) if weighted else None,
            "mean_absolute_probability_delta_when_ablated": _mean_or_none(ablations),
            "std_absolute_probability_delta_when_ablated": (
                float(np.std(ablations)) if ablations else None
            ),
            "windows": len(activations),
        }
    return summary


def _temporal_rows(
    labels: pd.DataFrame,
    diagnostics: Mapping[str, object],
    *,
    jump_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    video_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    videos = diagnostics["videos"]
    for label_row, video in zip(labels.itertuples(index=False), videos):
        if not bool(video["valid"]):
            raise Stage1EvaluationError(f"failed to decode every temporal window: {video['path']}")
        probabilities = [
            float(window["rerecorded_probability"])
            for window in video["windows"]
        ]
        temporal = temporal_probability_diagnostics(
            probabilities,
            jump_threshold=jump_threshold,
        )
        video_rows.append({"ID": str(label_row.ID), **temporal})
        for window in video["windows"]:
            flattened: dict[str, object] = {
                "ID": str(label_row.ID),
                "window_index": int(window["window_index"]),
                "relative_position": float(window["relative_position"]),
                "rerecorded_probability": float(window["rerecorded_probability"]),
                "frame_probability_mean_absolute_step": float(
                    window["frame_probability_mean_absolute_step"]
                ),
                "frame_probability_max_absolute_step": float(
                    window["frame_probability_max_absolute_step"]
                ),
                "frame_label_switches": int(window["frame_label_switches"]),
                "explainability_mask_mean": float(window["explainability_mask_mean"]),
                "explainability_mask_min": float(window["explainability_mask_min"]),
                "explainability_mask_max": float(window["explainability_mask_max"]),
                "flow_last_to_first_update_ratio": float(
                    window["flow_last_to_first_update_ratio"]
                ),
                "convgru_last_to_first_update_l2_ratio": float(
                    window["convgru_last_to_first_update_l2_ratio"]
                ),
            }
            for iteration, magnitude in enumerate(
                window["convgru_update_l2_by_iteration"],
                start=1,
            ):
                flattened[f"convgru_update_l2_iteration_{iteration}"] = float(magnitude)
            for branch in BRANCHES:
                flattened[f"{branch}_activation_rms"] = float(
                    window["activation_rms"][branch]
                )
                flattened[f"{branch}_weighted_activation_rms"] = float(
                    window["weighted_activation_rms"][branch]
                )
                flattened[f"{branch}_probability_delta_without"] = float(
                    window["probability_delta_without_branch"][branch]
                )
            window_rows.append(flattened)
    aggregate_fields = (
        "probability_std",
        "mean_absolute_step",
        "max_absolute_step",
        "total_variation",
        "smoothness_score",
        "label_switches",
        "large_jump_events",
    )
    aggregate = {
        f"mean_{field}": float(np.mean([float(row[field]) for row in video_rows]))
        for field in aggregate_fields
    }
    aggregate.update(
        {
            f"std_{field}": float(np.std([float(row[field]) for row in video_rows]))
            for field in aggregate_fields
        }
    )
    aggregate["videos"] = len(video_rows)
    aggregate["jump_threshold"] = jump_threshold
    for field in (
        "frame_probability_mean_absolute_step",
        "frame_probability_max_absolute_step",
        "frame_label_switches",
        "explainability_mask_mean",
        "flow_last_to_first_update_ratio",
        "convgru_last_to_first_update_l2_ratio",
    ):
        aggregate[f"mean_{field}"] = float(
            np.mean([float(row[field]) for row in window_rows])
        )
        aggregate[f"std_{field}"] = float(
            np.std([float(row[field]) for row in window_rows])
        )
    iteration_fields = sorted(
        {
            name
            for row in window_rows
            for name in row
            if name.startswith("convgru_update_l2_iteration_")
        }
    )
    aggregate["convgru_update_l2_by_iteration"] = {
        name.removeprefix("convgru_update_l2_iteration_"): {
            "mean": float(np.mean([float(row[name]) for row in window_rows])),
            "std": float(np.std([float(row[name]) for row in window_rows])),
        }
        for name in iteration_fields
    }
    aggregate["interpretation"] = (
        "ordered clip-window instability proxy; Stage 1 does not emit frame segmentation"
    )
    return video_rows, window_rows, aggregate


def _fold_summary(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise Stage1EvaluationError("fold metrics JSON must be a mapping keyed by fold")
    metrics = {
        str(name): value
        for name, value in raw.items()
        if isinstance(value, dict) and "macro_f1" in value
    }
    if not metrics:
        raise Stage1EvaluationError("fold metrics JSON contains no Macro F1 records")
    summary = summarize_fold_generalization(metrics)
    audit_path = path.parent.parent / "split_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(audit, dict):
            summary["group_audit"] = audit
    return summary


def _training_diagnostic_summary(
    history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not history:
        return {"epochs": 0, "status": "unavailable"}
    records = [dict(record) for record in history]

    def series(name: str) -> list[float]:
        return [
            float(record[name])
            for record in records
            if record.get(name) is not None
        ]

    smoothing_ratio = series(
        "train_weighted_smoothing_to_classification_ratio"
    )
    mask_mean = series("train_explainability_mask_mean")
    mask_near_zero = series("train_explainability_mask_near_zero_fraction")
    mask_near_one = series("train_explainability_mask_near_one_fraction")
    gate_alpha = series("train_mstcn_residual_alpha")
    if not gate_alpha:
        gate_alpha = series("mstcn_residual_alpha")
    observed_sequence_lengths = sorted(
        {
            int(length)
            for record in records
            for length in record.get("train_sequence_lengths_observed", [])
        }
    )
    update_fields = sorted(
        {
            name
            for record in records
            for name in record
            if name.startswith("train_convgru_update_l2_iteration_")
        },
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    update_by_iteration = {
        name.rsplit("_", 1)[-1]: {
            "mean": float(np.mean(series(name))),
            "std": float(np.std(series(name))),
            "last_epoch": series(name)[-1],
        }
        for name in update_fields
        if series(name)
    }
    tracked_fields = sorted(
        {
            name
            for record in records
            for name in record
            if (
                "classification" in name
                or "smoothing" in name
                or "explainability_mask" in name
                or "convgru_" in name
                or "mstcn_" in name
            )
            and isinstance(record.get(name), (int, float))
        }
    )
    return {
        "epochs": len(records),
        "status": "measured",
        "loss_balance": {
            "weighted_smoothing_to_classification_ratio_mean": (
                _mean_or_none(smoothing_ratio)
            ),
            "weighted_smoothing_to_classification_ratio_max": (
                max(smoothing_ratio) if smoothing_ratio else None
            ),
            "epochs_where_weighted_smoothing_exceeds_classification": sum(
                value > 1.0 for value in smoothing_ratio
            ),
        },
        "explainability_mask": {
            "mean_across_epochs": _mean_or_none(mask_mean),
            "min_epoch_mean": min(mask_mean) if mask_mean else None,
            "max_epoch_mean": max(mask_mean) if mask_mean else None,
            "near_zero_fraction_mean": _mean_or_none(mask_near_zero),
            "near_one_fraction_mean": _mean_or_none(mask_near_one),
        },
        "convgru_fixed_point": {
            "update_l2_by_iteration": update_by_iteration,
            "last_to_first_update_l2_ratio_mean": _mean_or_none(
                series("train_convgru_last_to_first_update_l2_ratio")
            ),
            "last_to_first_update_l2_ratio_last_epoch": (
                series("train_convgru_last_to_first_update_l2_ratio")[-1]
                if series("train_convgru_last_to_first_update_l2_ratio")
                else None
            ),
        },
        "mstcn_gate": {
            "trajectory": gate_alpha,
            "initial": gate_alpha[0] if gate_alpha else None,
            "final": gate_alpha[-1] if gate_alpha else None,
            "minimum": min(gate_alpha) if gate_alpha else None,
            "maximum": max(gate_alpha) if gate_alpha else None,
            "range": (
                max(gate_alpha) - min(gate_alpha)
                if gate_alpha
                else None
            ),
            "epochs_near_zero": sum(value <= 0.01 for value in gate_alpha),
            "epochs_overpowering": sum(value >= 0.5 for value in gate_alpha),
            "stable_soft_prior": (
                all(0.01 < value < 0.5 for value in gate_alpha)
                if gate_alpha
                else None
            ),
        },
        "sequence_length_training": {
            "observed_frame_counts": observed_sequence_lengths,
            "epochs": {
                str(record["epoch"]): [
                    int(length)
                    for length in record.get("train_sequence_lengths_observed", [])
                ]
                for record in records
            },
        },
        "epoch_series": {
            "epoch": [int(record["epoch"]) for record in records],
            **{name: series(name) for name in tracked_fields},
        },
    }


def _format_report(report: Mapping[str, object]) -> str:
    overall = report["classification"]
    lines = [
        "# Stage 1 model evaluation",
        "",
        f"- Macro F1: {float(overall['macro_f1']):.6f}",
        f"- Accuracy: {float(overall['accuracy']):.6f}",
        f"- Samples: {int(overall['samples'])}",
        "",
        "## Per-class metrics",
        "",
        "| class | precision | recall | F1 | support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in STAGE1_LABELS:
        values = overall["per_class"][label]
        lines.append(
            f"| {label} | {float(values['precision']):.6f} | "
            f"{float(values['recall']):.6f} | {float(values['f1']):.6f} | "
            f"{int(values['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "| true / predicted | ORIGINAL | RERECORDED |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, row in zip(STAGE1_LABELS, overall["confusion_matrix"]):
        lines.append(f"| {label} | {int(row[0])} | {int(row[1])} |")
    lines.extend(
        [
            "",
            "## Duration groups",
            "",
            "| group | seconds | samples | Macro F1 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name, values in report["duration_groups"].items():
        if int(values["samples"]) == 0:
            lines.append(f"| {name} | unavailable | 0 | unavailable |")
        else:
            span = (
                f"{float(values['duration_min_seconds']):.3f}–"
                f"{float(values['duration_max_seconds']):.3f}"
            )
            lines.append(
                f"| {name} | {span} | {int(values['samples'])} | "
                f"{float(values['macro_f1']):.6f} |"
            )
    lines.extend(
        [
            "",
            "## Branch contribution",
            "",
            "| branch | activation RMS | weighted RMS | mean abs probability delta when ablated |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["branch_contribution"]["branches"].items():
        lines.append(
            f"| {name} | {float(values['activation_rms_mean']):.6f} | "
            f"{float(values['weighted_activation_rms_mean']):.6f} | "
            f"{float(values['mean_absolute_probability_delta_when_ablated']):.6f} |"
        )
    temporal = report["temporal_consistency"]
    lines.extend(
        [
            "",
            "## Temporal consistency",
            "",
            f"- Mean probability step: {float(temporal['mean_mean_absolute_step']):.6f}",
            f"- Mean maximum jump: {float(temporal['mean_max_absolute_step']):.6f}",
            f"- Mean label switches: {float(temporal['mean_label_switches']):.6f}",
            f"- Mean large-jump events: {float(temporal['mean_large_jump_events']):.6f}",
            f"- Mean within-clip frame probability step: {float(temporal['mean_frame_probability_mean_absolute_step']):.6f}",
            f"- Mean frame label switches: {float(temporal['mean_frame_label_switches']):.6f}",
            f"- Mean explainability mask: {float(temporal['mean_explainability_mask_mean']):.6f}",
            f"- Mean ConvGRU last/first update L2 ratio: {float(temporal['mean_convgru_last_to_first_update_l2_ratio']):.6f}",
            "- This is an ordered clip-window instability proxy, not a frame-segmentation metric.",
            "",
            "## Training convergence",
            "",
            f"- Status: `{report['convergence']['status']}`",
            f"- Loss converged: `{report['convergence']['loss_converged']}`",
            f"- Prediction fixed point: `{report['convergence']['prediction_fixed_point']}`",
            f"- Stable: `{report['convergence']['stable']}`",
        ]
    )
    folds = report.get("cross_validation")
    lines.extend(["", "## GroupKFold generalization", ""])
    if folds is None:
        lines.append("- Not supplied. Pass `--fold-metrics-json` from `run_stage1_cv.py`.")
    else:
        audit = folds.get("group_audit")
        if isinstance(audit, dict):
            lines.extend(
                [
                    f"- Group source: `{audit.get('group_source', 'unknown')}`",
                    f"- Unique held-out domains: {int(audit.get('groups', 0))}",
                ]
            )
        lines.extend(
            [
                f"- Fold Macro F1: `{folds['fold_macro_f1']}`",
                f"- Mean: {float(folds['macro_f1_mean']):.6f}",
                f"- Sample standard deviation: {float(folds['macro_f1_std']):.6f}",
                f"- Range: {float(folds['macro_f1_range']):.6f}",
            ]
        )
    return "\n".join(lines) + "\n"


def evaluate_model(args: argparse.Namespace) -> dict[str, object]:
    labels_csv = args.labels_csv or args.data_dir / "labels.csv"
    labels = _load_labels(labels_csv, args.data_dir)
    checkpoint = load_checkpoint(args.checkpoint, required_keys=("model", "architecture"))
    history = _load_history(checkpoint, args.training_history)
    diagnostics = score_stage1_checkpoint_diagnostics(
        labels["video_path"].tolist(),
        args.checkpoint,
        temporal_windows=args.trace_windows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        require_cuda=not args.allow_cpu,
    )
    official_slots = resolve_tta_slots(dict(checkpoint))
    if args.trace_windows == official_slots:
        official_probabilities = [
            float(video["rerecorded_probability"])
            for video in diagnostics["videos"]
        ]
    else:
        official_probabilities = score_stage1_checkpoint(
            labels["video_path"].tolist(),
            args.checkpoint,
            tta_slots=official_slots,
            require_cuda=not args.allow_cpu,
        )
    labels["rerecorded_probability"] = official_probabilities
    labels["answer"] = np.where(
        labels["rerecorded_probability"] >= 0.5,
        "RERECORDED",
        "ORIGINAL",
    )
    metadata = [probe_video_duration(path) for path in labels["video_path"]]
    labels["frame_count"] = [item["frame_count"] for item in metadata]
    labels["fps"] = [item["fps"] for item in metadata]
    labels["duration_seconds"] = [item["duration_seconds"] for item in metadata]
    labels["duration_group"] = assign_duration_groups(labels["duration_seconds"])

    temporal_rows, window_rows, temporal_summary = _temporal_rows(
        labels,
        diagnostics,
        jump_threshold=args.jump_threshold,
    )
    temporal_frame = pd.DataFrame(temporal_rows)
    labels = labels.merge(temporal_frame, on="ID", how="left", validate="one_to_one")
    classification = evaluate_stage1_classification(
        labels["label"].tolist(),
        labels["answer"].tolist(),
    )
    report = {
        "classification": classification,
        "convergence": summarize_training_convergence(
            history,
            fixed_point_window=args.fixed_point_window,
            loss_relative_tolerance=args.loss_relative_tolerance,
            probability_delta_tolerance=args.probability_delta_tolerance,
        ),
        "training_diagnostics": _training_diagnostic_summary(history),
        "duration_groups": evaluate_duration_groups(labels),
        "branch_contribution": _branch_summary(diagnostics),
        "temporal_consistency": temporal_summary,
        "cross_validation": _fold_summary(args.fold_metrics_json),
        "official_inference_slots": official_slots,
        "trace_windows": args.trace_windows,
        "checkpoint_selection": checkpoint.get("selection"),
        "training_schedule": checkpoint.get("training_schedule"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(window_rows).to_csv(
        args.output_dir / "temporal_branch_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    confusion = pd.DataFrame(
        classification["confusion_matrix"],
        index=[f"true_{label}" for label in STAGE1_LABELS],
        columns=[f"pred_{label}" for label in STAGE1_LABELS],
    )
    confusion.to_csv(args.output_dir / "confusion_matrix.csv")
    if history:
        save_loss_curve_svg(args.output_dir / "loss_curve.svg", history)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = _format_report(report)
    (args.output_dir / "evaluation.md").write_text(markdown, encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(markdown, end="")
    return report


def evaluate_cv_experiment(
    experiment_root: str | Path,
    *,
    data_dir: str | Path,
    trace_windows: int = 9,
    jump_threshold: float = 0.25,
    fixed_point_window: int = 3,
    loss_relative_tolerance: float = 0.02,
    probability_delta_tolerance: float = 0.01,
    batch_size: int = 4,
    num_workers: int = 0,
    allow_cpu: bool = False,
) -> dict[str, object]:
    """Evaluate every fold-best checkpoint and aggregate OOF diagnostics."""

    root = Path(experiment_root)
    data = Path(data_dir)
    oof_path = root / "oof_predictions.csv"
    fold_metrics_path = root / "fold_metrics.json"
    if not oof_path.is_file() or not fold_metrics_path.is_file():
        raise FileNotFoundError(
            f"CV evaluation requires {oof_path} and {fold_metrics_path}"
        )
    oof = _load_labels(oof_path, data)
    if "answer" not in oof.columns:
        raise Stage1EvaluationError("OOF predictions must contain an answer column")
    metadata = [probe_video_duration(path) for path in oof["video_path"]]
    oof["frame_count"] = [item["frame_count"] for item in metadata]
    oof["fps"] = [item["fps"] for item in metadata]
    oof["duration_seconds"] = [item["duration_seconds"] for item in metadata]
    oof["duration_group"] = assign_duration_groups(oof["duration_seconds"])
    classification = evaluate_stage1_classification(
        oof["label"].tolist(),
        oof["answer"].tolist(),
    )

    fold_reports: dict[str, dict[str, object]] = {}
    fold_values = sorted(int(value) for value in oof["fold"].unique())
    for fold in fold_values:
        fold_root = root / f"fold_{fold}"
        checkpoint = fold_root / "model" / "best.pt"
        validation_csv = fold_root / "validation_predictions.csv"
        report = evaluate_model(
            argparse.Namespace(
                data_dir=data,
                labels_csv=validation_csv,
                checkpoint=checkpoint,
                output_dir=fold_root / "evaluation",
                training_history=None,
                fold_metrics_json=fold_metrics_path,
                trace_windows=trace_windows,
                jump_threshold=jump_threshold,
                fixed_point_window=fixed_point_window,
                loss_relative_tolerance=loss_relative_tolerance,
                probability_delta_tolerance=probability_delta_tolerance,
                batch_size=batch_size,
                num_workers=num_workers,
                allow_cpu=allow_cpu,
                quiet=True,
            )
        )
        fold_reports[str(fold)] = {
            "checkpoint": str(checkpoint),
            "classification": report["classification"],
            "duration_groups": report["duration_groups"],
            "convergence": report["convergence"],
            "training_diagnostics": report["training_diagnostics"],
            "branch_activation_rms": report["branch_contribution"],
            "temporal_consistency": report["temporal_consistency"],
            "checkpoint_selection": report["checkpoint_selection"],
        }

    cross_validation = _fold_summary(fold_metrics_path)
    duration_groups = evaluate_duration_groups(oof)
    report = {
        "classification": classification,
        "oof_confusion_matrix": classification["confusion_matrix"],
        "duration_groups": duration_groups,
        "macro_f1_by_duration_group": {
            name: values.get("macro_f1")
            for name, values in duration_groups.items()
        },
        "cross_validation": cross_validation,
        "fold_best_model_diagnostics": fold_reports,
        "branch_activation_rms_by_fold": {
            fold: values["branch_activation_rms"]
            for fold, values in fold_reports.items()
        },
        "temporal_consistency_stddev_by_fold": {
            fold: {
                name: value
                for name, value in values["temporal_consistency"].items()
                if name.startswith("std_")
            }
            for fold, values in fold_reports.items()
        },
        "samples": len(oof),
    }
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(
        classification["confusion_matrix"],
        index=[f"true_{label}" for label in STAGE1_LABELS],
        columns=[f"pred_{label}" for label in STAGE1_LABELS],
    ).to_csv(root / "oof_confusion_matrix.csv")
    (root / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    duration_lines = [
        "# Stage 1 GroupKFold OOF evaluation",
        "",
        f"- OOF Macro F1: {float(classification['macro_f1']):.6f}",
        f"- OOF accuracy: {float(classification['accuracy']):.6f}",
        f"- Samples: {len(oof)}",
        "",
        "| duration group | samples | Macro F1 |",
        "| --- | ---: | ---: |",
    ]
    for name, values in report["duration_groups"].items():
        macro_f1 = values.get("macro_f1")
        duration_lines.append(
            f"| {name} | {int(values['samples'])} | "
            f"{'unavailable' if macro_f1 is None else f'{float(macro_f1):.6f}'} |"
        )
    (root / "evaluation.md").write_text(
        "\n".join(duration_lines) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Stage 1 data root")
    parser.add_argument("--labels-csv", type=Path, help="Defaults to DATA_DIR/labels.csv")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument(
        "--cv-root",
        type=Path,
        help="Experiment root containing fold_*/ and oof_predictions.csv.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--training-history", type=Path, help="Optional JSON or JSONL override")
    parser.add_argument(
        "--fold-metrics-json",
        type=Path,
        help="GroupKFold fold_metrics.json emitted by run_stage1_cv.py",
    )
    parser.add_argument("--trace-windows", type=int, default=9)
    parser.add_argument("--jump-threshold", type=float, default=0.25)
    parser.add_argument("--fixed-point-window", type=int, default=3)
    parser.add_argument("--loss-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--probability-delta-tolerance", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if min(args.trace_windows, args.fixed_point_window) < 2:
        parser.error("--trace-windows and --fixed-point-window must be >= 2")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be >= 1 and --num-workers must be >= 0")
    if not 0.0 <= args.jump_threshold <= 1.0:
        parser.error("--jump-threshold must be in [0, 1]")
    if min(args.loss_relative_tolerance, args.probability_delta_tolerance) < 0.0:
        parser.error("convergence tolerances must be >= 0")
    if args.cv_root is not None:
        report = evaluate_cv_experiment(
            args.cv_root,
            data_dir=args.data_dir,
            trace_windows=args.trace_windows,
            jump_threshold=args.jump_threshold,
            fixed_point_window=args.fixed_point_window,
            loss_relative_tolerance=args.loss_relative_tolerance,
            probability_delta_tolerance=args.probability_delta_tolerance,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            allow_cpu=args.allow_cpu,
        )
        print(json.dumps(report["classification"], ensure_ascii=False, indent=2))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required with --checkpoint")
        evaluate_model(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
