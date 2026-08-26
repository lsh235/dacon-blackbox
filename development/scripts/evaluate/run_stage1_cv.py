#!/usr/bin/env python3
"""Run leakage-resistant Stage 1 local CV for RGB+CE versus RGB+FFT+Focal."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from blackbox.evaluation.stage1 import (
    evaluate_stage1_classification,
    save_stage1_evaluation,
)
from blackbox.stages.stage1.baseline import fit_stage1, score_stage1_videos
from blackbox.stages.stage1.splits import make_stratified_group_folds


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_mode: str
    focal_gamma: float


EXPERIMENTS = (
    Experiment("exp_a_rgb_cross_entropy", "rgb", 0.0),
    Experiment("exp_b_rgb_fft_focal", "rgb_fft", 2.0),
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


def _write_comparison(
    output_root: Path,
    *,
    group_source: str,
    folds: int,
    epochs: int,
    results: dict[str, dict[str, object]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(
        json.dumps(
            {
                "group_source": group_source,
                "folds": folds,
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
        "# Stage 1 Local CV: Experiment A vs B",
        "",
        f"- Group source: `{group_source}`",
        f"- Folds: {folds}",
        f"- Epochs per fold: {epochs}",
        "- Caution: supplied public examples are a structural smoke fixture, not a generalization benchmark.",
        "",
        "| experiment | feature mode | focal gamma | Macro F1 | accuracy | samples |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        metrics = result["metrics"]
        lines.append(
            "| {name} | {feature_mode} | {gamma:.1f} | {macro_f1:.6f} | {accuracy:.6f} | {samples} |".format(
                name=name,
                feature_mode=result["feature_mode"],
                gamma=float(result["focal_gamma"]),
                macro_f1=float(metrics["macro_f1"]),
                accuracy=float(metrics["accuracy"]),
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
    epochs: int,
    seed: int,
) -> dict[str, object]:
    experiment_root = output_root / experiment.name
    oof_parts: list[pd.DataFrame] = []
    fold_metrics: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for fold in range(folds):
        train_rows = assignments.loc[assignments["fold"] != fold].copy()
        valid_rows = assignments.loc[assignments["fold"] == fold].copy()
        if train_rows.empty or valid_rows.empty:
            raise ValueError(f"fold {fold} has an empty train or validation split")

        fold_root = experiment_root / f"fold_{fold}"
        checkpoint = fit_stage1(
            data_dir,
            fold_root / "model",
            epochs=epochs,
            seed=seed + fold,
            feature_mode=experiment.feature_mode,
            focal_gamma=experiment.focal_gamma,
            label_frame=train_rows,
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
    return {
        **asdict(experiment),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Stage 1 directory")
    parser.add_argument("--labels-csv", type=Path, help="Defaults to DATA_DIR/labels.csv")
    parser.add_argument("--group-column", help="source/scene column; auto-detected when omitted")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1 for a local CV measurement")

    labels_csv = args.labels_csv or args.data_dir / "labels.csv"
    labels = _load_labels(labels_csv, args.data_dir)
    plan = make_stratified_group_folds(
        labels,
        n_splits=args.folds,
        group_column=args.group_column,
        seed=args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan.assignments.to_csv(args.output_root / "split_assignments.csv", index=False)
    results = {
        experiment.name: _run_experiment(
            experiment,
            assignments=plan.assignments,
            data_dir=args.data_dir,
            output_root=args.output_root,
            folds=args.folds,
            epochs=args.epochs,
            seed=args.seed,
        )
        for experiment in EXPERIMENTS
    }
    _write_comparison(
        args.output_root,
        group_source=plan.group_source,
        folds=args.folds,
        epochs=args.epochs,
        results=results,
    )
    print((args.output_root / "comparison.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
