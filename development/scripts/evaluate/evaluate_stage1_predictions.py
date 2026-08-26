#!/usr/bin/env python3
"""Evaluate Stage 1 predictions with Macro F1 and leakage-auditable diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from blackbox.evaluation.stage1 import (
    Stage1EvaluationError,
    evaluate_stage1_classification,
    format_stage1_evaluation_report,
    save_stage1_evaluation,
)


def evaluate_prediction_csv(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join one prediction per labeled ID and calculate local Stage 1 metrics."""

    for name, frame, columns in (
        ("labels", labels, {"ID", "label"}),
        ("predictions", predictions, {"ID", "answer"}),
    ):
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise Stage1EvaluationError(f"{name} is missing columns: {missing}")
        if frame["ID"].isna().any() or frame["ID"].duplicated().any():
            raise Stage1EvaluationError(f"{name} IDs must be non-empty and unique")

    joined = labels[["ID", "label"]].merge(
        predictions[["ID", "answer"]],
        on="ID",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    missing = joined.loc[joined["_merge"] != "both", "ID"].tolist()
    if missing:
        raise Stage1EvaluationError(
            "labels and predictions must contain the same IDs; mismatch=" + str(missing)
        )
    joined = joined.drop(columns="_merge")
    metrics = evaluate_stage1_classification(joined["label"].tolist(), joined["answer"].tolist())
    return joined, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    joined, metrics = evaluate_prediction_csv(
        pd.read_csv(args.labels_csv),
        pd.read_csv(args.predictions_csv),
    )
    if args.output_dir is not None:
        save_stage1_evaluation(args.output_dir, metrics, title="Stage 1 local evaluation")
        joined.to_csv(args.output_dir / "evaluated_predictions.csv", index=False)
    print(format_stage1_evaluation_report(metrics, title="Stage 1 local evaluation"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
