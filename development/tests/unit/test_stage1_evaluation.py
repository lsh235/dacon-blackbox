from __future__ import annotations

import unittest

import pandas as pd

from blackbox.evaluation.stage1 import evaluate_stage1_classification
from blackbox.stages.stage1.splits import (
    Stage1SplitError,
    make_stratified_group_folds,
)


class Stage1MetricTests(unittest.TestCase):
    def test_macro_f1_confusion_matrix_and_per_class_metrics(self) -> None:
        metrics = evaluate_stage1_classification(
            ["ORIGINAL", "ORIGINAL", "RERECORDED", "RERECORDED"],
            ["ORIGINAL", "RERECORDED", "RERECORDED", "RERECORDED"],
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(float(metrics["accuracy"]), 0.75)
        self.assertAlmostEqual(float(metrics["macro_f1"]), (2 / 3 + 0.8) / 2)
        per_class = metrics["per_class"]
        self.assertAlmostEqual(float(per_class["ORIGINAL"]["precision"]), 1.0)
        self.assertAlmostEqual(float(per_class["RERECORDED"]["recall"]), 1.0)

    def test_missing_predicted_class_reports_zero_without_crashing(self) -> None:
        metrics = evaluate_stage1_classification(
            ["ORIGINAL", "RERECORDED"],
            ["ORIGINAL", "ORIGINAL"],
        )

        rerecorded = metrics["per_class"]["RERECORDED"]
        self.assertEqual(rerecorded["precision"], 0.0)
        self.assertEqual(rerecorded["recall"], 0.0)
        self.assertEqual(rerecorded["f1"], 0.0)


class Stage1GroupSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for source in range(6):
            for label, directory in (("ORIGINAL", "original"), ("RERECORDED", "rerecorded")):
                rows.append(
                    {
                        "ID": f"S{source}_{label[0]}",
                        "path": f"{directory}/{source:06d}.mp4",
                        "label": label,
                        "source_content_id": f"source-{source}",
                    }
                )
        self.metadata = pd.DataFrame(rows)

    def test_group_folds_never_split_one_source_between_train_and_valid(self) -> None:
        plan = make_stratified_group_folds(
            self.metadata,
            n_splits=3,
            group_column="source_content_id",
            seed=7,
        )

        self.assertEqual(plan.group_source, "source_content_id")
        self.assertTrue((plan.assignments.groupby("group_value")["fold"].nunique() == 1).all())
        for fold in range(3):
            valid_groups = set(plan.assignments.loc[plan.assignments["fold"] == fold, "group_value"])
            train_groups = set(plan.assignments.loc[plan.assignments["fold"] != fold, "group_value"])
            self.assertFalse(valid_groups & train_groups)
            valid_labels = plan.assignments.loc[plan.assignments["fold"] == fold, "label"]
            self.assertEqual(set(valid_labels), {"ORIGINAL", "RERECORDED"})

    def test_path_stem_fallback_groups_original_and_rerecorded_pairs(self) -> None:
        plan = make_stratified_group_folds(
            self.metadata.drop(columns="source_content_id"),
            n_splits=3,
        )

        self.assertEqual(plan.group_source, "path_stem")
        self.assertTrue((plan.assignments.groupby("group_value")["fold"].nunique() == 1).all())

    def test_missing_explicit_group_column_fails_closed(self) -> None:
        with self.assertRaisesRegex(Stage1SplitError, "group column"):
            make_stratified_group_folds(
                self.metadata,
                n_splits=3,
                group_column="scene_id",
            )


if __name__ == "__main__":
    unittest.main()
