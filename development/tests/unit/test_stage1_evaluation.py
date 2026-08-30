from __future__ import annotations

import tempfile
import unittest
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from blackbox.evaluation.stage1 import (
    assign_duration_groups,
    evaluate_duration_groups,
    evaluate_stage1_classification,
    save_loss_curve_svg,
    summarize_fold_generalization,
    summarize_training_convergence,
    temporal_probability_diagnostics,
)
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


class Stage1ConvergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = [
            {
                "epoch": 1,
                "train_loss": 0.8,
                "valid_loss": 0.7,
                "valid_macro_f1": 0.6,
                "valid_prediction_change_rate": None,
                "valid_probability_mean_abs_delta": None,
            },
            {
                "epoch": 2,
                "train_loss": 0.51,
                "valid_loss": 0.500,
                "valid_macro_f1": 0.8,
                "valid_prediction_change_rate": 0.0,
                "valid_probability_mean_abs_delta": 0.006,
            },
            {
                "epoch": 3,
                "train_loss": 0.50,
                "valid_loss": 0.501,
                "valid_macro_f1": 0.8,
                "valid_prediction_change_rate": 0.0,
                "valid_probability_mean_abs_delta": 0.004,
            },
            {
                "epoch": 4,
                "train_loss": 0.49,
                "valid_loss": 0.499,
                "valid_macro_f1": 0.8,
                "valid_prediction_change_rate": 0.0,
                "valid_probability_mean_abs_delta": 0.003,
            },
        ]

    def test_stable_loss_and_unchanged_predictions_reach_fixed_point(self) -> None:
        summary = summarize_training_convergence(self.history, fixed_point_window=3)

        self.assertEqual(summary["status"], "measured")
        self.assertTrue(summary["loss_converged"])
        self.assertTrue(summary["prediction_fixed_point"])
        self.assertTrue(summary["stable"])

    def test_loss_curve_is_written_as_svg_without_optional_plot_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "loss.svg"
            save_loss_curve_svg(output, self.history)
            contents = output.read_text(encoding="utf-8")

        self.assertIn("Stage 1 loss convergence", contents)
        self.assertIn("Train loss", contents)
        self.assertIn("Validation loss", contents)


class Stage1DiagnosticAggregationTests(unittest.TestCase):
    def test_training_diagnostic_summary_exports_epoch_series(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "evaluate"
            / "evaluate_stage1_model.py"
        )
        spec = importlib.util.spec_from_file_location("stage1_evaluation_script", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        history = [
            {
                "epoch": 1,
                "train_weighted_smoothing_to_classification_ratio": 0.02,
                "train_explainability_mask_mean": 0.51,
                "train_explainability_mask_near_zero_fraction": 0.0,
                "train_explainability_mask_near_one_fraction": 0.0,
                "train_convgru_update_l2_iteration_1": 3.0,
                "train_convgru_update_l2_iteration_2": 2.0,
                "train_convgru_update_l2_iteration_3": 1.0,
                "train_convgru_last_to_first_update_l2_ratio": 1.0 / 3.0,
                "train_mstcn_residual_alpha": 0.1,
                "train_sequence_lengths_observed": [16, 24, 32],
            }
        ]

        summary = module._training_diagnostic_summary(history)

        self.assertEqual(summary["epochs"], 1)
        self.assertEqual(summary["explainability_mask"]["mean_across_epochs"], 0.51)
        self.assertEqual(
            summary["convgru_fixed_point"]["update_l2_by_iteration"]["3"]["mean"],
            1.0,
        )
        self.assertIn(
            "train_weighted_smoothing_to_classification_ratio",
            summary["epoch_series"],
        )
        self.assertEqual(summary["mstcn_gate"]["trajectory"], [0.1])
        self.assertTrue(summary["mstcn_gate"]["stable_soft_prior"])
        self.assertEqual(
            summary["sequence_length_training"]["observed_frame_counts"],
            [16, 24, 32],
        )

    def test_duration_groups_are_equal_count_and_keep_original_row_order(self) -> None:
        groups = assign_duration_groups([30.0, 10.0, 60.0, 20.0, 50.0, 40.0])

        self.assertEqual(groups, ["medium", "short", "long", "short", "long", "medium"])

    def test_equal_durations_are_not_artificially_split(self) -> None:
        self.assertEqual(assign_duration_groups([5.0] * 6), ["medium"] * 6)

    def test_duration_group_metrics_include_all_three_buckets(self) -> None:
        rows = pd.DataFrame(
            {
                "label": ["ORIGINAL", "RERECORDED"] * 3,
                "answer": ["ORIGINAL", "RERECORDED", "ORIGINAL", "ORIGINAL", "RERECORDED", "RERECORDED"],
                "duration_seconds": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        rows["duration_group"] = assign_duration_groups(rows["duration_seconds"])

        metrics = evaluate_duration_groups(rows)

        self.assertEqual(set(metrics), {"short", "medium", "long"})
        self.assertEqual(sum(int(value["samples"]) for value in metrics.values()), 6)

    def test_temporal_probability_diagnostics_count_jumps_and_switches(self) -> None:
        metrics = temporal_probability_diagnostics(
            [0.1, 0.2, 0.8, 0.75],
            jump_threshold=0.25,
        )

        self.assertEqual(metrics["label_switches"], 1)
        self.assertEqual(metrics["large_jump_events"], 1)
        self.assertAlmostEqual(float(metrics["total_variation"]), 0.75)

    def test_group_fold_summary_reports_sample_standard_deviation(self) -> None:
        summary = summarize_fold_generalization(
            {
                "0": {"macro_f1": 0.6},
                "1": {"macro_f1": 0.8},
                "2": {"macro_f1": 1.0},
            }
        )

        self.assertAlmostEqual(float(summary["macro_f1_mean"]), 0.8)
        self.assertAlmostEqual(float(summary["macro_f1_std"]), 0.2)
        self.assertAlmostEqual(float(summary["macro_f1_range"]), 0.4)


class Stage1CvScriptConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "evaluate"
            / "run_stage1_cv.py"
        )
        module_name = "stage1_cv_script_under_test"
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cls.module = module

    def test_mode_i_and_j_keep_mode_g_structure_and_isolate_augmentations(self) -> None:
        mode_i = self.module._resolve_ablation_experiments("mode_i")[0]
        mode_j = self.module._resolve_ablation_experiments("mode_j")[0]

        self.assertTrue(mode_i.detach_mstcn_input)
        self.assertFalse(mode_i.zero_gate)
        self.assertFalse(mode_i.enable_photometric)
        self.assertFalse(mode_i.enable_occlusion)
        self.assertFalse(mode_i.enable_affine)
        self.assertTrue(mode_j.detach_mstcn_input)
        self.assertFalse(mode_j.zero_gate)
        self.assertTrue(mode_j.enable_photometric)
        self.assertFalse(mode_j.enable_occlusion)
        self.assertFalse(mode_j.enable_affine)

    def test_full_dataset_can_be_selected_from_environment(self) -> None:
        root, source = self.module.resolve_dataset_root(
            None,
            None,
            environment={"BLACKBOX_FULL_DATA_DIR": "/tmp/full-stage1"},
        )

        self.assertEqual(root, Path("/tmp/full-stage1"))
        self.assertEqual(source, "full_tournament_environment")

    def test_full_dataset_cli_flag_is_distinct_from_legacy_data_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.module.resolve_dataset_root(
                Path("/tmp/public"),
                Path("/tmp/full-stage1"),
                environment={},
            )


if __name__ == "__main__":
    unittest.main()
