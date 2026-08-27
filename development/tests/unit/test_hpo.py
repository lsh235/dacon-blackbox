from __future__ import annotations

import unittest

import optuna

from blackbox.hpo import materialize_best_config, suggest_stage3_hparams


class Stage3HPOTests(unittest.TestCase):
    def test_search_space_covers_iteration9_parameters(self) -> None:
        params = suggest_stage3_hparams(
            optuna.trial.FixedTrial(
                {
                    "flow_roi_top_ratio": 0.5,
                    "flow_grid_size": 3,
                    "learning_rate": 0.0002,
                    "focal_gamma": 2.0,
                }
            )
        )
        self.assertEqual(
            set(params),
            {"flow_roi_top_ratio", "flow_grid_size", "learning_rate", "focal_gamma"},
        )

    def test_best_parameters_are_written_into_reusable_stage3_yaml(self) -> None:
        best = {
            "flow_roi_top_ratio": 0.65,
            "flow_grid_size": 4,
            "learning_rate": 0.0001,
            "focal_gamma": 1.0,
        }
        output = materialize_best_config(
            {"stage3": {"architecture": "two_stream"}},
            best_params=best,
            fixed_stage3_params={"use_physics_vector": True},
            best_value=0.75,
            trial_number=2,
            epochs_per_trial=3,
        )
        self.assertEqual(output["stage3"]["flow_grid_size"], 4)
        self.assertEqual(output["stage3"]["focal_gamma"], 1.0)
        self.assertTrue(output["stage3"]["use_physics_vector"])
        self.assertEqual(output["hpo"]["metric"], "validation_accel_steer_mean_macro_f1")
        self.assertEqual(output["hpo"]["best_value"], 0.75)


if __name__ == "__main__":
    unittest.main()
