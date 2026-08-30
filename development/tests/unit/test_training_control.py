from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from blackbox.experiment_config import load_experiment_config, stage_paths
from blackbox.training_control import EarlyStopping, JsonlTrainingLogger, cosine_scheduler


class TrainingControlTests(unittest.TestCase):
    def test_yaml_configuration_resolves_paths_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "configs" / "trial.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                "data:\n  root: ../raw\n  processed_root: ../processed\nrun:\n  model_root: ../models\n",
                encoding="utf-8",
            )
            config, resolved = load_experiment_config(config_path)
            data_dir, model_dir, processed_dir = stage_paths(config, resolved, "stage2")
        self.assertEqual(data_dir, (Path(temporary) / "raw" / "stage2").resolve())
        self.assertEqual(model_dir, (Path(temporary) / "models" / "stage2").resolve())
        self.assertEqual(processed_dir, (Path(temporary) / "processed").resolve())

    def test_cosine_logger_and_early_stopping_record_epoch_history(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = cosine_scheduler(optimizer, epochs=2, minimum_learning_rate=0.01)
        stopper = EarlyStopping(mode="min", patience=2)
        with tempfile.TemporaryDirectory() as temporary:
            logger = JsonlTrainingLogger("fixture", Path(temporary))
            logger.log(
                epoch=1,
                train_loss=1.0,
                learning_rate=optimizer.param_groups[0]["lr"],
                valid_metric=None,
                monitor_name="train_loss_proxy_no_validation",
                monitor_value=1.0,
                valid_loss=1.2,
                diagnostics={"prediction_change_rate": 0.0},
            )
            (parameter.square()).backward()
            optimizer.step()
            scheduler.step()
            history = [json.loads(line) for line in logger.path.read_text().splitlines()]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["valid_metric"], None)
        self.assertEqual(history[0]["valid_loss"], 1.2)
        self.assertEqual(history[0]["prediction_change_rate"], 0.0)
        self.assertLess(optimizer.param_groups[0]["lr"], 0.1)
        self.assertFalse(stopper.step(1.0))
        self.assertFalse(stopper.step(1.0))
        self.assertTrue(stopper.step(1.0))


if __name__ == "__main__":
    unittest.main()
