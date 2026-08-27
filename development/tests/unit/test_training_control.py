from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from blackbox.training_control import EarlyStopping, JsonlTrainingLogger, cosine_scheduler


class TrainingControlTests(unittest.TestCase):
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
            )
            (parameter.square()).backward()
            optimizer.step()
            scheduler.step()
            history = [json.loads(line) for line in logger.path.read_text().splitlines()]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["valid_metric"], None)
        self.assertLess(optimizer.param_groups[0]["lr"], 0.1)
        self.assertFalse(stopper.step(1.0))
        self.assertFalse(stopper.step(1.0))
        self.assertTrue(stopper.step(1.0))


if __name__ == "__main__":
    unittest.main()
