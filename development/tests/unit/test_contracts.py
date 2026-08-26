from __future__ import annotations

import unittest

import pandas as pd

from blackbox.contracts import ContractError, validate_prediction_frame


class PredictionContractTests(unittest.TestCase):
    def test_stage1_accepts_documented_values(self) -> None:
        frame = pd.DataFrame(
            [
                {"ID": "A", "answer": "ORIGINAL"},
                {"ID": "B", "answer": "RERECORDED"},
            ]
        )
        self.assertIs(validate_prediction_frame("stage1", frame), frame)

    def test_rejects_wrong_column_order(self) -> None:
        frame = pd.DataFrame([{"answer": "ORIGINAL", "ID": "A"}])
        with self.assertRaisesRegex(ContractError, "columns must be"):
            validate_prediction_frame(1, frame)

    def test_stage2_rejects_negative_frame(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ID": "A",
                    "collision_frame": -1,
                    "entry_frame": 0,
                    "evasion_space": 0,
                    "entry_side": "LEFT",
                }
            ]
        )
        with self.assertRaisesRegex(ContractError, "non-negative integers"):
            validate_prediction_frame("stage2", frame)

    def test_stage3_requires_contiguous_sample_index(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ID": "A",
                    "sample_index": 0,
                    "accel_label": "CONSTANT",
                    "steer_label": "STRAIGHT",
                },
                {
                    "ID": "A",
                    "sample_index": 2,
                    "accel_label": "ACCELERATING",
                    "steer_label": "RIGHT",
                },
            ]
        )
        with self.assertRaisesRegex(ContractError, "contiguous from 0"):
            validate_prediction_frame("stage3", frame)

    def test_stage3_rejects_unknown_label(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ID": "A",
                    "sample_index": 0,
                    "accel_label": "UNKNOWN",
                    "steer_label": "STRAIGHT",
                }
            ]
        )
        with self.assertRaisesRegex(ContractError, "accel_label"):
            validate_prediction_frame(3, frame)


if __name__ == "__main__":
    unittest.main()
