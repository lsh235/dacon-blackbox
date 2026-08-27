from __future__ import annotations

import unittest

import numpy as np

from blackbox.stages.stage3.baseline import ACCEL_LABELS, STEER_LABELS
from blackbox.stages.stage3.postprocessing import (
    ACCEL_TRANSITIONS,
    STEER_TRANSITIONS,
    constrain_stage3_scores,
    transition_mask,
    viterbi_transition_logits,
)


class Stage3TransitionConstraintTests(unittest.TestCase):
    def test_acceleration_path_blocks_direct_accel_to_decel_flip(self) -> None:
        probabilities = np.asarray(
            [
                [0.95, 0.01, 0.03, 0.01],
                [0.01, 0.95, 0.03, 0.01],
                [0.01, 0.95, 0.03, 0.01],
            ],
            dtype=np.float32,
        )

        constrained = viterbi_transition_logits(
            probabilities,
            allowed_transitions=transition_mask(ACCEL_LABELS, ACCEL_TRANSITIONS),
        )
        path = constrained.argmax(axis=1)

        self.assertFalse(
            any(
                (left, right) in {(0, 1), (1, 0)}
                for left, right in zip(path, path[1:])
            )
        )
        self.assertTrue(np.all(constrained[np.arange(3), path] > -1e9))

    def test_steering_path_requires_straight_between_opposite_turns(self) -> None:
        probabilities = np.asarray(
            [
                [0.98, 0.01, 0.01],
                [0.01, 0.05, 0.94],
                [0.01, 0.05, 0.94],
            ],
            dtype=np.float32,
        )

        path = viterbi_transition_logits(
            probabilities,
            allowed_transitions=transition_mask(STEER_LABELS, STEER_TRANSITIONS),
        ).argmax(axis=1)

        self.assertNotIn((0, 2), list(zip(path, path[1:])))
        self.assertNotIn((2, 0), list(zip(path, path[1:])))

    def test_stage3_constraint_preserves_output_shapes(self) -> None:
        accel = np.full((4, len(ACCEL_LABELS)), 1 / len(ACCEL_LABELS), dtype=np.float32)
        steer = np.full((4, len(STEER_LABELS)), 1 / len(STEER_LABELS), dtype=np.float32)

        constrained = constrain_stage3_scores(
            {"S3_A": (accel, steer)},
            accel_labels=ACCEL_LABELS,
            steer_labels=STEER_LABELS,
        )

        self.assertEqual(constrained["S3_A"][0].shape, accel.shape)
        self.assertEqual(constrained["S3_A"][1].shape, steer.shape)
        self.assertTrue(np.isfinite(constrained["S3_A"][0]).all())


if __name__ == "__main__":
    unittest.main()
