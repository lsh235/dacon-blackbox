from __future__ import annotations

import unittest

import torch

from blackbox.stages.stage2.dataset_stage2 import IGNORE_INDEX, local_event_target, sliding_window_starts
from blackbox.stages.stage2.model_stage2 import Stage2CnnBiLSTM


class Stage2SlidingWindowTests(unittest.TestCase):
    def test_tail_window_is_retained(self) -> None:
        self.assertEqual(sliding_window_starts(100, 64, 32), [0, 32, 36])
        self.assertEqual(sliding_window_starts(10, 64, 32), [0])

    def test_local_target_never_relabels_an_event_outside_its_window(self) -> None:
        self.assertEqual(local_event_target(35, start_frame=32, valid_length=16), 3)
        self.assertEqual(
            local_event_target(31, start_frame=32, valid_length=16),
            IGNORE_INDEX,
        )
        self.assertEqual(
            local_event_target(-1, start_frame=32, valid_length=16),
            IGNORE_INDEX,
        )


class Stage2SequenceModelTests(unittest.TestCase):
    def test_cnn_bilstm_returns_window_local_contract(self) -> None:
        model = Stage2CnnBiLSTM(hidden_size=8, layers=1, frame_batch_size=1).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 2, 3, 64, 64), torch.tensor([1]))

        self.assertEqual(tuple(output["collision_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["entry_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["evasion_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["entry_side_logits"].shape), (1, 2))
        self.assertTrue(torch.isneginf(output["collision_logits"][0, 1]).item())


if __name__ == "__main__":
    unittest.main()
