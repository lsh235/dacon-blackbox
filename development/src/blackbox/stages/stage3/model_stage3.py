"""Seq2Seq RGB + dense-flow model skeleton for Stage 3 motion labels."""

from __future__ import annotations

import torch
from torch import nn

from blackbox.stages.two_stream import TwoStreamBiLSTMEncoder


ACCEL_CLASS_COUNT = 4
STEER_CLASS_COUNT = 3


class Stage3TwoStreamBiLSTM(TwoStreamBiLSTMEncoder):
    """Predict acceleration and steering logits at every valid sequence step.

    The Stage 3 dataset supplies one pooled RGB/flow map per metadata-derived
    0.1-second step, so each output position is a candidate official sample.
    It also propagates a metadata/label conflict flag where the public example
    cannot reconcile its reported FPS with sparse source-frame annotations.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 192,
        layers: int = 2,
        frame_batch_size: int = 8,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            layers=layers,
            frame_batch_size=frame_batch_size,
        )
        self.accel_head = nn.Linear(self.temporal_size, ACCEL_CLASS_COUNT)
        self.steer_head = nn.Linear(self.temporal_size, STEER_CLASS_COUNT)

    def forward(
        self,
        frames: torch.Tensor,
        flow: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden, valid_mask, _ = self.encode_sequence(frames, flow, valid_lengths)
        invalid = ~valid_mask.unsqueeze(-1)
        return {
            "accel_logits": self.accel_head(hidden).masked_fill(invalid, float("-inf")),
            "steer_logits": self.steer_head(hidden).masked_fill(invalid, float("-inf")),
            "valid_mask": valid_mask,
        }
