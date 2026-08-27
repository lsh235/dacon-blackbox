"""Seq2Seq RGB + dense-flow model skeleton for Stage 3 motion labels."""

from __future__ import annotations

import torch
from torch import nn

from blackbox.stages.two_stream import TwoStreamBiLSTMEncoder


ACCEL_CLASS_COUNT = 4
STEER_CLASS_COUNT = 3


class Stage3TwoStreamBiLSTM(TwoStreamBiLSTMEncoder):
    """Predict acceleration and steering logits at every valid sequence step.

    The model deliberately emits one result per supplied source-frame position.
    Converting those positions to the official 0.1-second ``sample_index`` is
    a separate, explicit time-axis operation because the supplied documents do
    not yet reconcile source FPS with that index.
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
