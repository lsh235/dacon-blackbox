"""Overlap-aware inference helpers for the experimental Stage 2 Two-Stream model.

They are deliberately not wired into ``predict_stage2`` yet: that submission
path consumes the baseline image-folder input and baseline checkpoint format.
This module keeps the RGB+Flow research contract explicit until end-to-end
submission validation is authorized and completed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import torch
from torch.utils.data import DataLoader

from blackbox.stages.stage2.model_stage2 import Stage2TwoStreamBiLSTM
from blackbox.stages.stage2.train_stage2 import select_aggregated_event_frame


@torch.inference_mode()
def predict_two_stream_event_frames(
    model: Stage2TwoStreamBiLSTM,
    loader: DataLoader,
    *,
    device: torch.device,
    event: Literal["collision", "entry"] = "collision",
    aggregation_policy: Literal["mean", "max"] = "mean",
) -> dict[str, int]:
    """Map each video window's probability peak back onto its original axis.

    Local padded positions are assigned ``-inf`` by the model.  Before global
    selection, each valid local softmax score is associated with the supplied
    ``frame_numbers`` and overlapping occurrences of the same original frame
    are aggregated using the configured policy.
    """

    logits_key = f"{event}_logits"
    model.eval()
    per_video: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"scores": [], "frame_numbers": [], "valid_lengths": []}
    )
    for batch in loader:
        outputs = model(
            batch["frames"].to(device, non_blocking=True),
            batch["flow"].to(device, non_blocking=True),
            batch["valid_length"].to(device, non_blocking=True),
        )
        probabilities = torch.softmax(outputs[logits_key], dim=1).cpu()
        frame_numbers = batch["frame_numbers"].cpu()
        valid_lengths = batch["valid_length"].cpu()
        for index, video_id in enumerate(batch["id"]):
            current = per_video[str(video_id)]
            current["scores"].append(probabilities[index])
            current["frame_numbers"].append(frame_numbers[index])
            current["valid_lengths"].append(valid_lengths[index].reshape(1))

    return {
        video_id: select_aggregated_event_frame(
            torch.stack(parts["scores"]),
            torch.stack(parts["frame_numbers"]),
            torch.cat(parts["valid_lengths"]),
            policy=aggregation_policy,
        )
        for video_id, parts in per_video.items()
    }
