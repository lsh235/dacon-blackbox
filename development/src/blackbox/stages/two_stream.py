"""Shared RGB + dense-flow spatial encoders for temporal video stages."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18


class TwoStreamBiLSTMEncoder(nn.Module):
    """Encode aligned RGB and dense ``(dx, dy)`` maps into valid time states.

    Both inputs retain their two-dimensional image grids through independent
    ResNet residual stages.  In particular, the flow stream never reduces a
    window to a scalar global-motion statistic before its spatial CNN.  The
    final per-frame CNN embeddings are fused only at matching local time steps,
    then passed to a BiLSTM shared by Stage 2 and Stage 3 research models.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 192,
        layers: int = 2,
        frame_batch_size: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or layers < 1 or frame_batch_size < 1:
            raise ValueError("hidden_size, layers, and frame_batch_size must be >= 1")
        self.frame_batch_size = frame_batch_size

        spatial_encoder = resnet18(weights=None)
        feature_size = spatial_encoder.fc.in_features
        spatial_encoder.fc = nn.Identity()
        self.spatial_encoder = spatial_encoder

        flow_encoder = resnet18(weights=None)
        flow_encoder.conv1 = nn.Conv2d(
            2,
            flow_encoder.conv1.out_channels,
            kernel_size=flow_encoder.conv1.kernel_size,
            stride=flow_encoder.conv1.stride,
            padding=flow_encoder.conv1.padding,
            bias=False,
        )
        flow_encoder.fc = nn.Identity()
        self.temporal_flow_encoder = flow_encoder

        self.temporal = nn.LSTM(
            feature_size * 2,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if layers > 1 else 0.0,
        )
        self.temporal_size = hidden_size * 2
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225])[None, None, :, None, None])

    def _encode_frames(self, encoder: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = inputs.shape
        flat = inputs.reshape(batch * time, channels, height, width)
        features = [encoder(chunk) for chunk in flat.split(self.frame_batch_size)]
        return torch.cat(features, dim=0).reshape(batch, time, -1)

    @staticmethod
    def _valid_mask(lengths: torch.Tensor, time: int) -> torch.Tensor:
        return torch.arange(time, device=lengths.device)[None, :] < lengths[:, None]

    def encode_sequence(
        self,
        frames: torch.Tensor,
        flow: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return padded hidden states, a valid mask, and clamped lengths."""

        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(
                "frames must have shape [batch, time, 3, height, width], "
                f"got {tuple(frames.shape)}"
            )
        if flow.ndim != 5 or flow.shape[2] != 2:
            raise ValueError(
                "flow must have shape [batch, time, 2, height, width], "
                f"got {tuple(flow.shape)}"
            )
        if flow.shape[:2] != frames.shape[:2] or flow.shape[3:] != frames.shape[3:]:
            raise ValueError("flow and frames must have matching batch/time/spatial dimensions")
        if valid_lengths.ndim != 1 or valid_lengths.shape[0] != frames.shape[0]:
            raise ValueError("valid_lengths must have one positive length per batch item")
        _, time = frames.shape[:2]
        lengths = valid_lengths.to(device=frames.device, dtype=torch.long).clamp(max=time)
        if bool((lengths < 1).any()):
            raise ValueError("valid_lengths must be >= 1")
        if not bool(torch.isfinite(flow).all()):
            raise ValueError("flow must contain only finite values")

        rgb_features = self._encode_frames(self.spatial_encoder, (frames - self.rgb_mean) / self.rgb_std)
        flow_features = self._encode_frames(self.temporal_flow_encoder, flow)
        fused = torch.cat([rgb_features, flow_features], dim=-1)
        packed = pack_padded_sequence(
            fused,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.temporal(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=time)
        return hidden, self._valid_mask(lengths, time), lengths
