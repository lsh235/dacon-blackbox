"""CNN + BiLSTM window model for the Stage 2 research skeleton."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18


class Stage2CnnBiLSTM(nn.Module):
    """Encode frame chunks, then predict local event and scene logits.

    Input: ``frames`` is ``[batch, time, 3, height, width]`` in RGB [0, 1].
    Output frame logits are window-local.  Caller must use the dataset's
    ``frame_numbers`` to map an argmax to the official original frame number.

    CNN features are computed in small batches, so a 64-frame window does not
    require materializing all ResNet activations at once.
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
        encoder = resnet18(weights=None)
        feature_size = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.frame_encoder = encoder
        self.temporal = nn.LSTM(
            feature_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if layers > 1 else 0.0,
        )
        temporal_size = hidden_size * 2
        self.collision_head = nn.Linear(temporal_size, 1)
        self.entry_head = nn.Linear(temporal_size, 1)
        self.evasion_head = nn.Linear(temporal_size, 2)
        self.entry_side_head = nn.Linear(temporal_size, 2)
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225])[None, None, :, None, None])

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        features = [self.frame_encoder(chunk) for chunk in flat.split(self.frame_batch_size)]
        return torch.cat(features, dim=0).reshape(batch, time, -1)

    @staticmethod
    def _valid_mask(lengths: torch.Tensor, time: int) -> torch.Tensor:
        return torch.arange(time, device=lengths.device)[None, :] < lengths[:, None]

    def forward(self, frames: torch.Tensor, valid_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(
                "frames must have shape [batch, time, 3, height, width], "
                f"got {tuple(frames.shape)}"
            )
        if valid_lengths.ndim != 1 or valid_lengths.shape[0] != frames.shape[0]:
            raise ValueError("valid_lengths must have one positive length per batch item")
        batch, time = frames.shape[:2]
        lengths = valid_lengths.to(device=frames.device, dtype=torch.long).clamp(max=time)
        if bool((lengths < 1).any()):
            raise ValueError("valid_lengths must be >= 1")

        normalized = (frames - self.rgb_mean) / self.rgb_std
        features = self._encode_frames(normalized)
        packed = pack_padded_sequence(
            features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.temporal(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=time)
        mask = self._valid_mask(lengths, time)
        invalid = ~mask
        collision_logits = self.collision_head(hidden).squeeze(-1).masked_fill(invalid, float("-inf"))
        entry_logits = self.entry_head(hidden).squeeze(-1).masked_fill(invalid, float("-inf"))
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        return {
            "collision_logits": collision_logits,
            "entry_logits": entry_logits,
            "evasion_logits": self.evasion_head(pooled),
            "entry_side_logits": self.entry_side_head(pooled),
            "valid_mask": mask,
        }


class Stage2TwoStreamBiLSTM(nn.Module):
    """Fuse RGB spatial and Farneback-flow CNN features before a BiLSTM.

    ``frames`` is ``[B, T, 3, H, W]`` in RGB ``[0, 1]`` and ``flow`` is
    ``[B, T, 2, H, W]`` in normalized ``(dx, dy)`` ``[-1, 1]``.  The two CNNs
    process matching local time indices, then their features are concatenated
    before sequence modelling.  This class deliberately leaves the existing
    RGB-only ``Stage2CnnBiLSTM`` untouched for a controlled ablation.
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

        temporal_encoder = resnet18(weights=None)
        temporal_encoder.conv1 = nn.Conv2d(
            2,
            temporal_encoder.conv1.out_channels,
            kernel_size=temporal_encoder.conv1.kernel_size,
            stride=temporal_encoder.conv1.stride,
            padding=temporal_encoder.conv1.padding,
            bias=False,
        )
        temporal_encoder.fc = nn.Identity()
        self.temporal_flow_encoder = temporal_encoder

        self.temporal = nn.LSTM(
            feature_size * 2,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.15 if layers > 1 else 0.0,
        )
        temporal_size = hidden_size * 2
        self.collision_head = nn.Linear(temporal_size, 1)
        self.entry_head = nn.Linear(temporal_size, 1)
        self.evasion_head = nn.Linear(temporal_size, 2)
        self.entry_side_head = nn.Linear(temporal_size, 2)
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

    def forward(
        self,
        frames: torch.Tensor,
        flow: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        batch, time = frames.shape[:2]
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
        mask = self._valid_mask(lengths, time)
        invalid = ~mask
        collision_logits = self.collision_head(hidden).squeeze(-1).masked_fill(invalid, float("-inf"))
        entry_logits = self.entry_head(hidden).squeeze(-1).masked_fill(invalid, float("-inf"))
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        return {
            "collision_logits": collision_logits,
            "entry_logits": entry_logits,
            "evasion_logits": self.evasion_head(pooled),
            "entry_side_logits": self.entry_side_head(pooled),
            "valid_mask": mask,
        }
