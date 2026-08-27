"""Shared RGB + dense-flow spatial encoders for temporal video stages."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18


DEFAULT_FLOW_PHYSICS_FEATURES = ("mean_dx", "mean_dy", "std_dx", "std_dy")
TWO_STREAM_ARCHITECTURE_VERSION = 2


def _flow_roi_start(height: int, top_ratio: float) -> int:
    if height < 1:
        raise ValueError("flow height must be >= 1")
    if not 0.0 <= top_ratio < 1.0:
        raise ValueError("flow_roi_top_ratio must be in [0, 1)")
    return min(int(height * top_ratio), height - 1)


def mask_flow_roi(flow: torch.Tensor, *, top_ratio: float) -> torch.Tensor:
    """Zero Flow above a bottom ROI without changing shape or dtype."""

    if flow.ndim < 4 or flow.shape[-3] != 2:
        raise ValueError("flow must end with [2, height, width]")
    roi_start = _flow_roi_start(flow.shape[-2], top_ratio)
    if roi_start == 0:
        return flow
    mask = flow.new_zeros((1,) * (flow.ndim - 2) + (flow.shape[-2], 1))
    mask[..., roi_start:, :] = 1
    return flow * mask


def flow_roi_statistics(
    flow: torch.Tensor,
    *,
    top_ratio: float,
    feature_names: Sequence[str] = DEFAULT_FLOW_PHYSICS_FEATURES,
) -> torch.Tensor:
    """Return normalized Flow statistics using only pixels inside the ROI."""

    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError("flow must have shape [batch, time, 2, height, width]")
    if not bool(torch.isfinite(flow).all()):
        raise ValueError("flow must contain only finite values")
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("physics feature names must be non-empty and unique")
    unsupported = set(names) - set(DEFAULT_FLOW_PHYSICS_FEATURES)
    if unsupported:
        raise ValueError(f"unsupported physics features: {sorted(unsupported)}")
    roi_start = _flow_roi_start(flow.shape[-2], top_ratio)
    pixels = flow[..., roi_start:, :].flatten(start_dim=-2)
    dx = pixels[:, :, 0]
    dy = pixels[:, :, 1]
    values = {
        "mean_dx": dx.mean(dim=-1),
        "mean_dy": dy.mean(dim=-1),
        "std_dx": dx.std(dim=-1, unbiased=False),
        "std_dy": dy.std(dim=-1, unbiased=False),
    }
    return torch.stack([values[name] for name in names], dim=-1)


def spatial_grid_pool(feature_maps: torch.Tensor, *, grid_size: int) -> torch.Tensor:
    """Pool a CNN map into a fixed row-major spatial grid and flatten it."""

    if feature_maps.ndim != 4:
        raise ValueError("feature_maps must have shape [batch, channels, height, width]")
    if grid_size < 1:
        raise ValueError("flow_grid_size must be >= 1")
    pooled = functional.adaptive_avg_pool2d(feature_maps, (grid_size, grid_size))
    return pooled.flatten(start_dim=1)


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
        flow_grid_size: int = 1,
        flow_roi_top_ratio: float = 0.0,
        flow_roi_mode: str = "mask",
        flow_projection_dim: int | None = None,
        use_physics_vector: bool = False,
        physics_features: Sequence[str] = DEFAULT_FLOW_PHYSICS_FEATURES,
        physics_projection_dim: int = 32,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or layers < 1 or frame_batch_size < 1:
            raise ValueError("hidden_size, layers, and frame_batch_size must be >= 1")
        if flow_grid_size < 1:
            raise ValueError("flow_grid_size must be >= 1")
        _flow_roi_start(1, flow_roi_top_ratio)
        if flow_roi_mode != "mask":
            raise ValueError("flow_roi_mode must be 'mask'")
        if flow_projection_dim is not None and flow_projection_dim < 1:
            raise ValueError("flow_projection_dim must be >= 1")
        if physics_projection_dim < 1:
            raise ValueError("physics_projection_dim must be >= 1")
        physics_features = tuple(physics_features)
        unsupported = set(physics_features) - set(DEFAULT_FLOW_PHYSICS_FEATURES)
        if not physics_features or len(set(physics_features)) != len(physics_features) or unsupported:
            raise ValueError("physics_features must be unique supported Flow statistics")
        self.frame_batch_size = frame_batch_size
        self.flow_grid_size = flow_grid_size
        self.flow_roi_top_ratio = float(flow_roi_top_ratio)
        self.flow_roi_mode = flow_roi_mode
        self.use_physics_vector = use_physics_vector
        self.physics_features = physics_features
        self.physics_projection_dim = physics_projection_dim

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

        resolved_flow_projection_dim = flow_projection_dim or feature_size
        if flow_grid_size == 1 and resolved_flow_projection_dim != feature_size:
            raise ValueError("flow_projection_dim must equal the ResNet feature size when flow_grid_size=1")
        self.flow_projection_dim = resolved_flow_projection_dim
        if flow_grid_size > 1:
            self.flow_projection: nn.Linear | None = nn.Linear(
                feature_size * flow_grid_size * flow_grid_size,
                resolved_flow_projection_dim,
            )
            self.flow_projection_norm: nn.LayerNorm | None = nn.LayerNorm(resolved_flow_projection_dim)
        else:
            # Keeping these absent from the default state dict preserves strict
            # loading for the frozen Stage 2 and legacy 1x1 checkpoints.
            self.flow_projection = None
            self.flow_projection_norm = None

        if use_physics_vector:
            self.physics_projection: nn.Sequential | None = nn.Sequential(
                nn.Linear(len(physics_features), physics_projection_dim),
                nn.GELU(),
                nn.LayerNorm(physics_projection_dim),
            )
        else:
            self.physics_projection = None

        self.temporal = nn.LSTM(
            feature_size
            + resolved_flow_projection_dim
            + (physics_projection_dim if use_physics_vector else 0),
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
    def _resnet_convolution_trunk(encoder: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        features = encoder.conv1(inputs)
        features = encoder.bn1(features)
        features = encoder.relu(features)
        features = encoder.maxpool(features)
        features = encoder.layer1(features)
        features = encoder.layer2(features)
        features = encoder.layer3(features)
        return encoder.layer4(features)

    def _encode_flow_frames(self, flow: torch.Tensor) -> torch.Tensor:
        roi_flow = mask_flow_roi(flow, top_ratio=self.flow_roi_top_ratio)
        if self.flow_grid_size == 1:
            return self._encode_frames(self.temporal_flow_encoder, roi_flow)
        batch, time, channels, height, width = roi_flow.shape
        flat = roi_flow.reshape(batch * time, channels, height, width)
        if self.flow_projection is None or self.flow_projection_norm is None:
            raise RuntimeError("grid Flow encoder is missing its projection modules")
        projected: list[torch.Tensor] = []
        for chunk in flat.split(self.frame_batch_size):
            maps = self._resnet_convolution_trunk(self.temporal_flow_encoder, chunk)
            pooled = spatial_grid_pool(maps, grid_size=self.flow_grid_size)
            projected.append(self.flow_projection_norm(self.flow_projection(pooled)))
        return torch.cat(projected, dim=0).reshape(batch, time, self.flow_projection_dim)

    def encode_physics_vector(self, flow: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Project ROI statistics and force padding positions back to zero."""

        if self.physics_projection is None:
            raise RuntimeError("physics vector support is disabled")
        if valid_mask.shape != flow.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [batch, time]")
        statistics = flow_roi_statistics(
            flow,
            top_ratio=self.flow_roi_top_ratio,
            feature_names=self.physics_features,
        )
        embedded = self.physics_projection(statistics)
        return embedded.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    def architecture_metadata(self) -> dict[str, object]:
        """Return checkpoint-safe architecture details for exact recreation."""

        return {
            "version": TWO_STREAM_ARCHITECTURE_VERSION,
            "flow_roi_mode": self.flow_roi_mode,
            "flow_roi_top_ratio": self.flow_roi_top_ratio,
            "flow_grid_size": self.flow_grid_size,
            "flow_projection_dim": self.flow_projection_dim,
            "use_physics_vector": self.use_physics_vector,
            "physics_features": list(self.physics_features),
            "physics_projection_dim": self.physics_projection_dim,
        }

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

        valid_mask = self._valid_mask(lengths, time)
        rgb_features = self._encode_frames(self.spatial_encoder, (frames - self.rgb_mean) / self.rgb_std)
        flow_features = self._encode_flow_frames(flow)
        feature_parts = [rgb_features, flow_features]
        if self.use_physics_vector:
            feature_parts.append(self.encode_physics_vector(flow, valid_mask))
        fused = torch.cat(feature_parts, dim=-1)
        packed = pack_padded_sequence(
            fused,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.temporal(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=time)
        return hidden, valid_mask, lengths
