"""Stage 1 RGB, spatial-frequency, and flicker multi-stream classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s

from blackbox.common.runtime import (
    CheckpointError,
    DEFAULT_SEED,
    autocast_context,
    choose_device,
    load_checkpoint,
    make_grad_scaler,
    release_device_cache,
    seed_everything,
    video_paths,
)
from blackbox.contracts import validate_prediction_frame
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT
from blackbox.training_control import (
    EarlyStopping,
    JsonlTrainingLogger,
    TrainingControlConfig,
    group_holdout_indices,
    macro_f1_score,
)
from blackbox.stages.stage1.dataset import (
    DEFAULT_FEATURE_MODE,
    DEFAULT_FFT_SIZE,
    DEFAULT_FORENSIC_SIZE,
    DEFAULT_JITTER_FRAMES,
    DEFAULT_ROW_PROFILE_BINS,
    DEFAULT_TEMPORAL_SLOTS,
    DEFAULT_TRAIN_SEQUENCE_LENGTHS,
    RGB_FFT_FEATURES,
    Stage1InferenceDataset,
    Stage1TrainAugmentation,
    Stage1TrainingDataset,
    Stage1ValidationDataset,
    feature_channels,
)
from blackbox.stages.stage1.losses import Stage1MultiTaskLoss


LABEL_TO_INDEX = {"ORIGINAL": 0, "RERECORDED": 1}
DEFAULT_TTA_SLOTS = DEFAULT_TEMPORAL_SLOTS
STAGE1_ARCHITECTURE = "stage1_rgb_fft_flicker_corr_gru_gated_mstcn_ablation_v5"
DEFAULT_CORRELATION_SCALES = (1, 2, 4, 8)
DEFAULT_CORRELATION_RADIUS = 2
DEFAULT_MOTION_ITERATIONS = 3
DEFAULT_MSTCN_STAGES = 3
DEFAULT_TEMPORAL_REFINEMENT_MODE = "gated_mstcn"
DEFAULT_MSTCN_GATE_INITIAL = 0.1
DEFAULT_DETACH_MSTCN_INPUT = True
DEFAULT_BASE_INITIALIZATION_SEED = 42
TEMPORAL_REFINEMENT_MODES = {"single_stage", "gated_mstcn"}
DEFAULT_MAX_EPOCHS = 30
DEFAULT_MINIMUM_EPOCHS = 10
DEFAULT_EARLY_STOPPING_PATIENCE = 7
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_MVIT_INPUT_FRAMES = 16
DEFAULT_BACKBONE_LEARNING_RATE = 1e-5
DEFAULT_AUXILIARY_LEARNING_RATE = 1e-4
DEFAULT_WARMUP_INITIAL_LEARNING_RATE = 1e-6
DEFAULT_MINIMUM_LEARNING_RATE = 1e-6


def inverse_frequency_focal_alpha(
    targets: Sequence[int],
    *,
    num_classes: int = len(LABEL_TO_INDEX),
) -> tuple[float, ...]:
    """Return mean-one inverse-frequency weights for one training split only."""

    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    parsed_targets = torch.as_tensor(list(targets), dtype=torch.long)
    if parsed_targets.ndim != 1 or parsed_targets.numel() == 0:
        raise ValueError("targets must be a non-empty one-dimensional sequence")
    if bool((parsed_targets < 0).any()) or bool((parsed_targets >= num_classes).any()):
        raise ValueError(f"targets must be in [0, {num_classes - 1}]")
    counts = torch.bincount(parsed_targets, minlength=num_classes).to(torch.float64)
    missing = [index for index, count in enumerate(counts.tolist()) if count == 0]
    if missing:
        raise ValueError(
            "inverse-frequency Focal alpha requires every class in the training split; "
            f"missing class indices={missing}"
        )
    weights = parsed_targets.numel() / (float(num_classes) * counts)
    return tuple(float(value) for value in weights.tolist())


def resolve_tta_slots(checkpoint: dict[str, object]) -> int:
    """Read temporal region count while validating checkpoint metadata."""

    sampling = checkpoint.get("sampling", {})
    if not isinstance(sampling, dict):
        raise CheckpointError("Stage 1 checkpoint sampling metadata must be a mapping")
    value = sampling.get("inference_tta_slots", DEFAULT_TTA_SLOTS)
    try:
        slots = int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"Stage 1 inference_tta_slots must be an integer, got {value!r}") from exc
    if slots < 1:
        raise CheckpointError(f"Stage 1 inference_tta_slots must be >= 1, got {slots}")
    return slots


class SpatialFftEncoder(nn.Module):
    """Encode each frame's FFT map with a 2-D CNN, then pool over time."""

    output_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, self.output_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, self.output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, fft_clip: torch.Tensor) -> torch.Tensor:
        if fft_clip.ndim != 5 or fft_clip.shape[1] != 3:
            raise ValueError(
                "fft_clip must have shape [batch, 3, time, height, width], "
                f"got {tuple(fft_clip.shape)}"
            )
        batch, channels, frames, height, width = fft_clip.shape
        per_frame = fft_clip.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            channels,
            height,
            width,
        )
        encoded = self.frame_encoder(per_frame).flatten(1)
        return encoded.reshape(batch, frames, self.output_dim).mean(dim=1)


class DilatedTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class FlickerTemporalEncoder(nn.Module):
    """Model periodic luminance and rolling-row signals with dilated Conv1d."""

    output_dim = 128

    def __init__(self, input_channels: int, *, hidden_channels: int = 64) -> None:
        super().__init__()
        if input_channels < 3:
            raise ValueError("flicker input needs Y(t), delta Y(t), and row profiles")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DilatedTemporalBlock(hidden_channels, dilation) for dilation in (1, 2, 4)]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, self.output_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def encode_sequence(self, flicker: torch.Tensor) -> torch.Tensor:
        if flicker.ndim != 3 or flicker.shape[1] != self.input_channels:
            raise ValueError(
                f"flicker must have shape [batch, {self.input_channels}, time], "
                f"got {tuple(flicker.shape)}"
            )
        encoded = self.input_projection(flicker)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded

    def forward(self, flicker: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(flicker)
        return self.pool_sequence(encoded)

    def pool_sequence(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3 or encoded.shape[1] != self.hidden_channels:
            raise ValueError("encoded flicker sequence has incompatible shape")
        pooled = torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=1)
        return self.output_projection(pooled)


class TemporalClassificationStage(nn.Module):
    """One dilated TCN stage producing frame-wise class logits."""

    def __init__(
        self,
        input_channels: int,
        *,
        hidden_channels: int = 64,
        classes: int = 2,
    ) -> None:
        super().__init__()
        if min(input_channels, hidden_channels, classes) < 1:
            raise ValueError("MS-TCN channel and class counts must be >= 1")
        self.input_channels = int(input_channels)
        self.classes = int(classes)
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DilatedTemporalBlock(hidden_channels, dilation) for dilation in (1, 2, 4)]
        )
        self.classifier = nn.Conv1d(hidden_channels, classes, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_channels:
            raise ValueError(
                "MS-TCN stage input must have shape [batch, time, channels], "
                f"expected channels={self.input_channels}, got {tuple(sequence.shape)}"
            )
        encoded = self.input_projection(sequence.transpose(1, 2))
        for block in self.blocks:
            encoded = block(encoded)
        return self.classifier(encoded).transpose(1, 2).contiguous()


class MultiStageTemporalRefinementHead(nn.Module):
    """Refine labels using only the previous stage's class probabilities.

    Visual/forensic frame features are consumed exclusively by ``initial_stage``.
    Every later stage has exactly ``classes`` input channels and receives only
    ``softmax(previous_logits)``.  This prevents later refinement from learning
    a shortcut through camera or background appearance features.
    """

    def __init__(
        self,
        input_channels: int,
        *,
        stages: int = DEFAULT_MSTCN_STAGES,
        hidden_channels: int = 64,
        classes: int = 2,
    ) -> None:
        super().__init__()
        if stages < 2:
            raise ValueError("MS-TCN requires at least two stages")
        self.stages = int(stages)
        self.classes = int(classes)
        self.initial_stage = TemporalClassificationStage(
            input_channels,
            hidden_channels=hidden_channels,
            classes=classes,
        )
        self.refinement_stages = nn.ModuleList(
            TemporalClassificationStage(
                classes,
                hidden_channels=hidden_channels,
                classes=classes,
            )
            for _ in range(stages - 1)
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        logits = self.initial_stage(frame_features)
        stage_logits = [logits]
        for refinement in self.refinement_stages:
            probabilities_only = torch.softmax(logits, dim=-1)
            logits = refinement(probabilities_only)
            stage_logits.append(logits)
        return torch.stack(stage_logits, dim=1)


class AllPairsCorrelationPyramid(nn.Module):
    """Build all-pairs HxWxHxW correlations and pool target coordinates."""

    def __init__(
        self,
        *,
        scales: Sequence[int] = DEFAULT_CORRELATION_SCALES,
        max_positions: int = 1024,
    ) -> None:
        super().__init__()
        parsed = tuple(int(scale) for scale in scales)
        if not parsed or any(scale < 1 for scale in parsed):
            raise ValueError("correlation scales must be positive integers")
        if len(set(parsed)) != len(parsed):
            raise ValueError("correlation scales must be unique")
        if max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        self.scales = parsed
        self.max_positions = int(max_positions)

    def forward(
        self,
        target_features: torch.Tensor,
        source_features: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if target_features.shape != source_features.shape or target_features.ndim != 4:
            raise ValueError(
                "correlation inputs must share shape [batch, channels, height, width]"
            )
        batch, _, height, width = target_features.shape
        positions = height * width
        if positions > self.max_positions:
            raise ValueError(
                "all-pairs correlation feature map is too large: "
                f"{height}x{width}={positions} > {self.max_positions}; downsample first"
            )
        if min(height, width) < max(self.scales):
            raise ValueError(
                f"correlation feature map must be at least {max(self.scales)}x"
                f"{max(self.scales)}, got {height}x{width}"
            )
        target = F.normalize(target_features.float(), dim=1)
        source = F.normalize(source_features.float(), dim=1)
        volume = torch.einsum("bcyx,bcij->byxij", target, source).to(
            target_features.dtype
        )
        flattened = volume.reshape(batch * height * width, 1, height, width)
        pyramid: list[torch.Tensor] = []
        for scale in self.scales:
            pooled = (
                flattened
                if scale == 1
                else F.avg_pool2d(flattened, kernel_size=scale, stride=scale)
            )
            pyramid.append(
                pooled.reshape(
                    batch,
                    height,
                    width,
                    pooled.shape[-2],
                    pooled.shape[-1],
                )
            )
        return tuple(pyramid)

    def lookup(
        self,
        pyramid: Sequence[torch.Tensor],
        coordinates: torch.Tensor,
        *,
        radius: int = DEFAULT_CORRELATION_RADIUS,
    ) -> torch.Tensor:
        """Bilinearly sample local target-coordinate neighborhoods at each scale."""

        if len(pyramid) != len(self.scales):
            raise ValueError("correlation pyramid level count does not match configured scales")
        if coordinates.ndim != 4 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape [batch, 2, height, width]")
        if radius < 0:
            raise ValueError("correlation lookup radius must be >= 0")
        batch, _, height, width = coordinates.shape
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(-radius, radius + 1, device=coordinates.device),
                torch.arange(-radius, radius + 1, device=coordinates.device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        offsets = offsets[:, [1, 0]].to(coordinates)
        outputs: list[torch.Tensor] = []
        base_coordinates = coordinates.permute(0, 2, 3, 1)
        for scale, volume in zip(self.scales, pyramid):
            if volume.shape[:3] != (batch, height, width):
                raise ValueError("correlation reference dimensions do not match coordinates")
            target_height, target_width = volume.shape[-2:]
            sample_coordinates = (
                base_coordinates[..., None, :] / float(scale)
                + offsets[None, None, None]
            )
            normalized_x = (
                2.0
                * offsets[None, None, None, :, 0].expand_as(
                    sample_coordinates[..., 0]
                )
                if target_width == 1
                else 2.0 * sample_coordinates[..., 0] / (target_width - 1) - 1.0
            )
            normalized_y = (
                2.0
                * offsets[None, None, None, :, 1].expand_as(
                    sample_coordinates[..., 1]
                )
                if target_height == 1
                else 2.0 * sample_coordinates[..., 1] / (target_height - 1) - 1.0
            )
            grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(
                batch * height * width,
                1,
                -1,
                2,
            )
            maps = volume.reshape(batch * height * width, 1, target_height, target_width)
            sampled = F.grid_sample(
                maps,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            outputs.append(
                sampled.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
            )
        return torch.cat(outputs, dim=1).to(coordinates.dtype)


def _coordinate_grid(
    batch: int,
    height: int,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=reference.device, dtype=reference.dtype),
        torch.arange(width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)


class ConvGRUCell(nn.Module):
    """Spatial GRU with sigmoid gates and a bounded tanh candidate state."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        if min(input_channels, hidden_channels) < 1:
            raise ValueError("ConvGRU channel counts must be >= 1")
        combined = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.update_gate = nn.Conv2d(combined, hidden_channels, 3, padding=1)
        self.reset_gate = nn.Conv2d(combined, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(combined, hidden_channels, 3, padding=1)

    def forward(self, hidden: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or inputs.ndim != 4 or hidden.shape[0:1] != inputs.shape[0:1]:
            raise ValueError("ConvGRU hidden and inputs must be 4-D with matching batches")
        if hidden.shape[1] != self.hidden_channels or hidden.shape[-2:] != inputs.shape[-2:]:
            raise ValueError("ConvGRU hidden shape is incompatible with its inputs")
        combined = torch.cat((hidden, inputs), dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        candidate = torch.tanh(
            self.candidate(torch.cat((reset * hidden, inputs), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class RecurrentCorrelationUpdate(nn.Module):
    """Use tied ConvGRU weights to iteratively refine bounded displacement."""

    def __init__(
        self,
        *,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        radius: int = DEFAULT_CORRELATION_RADIUS,
        iterations: int = DEFAULT_MOTION_ITERATIONS,
        max_delta: float = 1.5,
    ) -> None:
        super().__init__()
        if radius < 0 or iterations < 1 or max_delta <= 0.0:
            raise ValueError("radius must be >= 0, iterations >= 1, and max_delta > 0")
        self.radius = radius
        self.iterations = iterations
        self.max_delta = float(max_delta)
        self.correlation = AllPairsCorrelationPyramid()
        lookup_channels = len(self.correlation.scales) * (2 * radius + 1) ** 2
        self.context_projection = nn.Conv2d(feature_channels, hidden_channels, 3, padding=1)
        self.gru = ConvGRUCell(
            lookup_channels + feature_channels + 2,
            hidden_channels,
        )
        self.displacement_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )

    def forward(
        self,
        target_features: torch.Tensor,
        source_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pyramid = self.correlation(target_features, source_features)
        batch, _, height, width = target_features.shape
        base = _coordinate_grid(batch, height, width, target_features)
        displacement = target_features.new_zeros(batch, 2, height, width)
        hidden = torch.tanh(self.context_projection(target_features))
        update_magnitudes: list[torch.Tensor] = []
        for iteration in range(self.iterations):
            local_correlation = self.correlation.lookup(
                pyramid,
                base + displacement,
                radius=self.radius,
            )
            hidden = self.gru(
                hidden,
                torch.cat((local_correlation, target_features, displacement), dim=1),
            )
            damping = 1.0 / float(iteration + 1)
            delta = (
                self.max_delta
                * damping
                * torch.tanh(self.displacement_head(hidden))
            )
            displacement = displacement + delta
            update_magnitudes.append(
                torch.linalg.vector_norm(delta.float().flatten(1), ord=2, dim=1)
            )
        return hidden, displacement, torch.stack(update_magnitudes, dim=1)


class MotionConsistencyEncoder(nn.Module):
    """Track pairwise motion, reconstruct targets, and predict soft masks."""

    output_dim = 128
    feature_channels = 64

    def __init__(
        self,
        *,
        iterations: int = DEFAULT_MOTION_ITERATIONS,
        radius: int = DEFAULT_CORRELATION_RADIUS,
    ) -> None:
        super().__init__()
        self.iterations = iterations
        self.radius = radius
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, self.feature_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, self.feature_channels),
            nn.GELU(),
        )
        self.update_operator = RecurrentCorrelationUpdate(
            feature_channels=self.feature_channels,
            hidden_channels=self.feature_channels,
            radius=radius,
            iterations=iterations,
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.feature_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.feature_channels * 2, self.output_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    @staticmethod
    def _warp(source: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = source.shape
        coordinates = _coordinate_grid(batch, height, width, source) + displacement
        grid = coordinates.permute(0, 2, 3, 1)
        grid_x = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
        grid_y = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
        return F.grid_sample(
            source,
            torch.stack((grid_x, grid_y), dim=-1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(self, rgb_clip: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb_clip.ndim != 5 or rgb_clip.shape[1] != 3 or rgb_clip.shape[2] < 2:
            raise ValueError("motion branch expects [batch, 3, time>=2, height, width]")
        batch, channels, frames, height, width = rgb_clip.shape
        encoded = self.feature_encoder(
            rgb_clip.permute(0, 2, 1, 3, 4).reshape(
                batch * frames,
                channels,
                height,
                width,
            )
        )
        feature_height, feature_width = encoded.shape[-2:]
        encoded = encoded.reshape(
            batch,
            frames,
            self.feature_channels,
            feature_height,
            feature_width,
        )
        resized_frames = F.interpolate(
            rgb_clip.permute(0, 2, 1, 3, 4).reshape(
                batch * frames,
                channels,
                height,
                width,
            ),
            size=(feature_height, feature_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, channels, feature_height, feature_width)

        states: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        mask_logits: list[torch.Tensor] = []
        reconstructions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        displacements: list[torch.Tensor] = []
        update_magnitudes: list[torch.Tensor] = []
        for target_index in range(1, frames):
            state, displacement, magnitudes = self.update_operator(
                encoded[:, target_index],
                encoded[:, target_index - 1],
            )
            source_frame = resized_frames[:, target_index - 1]
            states.append(state)
            logits = self.mask_head(state)
            mask_logits.append(logits)
            masks.append(torch.sigmoid(logits))
            reconstructions.append(self._warp(source_frame, displacement))
            targets.append(resized_frames[:, target_index])
            displacements.append(displacement)
            update_magnitudes.append(magnitudes)

        state_sequence = torch.stack(states, dim=1)
        spatial_states = state_sequence.mean(dim=(-2, -1))
        first_state = spatial_states.new_zeros(batch, 1, self.feature_channels)
        frame_features = torch.cat((first_state, spatial_states), dim=1)
        pooled = torch.cat((frame_features.mean(dim=1), frame_features.amax(dim=1)), dim=1)
        update_l2_trace = torch.stack(update_magnitudes, dim=1)
        return {
            "clip_features": self.output_projection(pooled),
            "frame_features": frame_features,
            "explainability_masks": torch.stack(masks, dim=2),
            "explainability_mask_logits": torch.stack(mask_logits, dim=2),
            "reconstructed_targets": torch.stack(reconstructions, dim=2),
            "target_frames": torch.stack(targets, dim=2),
            "displacements": torch.stack(displacements, dim=2),
            "flow_update_magnitudes": update_l2_trace,
            "flow_update_l2_magnitudes": update_l2_trace,
        }


class Stage1MViT(nn.Module):
    """Fuse global streams with probability-only multi-stage temporal refinement."""

    def __init__(
        self,
        *,
        feature_mode: str = DEFAULT_FEATURE_MODE,
        row_profile_bins: int = DEFAULT_ROW_PROFILE_BINS,
        motion_iterations: int = DEFAULT_MOTION_ITERATIONS,
        correlation_radius: int = DEFAULT_CORRELATION_RADIUS,
        temporal_refinement_stages: int = DEFAULT_MSTCN_STAGES,
        temporal_refinement_mode: str = DEFAULT_TEMPORAL_REFINEMENT_MODE,
        mstcn_gate_initial: float = DEFAULT_MSTCN_GATE_INITIAL,
        zero_gate: bool = False,
        detach_mstcn_input: bool = DEFAULT_DETACH_MSTCN_INPUT,
        base_initialization_seed: int = DEFAULT_BASE_INITIALIZATION_SEED,
    ) -> None:
        super().__init__()
        feature_channels(feature_mode)
        if row_profile_bins < 1:
            raise ValueError("row_profile_bins must be >= 1")
        if temporal_refinement_mode not in TEMPORAL_REFINEMENT_MODES:
            raise ValueError(
                "temporal_refinement_mode must be one of "
                f"{sorted(TEMPORAL_REFINEMENT_MODES)}, got {temporal_refinement_mode!r}"
            )
        if not 0.0 < mstcn_gate_initial < 1.0:
            raise ValueError("mstcn_gate_initial must satisfy 0 < alpha < 1")
        if temporal_refinement_mode == "single_stage" and zero_gate:
            raise ValueError("zero_gate requires temporal_refinement_mode='gated_mstcn'")
        if base_initialization_seed < 0:
            raise ValueError("base_initialization_seed must be >= 0")
        self.feature_mode = feature_mode
        self.row_profile_bins = row_profile_bins
        self.temporal_refinement_mode = temporal_refinement_mode
        self.mstcn_gate_initial = float(mstcn_gate_initial)
        self.zero_gate = bool(zero_gate)
        self.detach_mstcn_input = bool(
            detach_mstcn_input and temporal_refinement_mode == "gated_mstcn"
        )
        self.base_initialization_seed = int(base_initialization_seed)
        self.head_initialization_seed = self.base_initialization_seed + 1
        self.register_buffer("_zero_alpha", torch.tensor(0.0), persistent=False)
        self.mvit_input_frames = DEFAULT_MVIT_INPUT_FRAMES
        with torch.random.fork_rng():
            torch.manual_seed(self.base_initialization_seed)
            self.rgb_backbone = mvit_v2_s(weights=None)
            rgb_dimension = self.rgb_backbone.head[1].in_features
            self.rgb_dimension = int(rgb_dimension)
            self.rgb_backbone.head = nn.Identity()
            self.spatial_branch = SpatialFftEncoder()
            self.temporal_branch = FlickerTemporalEncoder(2 + row_profile_bins)
            self.motion_branch = MotionConsistencyEncoder(
                iterations=motion_iterations,
                radius=correlation_radius,
            )
            fusion_dimension = (
                rgb_dimension
                + self.spatial_branch.output_dim
                + self.temporal_branch.output_dim
                + self.motion_branch.output_dim
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_dimension),
                nn.Linear(fusion_dimension, 256),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(256, 2),
            )

        frame_feature_channels = (
            self.temporal_branch.hidden_channels
            + self.motion_branch.feature_channels
        )
        with torch.random.fork_rng():
            torch.manual_seed(self.head_initialization_seed)
            if temporal_refinement_mode == "single_stage":
                self.frame_classifier: nn.Module | None = nn.Sequential(
                    nn.LayerNorm(frame_feature_channels),
                    nn.Linear(frame_feature_channels, 2),
                )
                self.temporal_refinement_head: nn.Module | None = None
                self.register_parameter("mstcn_gate_logit", None)
            else:
                self.frame_classifier = None
                self.temporal_refinement_head = MultiStageTemporalRefinementHead(
                    frame_feature_channels,
                    stages=temporal_refinement_stages,
                )
                if zero_gate:
                    self.register_parameter("mstcn_gate_logit", None)
                else:
                    initial = torch.tensor(float(mstcn_gate_initial)).logit()
                    self.mstcn_gate_logit = nn.Parameter(initial)

    @property
    def branch_dimensions(self) -> dict[str, int]:
        return {
            "rgb": self.rgb_dimension,
            "spatial": self.spatial_branch.output_dim,
            "temporal": self.temporal_branch.output_dim,
            "motion": self.motion_branch.output_dim,
        }

    @property
    def branch_slices(self) -> dict[str, slice]:
        start = 0
        output: dict[str, slice] = {}
        for name, dimension in self.branch_dimensions.items():
            output[name] = slice(start, start + dimension)
            start += dimension
        return output

    def extract_branch_features(
        self,
        rgb_clip: torch.Tensor,
        fft_clip: torch.Tensor,
        flicker: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return unfused features for activation and ablation diagnostics."""

        features, _, _, _ = self._extract_all_features(rgb_clip, fft_clip, flicker)
        return features

    def _extract_all_features(
        self,
        rgb_clip: torch.Tensor,
        fft_clip: torch.Tensor,
        flicker: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        if rgb_clip.ndim != 5 or rgb_clip.shape[1] != 3:
            raise ValueError(
                "rgb_clip must have shape [batch, 3, time, height, width], "
                f"got {tuple(rgb_clip.shape)}"
            )
        if fft_clip.ndim != 5 or fft_clip.shape[:3] != rgb_clip.shape[:3]:
            raise ValueError("fft_clip must share batch, channels, and time with rgb_clip")
        if flicker.ndim != 3 or flicker.shape[0] != rgb_clip.shape[0]:
            raise ValueError("flicker must have shape [batch, channels, time]")
        if flicker.shape[-1] != rgb_clip.shape[2]:
            raise ValueError("all Stage 1 branches must receive the same raw clip length")

        mvit_clip = self.interpolate_mvit_time(rgb_clip)
        rgb_features = self.rgb_backbone(mvit_clip)
        if self.feature_mode == RGB_FFT_FEATURES:
            spatial_features = self.spatial_branch(fft_clip)
        else:
            spatial_features = rgb_features.new_zeros(
                rgb_features.shape[0],
                self.spatial_branch.output_dim,
            )
        temporal_sequence = self.temporal_branch.encode_sequence(flicker)
        temporal_features = self.temporal_branch.pool_sequence(temporal_sequence)
        motion_outputs = self.motion_branch(rgb_clip)
        frame_features = torch.cat(
            (
                temporal_sequence.permute(0, 2, 1),
                motion_outputs["frame_features"],
            ),
            dim=2,
        )
        features = {
            "rgb": rgb_features,
            "spatial": spatial_features,
            "temporal": temporal_features,
            "motion": motion_outputs["clip_features"],
        }
        if self.temporal_refinement_mode == "single_stage":
            if self.frame_classifier is None:
                raise RuntimeError("single-stage temporal classifier is not initialized")
            frame_logits = self.frame_classifier(frame_features)
            stage_frame_logits = frame_logits[:, None]
        else:
            stage_frame_logits = self.refine_frame_features(frame_features)
            frame_logits = stage_frame_logits[:, -1]
        return (
            features,
            motion_outputs,
            frame_logits,
            stage_frame_logits,
        )

    def refine_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Run MS-TCN behind an optional one-way autograd boundary."""

        if self.temporal_refinement_mode != "gated_mstcn":
            raise RuntimeError("MS-TCN refinement requires gated_mstcn mode")
        if self.temporal_refinement_head is None:
            raise RuntimeError("MS-TCN refinement head is not initialized")
        mstcn_input = (
            frame_features.detach()
            if self.detach_mstcn_input
            else frame_features
        )
        return self.temporal_refinement_head(mstcn_input)

    def interpolate_mvit_time(self, rgb_clip: torch.Tensor) -> torch.Tensor:
        """Resample only the MViT view to its fixed 16-frame token geometry."""

        if rgb_clip.ndim != 5 or rgb_clip.shape[1] != 3:
            raise ValueError(
                "rgb_clip must have shape [batch, 3, time, height, width], "
                f"got {tuple(rgb_clip.shape)}"
            )
        if rgb_clip.shape[2] < 2:
            raise ValueError("Stage 1 clips must contain at least two frames")
        if rgb_clip.shape[2] == self.mvit_input_frames:
            return rgb_clip
        return F.interpolate(
            rgb_clip,
            size=(self.mvit_input_frames, rgb_clip.shape[-2], rgb_clip.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )

    @staticmethod
    def fuse_branch_features(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (
                features["rgb"],
                features["spatial"],
                features["temporal"],
                features["motion"],
            ),
            dim=1,
        )

    @property
    def mstcn_alpha(self) -> torch.Tensor:
        if self.temporal_refinement_mode == "single_stage" or self.zero_gate:
            return self._zero_alpha
        if self.mstcn_gate_logit is None:
            raise RuntimeError("learnable MS-TCN gate parameter is not initialized")
        return torch.sigmoid(self.mstcn_gate_logit)

    def combine_clip_logits(
        self,
        fusion_logits: torch.Tensor,
        refined_frame_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Make the probability-only refinement part of the final decision."""

        if fusion_logits.ndim != 2 or refined_frame_logits.ndim != 3:
            raise ValueError("clip/frame logits have incompatible ranks")
        if (
            fusion_logits.shape[0] != refined_frame_logits.shape[0]
            or fusion_logits.shape[1] != refined_frame_logits.shape[2]
        ):
            raise ValueError("clip/frame logits have incompatible class geometry")
        if self.temporal_refinement_mode == "single_stage":
            return fusion_logits
        return fusion_logits + self.mstcn_alpha * refined_frame_logits.mean(dim=1)

    def forward(
        self,
        rgb_clip: torch.Tensor,
        fft_clip: torch.Tensor,
        flicker: torch.Tensor,
        *,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features, motion_outputs, frame_logits, stage_frame_logits = self._extract_all_features(
            rgb_clip,
            fft_clip,
            flicker,
        )
        fusion_logits = self.classifier(self.fuse_branch_features(features))
        logits = self.combine_clip_logits(fusion_logits, frame_logits)
        if not return_auxiliary:
            return logits
        return {
            "logits": logits,
            "frame_logits": frame_logits,
            "stage_frame_logits": stage_frame_logits,
            "mstcn_residual_alpha": self.mstcn_alpha,
            "explainability_masks": motion_outputs["explainability_masks"],
            "explainability_mask_logits": motion_outputs[
                "explainability_mask_logits"
            ],
            "reconstructed_targets": motion_outputs["reconstructed_targets"],
            "target_frames": motion_outputs["target_frames"],
            "displacements": motion_outputs["displacements"],
            "flow_update_magnitudes": motion_outputs["flow_update_magnitudes"],
            "flow_update_l2_magnitudes": motion_outputs[
                "flow_update_l2_magnitudes"
            ],
        }


class BestStateTracker:
    """Keep the true best state_dict on CPU independently of early stopping."""

    def __init__(self, model: nn.Module, *, mode: Literal["min", "max"]) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best_value: float | None = None
        self.best_epoch = 0
        self._state = self._copy(model)

    @staticmethod
    def _copy(model: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    def consider(self, model: nn.Module, *, value: float, epoch: int) -> bool:
        improved = (
            self.best_value is None
            or (value > self.best_value if self.mode == "max" else value < self.best_value)
        )
        if improved:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self._state = self._copy(model)
        return improved

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self._state.items()}

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self._state, strict=True)


def linear_warmup_learning_rate(
    *,
    epoch: int,
    warmup_epochs: int,
    initial_learning_rate: float,
    target_learning_rate: float,
) -> float:
    """Return the one-based epoch LR for an inclusive linear warm-up."""

    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    if warmup_epochs < 1:
        raise ValueError("warmup_epochs must be >= 1")
    if min(initial_learning_rate, target_learning_rate) < 0.0:
        raise ValueError("learning rates must be >= 0")
    if warmup_epochs == 1:
        return float(target_learning_rate)
    progress = min(epoch - 1, warmup_epochs - 1) / (warmup_epochs - 1)
    return float(
        initial_learning_rate
        + progress * (target_learning_rate - initial_learning_rate)
    )


def stage1_early_stopping_triggered(
    *,
    epoch: int,
    minimum_epochs: int,
    stop_requested: bool,
) -> bool:
    """Gate a patience signal until the required minimum epoch is complete."""

    if epoch < 1 or minimum_epochs < 1:
        raise ValueError("epoch and minimum_epochs must be >= 1")
    return bool(stop_requested and epoch >= minimum_epochs)


def _set_optimizer_group_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    warmup_epochs: int,
    initial_learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        target_learning_rate = float(group["target_learning_rate"])
        group["lr"] = linear_warmup_learning_rate(
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            initial_learning_rate=initial_learning_rate,
            target_learning_rate=target_learning_rate,
        )


def load_local_mvit_backbone(
    model: Stage1MViT,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Load shape-compatible MViTv2-S tensors from an explicit local file."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"local MViTv2-S checkpoint not found: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state: Mapping[str, object] | None = raw if isinstance(raw, Mapping) else None
    if state is not None:
        for container_key in ("model", "state_dict", "model_state_dict"):
            candidate = state.get(container_key)
            if isinstance(candidate, Mapping):
                state = candidate
                break
    if state is None:
        raise ValueError("local MViTv2-S checkpoint must contain a state_dict mapping")

    expected = model.rgb_backbone.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        if not isinstance(raw_name, str) or not isinstance(value, torch.Tensor):
            continue
        name = raw_name
        for prefix in ("module.rgb_backbone.", "rgb_backbone.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name in expected and expected[name].shape == value.shape:
            compatible[name] = value.detach().cpu()
    if not compatible:
        raise ValueError(
            f"no shape-compatible MViTv2-S tensors found in local checkpoint: {path}"
        )
    model.rgb_backbone.load_state_dict(compatible, strict=False)
    return {
        "path": str(path.resolve()),
        "loaded_tensors": len(compatible),
        "available_backbone_tensors": len(expected),
    }


def _move_inputs(
    inputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in inputs.items()
    }


def _video_level_predictions(
    probabilities: Sequence[Sequence[float]],
) -> list[int]:
    if any(not values for values in probabilities):
        raise ValueError("every validation video must have at least one temporal region")
    return [int(float(np.mean(values)) >= 0.5) for values in probabilities]


def _auxiliary_batch_diagnostics(
    outputs: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    masks = outputs["explainability_masks"].detach().float()
    update_l2 = outputs["flow_update_l2_magnitudes"].detach().float()
    if masks.ndim != 5 or update_l2.ndim != 3:
        raise ValueError("Stage 1 auxiliary diagnostics received invalid tensor shapes")
    if not torch.isfinite(masks).all() or not torch.isfinite(update_l2).all():
        raise ValueError("Stage 1 auxiliary diagnostics must be finite")
    per_iteration = update_l2.mean(dim=(0, 1))
    diagnostics = {
        "explainability_mask_mean": float(masks.mean().cpu()),
        "explainability_mask_std": float(masks.std(unbiased=False).cpu()),
        "explainability_mask_near_zero_fraction": float((masks <= 0.05).float().mean().cpu()),
        "explainability_mask_near_one_fraction": float((masks >= 0.95).float().mean().cpu()),
        "convgru_last_to_first_update_l2_ratio": float(
            (
                update_l2[..., -1]
                / update_l2[..., 0].clamp_min(1e-8)
            ).mean().cpu()
        ),
        "mstcn_residual_alpha": float(
            outputs.get("mstcn_residual_alpha", masks.new_zeros(())).detach().float().cpu()
        ),
    }
    diagnostics.update(
        {
            f"convgru_update_l2_iteration_{iteration}": float(value.cpu())
            for iteration, value in enumerate(per_iteration, start=1)
        }
    )
    return diagnostics


def _average_diagnostic_batches(
    batches: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not batches:
        return {}
    names = tuple(batches[0])
    if any(tuple(batch) != names for batch in batches):
        raise ValueError("Stage 1 diagnostic batches have inconsistent fields")
    return {
        name: float(np.mean([float(batch[name]) for batch in batches]))
        for name in names
    }


def _loss_balance_diagnostics(
    components: Mapping[str, float],
    *,
    frame_classification_weight: float,
    smoothing_weight: float,
    explainability_weight: float,
    mask_regularization_weight: float,
    mask_sparsity_weight: float,
) -> dict[str, float]:
    classification = (
        float(components["clip_classification"])
        + frame_classification_weight * float(components["frame_classification"])
    )
    smoothing = float(components["smoothing"])
    weighted_smoothing = smoothing_weight * smoothing
    explainability = explainability_weight * (
        float(components["reconstruction"])
        + mask_regularization_weight * float(components["mask_regularization"])
    )
    mask_sparsity = mask_sparsity_weight * float(components["mask_sparsity"])
    explainability_total = explainability + mask_sparsity
    denominator = max(abs(classification), 1e-12)
    return {
        "classification_focal_loss": classification,
        "weighted_smoothing_loss": weighted_smoothing,
        "weighted_explainability_loss": explainability_total,
        "weighted_mask_sparsity_loss": mask_sparsity,
        "smoothing_to_classification_ratio": smoothing / denominator,
        "weighted_smoothing_to_classification_ratio": weighted_smoothing / denominator,
        "weighted_explainability_to_classification_ratio": (
            explainability_total / denominator
        ),
    }


def fit_stage1(
    data_dir: str | Path,
    model_dir: str | Path,
    *,
    epochs: int = DEFAULT_MAX_EPOCHS,
    minimum_epochs: int = DEFAULT_MINIMUM_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    backbone_learning_rate: float = DEFAULT_BACKBONE_LEARNING_RATE,
    auxiliary_learning_rate: float = DEFAULT_AUXILIARY_LEARNING_RATE,
    warmup_initial_learning_rate: float = DEFAULT_WARMUP_INITIAL_LEARNING_RATE,
    pretrained_backbone_checkpoint: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    focal_gamma: float = 2.0,
    focal_alpha: Sequence[float] | torch.Tensor | None = None,
    frame_classification_weight: float = 0.25,
    smoothing_weight: float = 0.05,
    smoothing_truncation: float = 4.0,
    explainability_weight: float = 0.05,
    mask_regularization_weight: float = 0.02,
    mask_sparsity_weight: float = 1e-3,
    motion_iterations: int = DEFAULT_MOTION_ITERATIONS,
    correlation_radius: int = DEFAULT_CORRELATION_RADIUS,
    temporal_refinement_stages: int = DEFAULT_MSTCN_STAGES,
    temporal_refinement_mode: str = DEFAULT_TEMPORAL_REFINEMENT_MODE,
    mstcn_gate_initial: float = DEFAULT_MSTCN_GATE_INITIAL,
    zero_gate: bool = False,
    detach_mstcn_input: bool = DEFAULT_DETACH_MSTCN_INPUT,
    base_initialization_seed: int = DEFAULT_BASE_INITIALIZATION_SEED,
    size: int = 224,
    frames: int = 16,
    training_sequence_lengths: Sequence[int] = DEFAULT_TRAIN_SEQUENCE_LENGTHS,
    batch_size: int = 1,
    train_slots: int = DEFAULT_TEMPORAL_SLOTS,
    jitter_frames: int = DEFAULT_JITTER_FRAMES,
    random_temporal_jitter: bool = True,
    forensic_size: int = DEFAULT_FORENSIC_SIZE,
    fft_size: int = DEFAULT_FFT_SIZE,
    row_profile_bins: int = DEFAULT_ROW_PROFILE_BINS,
    num_workers: int = 0,
    enable_augmentation: bool = True,
    enable_photometric_augmentation: bool = True,
    enable_occlusion_augmentation: bool = True,
    enable_affine_augmentation: bool = True,
    inference_tta_slots: int = DEFAULT_TTA_SLOTS,
    label_frame: pd.DataFrame | None = None,
    validation_label_frame: pd.DataFrame | None = None,
    training_control: TrainingControlConfig = TrainingControlConfig(),
    processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
) -> Path:
    if not 0 <= epochs <= DEFAULT_MAX_EPOCHS:
        raise ValueError(f"epochs must be in [0, {DEFAULT_MAX_EPOCHS}]")
    if minimum_epochs < 1:
        raise ValueError("minimum_epochs must be >= 1")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1")
    if warmup_epochs < 1:
        raise ValueError("warmup_epochs must be >= 1")
    learning_rates = {
        "backbone_learning_rate": backbone_learning_rate,
        "auxiliary_learning_rate": auxiliary_learning_rate,
        "warmup_initial_learning_rate": warmup_initial_learning_rate,
        "minimum_learning_rate": training_control.min_learning_rate,
    }
    invalid_learning_rates = {
        name: value for name, value in learning_rates.items() if value < 0.0
    }
    if invalid_learning_rates:
        raise ValueError(f"Stage 1 learning rates must be >= 0: {invalid_learning_rates}")
    if warmup_initial_learning_rate > min(
        backbone_learning_rate,
        auxiliary_learning_rate,
    ):
        raise ValueError("warm-up initial LR cannot exceed either target LR")
    if training_control.min_learning_rate > min(
        backbone_learning_rate,
        auxiliary_learning_rate,
    ):
        raise ValueError("minimum LR cannot exceed either target LR")
    if focal_gamma < 0 or min(
        frame_classification_weight,
        smoothing_weight,
        explainability_weight,
        mask_regularization_weight,
        mask_sparsity_weight,
    ) < 0:
        raise ValueError("focal_gamma and Stage 1 auxiliary loss weights must be >= 0")
    if smoothing_truncation <= 0.0:
        raise ValueError("smoothing_truncation must be > 0")
    if temporal_refinement_stages < 2:
        raise ValueError("temporal_refinement_stages must be >= 2")
    if temporal_refinement_mode not in TEMPORAL_REFINEMENT_MODES:
        raise ValueError(
            "temporal_refinement_mode must be one of "
            f"{sorted(TEMPORAL_REFINEMENT_MODES)}"
        )
    if not 0.0 < mstcn_gate_initial < 1.0:
        raise ValueError("mstcn_gate_initial must satisfy 0 < alpha < 1")
    if temporal_refinement_mode == "single_stage" and zero_gate:
        raise ValueError("zero_gate requires temporal_refinement_mode='gated_mstcn'")
    parsed_sequence_lengths = tuple(dict.fromkeys(int(value) for value in training_sequence_lengths))
    if parsed_sequence_lengths != DEFAULT_TRAIN_SEQUENCE_LENGTHS:
        raise ValueError(
            "Stage 1 v3 training_sequence_lengths must be exactly "
            f"{DEFAULT_TRAIN_SEQUENCE_LENGTHS}, got {parsed_sequence_lengths}"
        )
    if min(
        size,
        batch_size,
        train_slots,
        forensic_size,
        fft_size,
        row_profile_bins,
        motion_iterations,
    ) < 1 or frames < 2:
        raise ValueError("Stage 1 geometry and batch values must be >= 1")
    if frames != DEFAULT_MVIT_INPUT_FRAMES:
        raise ValueError(
            f"Stage 1 validation/inference frames must be {DEFAULT_MVIT_INPUT_FRAMES}"
        )
    if batch_size != 1:
        raise ValueError(
            "variable-length Stage 1 training requires batch_size=1 without padding"
        )
    if jitter_frames < 0 or num_workers < 0 or correlation_radius < 0:
        raise ValueError("jitter_frames, num_workers, and correlation_radius must be >= 0")
    if inference_tta_slots != train_slots:
        raise ValueError(
            "training, validation, and inference must use the same temporal region count: "
            f"train_slots={train_slots}, inference_tta_slots={inference_tta_slots}"
        )
    feature_channels(feature_mode)
    seed_everything(seed)
    data_root = Path(data_dir)
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data_root / "labels.csv") if label_frame is None else label_frame.copy()
    required_columns = {"path", "label"}
    missing_columns = sorted(required_columns - set(labels.columns))
    if missing_columns:
        raise ValueError(f"Stage 1 labels are missing columns: {missing_columns}")
    labels["label"] = labels["label"].astype(str)
    unknown_labels = sorted(set(labels["label"]) - set(LABEL_TO_INDEX))
    if unknown_labels:
        raise ValueError(f"unsupported Stage 1 labels: {unknown_labels}")
    samples = [
        (data_root / str(row.path), LABEL_TO_INDEX[str(row.label)])
        for row in labels.itertuples(index=False)
    ]
    missing = [str(path) for path, _ in samples if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 1 training videos not found: {missing}")

    if validation_label_frame is None:
        train_indices, valid_indices = group_holdout_indices(
            [path.stem for path, _ in samples],
            validation_fraction=training_control.validation_fraction,
        )
        train_samples = [
            sample for index, sample in enumerate(samples) if index in train_indices
        ]
        valid_samples = [
            sample for index, sample in enumerate(samples) if index in valid_indices
        ]
        validation_source = "internal_group_holdout"
    else:
        validation_labels = validation_label_frame.copy()
        validation_missing_columns = sorted(
            required_columns - set(validation_labels.columns)
        )
        if validation_missing_columns:
            raise ValueError(
                "Stage 1 validation labels are missing columns: "
                f"{validation_missing_columns}"
            )
        validation_labels["label"] = validation_labels["label"].astype(str)
        unknown_validation_labels = sorted(
            set(validation_labels["label"]) - set(LABEL_TO_INDEX)
        )
        if unknown_validation_labels:
            raise ValueError(
                f"unsupported Stage 1 validation labels: {unknown_validation_labels}"
            )
        train_samples = samples
        valid_samples = [
            (data_root / str(row.path), LABEL_TO_INDEX[str(row.label)])
            for row in validation_labels.itertuples(index=False)
        ]
        missing_validation = [
            str(path) for path, _ in valid_samples if not path.is_file()
        ]
        if missing_validation:
            raise FileNotFoundError(
                f"Stage 1 validation videos not found: {missing_validation}"
            )
        validation_source = "explicit_group_fold"
    device = choose_device()
    augmentation = (
        Stage1TrainAugmentation(
            enable_photometric=enable_photometric_augmentation,
            enable_occlusion=enable_occlusion_augmentation,
            affine_probability=0.35 if enable_affine_augmentation else 0.0,
        )
        if enable_augmentation
        else None
    )
    cache_frames = max(parsed_sequence_lengths)
    train_dataset = Stage1TrainingDataset(
        train_samples,
        size=size,
        frames=cache_frames,
        feature_mode=feature_mode,
        slots=train_slots,
        jitter_frames=jitter_frames,
        forensic_size=forensic_size,
        fft_size=fft_size,
        row_profile_bins=row_profile_bins,
        sequence_lengths=parsed_sequence_lengths,
        random_jitter=random_temporal_jitter,
        augmentation=augmentation,
        processed_root=processed_root,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    validation_dataset = (
        Stage1ValidationDataset(
            valid_samples,
            size=size,
            frames=frames,
            feature_mode=feature_mode,
            slots=train_slots,
            forensic_size=forensic_size,
            fft_size=fft_size,
            row_profile_bins=row_profile_bins,
        )
        if valid_samples
        else None
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
        if validation_dataset is not None
        else None
    )

    model = Stage1MViT(
        feature_mode=feature_mode,
        row_profile_bins=row_profile_bins,
        motion_iterations=motion_iterations,
        correlation_radius=correlation_radius,
        temporal_refinement_stages=temporal_refinement_stages,
        temporal_refinement_mode=temporal_refinement_mode,
        mstcn_gate_initial=mstcn_gate_initial,
        zero_gate=zero_gate,
        detach_mstcn_input=detach_mstcn_input,
        base_initialization_seed=base_initialization_seed,
    )
    pretrained_backbone = (
        None
        if pretrained_backbone_checkpoint is None
        else load_local_mvit_backbone(model, pretrained_backbone_checkpoint)
    )
    model.to(device)
    backbone_parameters = list(model.rgb_backbone.parameters())
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": warmup_initial_learning_rate,
                "name": "mvit_backbone",
                "target_learning_rate": backbone_learning_rate,
            },
            {
                "params": auxiliary_parameters,
                "lr": warmup_initial_learning_rate,
                "name": "head_auxiliary",
                "target_learning_rate": auxiliary_learning_rate,
            },
        ]
    )
    scaler = make_grad_scaler(device, enabled=training_control.use_amp)
    criterion = Stage1MultiTaskLoss(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        frame_classification_weight=frame_classification_weight,
        smoothing_weight=smoothing_weight,
        smoothing_truncation=smoothing_truncation,
        explainability_weight=explainability_weight,
        mask_regularization_weight=mask_regularization_weight,
        mask_sparsity_weight=mask_sparsity_weight,
    )
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
    monitor_mode: Literal["min", "max"] = "max" if validation_loader is not None else "min"
    tracker = BestStateTracker(model, mode=monitor_mode)
    logger = JsonlTrainingLogger("stage1", training_control.log_dir)
    early_stopping = EarlyStopping(
        mode=monitor_mode,
        patience=early_stopping_patience,
        min_delta=training_control.early_stopping_min_delta,
    )
    history: list[dict[str, object]] = []
    previous_valid_probabilities: list[float] | None = None
    previous_valid_predictions: list[int] | None = None
    previous_mstcn_alpha: float | None = None

    for epoch in range(max(0, epochs)):
        epoch_number = epoch + 1
        if epoch_number <= warmup_epochs:
            _set_optimizer_group_learning_rates(
                optimizer,
                epoch=epoch_number,
                warmup_epochs=warmup_epochs,
                initial_learning_rate=warmup_initial_learning_rate,
            )
        elif scheduler is None:
            for group in optimizer.param_groups:
                group["lr"] = float(group["target_learning_rate"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=training_control.min_learning_rate,
            )
        model.train()
        losses: list[float] = []
        train_components: dict[str, list[float]] = {
            name: []
            for name in (
                "clip_classification",
                "frame_classification",
                "smoothing",
                "reconstruction",
                "mask_regularization",
                "mask_sparsity",
            )
        }
        observed_train_sequence_lengths: set[int] = set()
        train_auxiliary_diagnostic_batches: list[dict[str, float]] = []
        for inputs, targets, _ in train_loader:
            observed_train_sequence_lengths.add(int(inputs["rgb_clip"].shape[2]))
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=training_control.use_amp):
                outputs = model(
                    **_move_inputs(inputs, device),
                    return_auxiliary=True,
                )
                if not isinstance(outputs, dict):
                    raise TypeError("Stage 1 auxiliary training forward must return a mapping")
                loss_terms = criterion(outputs, targets)
                loss = loss_terms["total"]
            train_auxiliary_diagnostic_batches.append(
                _auxiliary_batch_diagnostics(outputs)
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            for name in train_components:
                train_components[name].append(float(loss_terms[name].detach().cpu()))
        average_loss = sum(losses) / max(1, len(losses))
        average_train_components = {
            name: sum(values) / max(1, len(values))
            for name, values in train_components.items()
        }
        average_train_auxiliary_diagnostics = _average_diagnostic_batches(
            train_auxiliary_diagnostic_batches
        )
        average_train_loss_balance = _loss_balance_diagnostics(
            average_train_components,
            frame_classification_weight=frame_classification_weight,
            smoothing_weight=smoothing_weight,
            explainability_weight=explainability_weight,
            mask_regularization_weight=mask_regularization_weight,
            mask_sparsity_weight=mask_sparsity_weight,
        )

        valid_loss: float | None = None
        valid_metric: float | None = None
        valid_prediction_change_rate: float | None = None
        valid_probability_mean_abs_delta: float | None = None
        if validation_loader is not None:
            model.eval()
            valid_probabilities: list[list[float]] = [[] for _ in valid_samples]
            valid_losses: list[float] = []
            valid_components: dict[str, list[float]] = {
                name: [] for name in train_components
            }
            valid_auxiliary_diagnostic_batches: list[dict[str, float]] = []
            with torch.inference_mode():
                for inputs, targets, video_indices, valid in validation_loader:
                    if not bool(valid.all()):
                        failed = [
                            str(valid_samples[index][0])
                            for index, ok in zip(video_indices.tolist(), valid.tolist())
                            if not ok
                        ]
                        raise ValueError(
                            "Stage 1 validation could not decode videos: "
                            f"{failed}"
                        )
                    targets = targets.to(device, non_blocking=True)
                    with autocast_context(device, enabled=training_control.use_amp):
                        outputs = model(
                            **_move_inputs(inputs, device),
                            return_auxiliary=True,
                        )
                        if not isinstance(outputs, dict):
                            raise TypeError(
                                "Stage 1 auxiliary validation forward must return a mapping"
                            )
                        loss_terms = criterion(outputs, targets)
                        batch_loss = loss_terms["total"]
                        logits = outputs["logits"]
                        probabilities = torch.softmax(logits, dim=1)[:, 1]
                    valid_auxiliary_diagnostic_batches.append(
                        _auxiliary_batch_diagnostics(outputs)
                    )
                    valid_losses.append(float(batch_loss.detach().cpu()))
                    for name in valid_components:
                        valid_components[name].append(
                            float(loss_terms[name].detach().cpu())
                        )
                    for video_index, probability in zip(
                        video_indices.tolist(),
                        probabilities.float().cpu().tolist(),
                    ):
                        valid_probabilities[video_index].append(float(probability))
            valid_targets = [label for _, label in valid_samples]
            valid_predictions = _video_level_predictions(valid_probabilities)
            valid_mean_probabilities = [
                float(np.mean(probabilities)) for probabilities in valid_probabilities
            ]
            valid_loss = sum(valid_losses) / max(1, len(valid_losses))
            average_valid_components = {
                name: sum(values) / max(1, len(values))
                for name, values in valid_components.items()
            }
            average_valid_auxiliary_diagnostics = _average_diagnostic_batches(
                valid_auxiliary_diagnostic_batches
            )
            average_valid_loss_balance = _loss_balance_diagnostics(
                average_valid_components,
                frame_classification_weight=frame_classification_weight,
                smoothing_weight=smoothing_weight,
                explainability_weight=explainability_weight,
                mask_regularization_weight=mask_regularization_weight,
                mask_sparsity_weight=mask_sparsity_weight,
            )
            valid_metric = macro_f1_score(valid_targets, valid_predictions, labels=range(2))
            if previous_valid_probabilities is not None:
                valid_probability_mean_abs_delta = float(
                    np.mean(
                        np.abs(
                            np.asarray(valid_mean_probabilities)
                            - np.asarray(previous_valid_probabilities)
                        )
                    )
                )
            if previous_valid_predictions is not None:
                valid_prediction_change_rate = float(
                    np.mean(
                        np.asarray(valid_predictions)
                        != np.asarray(previous_valid_predictions)
                    )
                )
            previous_valid_probabilities = valid_mean_probabilities
            previous_valid_predictions = valid_predictions

        group_learning_rates = {
            str(group["name"]): float(group["lr"])
            for group in optimizer.param_groups
        }
        backbone_epoch_learning_rate = group_learning_rates["mvit_backbone"]
        auxiliary_epoch_learning_rate = group_learning_rates["head_auxiliary"]
        monitor_name = (
            "valid_macro_f1_three_region_mean_probability"
            if valid_metric is not None
            else "train_loss_proxy_no_validation"
        )
        monitor_value = valid_metric if valid_metric is not None else average_loss
        mstcn_alpha = average_train_auxiliary_diagnostics[
            "mstcn_residual_alpha"
        ]
        mstcn_alpha_delta = (
            None
            if previous_mstcn_alpha is None
            else mstcn_alpha - previous_mstcn_alpha
        )
        tracker.consider(model, value=monitor_value, epoch=epoch + 1)
        epoch_record: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "valid_loss": valid_loss,
            "valid_macro_f1": valid_metric,
            "valid_prediction_change_rate": valid_prediction_change_rate,
            "valid_probability_mean_abs_delta": valid_probability_mean_abs_delta,
            "learning_rate": auxiliary_epoch_learning_rate,
            "backbone_learning_rate": backbone_epoch_learning_rate,
            "auxiliary_learning_rate": auxiliary_epoch_learning_rate,
            "learning_rate_phase": (
                "linear_warmup" if epoch_number <= warmup_epochs else "cosine_annealing"
            ),
            "train_sequence_lengths_observed": sorted(
                observed_train_sequence_lengths
            ),
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            "mstcn_residual_alpha": mstcn_alpha,
            "mstcn_residual_alpha_delta": mstcn_alpha_delta,
        }
        epoch_record.update(
            {
                f"train_{name}_loss": value
                for name, value in average_train_components.items()
            }
        )
        epoch_record.update(
            {
                f"train_{name}": value
                for name, value in {
                    **average_train_loss_balance,
                    **average_train_auxiliary_diagnostics,
                }.items()
            }
        )
        if validation_loader is not None:
            epoch_record.update(
                {
                    f"valid_{name}_loss": value
                    for name, value in average_valid_components.items()
                }
            )
            epoch_record.update(
                {
                    f"valid_{name}": value
                    for name, value in {
                        **average_valid_loss_balance,
                        **average_valid_auxiliary_diagnostics,
                    }.items()
                }
            )
        history.append(epoch_record)
        logger.log(
            epoch=epoch + 1,
            train_loss=average_loss,
            learning_rate=auxiliary_epoch_learning_rate,
            valid_metric=valid_metric,
            monitor_name=monitor_name,
            monitor_value=monitor_value,
            valid_loss=valid_loss,
            diagnostics={
                "valid_prediction_change_rate": valid_prediction_change_rate,
                "valid_probability_mean_abs_delta": valid_probability_mean_abs_delta,
                "backbone_learning_rate": backbone_epoch_learning_rate,
                "auxiliary_learning_rate": auxiliary_epoch_learning_rate,
                "mstcn_residual_alpha": mstcn_alpha,
                "mstcn_residual_alpha_delta": mstcn_alpha_delta,
                **{
                    f"train_{name}_loss": value
                    for name, value in average_train_components.items()
                },
                **{
                    f"train_{name}": value
                    for name, value in {
                        **average_train_loss_balance,
                        **average_train_auxiliary_diagnostics,
                    }.items()
                },
                **(
                    {
                        **{
                            f"valid_{name}_loss": value
                            for name, value in average_valid_components.items()
                        },
                        **{
                            f"valid_{name}": value
                            for name, value in {
                                **average_valid_loss_balance,
                                **average_valid_auxiliary_diagnostics,
                            }.items()
                        },
                    }
                    if validation_loader is not None
                    else {}
                ),
            },
        )
        print(
            f"[Stage1][epoch {epoch + 1}/{epochs}] loss={average_loss:.6f} "
            f"valid_loss={'unavailable' if valid_loss is None else f'{valid_loss:.6f}'} "
            f"backbone_lr={backbone_epoch_learning_rate:.3e} "
            f"aux_lr={auxiliary_epoch_learning_rate:.3e} "
            f"smooth/cls={average_train_loss_balance['weighted_smoothing_to_classification_ratio']:.3e} "
            f"mask={average_train_auxiliary_diagnostics['explainability_mask_mean']:.3f} "
            f"gru_ratio={average_train_auxiliary_diagnostics['convgru_last_to_first_update_l2_ratio']:.3f} "
            f"mstcn_alpha={mstcn_alpha:.6f} "
            "mstcn_delta="
            f"{'unavailable' if mstcn_alpha_delta is None else f'{mstcn_alpha_delta:+.3e}'} "
            "valid_macro_f1="
            f"{'unavailable' if valid_metric is None else f'{valid_metric:.6f}'}"
        )
        previous_mstcn_alpha = mstcn_alpha
        if scheduler is not None:
            scheduler.step()
        stop_requested = early_stopping.step(monitor_value)
        if stage1_early_stopping_triggered(
            epoch=epoch_number,
            minimum_epochs=minimum_epochs,
            stop_requested=stop_requested,
        ):
            print(f"[Stage1] early stopping at epoch {epoch + 1}")
            break

    tracker.restore(model)
    checkpoint = output / "best.pt"
    torch.save(
        {
            "architecture": STAGE1_ARCHITECTURE,
            "model": tracker.state_dict(),
            "size": size,
            "frames": frames,
            "feature_mode": feature_mode,
            "model_config": {
                "rgb_channels": 3,
                "mvit_input_frames": DEFAULT_MVIT_INPUT_FRAMES,
                "mvit_temporal_adapter": "trilinear_interpolation",
                "pretrained_backbone": pretrained_backbone,
                "spatial_branch": feature_mode == RGB_FFT_FEATURES,
                "forensic_size": forensic_size,
                "fft_size": fft_size,
                "flicker_channels": 2 + row_profile_bins,
                "row_profile_bins": row_profile_bins,
                "motion_branch": "all_pairs_correlation_pyramid_tied_conv_gru",
                "correlation_scales": list(DEFAULT_CORRELATION_SCALES),
                "correlation_radius": correlation_radius,
                "motion_iterations": motion_iterations,
                "motion_update_damping": "inverse_iteration",
                "motion_feature_stride": 16,
                "explainability_mask": True,
                "temporal_refinement": temporal_refinement_mode,
                "temporal_refinement_stages": temporal_refinement_stages,
                "temporal_refinement_dilations": [1, 2, 4],
                "temporal_refinement_input": (
                    "stage_1_temporal_motion_features_then_previous_stage_softmax_only"
                ),
                "mstcn_gate_initial": mstcn_gate_initial,
                "mstcn_gate_final": float(model.mstcn_alpha.detach().cpu()),
                "mstcn_zero_gate": zero_gate,
                "mstcn_input_detached": model.detach_mstcn_input,
                "base_initialization_seed": model.base_initialization_seed,
                "head_initialization_seed": model.head_initialization_seed,
                "rng_synchronization": "isolated_base_then_ablation_head_contexts",
                "fusion": (
                    "base_clip_logits_only"
                    if temporal_refinement_mode == "single_stage"
                    else (
                        "base_clip_logits_only_with_mstcn_deep_supervision"
                        if zero_gate
                        else "base_clip_logits_plus_gated_refined_frame_logits"
                    )
                ),
            },
            "loss": {
                "name": "stage1_multitask",
                "classification": {
                    "name": "focal",
                    "gamma": float(focal_gamma),
                    "alpha": (
                        None
                        if criterion.classification.alpha is None
                        else criterion.classification.alpha.detach().cpu().tolist()
                    ),
                },
                "frame_classification_weight": frame_classification_weight,
                "smoothing": {
                    "name": "truncated_log_probability_mse",
                    "weight": smoothing_weight,
                    "truncation": smoothing_truncation,
                },
                "explainability": {
                    "formula": (
                        "weight * (reconstruction + mask_regularization_weight * "
                        "bce_to_one) + mask_sparsity_weight * mean(mask)"
                    ),
                    "weight": explainability_weight,
                    "reconstruction": "mask_weighted_charbonnier",
                    "mask_regularization": {
                        "name": "binary_cross_entropy_to_one",
                        "weight_inside_explainability": mask_regularization_weight,
                        "effective_weight": (
                            explainability_weight * mask_regularization_weight
                        ),
                    },
                    "mask_sparsity": {
                        "name": "l1_mean_mask",
                        "weight": mask_sparsity_weight,
                    },
                },
            },
            "sampling": {
                "name": "centered_contiguous_regions",
                "frames_per_region": frames,
                "training_cache_frames_per_region": cache_frames,
                "training_sequence_lengths": list(parsed_sequence_lengths),
                "training_sequence_length_selection": "uniform_random_per_sample",
                "train_slots": train_slots,
                "train_random_jitter": bool(random_temporal_jitter),
                "jitter_frames": jitter_frames,
                "inference_tta_slots": int(inference_tta_slots),
                "aggregation": "mean_rerecorded_probability",
                "validation_source": validation_source,
                "validation_decoder": "online_inference_equivalent",
            },
            "selection": {
                "monitor": (
                    "valid_macro_f1_three_region_mean_probability"
                    if validation_loader is not None
                    else "train_loss_proxy_no_validation"
                ),
                "mode": monitor_mode,
                "best_epoch": tracker.best_epoch,
                "best_value": tracker.best_value,
                "restored_before_save": True,
            },
            "training_schedule": {
                "configured_epochs": epochs,
                "maximum_epochs": DEFAULT_MAX_EPOCHS,
                "minimum_epochs_before_early_stopping": minimum_epochs,
                "early_stopping_patience": early_stopping_patience,
                "warmup": {
                    "name": "linear",
                    "epochs": warmup_epochs,
                    "initial_learning_rate": warmup_initial_learning_rate,
                    "backbone_target_learning_rate": backbone_learning_rate,
                    "head_auxiliary_target_learning_rate": auxiliary_learning_rate,
                },
                "after_warmup": {
                    "name": "CosineAnnealingLR",
                    "minimum_learning_rate": training_control.min_learning_rate,
                },
                "batch_size": batch_size,
            },
            "augmentation": (
                {"enabled": augmentation.enabled, **augmentation.checkpoint_config()}
                if augmentation is not None
                else {"enabled": False}
            ),
            "amp": {
                "requested": training_control.use_amp,
                "enabled": scaler.is_enabled(),
            },
            "training_history": history,
        },
        checkpoint,
    )
    (output / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    del (
        criterion,
        train_dataset,
        train_loader,
        validation_dataset,
        validation_loader,
        scaler,
        optimizer,
        model,
    )
    release_device_cache(device)
    return checkpoint


def _model_config(checkpoint: dict[str, object]) -> dict[str, object]:
    if checkpoint.get("architecture") != STAGE1_ARCHITECTURE:
        raise CheckpointError(
            "Stage 1 checkpoint is incompatible with the gated MS-TCN ablation "
            "architecture; retrain Stage 1 with the current model"
        )
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise CheckpointError("Stage 1 checkpoint model_config must be a mapping")
    return config


def load_stage1_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    require_cuda: bool = True,
) -> tuple[Stage1MViT, dict[str, object], torch.device]:
    """Load one strict multi-stream checkpoint for inference diagnostics."""

    device = choose_device(require_cuda=require_cuda)
    checkpoint = load_checkpoint(
        checkpoint_path,
        required_keys=("architecture", "model", "size", "frames", "model_config"),
    )
    config = _model_config(checkpoint)
    feature_mode = str(checkpoint.get("feature_mode", DEFAULT_FEATURE_MODE))
    feature_channels(feature_mode)
    row_profile_bins = int(config.get("row_profile_bins", DEFAULT_ROW_PROFILE_BINS))
    model = Stage1MViT(
        feature_mode=feature_mode,
        row_profile_bins=row_profile_bins,
        motion_iterations=int(config.get("motion_iterations", DEFAULT_MOTION_ITERATIONS)),
        correlation_radius=int(config.get("correlation_radius", DEFAULT_CORRELATION_RADIUS)),
        temporal_refinement_stages=int(
            config.get("temporal_refinement_stages", DEFAULT_MSTCN_STAGES)
        ),
        temporal_refinement_mode=str(
            config.get("temporal_refinement", DEFAULT_TEMPORAL_REFINEMENT_MODE)
        ),
        mstcn_gate_initial=float(
            config.get("mstcn_gate_initial", DEFAULT_MSTCN_GATE_INITIAL)
        ),
        zero_gate=bool(config.get("mstcn_zero_gate", False)),
        detach_mstcn_input=bool(
            config.get("mstcn_input_detached", DEFAULT_DETACH_MSTCN_INPUT)
        ),
        base_initialization_seed=int(
            config.get(
                "base_initialization_seed",
                DEFAULT_BASE_INITIALIZATION_SEED,
            )
        ),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint, device


def classifier_branch_weight_rms(model: Stage1MViT) -> dict[str, float]:
    """Summarize first fusion-layer weights with dimension-normalized RMS."""

    first_linear = model.classifier[1]
    if not isinstance(first_linear, nn.Linear):
        raise TypeError("Stage 1 classifier[1] must be a linear fusion layer")
    return {
        name: float(
            first_linear.weight[:, feature_slice]
            .detach()
            .float()
            .square()
            .mean()
            .sqrt()
            .cpu()
        )
        for name, feature_slice in model.branch_slices.items()
    }


def score_stage1_checkpoint_diagnostics(
    videos: Sequence[str | Path],
    checkpoint_path: str | Path,
    *,
    temporal_windows: int | None = None,
    batch_size: int = 4,
    num_workers: int | None = None,
    require_cuda: bool = True,
) -> dict[str, object]:
    """Return ordered window probabilities and per-branch contribution diagnostics.

    The probability trace is clip-level because Stage 1 produces one class per
    clip, not a frame segmentation. Adjacent windows therefore provide an
    over-segmentation/noise proxy rather than a frame-level prediction claim.
    """

    paths = [Path(video) for video in videos]
    if not paths:
        raise ValueError("Stage 1 diagnostics require at least one video")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    model, checkpoint, device = load_stage1_checkpoint_model(
        checkpoint_path,
        require_cuda=require_cuda,
    )
    config = _model_config(checkpoint)
    slots = resolve_tta_slots(checkpoint) if temporal_windows is None else int(temporal_windows)
    if slots < 2:
        raise ValueError("temporal_windows must be >= 2 for a temporal variability trace")
    workers = min(4, len(paths)) if num_workers is None else int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")
    dataset = Stage1InferenceDataset(
        paths,
        slots=slots,
        size=int(checkpoint["size"]),
        frames=int(checkpoint["frames"]),
        feature_mode=str(checkpoint.get("feature_mode", DEFAULT_FEATURE_MODE)),
        forensic_size=int(config.get("forensic_size", DEFAULT_FORENSIC_SIZE)),
        fft_size=int(config.get("fft_size", DEFAULT_FFT_SIZE)),
        row_profile_bins=int(config.get("row_profile_bins", DEFAULT_ROW_PROFILE_BINS)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    video_windows: list[list[dict[str, object]]] = [[] for _ in paths]
    failures: set[int] = set()
    first_linear = model.classifier[1]
    if not isinstance(first_linear, nn.Linear):
        raise TypeError("Stage 1 classifier[1] must be a linear fusion layer")

    with torch.inference_mode():
        for inputs, video_indices, valid in loader:
            moved = _move_inputs(inputs, device)
            with autocast_context(device):
                (
                    features,
                    motion_outputs,
                    frame_logits,
                    _,
                ) = model._extract_all_features(**moved)
                fused = model.fuse_branch_features(features)
                logits = model.combine_clip_logits(
                    model.classifier(fused),
                    frame_logits,
                )
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                frame_probabilities = torch.softmax(frame_logits, dim=2)[:, :, 1]
                frame_steps = torch.diff(frame_probabilities, dim=1).abs()
                frame_switches = torch.diff(
                    (frame_probabilities >= 0.5).to(torch.int8),
                    dim=1,
                ).abs().sum(dim=1)
                masks = motion_outputs["explainability_masks"]
                update_magnitudes = motion_outputs["flow_update_magnitudes"].mean(dim=1)
                normalized = model.classifier[0](fused)
                activation_rms = {
                    name: values.float().square().mean(dim=1).sqrt()
                    for name, values in features.items()
                }
                weighted_rms = {
                    name: F.linear(
                        normalized[:, feature_slice],
                        first_linear.weight[:, feature_slice],
                        bias=None,
                    ).float().square().mean(dim=1).sqrt()
                    for name, feature_slice in model.branch_slices.items()
                }
                ablation_deltas: dict[str, torch.Tensor] = {}
                for name, feature_slice in model.branch_slices.items():
                    ablated = fused.clone()
                    ablated[:, feature_slice] = 0.0
                    ablated_probability = torch.softmax(
                        model.combine_clip_logits(
                            model.classifier(ablated),
                            frame_logits,
                        ),
                        dim=1,
                    )[:, 1]
                    ablation_deltas[name] = probabilities - ablated_probability

            for batch_index, (video_index, ok) in enumerate(
                zip(video_indices.tolist(), valid.tolist())
            ):
                if not ok:
                    failures.add(int(video_index))
                    continue
                window_index = len(video_windows[video_index])
                video_windows[video_index].append(
                    {
                        "window_index": window_index,
                        "relative_position": (
                            window_index / (slots - 1) if slots > 1 else 0.0
                        ),
                        "rerecorded_probability": float(probabilities[batch_index].float().cpu()),
                        "frame_probability_mean_absolute_step": float(
                            frame_steps[batch_index].mean().float().cpu()
                        ),
                        "frame_probability_max_absolute_step": float(
                            frame_steps[batch_index].max().float().cpu()
                        ),
                        "frame_label_switches": int(frame_switches[batch_index].cpu()),
                        "explainability_mask_mean": float(
                            masks[batch_index].mean().float().cpu()
                        ),
                        "explainability_mask_min": float(
                            masks[batch_index].amin().float().cpu()
                        ),
                        "explainability_mask_max": float(
                            masks[batch_index].amax().float().cpu()
                        ),
                        "flow_update_magnitude_by_iteration": [
                            float(value)
                            for value in update_magnitudes[batch_index].float().cpu().tolist()
                        ],
                        "convgru_update_l2_by_iteration": [
                            float(value)
                            for value in update_magnitudes[batch_index].float().cpu().tolist()
                        ],
                        "flow_last_to_first_update_ratio": float(
                            (
                                update_magnitudes[batch_index, -1]
                                / update_magnitudes[batch_index, 0].clamp_min(1e-8)
                            ).float().cpu()
                        ),
                        "convgru_last_to_first_update_l2_ratio": float(
                            (
                                update_magnitudes[batch_index, -1]
                                / update_magnitudes[batch_index, 0].clamp_min(1e-8)
                            ).float().cpu()
                        ),
                        "activation_rms": {
                            name: float(values[batch_index].cpu())
                            for name, values in activation_rms.items()
                        },
                        "weighted_activation_rms": {
                            name: float(values[batch_index].cpu())
                            for name, values in weighted_rms.items()
                        },
                        "probability_delta_without_branch": {
                            name: float(values[batch_index].float().cpu())
                            for name, values in ablation_deltas.items()
                        },
                    }
                )

    records: list[dict[str, object]] = []
    for index, (path, windows) in enumerate(zip(paths, video_windows)):
        records.append(
            {
                "path": str(path),
                "valid": index not in failures and len(windows) == slots,
                "rerecorded_probability": (
                    float(np.mean([window["rerecorded_probability"] for window in windows]))
                    if windows
                    else None
                ),
                "windows": windows,
            }
        )
    weights = classifier_branch_weight_rms(model)
    del model
    release_device_cache(device)
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "temporal_windows": slots,
        "classifier_branch_weight_rms": weights,
        "videos": records,
    }


def score_stage1_checkpoint(
    videos: Sequence[str | Path],
    checkpoint_path: str | Path,
    *,
    tta_slots: int | None = None,
    require_cuda: bool = True,
) -> list[float]:
    """Score one fold and release it before another checkpoint is loaded."""

    paths = [Path(video) for video in videos]
    if not paths:
        raise ValueError("Stage 1 scoring requires at least one video")
    device = choose_device(require_cuda=require_cuda)
    checkpoint = load_checkpoint(
        checkpoint_path,
        required_keys=("architecture", "model", "size", "frames", "model_config"),
    )
    config = _model_config(checkpoint)
    size = int(checkpoint["size"])
    frame_count = int(checkpoint["frames"])
    feature_mode = str(checkpoint.get("feature_mode", DEFAULT_FEATURE_MODE))
    feature_channels(feature_mode)
    forensic_size = int(config.get("forensic_size", DEFAULT_FORENSIC_SIZE))
    fft_size = int(config.get("fft_size", DEFAULT_FFT_SIZE))
    row_profile_bins = int(config.get("row_profile_bins", DEFAULT_ROW_PROFILE_BINS))
    model = Stage1MViT(
        feature_mode=feature_mode,
        row_profile_bins=row_profile_bins,
        motion_iterations=int(config.get("motion_iterations", DEFAULT_MOTION_ITERATIONS)),
        correlation_radius=int(config.get("correlation_radius", DEFAULT_CORRELATION_RADIUS)),
        temporal_refinement_stages=int(
            config.get("temporal_refinement_stages", DEFAULT_MSTCN_STAGES)
        ),
        temporal_refinement_mode=str(
            config.get("temporal_refinement", DEFAULT_TEMPORAL_REFINEMENT_MODE)
        ),
        mstcn_gate_initial=float(
            config.get("mstcn_gate_initial", DEFAULT_MSTCN_GATE_INITIAL)
        ),
        zero_gate=bool(config.get("mstcn_zero_gate", False)),
        detach_mstcn_input=bool(
            config.get("mstcn_input_detached", DEFAULT_DETACH_MSTCN_INPUT)
        ),
        base_initialization_seed=int(
            config.get(
                "base_initialization_seed",
                DEFAULT_BASE_INITIALIZATION_SEED,
            )
        ),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    slots = resolve_tta_slots(checkpoint) if tta_slots is None else int(tta_slots)
    if slots < 1:
        raise ValueError("tta_slots must be >= 1")
    dataset = Stage1InferenceDataset(
        paths,
        slots=slots,
        size=size,
        frames=frame_count,
        feature_mode=feature_mode,
        forensic_size=forensic_size,
        fft_size=fft_size,
        row_profile_bins=row_profile_bins,
    )
    workers = min(4, len(paths))
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    scores: list[list[float]] = [[] for _ in paths]
    with torch.inference_mode():
        for inputs, video_indices, valid in loader:
            with autocast_context(device):
                probabilities = torch.softmax(
                    model(**_move_inputs(inputs, device)),
                    dim=1,
                )[:, 1]
            for index, value, ok in zip(
                video_indices.tolist(),
                probabilities.float().cpu().tolist(),
                valid.tolist(),
            ):
                if ok:
                    scores[index].append(float(value))

    del model
    release_device_cache(device)
    return [float(np.mean(values)) if values else 1.0 for values in scores]


def score_stage1_videos(
    videos: Sequence[str | Path],
    model_dir: str | Path,
    *,
    tta_slots: int | None = None,
) -> list[float]:
    """Average RERECORDED probabilities over early/middle/late regions."""

    return score_stage1_checkpoint(
        videos,
        Path(model_dir) / "best.pt",
        tta_slots=tta_slots,
    )


def predict_stage1(data_dir, model_dir):
    videos = video_paths(Path(data_dir) / "videos")
    probabilities = score_stage1_videos(videos, model_dir)
    rows = [
        {
            "ID": path.stem,
            "answer": "RERECORDED" if probability >= 0.5 else "ORIGINAL",
        }
        for path, probability in zip(videos, probabilities)
    ]
    frame = pd.DataFrame(rows, columns=["ID", "answer"])
    return validate_prediction_frame("stage1", frame)
