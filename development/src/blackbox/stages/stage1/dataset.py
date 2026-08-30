"""Stage 1 contiguous sampling and multi-stream forensic features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from blackbox.common.runtime import S1_MEAN, S1_STD, video_paths
from blackbox.preprocessing import DEFAULT_PROCESSED_ROOT, load_stage1_clip


RGB_FEATURES = "rgb"
RGB_FFT_FEATURES = "rgb_fft"
DEFAULT_FEATURE_MODE = RGB_FFT_FEATURES
DEFAULT_TEMPORAL_SLOTS = 3
DEFAULT_FORENSIC_SIZE = 320
DEFAULT_FFT_SIZE = 112
DEFAULT_ROW_PROFILE_BINS = 16
DEFAULT_JITTER_FRAMES = 4
DEFAULT_TRAIN_SEQUENCE_LENGTHS = (16, 24, 32)
_FEATURE_MODES = {RGB_FEATURES, RGB_FFT_FEATURES}

STAGE1_AUGMENTATION_PROFILES: dict[str, dict[str, bool]] = {
    "mode_g_full": {
        "enable_photometric": True,
        "enable_occlusion": True,
        "enable_affine": True,
    },
    "mode_g_no_aug": {
        "enable_photometric": False,
        "enable_occlusion": False,
        "enable_affine": False,
    },
    "mode_g_photo_only": {
        "enable_photometric": True,
        "enable_occlusion": False,
        "enable_affine": False,
    },
}


def stage1_augmentation_profile(name: str) -> dict[str, bool]:
    """Return a copy of a named Mode G augmentation ablation profile."""

    try:
        return dict(STAGE1_AUGMENTATION_PROFILES[name])
    except KeyError as exc:
        raise ValueError(
            f"unknown Stage 1 augmentation profile {name!r}; "
            f"expected one of {sorted(STAGE1_AUGMENTATION_PROFILES)}"
        ) from exc


@dataclass(frozen=True)
class _AugmentationParameters:
    color_operations: tuple[tuple[str, float], ...]
    affine: tuple[float, float, float] | None
    occlusion: tuple[float, float, float, float] | None


class Stage1TrainAugmentation:
    """Apply one clip-consistent transform to RGB and forensic views.

    Parameters are sampled once and then reused for every frame and for both
    spatial resolutions. FFT and flicker features are intentionally computed
    *after* this transform, so they always describe the augmented RGB sample.
    """

    def __init__(
        self,
        *,
        enable_photometric: bool = True,
        enable_occlusion: bool = True,
        color_jitter_probability: float = 0.8,
        affine_probability: float = 0.35,
        occlusion_probability: float = 0.5,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        max_degrees: float = 2.0,
        max_translate: float = 0.02,
        occlusion_scale: tuple[float, float] = (0.02, 0.20),
        occlusion_aspect_ratio: tuple[float, float] = (0.3, 3.3),
    ) -> None:
        probabilities = {
            "color_jitter_probability": color_jitter_probability,
            "affine_probability": affine_probability,
            "occlusion_probability": occlusion_probability,
        }
        invalid_probabilities = {
            name: value for name, value in probabilities.items() if not 0.0 <= value <= 1.0
        }
        if invalid_probabilities:
            raise ValueError(f"augmentation probabilities must be in [0, 1]: {invalid_probabilities}")
        magnitudes = {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "max_degrees": max_degrees,
            "max_translate": max_translate,
        }
        invalid_magnitudes = {name: value for name, value in magnitudes.items() if value < 0.0}
        if invalid_magnitudes:
            raise ValueError(f"augmentation magnitudes must be non-negative: {invalid_magnitudes}")
        if brightness >= 1.0 or contrast >= 1.0 or saturation >= 1.0:
            raise ValueError("brightness, contrast, and saturation must be < 1.0")
        if hue > 0.5:
            raise ValueError("hue must be <= 0.5")
        if (
            len(occlusion_scale) != 2
            or not 0.0 < occlusion_scale[0] <= occlusion_scale[1] <= 1.0
        ):
            raise ValueError("occlusion_scale must satisfy 0 < min <= max <= 1")
        if (
            len(occlusion_aspect_ratio) != 2
            or not 0.0 < occlusion_aspect_ratio[0] <= occlusion_aspect_ratio[1]
        ):
            raise ValueError("occlusion_aspect_ratio must satisfy 0 < min <= max")

        self.enable_photometric = bool(enable_photometric)
        self.enable_occlusion = bool(enable_occlusion)
        self.color_jitter_probability = float(color_jitter_probability)
        self.affine_probability = float(affine_probability)
        self.occlusion_probability = float(occlusion_probability)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.max_degrees = float(max_degrees)
        self.max_translate = float(max_translate)
        self.occlusion_scale = tuple(float(value) for value in occlusion_scale)
        self.occlusion_aspect_ratio = tuple(
            float(value) for value in occlusion_aspect_ratio
        )

    @staticmethod
    def _apply_to_frames(frames: torch.Tensor, operation, *args, **kwargs) -> torch.Tensor:
        return torch.stack([operation(frame, *args, **kwargs) for frame in frames], dim=0)

    def _sample_parameters(self) -> _AugmentationParameters:
        color_operations: tuple[tuple[str, float], ...] = ()
        if self.enable_photometric and bool(
            torch.rand(()) < self.color_jitter_probability
        ):
            candidates = (
                (
                    "brightness",
                    1.0 + float(torch.empty(()).uniform_(-self.brightness, self.brightness)),
                ),
                (
                    "contrast",
                    1.0 + float(torch.empty(()).uniform_(-self.contrast, self.contrast)),
                ),
                (
                    "saturation",
                    1.0 + float(torch.empty(()).uniform_(-self.saturation, self.saturation)),
                ),
                (
                    "hue",
                    float(torch.empty(()).uniform_(-self.hue, self.hue)),
                ),
            )
            color_operations = tuple(candidates[index] for index in torch.randperm(len(candidates)).tolist())

        affine: tuple[float, float, float] | None = None
        if self.affine_probability > 0.0 and bool(
            torch.rand(()) < self.affine_probability
        ):
            affine = (
                float(torch.empty(()).uniform_(-self.max_degrees, self.max_degrees)),
                float(torch.empty(()).uniform_(-self.max_translate, self.max_translate)),
                float(torch.empty(()).uniform_(-self.max_translate, self.max_translate)),
            )

        occlusion: tuple[float, float, float, float] | None = None
        if self.enable_occlusion and bool(
            torch.rand(()) < self.occlusion_probability
        ):
            log_ratio_min = math.log(self.occlusion_aspect_ratio[0])
            log_ratio_max = math.log(self.occlusion_aspect_ratio[1])
            for _ in range(10):
                area = float(
                    torch.empty(()).uniform_(
                        self.occlusion_scale[0],
                        self.occlusion_scale[1],
                    )
                )
                ratio = math.exp(
                    float(torch.empty(()).uniform_(log_ratio_min, log_ratio_max))
                )
                height_fraction = math.sqrt(area / ratio)
                width_fraction = math.sqrt(area * ratio)
                if height_fraction <= 1.0 and width_fraction <= 1.0:
                    top_fraction = float(
                        torch.empty(()).uniform_(0.0, 1.0 - height_fraction)
                    )
                    left_fraction = float(
                        torch.empty(()).uniform_(0.0, 1.0 - width_fraction)
                    )
                    occlusion = (
                        top_fraction,
                        left_fraction,
                        height_fraction,
                        width_fraction,
                    )
                    break
        return _AugmentationParameters(
            color_operations=color_operations,
            affine=affine,
            occlusion=occlusion,
        )

    def _apply_parameters(
        self,
        clip: torch.Tensor,
        parameters: _AugmentationParameters,
    ) -> torch.Tensor:
        if clip.ndim != 4 or clip.shape[0] != 3:
            raise ValueError(
                "RGB clips must have shape [3, time, height, width], "
                f"got {tuple(clip.shape)}"
            )
        frames = clip.permute(1, 0, 2, 3).contiguous()
        operations = {
            "brightness": transform_functional.adjust_brightness,
            "contrast": transform_functional.adjust_contrast,
            "saturation": transform_functional.adjust_saturation,
            "hue": transform_functional.adjust_hue,
        }
        for name, factor in parameters.color_operations:
            frames = self._apply_to_frames(frames, operations[name], factor)

        if parameters.affine is not None:
            angle, translate_x, translate_y = parameters.affine
            height, width = frames.shape[-2:]
            frames = self._apply_to_frames(
                frames,
                transform_functional.affine,
                angle,
                [int(round(translate_x * width)), int(round(translate_y * height))],
                1.0,
                [0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
        if parameters.occlusion is not None:
            top_fraction, left_fraction, height_fraction, width_fraction = (
                parameters.occlusion
            )
            height, width = frames.shape[-2:]
            top = min(height - 1, int(round(top_fraction * height)))
            left = min(width - 1, int(round(left_fraction * width)))
            erase_height = max(1, min(height - top, int(round(height_fraction * height))))
            erase_width = max(1, min(width - left, int(round(width_fraction * width))))
            frames = frames.clone()
            frames[:, :, top : top + erase_height, left : left + erase_width] = 0.0
        return frames.clamp(0.0, 1.0).permute(1, 0, 2, 3).contiguous()

    def apply_views(self, *clips: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not clips:
            raise ValueError("at least one RGB view is required")
        parameters = self._sample_parameters()
        return tuple(self._apply_parameters(clip, parameters) for clip in clips)

    def __call__(self, rgb_clip: torch.Tensor) -> torch.Tensor:
        return self.apply_views(rgb_clip)[0]

    @property
    def enabled(self) -> bool:
        return bool(
            (self.enable_photometric and self.color_jitter_probability > 0.0)
            or self.affine_probability > 0.0
            or (self.enable_occlusion and self.occlusion_probability > 0.0)
        )

    @property
    def profile_name(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.enable_photometric and self.enable_occlusion:
            return "aggressive_photometric_occlusion"
        if self.enable_photometric:
            return "photometric_only"
        if self.enable_occlusion:
            return "occlusion_only"
        return "geometric_only"

    def checkpoint_config(self) -> dict[str, float | bool | str]:
        return {
            "profile": self.profile_name,
            "enable_photometric": self.enable_photometric,
            "enable_occlusion": self.enable_occlusion,
            "color_jitter_probability": self.color_jitter_probability,
            "affine_probability": self.affine_probability,
            "occlusion_probability": self.occlusion_probability,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue,
            "max_degrees": self.max_degrees,
            "max_translate": self.max_translate,
            "occlusion_scale_min": self.occlusion_scale[0],
            "occlusion_scale_max": self.occlusion_scale[1],
            "occlusion_aspect_ratio_min": self.occlusion_aspect_ratio[0],
            "occlusion_aspect_ratio_max": self.occlusion_aspect_ratio[1],
        }


def feature_channels(feature_mode: str) -> int:
    """Return RGB backbone channels; FFT now has its own encoder."""

    if feature_mode not in _FEATURE_MODES:
        raise ValueError(
            f"unsupported Stage 1 feature mode: {feature_mode!r}; "
            f"expected one of {sorted(_FEATURE_MODES)}"
        )
    return 3


def contiguous_frame_indices(
    total_frames: int,
    frames: int,
    *,
    slot: int,
    slots: int = DEFAULT_TEMPORAL_SLOTS,
    context_jitter_frames: int = 0,
) -> np.ndarray:
    """Return a centered contiguous clip plus optional jitter context.

    The video is divided into temporal regions. Within each region the nominal
    clip is centered, and preprocessing may retain equal context on both sides.
    Training later selects a random contiguous crop from that context. Clipping
    repeats boundary frames for very short videos while keeping a fixed shape.
    """

    if total_frames < 1 or frames < 1 or slots < 1:
        raise ValueError("total_frames, frames, and slots must be >= 1")
    if slot < 0 or slot >= slots:
        raise ValueError(f"slot must satisfy 0 <= slot < slots: slot={slot}, slots={slots}")
    if context_jitter_frames < 0:
        raise ValueError("context_jitter_frames must be >= 0")

    boundaries = np.linspace(0, total_frames, slots + 1)
    slot_start = min(total_frames - 1, int(np.floor(boundaries[slot])))
    slot_stop = min(total_frames, max(slot_start + 1, int(np.ceil(boundaries[slot + 1]))))
    slot_length = slot_stop - slot_start
    nominal_start = slot_start + max(0, (slot_length - frames) // 2)
    context_start = nominal_start - context_jitter_frames
    count = frames + 2 * context_jitter_frames
    indices = context_start + np.arange(count, dtype=np.int64)
    return np.clip(indices, slot_start, slot_stop - 1)


def _as_float_clip(clip: torch.Tensor) -> torch.Tensor:
    if clip.ndim != 4 or clip.shape[0] != 3:
        raise ValueError(f"clip must have shape [3, time, height, width], got {tuple(clip.shape)}")
    if clip.dtype == torch.uint8:
        return clip.float().div_(255.0)
    value = clip.float()
    if not torch.isfinite(value).all():
        raise ValueError("clip must contain finite values")
    if value.numel() and (value.min() < 0.0 or value.max() > 1.0):
        raise ValueError("floating RGB clips must be in [0, 1]")
    return value


def spatial_log_spectrum(
    rgb_clip: torch.Tensor,
    *,
    output_size: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-frame FFT maps before reducing them for the 2-D CNN."""

    rgb_clip = _as_float_clip(rgb_clip)
    if output_size is not None and output_size < 1:
        raise ValueError("output_size must be >= 1")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    centered = rgb_clip - rgb_clip.mean(dim=(-2, -1), keepdim=True)
    coefficients = torch.fft.fft2(centered, dim=(-2, -1), norm="ortho")
    coefficients = torch.fft.fftshift(coefficients, dim=(-2, -1))
    spectrum = torch.log1p(coefficients.abs())
    mean = spectrum.mean(dim=(-2, -1), keepdim=True)
    std = spectrum.std(dim=(-2, -1), keepdim=True, unbiased=False)
    spectrum = (spectrum - mean) / std.clamp_min(eps)
    if output_size is not None and spectrum.shape[-2:] != (output_size, output_size):
        spectrum = F.interpolate(
            spectrum.permute(1, 0, 2, 3),
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    return spectrum.contiguous()


def temporal_flicker_features(
    rgb_clip: torch.Tensor,
    *,
    row_profile_bins: int = DEFAULT_ROW_PROFILE_BINS,
) -> torch.Tensor:
    """Extract Y(t), delta Y(t), and row-wise luminance time series."""

    rgb_clip = _as_float_clip(rgb_clip)
    if row_profile_bins < 1:
        raise ValueError("row_profile_bins must be >= 1")
    luminance = (
        0.299 * rgb_clip[0]
        + 0.587 * rgb_clip[1]
        + 0.114 * rgb_clip[2]
    )
    global_y = luminance.mean(dim=(-2, -1))
    delta_y = torch.diff(global_y, prepend=global_y[:1])
    row_profiles = F.adaptive_avg_pool2d(
        luminance[:, None],
        (row_profile_bins, 1),
    )[:, 0, :, 0].transpose(0, 1)
    return torch.cat((global_y[None], delta_y[None], row_profiles), dim=0).contiguous()


def prepare_stage1_inputs(
    rgb_clip: torch.Tensor,
    forensic_rgb_clip: torch.Tensor,
    *,
    feature_mode: str,
    fft_size: int = DEFAULT_FFT_SIZE,
    row_profile_bins: int = DEFAULT_ROW_PROFILE_BINS,
) -> dict[str, torch.Tensor]:
    """Create separate RGB, spatial-frequency, and temporal model inputs."""

    feature_channels(feature_mode)
    rgb_clip = _as_float_clip(rgb_clip)
    forensic_rgb_clip = _as_float_clip(forensic_rgb_clip)
    if rgb_clip.shape[1] != forensic_rgb_clip.shape[1]:
        raise ValueError("RGB and forensic views must contain the same number of frames")
    mean = S1_MEAN.to(device=rgb_clip.device, dtype=rgb_clip.dtype)
    std = S1_STD.to(device=rgb_clip.device, dtype=rgb_clip.dtype)
    normalized_rgb = (rgb_clip - mean) / std
    if feature_mode == RGB_FFT_FEATURES:
        fft_clip = spatial_log_spectrum(forensic_rgb_clip, output_size=fft_size)
    else:
        fft_clip = torch.zeros(
            3,
            rgb_clip.shape[1],
            fft_size,
            fft_size,
            dtype=rgb_clip.dtype,
            device=rgb_clip.device,
        )
    return {
        "rgb_clip": normalized_rgb.contiguous(),
        "fft_clip": fft_clip.contiguous(),
        "flicker": temporal_flicker_features(
            rgb_clip,
            row_profile_bins=row_profile_bins,
        ),
    }


def _resize_center_crop(rgb: np.ndarray, size: int) -> torch.Tensor:
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized_height = max(size, round(height * scale))
    resized_width = max(size, round(width * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    y = (resized_height - size) // 2
    x = (resized_width - size) // 2
    cropped = resized[y : y + size, x : x + size].copy()
    return torch.from_numpy(cropped).permute(2, 0, 1).to(torch.uint8)


def _native_center_crop(rgb: np.ndarray, size: int) -> torch.Tensor:
    """Crop the native-resolution center square before forensic resizing."""

    height, width = rgb.shape[:2]
    edge = min(height, width)
    y = (height - edge) // 2
    x = (width - edge) // 2
    cropped = rgb[y : y + edge, x : x + edge]
    if cropped.shape[:2] != (size, size):
        interpolation = cv2.INTER_AREA if edge >= size else cv2.INTER_CUBIC
        cropped = cv2.resize(cropped, (size, size), interpolation=interpolation)
    return torch.from_numpy(cropped.copy()).permute(2, 0, 1).to(torch.uint8)


def decode_contiguous_views(
    path: str | Path,
    *,
    size: int,
    forensic_size: int,
    frames: int,
    slot: int,
    slots: int = DEFAULT_TEMPORAL_SLOTS,
    context_jitter_frames: int = 0,
) -> dict[str, torch.Tensor]:
    """Decode one bounded contiguous region into RGB and forensic views."""

    if min(size, forensic_size) < 1:
        raise ValueError("size and forensic_size must be >= 1")
    video = Path(path)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")
        total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        wanted = contiguous_frame_indices(
            total,
            frames,
            slot=slot,
            slots=slots,
            context_jitter_frames=context_jitter_frames,
        ).tolist()
        rgb_output: list[torch.Tensor] = []
        forensic_output: list[torch.Tensor] = []
        capture.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
        position = wanted[0]
        cached_index: int | None = None
        cached_rgb: torch.Tensor | None = None
        cached_forensic: torch.Tensor | None = None

        for index in wanted:
            if index == cached_index and cached_rgb is not None and cached_forensic is not None:
                rgb_output.append(cached_rgb)
                forensic_output.append(cached_forensic)
                continue
            ok = False
            bgr = None
            while position <= index:
                ok, bgr = capture.read()
                position += 1
                if not ok:
                    break
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cached_rgb = _resize_center_crop(rgb, size)
            cached_forensic = _native_center_crop(rgb, forensic_size)
            cached_index = index
            rgb_output.append(cached_rgb)
            forensic_output.append(cached_forensic)
    finally:
        capture.release()

    if not rgb_output:
        raise ValueError(f"cannot decode video: {video.name}")
    expected = frames + 2 * context_jitter_frames
    while len(rgb_output) < expected:
        rgb_output.append(rgb_output[-1])
        forensic_output.append(forensic_output[-1])
    return {
        "rgb": torch.stack(rgb_output, dim=1),
        "forensic_rgb": torch.stack(forensic_output, dim=1),
    }


class Stage1TrainingDataset(Dataset):
    """Three-region samples with variable contiguous crops from offline caches.

    ``frames`` is the length of the cached central region, while
    ``sequence_lengths`` controls the length returned to the model.  Training
    randomly chooses one configured length for every sample.  A fixed-length
    validation dataset is obtained by passing a one-element sequence.
    """

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        *,
        size: int,
        frames: int,
        feature_mode: str,
        slots: int = DEFAULT_TEMPORAL_SLOTS,
        jitter_frames: int = DEFAULT_JITTER_FRAMES,
        forensic_size: int = DEFAULT_FORENSIC_SIZE,
        fft_size: int = DEFAULT_FFT_SIZE,
        row_profile_bins: int = DEFAULT_ROW_PROFILE_BINS,
        sequence_lengths: Sequence[int] | None = None,
        random_jitter: bool = True,
        augmentation: Stage1TrainAugmentation | None = None,
        processed_root: str | Path = DEFAULT_PROCESSED_ROOT,
    ) -> None:
        if not samples:
            raise ValueError("Stage 1 training samples must not be empty")
        if min(size, frames, slots, forensic_size, fft_size, row_profile_bins) < 1:
            raise ValueError("Stage 1 geometry values must be >= 1")
        if jitter_frames < 0:
            raise ValueError("jitter_frames must be >= 0")
        parsed_sequence_lengths = tuple(
            dict.fromkeys(
                int(length)
                for length in (
                    (frames,) if sequence_lengths is None else sequence_lengths
                )
            )
        )
        if not parsed_sequence_lengths:
            raise ValueError("sequence_lengths must contain at least one clip length")
        if min(parsed_sequence_lengths) < 2:
            raise ValueError("Stage 1 sequence lengths must be >= 2")
        if max(parsed_sequence_lengths) > frames:
            raise ValueError(
                "sequence lengths cannot exceed the cached central region: "
                f"max={max(parsed_sequence_lengths)}, cache_frames={frames}"
            )
        feature_channels(feature_mode)
        invalid_labels = sorted({label for _, label in samples if label not in (0, 1)})
        if invalid_labels:
            raise ValueError(f"Stage 1 labels must be 0 or 1: {invalid_labels}")
        self.samples = list(samples)
        self.size = size
        self.frames = frames
        self.feature_mode = feature_mode
        self.slots = slots
        self.jitter_frames = jitter_frames
        self.forensic_size = forensic_size
        self.fft_size = fft_size
        self.row_profile_bins = row_profile_bins
        self.sequence_lengths = parsed_sequence_lengths
        self.random_jitter = random_jitter
        self.augmentation = augmentation
        self.processed_root = Path(processed_root)

    def __len__(self) -> int:
        return len(self.samples) * self.slots

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int, int]:
        video_index, slot = divmod(index, self.slots)
        path, label = self.samples[video_index]
        cached = load_stage1_clip(
            self.processed_root,
            path,
            size=self.size,
            frames=self.frames,
            slot=slot,
            slots=self.slots,
            jitter_frames=self.jitter_frames,
            forensic_size=self.forensic_size,
        )
        sequence_index = (
            int(torch.randint(len(self.sequence_lengths), ()).item())
            if len(self.sequence_lengths) > 1
            else 0
        )
        sequence_length = self.sequence_lengths[sequence_index]
        jitter_offset = (
            int(torch.randint(2 * self.jitter_frames + 1, ()).item())
            if self.random_jitter and self.jitter_frames > 0
            else self.jitter_frames
        )
        centered_offset = (self.frames - sequence_length) // 2
        offset = centered_offset + jitter_offset
        stop = offset + sequence_length
        rgb_clip = cached["rgb"][:, offset:stop].float().div_(255.0)
        forensic_clip = cached["forensic_rgb"][:, offset:stop].float().div_(255.0)
        if self.augmentation is not None:
            rgb_clip, forensic_clip = self.augmentation.apply_views(rgb_clip, forensic_clip)
        inputs = prepare_stage1_inputs(
            rgb_clip,
            forensic_clip,
            feature_mode=self.feature_mode,
            fft_size=self.fft_size,
            row_profile_bins=self.row_profile_bins,
        )
        return inputs, label, video_index


class Stage1InferenceDataset(Dataset):
    """Online deterministic early/middle/late clips for submission inference."""

    def __init__(
        self,
        videos: Sequence[Path],
        *,
        slots: int,
        size: int,
        frames: int,
        feature_mode: str,
        forensic_size: int,
        fft_size: int,
        row_profile_bins: int,
    ) -> None:
        if min(slots, size, frames, forensic_size, fft_size, row_profile_bins) < 1:
            raise ValueError("Stage 1 inference geometry values must be >= 1")
        feature_channels(feature_mode)
        self.videos = list(videos)
        self.slots = slots
        self.size = size
        self.frames = frames
        self.feature_mode = feature_mode
        self.forensic_size = forensic_size
        self.fft_size = fft_size
        self.row_profile_bins = row_profile_bins

    def __len__(self) -> int:
        return len(self.videos) * self.slots

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int, int]:
        video_index, slot = divmod(index, self.slots)
        path = self.videos[video_index]
        try:
            views = decode_contiguous_views(
                path,
                size=self.size,
                forensic_size=self.forensic_size,
                frames=self.frames,
                slot=slot,
                slots=self.slots,
            )
            inputs = prepare_stage1_inputs(
                views["rgb"],
                views["forensic_rgb"],
                feature_mode=self.feature_mode,
                fft_size=self.fft_size,
                row_profile_bins=self.row_profile_bins,
            )
            valid = 1
        except (OSError, ValueError, cv2.error):
            inputs = {
                "rgb_clip": torch.zeros(3, self.frames, self.size, self.size),
                "fft_clip": torch.zeros(3, self.frames, self.fft_size, self.fft_size),
                "flicker": torch.zeros(2 + self.row_profile_bins, self.frames),
            }
            valid = 0
        return inputs, video_index, valid


class Stage1TestDataset(Stage1InferenceDataset):
    """Discover test videos and apply validation-equivalent preprocessing.

    This dataset deliberately has no augmentation argument.  Every item uses
    the deterministic centered clip from each of the three temporal regions,
    followed only by center cropping, feature extraction, and normalization.
    ``test_data_dir`` may be either the Stage 1 directory containing ``videos``
    or the video directory itself because discovery is recursive.
    """

    def __init__(
        self,
        test_data_dir: str | Path,
        *,
        size: int,
        frames: int,
        feature_mode: str = DEFAULT_FEATURE_MODE,
        slots: int = DEFAULT_TEMPORAL_SLOTS,
        forensic_size: int = DEFAULT_FORENSIC_SIZE,
        fft_size: int = DEFAULT_FFT_SIZE,
        row_profile_bins: int = DEFAULT_ROW_PROFILE_BINS,
    ) -> None:
        if slots != DEFAULT_TEMPORAL_SLOTS:
            raise ValueError(
                "Stage 1 test inference must use the validation-equivalent "
                f"three-region policy, got slots={slots}"
            )
        self.test_data_dir = Path(test_data_dir)
        videos = video_paths(self.test_data_dir)
        video_ids = [path.stem for path in videos]
        duplicate_ids = sorted(
            video_id for video_id, count in Counter(video_ids).items() if count > 1
        )
        if duplicate_ids:
            raise ValueError(
                "Stage 1 test video stems must be unique for sample-submission "
                f"alignment: {duplicate_ids}"
            )
        super().__init__(
            videos,
            slots=slots,
            size=size,
            frames=frames,
            feature_mode=feature_mode,
            forensic_size=forensic_size,
            fft_size=fft_size,
            row_profile_bins=row_profile_bins,
        )

    @property
    def video_ids(self) -> list[str]:
        return [path.stem for path in self.videos]


class Stage1ValidationDataset(Stage1InferenceDataset):
    """Labeled validation view using the exact online inference decoder."""

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        *,
        slots: int,
        size: int,
        frames: int,
        feature_mode: str,
        forensic_size: int,
        fft_size: int,
        row_profile_bins: int,
    ) -> None:
        if not samples:
            raise ValueError("Stage 1 validation samples must not be empty")
        invalid_labels = sorted({label for _, label in samples if label not in (0, 1)})
        if invalid_labels:
            raise ValueError(f"Stage 1 labels must be 0 or 1: {invalid_labels}")
        self.labels = [int(label) for _, label in samples]
        super().__init__(
            [path for path, _ in samples],
            slots=slots,
            size=size,
            frames=frames,
            feature_mode=feature_mode,
            forensic_size=forensic_size,
            fft_size=fft_size,
            row_profile_bins=row_profile_bins,
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[dict[str, torch.Tensor], int, int, int]:
        inputs, video_index, valid = super().__getitem__(index)
        return inputs, self.labels[video_index], video_index, valid
