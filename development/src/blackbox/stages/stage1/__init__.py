"""Stage 1 training and inference API within the stage-specific package."""

from .baseline import Stage1MViT, fit_stage1, predict_stage1, score_stage1_videos
from .dataset import RGB_FEATURES, RGB_FFT_FEATURES
from .losses import (
    FocalLoss,
    Stage1MultiTaskLoss,
    explainability_reconstruction_terms,
    truncated_temporal_mse,
)

__all__ = [
    "FocalLoss",
    "RGB_FEATURES",
    "RGB_FFT_FEATURES",
    "Stage1MViT",
    "Stage1MultiTaskLoss",
    "explainability_reconstruction_terms",
    "fit_stage1",
    "predict_stage1",
    "score_stage1_videos",
    "truncated_temporal_mse",
]
