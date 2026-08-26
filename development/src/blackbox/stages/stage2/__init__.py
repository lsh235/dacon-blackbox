"""Stage 2 baseline API within the stage-specific package."""

from .baseline import Stage2Temporal, fit_stage2, predict_stage2

__all__ = ["Stage2Temporal", "fit_stage2", "predict_stage2"]
