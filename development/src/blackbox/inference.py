"""Development entry point matching the competition inference function contract."""

from blackbox.stages.stage1 import predict_stage1
from blackbox.stages.stage2 import predict_stage2
from blackbox.stages.stage3 import predict_stage3

__all__ = ["predict_stage1", "predict_stage2", "predict_stage3"]
