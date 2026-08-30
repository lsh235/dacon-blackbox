"""Top-level DACON inference entrypoints packaged by ``build_submission.sh``."""

# __BLACKBOX_EMBEDDED_RUNTIME__

from blackbox.inference import (
    predict_stage1 as _predict_stage1,
    predict_stage2 as _predict_stage2,
    predict_stage3 as _predict_stage3,
)


def predict_stage1(data_dir, model_dir):
    return _predict_stage1(data_dir, model_dir)


def predict_stage2(data_dir, model_dir):
    return _predict_stage2(data_dir, model_dir)


def predict_stage3(data_dir, model_dir):
    return _predict_stage3(data_dir, model_dir)
