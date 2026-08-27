#!/usr/bin/env bash
# Train the three stable baselines sequentially, then generate all Stage CSVs.
# Each Python inference call releases GPU memory between Stage 1, 2, and 3.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-${PROJECT_ROOT}/development/.venv/bin/python}"
TRAIN_DATA_ROOT=""
INFERENCE_DATA_ROOT=""
ARTIFACT_ROOT="${PROJECT_ROOT}/development/artifacts/iteration5-run-all"
EPOCHS=1
STAGE3_FRAMES_PER_SAMPLE=""

usage() {
    cat <<'EOF'
Usage: ./run_all.sh --data-root TRAIN_DATA --inference-data-root INFERENCE_DATA [options]

Required:
  --data-root PATH                 root containing stage1/, stage2/, stage3/ labels and videos
  --inference-data-root PATH       root containing Stage 1/2/3 inference directories

Options:
  --epochs N                       epochs per stage (default: 1)
  --artifact-root PATH             output root under development/artifacts/ by default
  --stage3-frames-per-sample N     diagnostic override; default uses CAP_PROP_FPS / 10 per video
  -h, --help                       show this help
EOF
}

while (($#)); do
    case "$1" in
        --data-root)
            TRAIN_DATA_ROOT="$2"
            shift 2
            ;;
        --inference-data-root)
            INFERENCE_DATA_ROOT="$2"
            shift 2
            ;;
        --artifact-root)
            ARTIFACT_ROOT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --stage3-frames-per-sample)
            STAGE3_FRAMES_PER_SAMPLE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$TRAIN_DATA_ROOT" || -z "$INFERENCE_DATA_ROOT" ]]; then
    echo "--data-root and --inference-data-root are required." >&2
    usage >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/development/src${PYTHONPATH:+:${PYTHONPATH}}"
MODEL_ROOT="${ARTIFACT_ROOT}/models"
SUBMISSION_ROOT="${ARTIFACT_ROOT}/submissions"
LOG_ROOT="${PROJECT_ROOT}/development/logs"

"$PYTHON_BIN" "${PROJECT_ROOT}/development/scripts/train/train_baseline.py" \
    --data-root "$TRAIN_DATA_ROOT" --model-root "$MODEL_ROOT" --stages 1 \
    --epochs "$EPOCHS" --log-dir "$LOG_ROOT"

"$PYTHON_BIN" "${PROJECT_ROOT}/development/scripts/train/train_baseline.py" \
    --data-root "$TRAIN_DATA_ROOT" --model-root "$MODEL_ROOT" --stages 2 \
    --epochs "$EPOCHS" --log-dir "$LOG_ROOT"

"$PYTHON_BIN" "${PROJECT_ROOT}/development/scripts/train/train_baseline.py" \
    --data-root "$TRAIN_DATA_ROOT" --model-root "$MODEL_ROOT" --stages 3 \
    --epochs "$EPOCHS" --log-dir "$LOG_ROOT"

SUBMISSION_ARGS=(
    --data-root "$INFERENCE_DATA_ROOT"
    --model-root "$MODEL_ROOT"
    --output-root "$SUBMISSION_ROOT"
)
if [[ -n "$STAGE3_FRAMES_PER_SAMPLE" ]]; then
    SUBMISSION_ARGS+=(--stage3-frames-per-sample "$STAGE3_FRAMES_PER_SAMPLE")
fi
"$PYTHON_BIN" "${PROJECT_ROOT}/development/scripts/submission/generate_submission.py" "${SUBMISSION_ARGS[@]}"

echo "[OK] Stage 1/2/3 models and CSVs: ${ARTIFACT_ROOT}"
