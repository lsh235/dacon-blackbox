#!/usr/bin/env bash
# Train the three stable baselines sequentially from one YAML configuration.
# The final inference runner releases GPU memory between Stage 1, 2, and 3.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-${PROJECT_ROOT}/development/.venv/bin/python}"
CONFIG_PATH=""

usage() {
    cat <<'EOF'
Usage: ./run_all.sh --config development/configs/baseline.yaml

Required:
  --config PATH                    YAML experiment configuration

Notes:
  Run ./preprocess_data.py with matching Stage 1 settings before baseline training.
  Two-Stream YAML files are for the stage-specific experiment CLIs; this runner
  deliberately writes submission-compatible baseline checkpoints only.

Options:
  -h, --help                       show this help
EOF
}

while (($#)); do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
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

if [[ -z "$CONFIG_PATH" ]]; then
    echo "--config is required." >&2
    usage >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/development/src${PYTHONPATH:+:${PYTHONPATH}}"
"$PYTHON_BIN" "${PROJECT_ROOT}/development/scripts/run_all.py" --config "$CONFIG_PATH"
