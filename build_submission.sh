#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_python="${project_root}/development/.venv/bin/python"
if [[ ! -x "${project_python}" ]]; then
  project_python="python3"
fi

export PYTHONPATH="${project_root}/development/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${project_python}" "${project_root}/development/scripts/submission/build_submission.py" "$@"
