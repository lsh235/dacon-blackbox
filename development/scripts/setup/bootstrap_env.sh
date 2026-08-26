#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${project_root}/.venv"
system_python="${SYSTEM_PYTHON:-python3}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  if ! "${system_python}" -m venv "${venv_dir}"; then
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
      echo "Failed to create the virtual environment: ${venv_dir}" >&2
      exit 1
    fi
  fi
fi

if ! "${venv_dir}/bin/python" -m pip --version >/dev/null 2>&1; then
  "${system_python}" -m pip --python "${venv_dir}" install --upgrade pip setuptools wheel
fi

"${venv_dir}/bin/python" -m pip install -r "${project_root}/requirements.txt"
PYTHONPATH="${project_root}/src" \
  "${venv_dir}/bin/python" "${project_root}/scripts/setup/check_environment.py" \
  --requirements "${project_root}/requirements.txt"
