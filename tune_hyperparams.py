#!/usr/bin/env python3
"""Run Optuna HPO for the Stage 3 ego-motion architecture."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_PYTHON = ROOT / "development" / ".venv" / "bin" / "python"
if PROJECT_PYTHON.is_file() and Path(sys.executable).absolute() != PROJECT_PYTHON:
    os.execv(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )
sys.path.insert(0, str(ROOT / "development" / "src"))

from blackbox.hpo import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
