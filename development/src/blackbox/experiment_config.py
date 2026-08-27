"""YAML experiment configuration loading with paths relative to the config."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load one mapping-only YAML file and return its resolved path."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"experiment config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML experiment config: {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config must be a mapping: {config_path}")
    return payload, config_path


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def config_path_value(config_path: Path, value: str | Path | None, *, field: str) -> Path:
    if value is None:
        raise ValueError(f"config is missing required path: {field}")
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path


def stage_paths(config: Mapping[str, Any], config_path: Path, stage: str) -> tuple[Path, Path, Path]:
    """Resolve data, model, and processed paths for one Stage training command."""

    data = section(config, "data")
    run = section(config, "run")
    root = config_path_value(config_path, data.get("root"), field="data.root")
    model_root = config_path_value(config_path, run.get("model_root"), field="run.model_root")
    processed_root = config_path_value(
        config_path,
        data.get("processed_root", "../data/processed"),
        field="data.processed_root",
    )
    return root / stage, model_root / stage, processed_root
