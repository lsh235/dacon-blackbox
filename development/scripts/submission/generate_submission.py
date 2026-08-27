#!/usr/bin/env python3
"""Generate Stage 1/2/3 contract CSVs with sequential model memory release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from blackbox.experiment_config import (
    config_path_list,
    config_path_value,
    load_experiment_config,
    section,
)
from blackbox.submission_pipeline import generate_submission_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML experiment configuration")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--stage1-checkpoints",
        type=Path,
        nargs="+",
        help="Stage 1 fold best.pt paths (or directories containing best.pt)",
    )
    parser.add_argument(
        "--stage2-checkpoints",
        type=Path,
        nargs="+",
        help="Stage 2 fold best.pt paths; each backbone must be beside best.pt",
    )
    parser.add_argument(
        "--stage3-checkpoints",
        type=Path,
        nargs="+",
        help="Stage 3 fold best.pt paths (or directories containing best.pt)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        help="Positive odd moving-average window on projected Stage 3 probabilities",
    )
    parser.add_argument(
        "--use-transition-constraints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply physics-aware Viterbi transition constraints to Stage 3",
    )
    parser.add_argument("--transition-penalty", type=float)
    parser.add_argument(
        "--stage3-frames-per-sample",
        type=int,
        help=(
            "Optional override for the source-frame count per Stage 3 0.1-second sample. "
            "Default: retain every decoded frame from the official 10 Hz evaluation video."
        ),
    )
    parser.add_argument("--stage1-sample-submission", type=Path)
    parser.add_argument("--stage2-sample-submission", type=Path)
    parser.add_argument("--stage3-sample-submission", type=Path)
    args = parser.parse_args()
    config: dict[str, object] = {}
    config_path: Path | None = None
    if args.config is not None:
        config, config_path = load_experiment_config(args.config)
    data = section(config, "data")
    run = section(config, "run")
    inference = section(config, "inference")
    stage3 = section(config, "stage3")
    configured_checkpoints = inference.get("checkpoints", {})
    if not isinstance(configured_checkpoints, dict):
        parser.error("inference.checkpoints must be a mapping")
    if config_path is not None:
        configured_data_root = config_path_value(
            config_path,
            data.get("inference_root"),
            field="data.inference_root",
        )
        configured_model_root = config_path_value(
            config_path,
            run.get("model_root"),
            field="run.model_root",
        )
        configured_output_root = config_path_value(
            config_path,
            run.get("output_root"),
            field="run.output_root",
        )
        checkpoint_paths = {
            stage_number: config_path_list(
                config_path,
                configured_checkpoints.get(f"stage{stage_number}", []),
                field=f"inference.checkpoints.stage{stage_number}",
            )
            for stage_number in (1, 2, 3)
        }
    else:
        configured_data_root = configured_model_root = configured_output_root = None
        checkpoint_paths = {1: [], 2: [], 3: []}
    data_root = args.data_root or configured_data_root
    model_root = args.model_root or configured_model_root
    output_root = args.output_root or configured_output_root
    if data_root is None or model_root is None or output_root is None:
        parser.error("--config or --data-root, --model-root, and --output-root are required")
    for stage_number, cli_paths in (
        (1, args.stage1_checkpoints),
        (2, args.stage2_checkpoints),
        (3, args.stage3_checkpoints),
    ):
        if cli_paths is not None:
            checkpoint_paths[stage_number] = cli_paths
    frames_per_sample = (
        args.stage3_frames_per_sample
        if args.stage3_frames_per_sample is not None
        else stage3.get("frames_per_sample")
    )
    smoothing_window = (
        args.smoothing_window
        if args.smoothing_window is not None
        else int(inference.get("smoothing_window", 1))
    )
    use_transition_constraints = (
        args.use_transition_constraints
        if args.use_transition_constraints is not None
        else bool(inference.get("use_transition_constraints", True))
    )
    transition_penalty = (
        args.transition_penalty
        if args.transition_penalty is not None
        else float(inference.get("transition_penalty", -1e9))
    )
    sample_submissions = {
        stage: path
        for stage, path in (
            (1, args.stage1_sample_submission),
            (2, args.stage2_sample_submission),
            (3, args.stage3_sample_submission),
        )
        if path is not None
    }
    summary = generate_submission_bundle(
        data_root,
        model_root,
        output_root,
        stage3_frames_per_sample=None if frames_per_sample is None else int(frames_per_sample),
        smoothing_window=smoothing_window,
        use_transition_constraints=use_transition_constraints,
        transition_penalty=transition_penalty,
        checkpoint_paths={stage: paths for stage, paths in checkpoint_paths.items() if paths},
        sample_submissions=sample_submissions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
