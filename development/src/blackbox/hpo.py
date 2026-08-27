"""Optuna search driver for the Stage 3 ego-motion two-stream model."""

from __future__ import annotations

import argparse
import gc
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from blackbox.common.runtime import (
    DEFAULT_SEED,
    choose_device,
    make_grad_scaler,
    release_device_cache,
    seed_everything,
)
from blackbox.experiment_config import load_experiment_config, section, stage_paths
from blackbox.stages.stage3.dataset_stage3 import (
    Stage3SequenceWindowDataset,
    collate_stage3_windows,
    read_stage3_records,
)
from blackbox.stages.stage3.model_stage3 import Stage3TwoStreamBiLSTM
from blackbox.stages.stage3.train_stage3 import (
    evaluate_stage3_macro_f1,
    run_stage3_epoch,
)
from blackbox.training_control import group_holdout_indices


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "experiment_two_stream.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "hpo-stage3"
SEARCH_SPACE_VERSION = 1


def suggest_stage3_hparams(trial: optuna.trial.BaseTrial) -> dict[str, float | int]:
    """Sample the four Iteration 9 parameters from bounded practical ranges."""

    return {
        "flow_roi_top_ratio": trial.suggest_float(
            "flow_roi_top_ratio", 0.35, 0.65, step=0.15
        ),
        "flow_grid_size": trial.suggest_categorical("flow_grid_size", [2, 3, 4]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "focal_gamma": trial.suggest_categorical("focal_gamma", [0.0, 1.0, 2.0, 3.0]),
    }


def materialize_best_config(
    base_config: dict[str, Any],
    *,
    best_params: dict[str, Any],
    fixed_stage3_params: dict[str, Any] | None = None,
    best_value: float,
    trial_number: int,
    epochs_per_trial: int,
    run_mode: str = "stage3_data",
) -> dict[str, Any]:
    """Merge one study result into a directly reusable experiment YAML."""

    output = deepcopy(base_config)
    stage3 = output.setdefault("stage3", {})
    if not isinstance(stage3, dict):
        raise ValueError("base config stage3 section must be a mapping")
    stage3.update(fixed_stage3_params or {})
    stage3.update(best_params)
    output["hpo"] = {
        "backend": "optuna",
        "mode": run_mode,
        "direction": "maximize",
        "metric": "validation_accel_steer_mean_macro_f1",
        "search_space_version": SEARCH_SPACE_VERSION,
        "best_value": float(best_value),
        "best_trial": int(trial_number),
        "epochs_per_trial": int(epochs_per_trial),
        "fixed_stage3_params": fixed_stage3_params or {},
    }
    return output


def _make_synthetic_loader(*, samples: int, seed: int, shuffle: bool) -> DataLoader:
    """Create deterministic tensors that exercise the real Stage 3 train path."""

    generator = torch.Generator().manual_seed(seed)
    items: list[dict[str, torch.Tensor]] = []
    time = 3
    size = 64
    for sample_index in range(samples):
        accel = (torch.arange(time) + sample_index) % 4
        steer = (torch.arange(time) + sample_index) % 3
        frames = torch.rand((time, 3, size, size), generator=generator) * 0.08
        flow = torch.randn((time, 2, size, size), generator=generator) * 0.02
        for step in range(time):
            frames[step] += float(accel[step]) / 12.0
            flow[step, 0] += float(accel[step] - 1) * 0.08
            flow[step, 1] += float(steer[step] - 1) * 0.08
        items.append(
            {
                "frames": frames,
                "flow": flow,
                "valid_length": torch.tensor(time, dtype=torch.long),
                "accel_targets": accel.long(),
                "steer_targets": steer.long(),
            }
        )
    loader_generator = torch.Generator().manual_seed(seed + 1)
    return DataLoader(
        items,
        batch_size=2,
        shuffle=shuffle,
        generator=loader_generator,
        num_workers=0,
    )


def _train_trial(
    trial: optuna.Trial,
    *,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    seed: int,
    use_amp: bool,
    flow_projection_dim: int,
    hidden_size: int,
    layers: int,
    frame_batch_size: int,
    use_physics_vector: bool,
    physics_projection_dim: int,
) -> float:
    params = suggest_stage3_hparams(trial)
    seed_everything(seed + trial.number)
    device = choose_device()
    model = Stage3TwoStreamBiLSTM(
        hidden_size=hidden_size,
        layers=layers,
        frame_batch_size=frame_batch_size,
        flow_roi_top_ratio=float(params["flow_roi_top_ratio"]),
        flow_grid_size=int(params["flow_grid_size"]),
        flow_projection_dim=flow_projection_dim,
        use_physics_vector=use_physics_vector,
        physics_projection_dim=physics_projection_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
    scaler = make_grad_scaler(device, enabled=use_amp)
    best_f1 = 0.0
    try:
        for epoch in range(epochs):
            loss = run_stage3_epoch(
                model,
                train_loader,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
                focal_gamma=float(params["focal_gamma"]),
            )
            if loss is None:
                raise ValueError("HPO trial found no Stage 3 training targets")
            valid_f1 = evaluate_stage3_macro_f1(
                model,
                valid_loader,
                device=device,
                use_amp=use_amp,
            )
            if valid_f1 is None:
                raise ValueError("HPO trial found no Stage 3 validation targets")
            best_f1 = max(best_f1, valid_f1)
            trial.report(valid_f1, step=epoch)
            trial.set_user_attr(f"epoch_{epoch + 1}_train_loss", loss)
            if trial.should_prune():
                raise optuna.TrialPruned(f"validation Macro F1={valid_f1:.6f}")
        return best_f1
    finally:
        del scaler, optimizer, model
        gc.collect()
        release_device_cache(device)


def _smoke_objective(args: argparse.Namespace) -> Callable[[optuna.Trial], float]:
    train_loader = _make_synthetic_loader(samples=8, seed=args.seed, shuffle=True)
    valid_loader = _make_synthetic_loader(samples=4, seed=args.seed + 100, shuffle=False)

    def objective(trial: optuna.Trial) -> float:
        return _train_trial(
            trial,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=args.epochs,
            seed=args.seed,
            use_amp=args.use_amp,
            flow_projection_dim=64,
            hidden_size=16,
            layers=1,
            frame_batch_size=4,
            use_physics_vector=True,
            physics_projection_dim=8,
        )

    return objective


def _real_objective(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    config_path: Path,
) -> Callable[[optuna.Trial], float]:
    configured_data, _, configured_processed = stage_paths(config, config_path, "stage3")
    data_dir = (args.data_dir or configured_data).resolve()
    processed_root = (args.processed_root or configured_processed).resolve()
    records = read_stage3_records(data_dir)
    train_indices, valid_indices = group_holdout_indices(
        [record.video_id for record in records],
        validation_fraction=args.validation_fraction,
    )
    if not valid_indices:
        raise ValueError("Stage 3 HPO requires at least two video groups for validation")
    train_records = [record for index, record in enumerate(records) if index in train_indices]
    valid_records = [record for index, record in enumerate(records) if index in valid_indices]
    train_dataset = Stage3SequenceWindowDataset(
        train_records,
        window_frames=args.window_frames,
        stride=args.stride,
        size=args.size,
        processed_root=processed_root,
    )
    valid_dataset = Stage3SequenceWindowDataset(
        valid_records,
        window_frames=args.window_frames,
        stride=args.stride,
        size=args.size,
        processed_root=processed_root,
    )

    def objective(trial: optuna.Trial) -> float:
        loader_generator = torch.Generator().manual_seed(args.seed + trial.number)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_stage3_windows,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=loader_generator,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_stage3_windows,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return _train_trial(
            trial,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=args.epochs,
            seed=args.seed,
            use_amp=args.use_amp,
            flow_projection_dim=args.flow_projection_dim,
            hidden_size=args.hidden_size,
            layers=args.layers,
            frame_batch_size=args.frame_batch_size,
            use_physics_vector=args.use_physics_vector,
            physics_projection_dim=args.physics_projection_dim,
        )

    return objective


def _write_study_outputs(
    study: optuna.Study,
    *,
    config: dict[str, Any],
    output_dir: Path,
    epochs: int,
    fixed_stage3_params: dict[str, Any],
    run_mode: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_config = materialize_best_config(
        config,
        best_params=study.best_trial.params,
        fixed_stage3_params=fixed_stage3_params,
        best_value=study.best_value,
        trial_number=study.best_trial.number,
        epochs_per_trial=epochs,
        run_mode=run_mode,
    )
    yaml_path = output_dir / "best_stage3_hparams.yaml"
    yaml_path.write_text(
        yaml.safe_dump(best_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    trials_path = output_dir / "trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    summary_path = output_dir / "study_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "direction": "maximize",
                "mode": run_mode,
                "metric": "validation_accel_steer_mean_macro_f1",
                "completed_trials": sum(
                    trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
                ),
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
                "fixed_stage3_params": fixed_stage3_params,
                "best_yaml": str(yaml_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"yaml": yaml_path, "trials": trials_path, "summary": summary_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--study-name", default="stage3-ego-motion-hpo")
    parser.add_argument("--storage", help="Optional Optuna storage URL, for example sqlite:///study.db")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true", help="Train on deterministic synthetic tensors")
    parser.add_argument("--window-frames", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--flow-projection-dim", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--physics-projection-dim", type=int, default=32)
    parser.add_argument("--use-physics-vector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    positive_values = (
        args.trials,
        args.epochs,
        args.window_frames,
        args.stride,
        args.size,
        args.batch_size,
        args.frame_batch_size,
        args.flow_projection_dim,
        args.hidden_size,
        args.layers,
        args.physics_projection_dim,
    )
    if min(positive_values) < 1:
        parser.error("trial, epoch, geometry, batch, projection, and model sizes must be >= 1")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")

    config, config_path = load_experiment_config(args.config)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=sampler,
        pruner=pruner,
    )
    objective = _smoke_objective(args) if args.smoke else _real_objective(
        args,
        config=config,
        config_path=config_path,
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    run_mode = "synthetic_smoke" if args.smoke else "stage3_data"
    fixed_stage3_params = (
        {
            "window_frames": 3,
            "stride": 3,
            "size": 64,
            "batch_size": 2,
            "frame_batch_size": 4,
            "flow_projection_dim": 64,
            "hidden_size": 16,
            "layers": 1,
            "use_physics_vector": True,
            "physics_projection_dim": 8,
        }
        if args.smoke
        else {
            "window_frames": args.window_frames,
            "stride": args.stride,
            "size": args.size,
            "batch_size": args.batch_size,
            "frame_batch_size": args.frame_batch_size,
            "flow_projection_dim": args.flow_projection_dim,
            "hidden_size": args.hidden_size,
            "layers": args.layers,
            "use_physics_vector": args.use_physics_vector,
            "physics_projection_dim": args.physics_projection_dim,
        }
    )
    outputs = _write_study_outputs(
        study,
        config=config,
        output_dir=args.output_dir.resolve(),
        epochs=args.epochs,
        fixed_stage3_params=fixed_stage3_params,
        run_mode=run_mode,
    )
    print(
        json.dumps(
            {
                "mode": run_mode,
                "trials": len(study.trials),
                "best_value": study.best_value,
                "best_params": study.best_trial.params,
                "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
