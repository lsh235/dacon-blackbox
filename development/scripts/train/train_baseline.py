#!/usr/bin/env python3
"""Train one or more supplied Stage baselines outside DOC/."""

from __future__ import annotations

import argparse
from pathlib import Path

from blackbox.data_validation import validate_public_example
from blackbox.training import train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--stages", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pretrained-stage2", action="store_true")
    args = parser.parse_args()
    if args.epochs < 0:
        parser.error("--epochs must be >= 0")

    validate_public_example(args.data_root)
    artifacts = train_baseline(
        args.data_root,
        args.model_root,
        stages=tuple(dict.fromkeys(args.stages)),
        epochs=args.epochs,
        pretrained_stage2=args.pretrained_stage2,
    )
    for artifact in artifacts:
        print(f"[OK] {artifact} ({artifact.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
