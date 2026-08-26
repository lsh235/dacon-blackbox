#!/usr/bin/env python3
"""Copy the supplied public example into an isolated development directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from blackbox.data_validation import validate_public_example


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    if destination.exists():
        raise SystemExit(f"destination already exists; refusing to overwrite: {destination}")
    if _contains(source, destination) or _contains(destination, source):
        raise SystemExit("source and destination must not contain each other")

    source_report = validate_public_example(source, decode=args.decode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    destination_report = validate_public_example(destination, decode=args.decode)
    if source_report != destination_report:
        raise SystemExit(
            f"copied data summary differs: source={source_report}, destination={destination_report}"
        )
    print(json.dumps({"destination": str(destination), **destination_report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
