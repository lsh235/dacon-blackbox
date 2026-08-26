"""Create traceable experiment manifests for models, predictions and configs."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from blackbox.inventory import sha256_file


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _environment() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        import torchvision

        result.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        result["torch"] = None
    return result


def create_manifest(
    project_root: str | Path,
    artifacts: list[str | Path],
    *,
    command: str,
    note: str = "",
) -> dict[str, object]:
    root = Path(project_root).resolve()
    files = []
    for artifact in artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact not found: {path}")
        try:
            display_path = path.relative_to(root).as_posix()
        except ValueError:
            display_path = str(path)
        files.append(
            {
                "path": display_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "command": command,
        "note": note,
        "environment": _environment(),
        "artifacts": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    manifest = create_manifest(
        args.project_root,
        list(args.artifact),
        command=args.command,
        note=args.note,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] manifest written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
