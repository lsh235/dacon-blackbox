"""Audit the local Python and CUDA environment against pinned requirements."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def read_pinned_requirements(path: str | Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"requirement must use an exact == pin: {line}")
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()
    if not pins:
        raise ValueError(f"no pinned requirements found: {path}")
    return pins


def _os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def audit_environment(
    requirements_path: str | Path,
    *,
    require_cuda: bool = True,
) -> dict[str, object]:
    pins = read_pinned_requirements(requirements_path)
    installed: dict[str, str | None] = {}
    errors: list[str] = []
    for package, expected in pins.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[package] = actual
        if actual != expected:
            errors.append(f"{package}: expected {expected}, installed {actual}")

    cuda: dict[str, object] = {
        "available": False,
        "runtime": None,
        "cudnn": None,
        "device": None,
        "capability": None,
    }
    try:
        import torch

        available = torch.cuda.is_available()
        cuda.update(
            {
                "available": available,
                "runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "device": torch.cuda.get_device_name(0) if available else None,
                "capability": list(torch.cuda.get_device_capability(0)) if available else None,
            }
        )
        if require_cuda and not available:
            errors.append("CUDA GPU is not available")
    except ImportError:
        if require_cuda:
            errors.append("torch is unavailable, so CUDA cannot be checked")

    os_release = _os_release()
    return {
        "ok": not errors,
        "os": {
            "name": os_release.get("PRETTY_NAME", platform.system()),
            "version_id": os_release.get("VERSION_ID"),
            "kernel": platform.release(),
            "libc": platform.libc_ver()[1],
        },
        "python": platform.python_version(),
        "requirements": pins,
        "installed": installed,
        "cuda": cuda,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = audit_environment(args.requirements, require_cuda=not args.allow_cpu)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
