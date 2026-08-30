"""Build the explicit Iteration 9 combined CSV scaffold and submission ZIP."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import textwrap
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from blackbox.contracts import STAGE_COLUMNS, validate_prediction_frame
from blackbox.stages.stage1.inference_stage1 import discover_fold_checkpoints
from blackbox.submission import REQUIRED_FILES, validate_submission_zip


DEVELOPMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = DEVELOPMENT_ROOT / "artifacts" / "config-baseline" / "submissions"
DEFAULT_MODEL_ROOT = DEVELOPMENT_ROOT / "artifacts" / "config-baseline" / "models"
DEFAULT_STAGE1_CHECKPOINT_DIR = (
    DEVELOPMENT_ROOT
    / "artifacts"
    / "experiments"
    / "stage1-v6-mode-g-full-5fold-20260830"
    / "stage1_v6_mode_g_full_augmented"
)
DEFAULT_OUTPUT_DIR = DEVELOPMENT_ROOT / "artifacts" / "final-submission"
DEFAULT_SOURCE_ROOT = DEVELOPMENT_ROOT / "src"
DEFAULT_REQUIREMENTS = DEVELOPMENT_ROOT / "requirements.txt"
DEFAULT_ENTRYPOINT = DEVELOPMENT_ROOT / "scripts" / "submission" / "inference_entrypoint.py"
STAGE_FILENAMES = {
    "stage1": "stage1_submission.csv",
    "stage2": "stage2_submission.csv",
    "stage3": "stage3_submission.csv",
}
MODEL_SOURCES = {
    "model/stage2/best.pt": Path("stage2/best.pt"),
    "model/stage2/resnet18-f37072fd.pth": Path("stage2/resnet18-f37072fd.pth"),
    "model/stage3/best.pt": Path("stage3/best.pt"),
}
COMBINED_COLUMNS = [
    "stage",
    "ID",
    "answer",
    "collision_frame",
    "entry_frame",
    "evasion_space",
    "entry_side",
    "sample_index",
    "accel_label",
    "steer_label",
]


@dataclass(frozen=True)
class SubmissionBuildReport:
    combined_csv: str
    archive: str
    manifest: str
    stage_rows: dict[str, int]
    combined_rows: int
    archive_files: int
    archive_bytes: int
    archive_sha256: str


def merge_stage_csvs(input_dir: str | Path, output_path: str | Path) -> dict[str, int]:
    """Validate all Stage CSVs and emit a provisional long-form union scaffold."""

    source = Path(input_dir).resolve()
    frames: list[pd.DataFrame] = []
    stage_rows: dict[str, int] = {}
    for stage, filename in STAGE_FILENAMES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage CSV: {path}")
        frame = pd.read_csv(path)
        validate_prediction_frame(stage, frame)
        stage_rows[stage] = len(frame)
        tagged = frame.copy()
        tagged.insert(0, "stage", stage)
        frames.append(tagged)
    combined = pd.concat(frames, ignore_index=True).reindex(columns=COMBINED_COLUMNS)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False, encoding="utf-8-sig")
    return stage_rows


def _require_nonempty(path: Path, *, description: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_EMBEDDED_RUNTIME_MARKER = "# __BLACKBOX_EMBEDDED_RUNTIME__"


def _module_name(relative: Path) -> tuple[str, bool]:
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    if not parts:
        raise ValueError(f"invalid Python source path: {relative}")
    return ".".join(parts), is_package


def _build_embedded_inference(entrypoint: Path, source_root: Path) -> str:
    """Embed project Python sources in the required top-level ``inference.py``."""

    entrypoint_source = entrypoint.read_text(encoding="utf-8")
    if entrypoint_source.count(_EMBEDDED_RUNTIME_MARKER) != 1:
        raise ValueError(
            f"entrypoint must contain exactly one {_EMBEDDED_RUNTIME_MARKER!r}: {entrypoint}"
        )

    modules: dict[str, dict[str, object]] = {}
    for source_path in sorted(source_root.rglob("*.py")):
        relative = source_path.relative_to(source_root)
        module_name, is_package = _module_name(relative)
        if module_name in modules:
            raise ValueError(f"duplicate embedded module: {module_name}")
        modules[module_name] = {
            "filename": relative.as_posix(),
            "is_package": is_package,
            "source": source_path.read_text(encoding="utf-8"),
        }
    if "blackbox" not in modules or not modules["blackbox"]["is_package"]:
        raise FileNotFoundError(
            f"source root must contain the blackbox package: {source_root / 'blackbox'}"
        )

    serialized = json.dumps(
        modules,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.b85encode(zlib.compress(serialized, level=9)).decode("ascii")
    payload_lines = "\n".join(f'    "{chunk}"' for chunk in textwrap.wrap(encoded, 100))
    bootstrap = f'''import base64 as _embedded_base64
import importlib.abc as _embedded_importlib_abc
import importlib.util as _embedded_importlib_util
import json as _embedded_json
import sys as _embedded_sys
import zlib as _embedded_zlib

_EMBEDDED_SOURCE_PAYLOAD = (
{payload_lines}
)
_EMBEDDED_SOURCES = _embedded_json.loads(
    _embedded_zlib.decompress(
        _embedded_base64.b85decode(_EMBEDDED_SOURCE_PAYLOAD.encode("ascii"))
    ).decode("utf-8")
)


class _EmbeddedBlackboxLoader(
    _embedded_importlib_abc.MetaPathFinder,
    _embedded_importlib_abc.Loader,
):
    def find_spec(self, fullname, path=None, target=None):
        source = _EMBEDDED_SOURCES.get(fullname)
        if source is None:
            return None
        return _embedded_importlib_util.spec_from_loader(
            fullname,
            self,
            is_package=source["is_package"],
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source = _EMBEDDED_SOURCES[module.__name__]
        module.__file__ = "<embedded>/" + source["filename"]
        if source["is_package"]:
            module.__path__ = ["<embedded>/" + source["filename"].rsplit("/", 1)[0]]
        exec(compile(source["source"], module.__file__, "exec"), module.__dict__)


_embedded_sys.meta_path.insert(0, _EmbeddedBlackboxLoader())
'''
    return entrypoint_source.replace(_EMBEDDED_RUNTIME_MARKER, bootstrap.rstrip())


def build_submission_package(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    stage1_checkpoint_dir: str | Path = DEFAULT_STAGE1_CHECKPOINT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    requirements: str | Path = DEFAULT_REQUIREMENTS,
    entrypoint: str | Path = DEFAULT_ENTRYPOINT,
) -> SubmissionBuildReport:
    """Build, validate, and describe a code-submission archive."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined_csv = output / "submission.csv"
    stage_rows = merge_stage_csvs(input_dir, combined_csv)

    models = Path(model_root).resolve()
    sources = Path(source_root).resolve()
    requirements_path = Path(requirements).resolve()
    entrypoint_path = Path(entrypoint).resolve()
    _require_nonempty(requirements_path, description="requirements file")
    _require_nonempty(entrypoint_path, description="inference entrypoint")
    if not sources.is_dir():
        raise FileNotFoundError(f"source root not found: {sources}")
    resolved_models: dict[str, Path] = {}
    stage1_checkpoints = discover_fold_checkpoints(stage1_checkpoint_dir)
    for fold_index, model_path in enumerate(stage1_checkpoints):
        _require_nonempty(
            model_path,
            description=f"model/stage1/fold_{fold_index}/best.pt",
        )
        resolved_models[f"model/stage1/fold_{fold_index}/best.pt"] = model_path
    for archive_name, relative_path in MODEL_SOURCES.items():
        model_path = models / relative_path
        _require_nonempty(model_path, description=archive_name)
        resolved_models[archive_name] = model_path
    embedded_inference = _build_embedded_inference(entrypoint_path, sources)

    archive_path = output / "submit.zip"
    temporary_archive = output / ".submit.zip.tmp"
    manifest_payload = {
        "format": "dacon_blackbox_mode_g_5fold_submission_bundle_v3",
        "combined_csv": {
            "path": "submission.csv",
            "format": "provisional_long_form_scaffold",
            "warning": (
                "The three official Stage contracts are heterogeneous; this union CSV is a local "
                "packaging scaffold, while inference.py remains the code-submission contract."
            ),
            "stage_rows": stage_rows,
            "columns": COMBINED_COLUMNS,
        },
        "required_archive_files": sorted(REQUIRED_FILES),
    }
    try:
        with ZipFile(temporary_archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("inference.py", embedded_inference)
            archive.write(requirements_path, "requirements.txt")
            for archive_name, model_path in resolved_models.items():
                archive.write(model_path, archive_name)
        temporary_archive.replace(archive_path)
    finally:
        if temporary_archive.exists():
            temporary_archive.unlink()

    validation = validate_submission_zip(archive_path)
    archive_sha256 = _sha256(archive_path)
    final_manifest = {
        **manifest_payload,
        "archive": {
            "path": str(archive_path),
            "sha256": archive_sha256,
            "zip_bytes": validation.zip_bytes,
            "uncompressed_bytes": validation.uncompressed_bytes,
            "file_count": validation.file_count,
            "functions": validation.functions,
        },
    }
    manifest_path = output / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SubmissionBuildReport(
        combined_csv=str(combined_csv),
        archive=str(archive_path),
        manifest=str(manifest_path),
        stage_rows=stage_rows,
        combined_rows=sum(stage_rows.values()),
        archive_files=validation.file_count,
        archive_bytes=validation.zip_bytes,
        archive_sha256=archive_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--stage1-checkpoint-dir",
        type=Path,
        default=DEFAULT_STAGE1_CHECKPOINT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--entrypoint", type=Path, default=DEFAULT_ENTRYPOINT)
    args = parser.parse_args(argv)
    report = build_submission_package(
        input_dir=args.input_dir,
        model_root=args.model_root,
        stage1_checkpoint_dir=args.stage1_checkpoint_dir,
        output_dir=args.output_dir,
        source_root=args.source_root,
        requirements=args.requirements,
        entrypoint=args.entrypoint,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
