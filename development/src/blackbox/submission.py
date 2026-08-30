"""Static validation for competition submission ZIP files."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


class SubmissionValidationError(ValueError):
    """Raised when a submission archive violates the documented contract."""


REQUIRED_FUNCTIONS = {"predict_stage1", "predict_stage2", "predict_stage3"}
REQUIRED_FILES = {
    "inference.py",
    "requirements.txt",
    "model/stage1/fold_0/best.pt",
    "model/stage1/fold_1/best.pt",
    "model/stage1/fold_2/best.pt",
    "model/stage1/fold_3/best.pt",
    "model/stage1/fold_4/best.pt",
    "model/stage2/best.pt",
    "model/stage2/resnet18-f37072fd.pth",
    "model/stage3/best.pt",
}
MAX_ZIP_BYTES = 10 * 1024**3
MAX_UNCOMPRESSED_BYTES = 32 * 1024**3


@dataclass(frozen=True)
class SubmissionReport:
    path: str
    file_count: int
    zip_bytes: int
    uncompressed_bytes: int
    functions: list[str]


def _top_level_functions(source: str) -> set[str]:
    try:
        tree = ast.parse(source, filename="inference.py")
    except SyntaxError as exc:
        raise SubmissionValidationError(f"inference.py syntax error: {exc}") from exc
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate_submission_zip(path: str | Path) -> SubmissionReport:
    """Validate paths, required files, function definitions and archive sizes."""

    archive_path = Path(path).resolve()
    if not archive_path.is_file():
        raise SubmissionValidationError(f"submission ZIP not found: {archive_path}")
    zip_bytes = archive_path.stat().st_size
    if zip_bytes > MAX_ZIP_BYTES:
        raise SubmissionValidationError(
            f"submission ZIP exceeds 10GB: {zip_bytes} bytes"
        )

    try:
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicates:
                raise SubmissionValidationError(f"duplicate archive entries: {duplicates}")
            for name in names:
                pure = PurePosixPath(name)
                if name.startswith("/") or "\\" in name or ".." in pure.parts:
                    raise SubmissionValidationError(f"unsafe archive path: {name!r}")
                if pure.parts[0] not in {"model", "inference.py", "requirements.txt"}:
                    raise SubmissionValidationError(
                        f"unexpected top-level submission entry: {name!r}"
                    )
                if pure.parts[0] == "model" and (
                    len(pure.parts) < 3
                    or pure.parts[1] not in {"stage1", "stage2", "stage3"}
                ):
                    raise SubmissionValidationError(
                        f"unexpected model submission entry: {name!r}"
                    )

            missing = sorted(REQUIRED_FILES - set(names))
            if missing:
                raise SubmissionValidationError(f"missing required files: {missing}")

            uncompressed_bytes = sum(info.file_size for info in infos)
            if uncompressed_bytes > MAX_UNCOMPRESSED_BYTES:
                raise SubmissionValidationError(
                    f"uncompressed submission exceeds 32GB: {uncompressed_bytes} bytes"
                )

            for required in REQUIRED_FILES - {"inference.py"}:
                if archive.getinfo(required).file_size <= 0:
                    raise SubmissionValidationError(f"required file is empty: {required}")

            try:
                inference_source = archive.read("inference.py").decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SubmissionValidationError("inference.py must be UTF-8") from exc
    except BadZipFile as exc:
        raise SubmissionValidationError(f"invalid ZIP archive: {archive_path}") from exc

    functions = _top_level_functions(inference_source)
    missing_functions = sorted(REQUIRED_FUNCTIONS - functions)
    if missing_functions:
        raise SubmissionValidationError(
            f"missing top-level inference functions: {missing_functions}"
        )
    return SubmissionReport(
        path=str(archive_path),
        file_count=len(names),
        zip_bytes=zip_bytes,
        uncompressed_bytes=uncompressed_bytes,
        functions=sorted(REQUIRED_FUNCTIONS),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a competition submit.zip archive.")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    report = validate_submission_zip(args.archive)
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
