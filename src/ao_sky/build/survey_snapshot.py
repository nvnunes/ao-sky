"""Build-local survey overlay snapshot helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ..survey import SurveyExtentOverlaySpec
from ._constants import (
    SURVEY_SNAPSHOT_DIRNAME,
    SURVEY_SNAPSHOT_MANIFEST_FILENAME,
    WORK_STATUS_PENDING,
)
from ._exceptions import BuildError
from .config import resolve_survey_root_only
from .control import (
    load_state,
    load_survey_extent_overlay_sources,
    set_survey_extent_overlays,
)

_HASH_CHUNK_SIZE = 1024 * 1024


def fetch_survey_data(
    build_path: Path,
    *,
    survey_root: Path | None = None,
    aosky_yaml: Path | None = None,
    force: bool = False,
    cwd: Path | None = None,
) -> Path:
    """Snapshot configured survey MOC files into one build-local surveys directory."""

    resolved_build_path = Path(build_path).expanduser().resolve()
    _require_traversal_not_started(resolved_build_path)

    source_root = resolve_survey_root_only(
        survey_root=survey_root,
        aosky_yaml=aosky_yaml,
        cwd=cwd,
    )
    source_overlays = load_survey_extent_overlay_sources(resolved_build_path)
    if not source_overlays:
        raise BuildError("Build has no survey_overlays to snapshot")

    destination_root = (resolved_build_path / SURVEY_SNAPSHOT_DIRNAME).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / SURVEY_SNAPSHOT_MANIFEST_FILENAME

    updated_overlays: list[SurveyExtentOverlaySpec] = []
    manifest_overlays: list[dict[str, object]] = []
    for overlay in source_overlays:
        copied_files: list[dict[str, object]] = []
        updated_files: list[Path] = []
        for source_path in overlay.moc_files:
            source, destination_relative = _resolve_survey_file(
                source_path,
                build_path=resolved_build_path,
                source_root=source_root,
            )
            destination = resolved_build_path / destination_relative
            copied_files.append(
                _copy_survey_file(
                    source=source,
                    destination=destination,
                    build_path=resolved_build_path,
                    force=force,
                )
            )
            updated_files.append(destination_relative)
        updated_overlays.append(
            SurveyExtentOverlaySpec(name=overlay.name, moc_files=tuple(updated_files))
        )
        manifest_overlays.append(
            {
                "name": overlay.name,
                "files": copied_files,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_survey_root": None if source_root is None else str(source_root),
        "survey_root": SURVEY_SNAPSHOT_DIRNAME,
        "overlays": manifest_overlays,
    }
    _write_json_atomically(manifest_path, manifest)
    set_survey_extent_overlays(resolved_build_path, tuple(updated_overlays))
    return manifest_path


def _require_traversal_not_started(build_path: Path) -> None:
    state = load_state(build_path)
    if np.any(state["traversal_status"] != WORK_STATUS_PENDING):
        raise BuildError(
            "Survey snapshots can only be refreshed before Traversal has started; "
            "create a new build to snapshot surveys for completed or partial output"
        )


def _resolve_survey_file(
    source_path: Path,
    *,
    build_path: Path,
    source_root: Path | None,
) -> tuple[Path, Path]:
    path = Path(source_path).expanduser()
    if path.is_absolute():
        source = path.resolve()
        filename = source.name
    else:
        filename = path.name
        if _is_build_local_survey_path(path):
            if source_root is not None:
                source = (source_root / filename).resolve()
                return source, _survey_destination_relative(filename)
            source = (build_path / SURVEY_SNAPSHOT_DIRNAME / filename).resolve()
            return source, _survey_destination_relative(filename)
        if source_root is None:
            raise BuildError(
                f"survey_root is required to snapshot relative survey file: {path}"
            )
        source = (source_root / filename).resolve()

    return source, _survey_destination_relative(filename)


def _is_build_local_survey_path(path: Path) -> bool:
    return len(path.parts) > 0 and path.parts[0] == SURVEY_SNAPSHOT_DIRNAME


def _survey_destination_relative(filename: str) -> Path:
    return Path(SURVEY_SNAPSHOT_DIRNAME) / filename


def _copy_survey_file(
    *,
    source: Path,
    destination: Path,
    build_path: Path,
    force: bool,
) -> dict[str, object]:
    if not source.is_file():
        raise BuildError(f"Required survey MOC file is missing: {source}")

    if destination.exists():
        if not destination.is_file():
            raise BuildError(f"Survey destination exists and is not a file: {destination}")
        if _same_file_content(source, destination):
            return _file_manifest_entry(source, destination, build_path=build_path)
        if not force:
            raise BuildError(
                f"Survey file already exists with different content: {destination}; "
                "rerun with --force to replace it"
            )

    if source.resolve() != destination.resolve():
        _copy_file_atomically(source, destination)
    return _file_manifest_entry(source, destination, build_path=build_path)


def _same_file_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    return _sha256(left) == _sha256(right)


def _copy_file_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temp)
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()


def _write_json_atomically(filename: Path, payload: dict[str, object]) -> None:
    temp = filename.with_name(f".{filename.name}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(filename)
    finally:
        if temp.exists():
            temp.unlink()


def _file_manifest_entry(
    source: Path,
    destination: Path,
    *,
    build_path: Path,
) -> dict[str, object]:
    return {
        "source_path": str(source.resolve()),
        "path": destination.resolve().relative_to(build_path.resolve()).as_posix(),
        "size": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
