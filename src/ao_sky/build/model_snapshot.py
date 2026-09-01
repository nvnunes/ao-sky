"""Build-local AO model snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ._constants import (
    MODEL_SNAPSHOT_DIRNAME,
    MODEL_SNAPSHOT_MANIFEST_FILENAME,
    WORK_STATUS_PENDING,
)
from ._exceptions import BuildError
from ._schema_compat import require_current_build_layout
from .config import resolve_runtime_root_candidates
from .control import (
    load_build_roots,
    load_runtime_config_path,
    load_state,
    set_model_root,
)
from .runtime_config import load_runtime_config

_REQUIRED_MODEL_SUFFIXES = (".pt", "_metadata.pkl")
_HASH_CHUNK_SIZE = 1024 * 1024


def fetch_model_data(
    build_path: Path,
    *,
    model_root: Path | None = None,
    aosky_yaml: Path | None = None,
    force: bool = False,
    cwd: Path | None = None,
) -> Path:
    """Snapshot configured AO model files into one build-local models directory."""

    resolved_build_path = Path(build_path).expanduser().resolve()
    require_current_build_layout(resolved_build_path, operation="refresh models for")
    _require_traversal_not_started(resolved_build_path)

    destination_root = (resolved_build_path / MODEL_SNAPSHOT_DIRNAME).resolve()
    source_root, source_is_override = _resolve_source_model_root(
        resolved_build_path,
        model_root=model_root,
        aosky_yaml=aosky_yaml,
        cwd=cwd,
    )

    runtime_config_path = load_runtime_config_path(resolved_build_path)
    runtime = load_runtime_config(
        runtime_config_path,
        model_root=source_root,
        allow_legacy=False,
    )
    model_roles = _required_model_roles(runtime)

    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_root / MODEL_SNAPSHOT_MANIFEST_FILENAME
    if (
        not source_is_override
        and source_root == destination_root
        and manifest_path.is_file()
    ):
        _validate_required_model_files(destination_root, model_roles)
        return manifest_path

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_root": str(source_root),
        "model_root": MODEL_SNAPSHOT_DIRNAME,
        "runtime_config_path": str(runtime_config_path.resolve()),
        "models": [],
    }

    for model_name in sorted(model_roles):
        model_entry = {
            "name": model_name,
            "roles": sorted(model_roles[model_name]),
            "files": [],
        }
        for suffix in _REQUIRED_MODEL_SUFFIXES:
            model_entry["files"].append(
                _copy_model_file(
                    source_root=source_root,
                    destination_root=destination_root,
                    build_path=resolved_build_path,
                    model_name=model_name,
                    suffix=suffix,
                    required=True,
                    force=force,
                )
            )
        manifest["models"].append(model_entry)

    _write_json_atomically(manifest_path, manifest)
    set_model_root(resolved_build_path, destination_root)
    return manifest_path


def _resolve_source_model_root(
    build_path: Path,
    *,
    model_root: Path | None,
    aosky_yaml: Path | None,
    cwd: Path | None,
) -> tuple[Path, bool]:
    candidates = resolve_runtime_root_candidates(
        gaia_root=None,
        build_root=None,
        model_root=model_root,
        aosky_yaml=aosky_yaml,
        cwd=cwd,
    )
    resolved = candidates["model_root"]
    if resolved is not None:
        return resolved.expanduser().resolve(), True
    return load_build_roots(build_path).model_root.expanduser().resolve(), False


def _require_traversal_not_started(build_path: Path) -> None:
    state = load_state(build_path)
    if np.any(state["traversal_status"] != WORK_STATUS_PENDING):
        raise BuildError(
            "Model snapshots can only be refreshed before Traversal has started; "
            "create a new build to snapshot models for completed or partial Traversal output"
        )


def _required_model_roles(runtime) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = defaultdict(set)
    for num_stars in range(runtime.ao_system.min_wfs, runtime.ao_system.max_wfs + 1):
        key = f"{int(num_stars)}star"
        model = runtime.models.get(key)
        if model:
            roles[model].add(f"model:{key}")
        legacy_model = runtime.legacy_field_averaged_models.get(key)
        if legacy_model:
            roles[legacy_model].add(f"legacy_field_averaged:{key}")
    return dict(roles)


def _validate_required_model_files(
    model_root: Path,
    model_roles: dict[str, set[str]],
) -> None:
    for model_name in model_roles:
        for suffix in _REQUIRED_MODEL_SUFFIXES:
            filename = model_root / f"{model_name}{suffix}"
            if not filename.is_file():
                raise BuildError(f"Required model file is missing: {filename}")


def _copy_model_file(
    *,
    source_root: Path,
    destination_root: Path,
    build_path: Path,
    model_name: str,
    suffix: str,
    required: bool,
    force: bool,
) -> dict[str, object] | None:
    filename = f"{model_name}{suffix}"
    source = source_root / filename
    destination = destination_root / filename
    if not source.is_file():
        if required:
            raise BuildError(f"Required model file is missing: {source}")
        return None

    if destination.exists():
        if not destination.is_file():
            raise BuildError(f"Model destination exists and is not a file: {destination}")
        if _same_file_content(source, destination):
            return _file_manifest_entry(destination, build_path=build_path)
        if not force:
            raise BuildError(
                f"Model file already exists with different content: {destination}; "
                "rerun with --force to replace it"
            )

    if source.resolve() != destination.resolve():
        _copy_file_atomically(source, destination)
    return _file_manifest_entry(destination, build_path=build_path)


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


def _file_manifest_entry(filename: Path, *, build_path: Path) -> dict[str, object]:
    return {
        "name": filename.name,
        "path": filename.resolve().relative_to(build_path.resolve()).as_posix(),
        "size": filename.stat().st_size,
        "sha256": _sha256(filename),
    }


def _sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
