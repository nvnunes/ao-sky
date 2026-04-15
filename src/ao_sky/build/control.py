"""Build root metadata and state persistence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re

import h5py
import numpy as np

from .._paths import get_outer_pixel_bucket_path
from ..spatial import get_pixel_skycoord
from ._constants import (
    ARTIFACT_STATE_PENDING,
    ARTIFACT_STATE_SKIPPED,
    BUILD_FILENAME,
    BUILD_LAYOUT_VERSION,
    BUILD_LOG_FILENAME,
    BUILD_STATUS_COMPLETED,
    BUILD_STATUS_FAILED,
    BUILD_STATUS_INITIALIZED,
    STATE_DTYPE,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
)
from ._exceptions import BuildError
from ._models import BuildDefinition, BuildPaths, LegacyBuildRuntime


def _write_scalar_dataset(group: h5py.Group, name: str, value: object) -> None:
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(name, data=value)


def _decode_bytes(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return value


def _encode_fixed_bytes(
    value: object,
    *,
    max_bytes: int,
    field_name: str,
) -> bytes:
    if value is None:
        return b""
    encoded = str(value).encode("utf-8")
    if len(encoded) > max_bytes:
        raise BuildError(
            f"{field_name} exceeds persisted limit of {max_bytes} bytes"
        )
    return encoded


def build_root_name(definition: BuildDefinition, version: int) -> str:
    """Return the canonical folder name for one build version."""

    return (
        f"{definition.ao_system_short_name}-"
        f"{definition.config_short_name}-v{int(version)}"
    )


def next_lineage_version(build_root: Path, definition: BuildDefinition) -> int:
    """Return the next free version number for one build lineage."""

    pattern = re.compile(
        rf"^{re.escape(definition.ao_system_short_name)}-"
        rf"{re.escape(definition.config_short_name)}-v(?P<version>\d+)$"
    )
    max_version = 0
    if build_root.is_dir():
        for child in build_root.iterdir():
            match = pattern.match(child.name)
            if match is not None:
                max_version = max(max_version, int(match.group("version")))
    return max_version + 1


def latest_build_path(build_root: Path, ao_system_short_name: str, config_short_name: str) -> Path:
    """Return the latest build directory for one lineage."""

    pattern = re.compile(
        rf"^{re.escape(ao_system_short_name)}-"
        rf"{re.escape(config_short_name)}-v(?P<version>\d+)$"
    )
    candidates: list[tuple[int, Path]] = []
    if build_root.is_dir():
        for child in build_root.iterdir():
            match = pattern.match(child.name)
            if match is not None:
                candidates.append((int(match.group("version")), child))
    if not candidates:
        raise BuildError(
            f"No builds found for lineage {ao_system_short_name}-{config_short_name}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def build_artifact_root(build_path: Path, definition: BuildDefinition) -> Path:
    """Return the root directory for per-outer-pixel build artifacts."""

    return build_path / f"hpx{definition.outer_level}-{definition.inner_level}"


def outer_artifact_filename(build_path: Path, definition: BuildDefinition, outer_pix: int) -> Path:
    """Return the persisted outer-pixel artifact filename."""

    return (
        build_artifact_root(build_path, definition)
        / get_outer_pixel_bucket_path(definition.outer_level, outer_pix)
        / "outer.h5"
    )


def create_build_root(
    *,
    definition: BuildDefinition,
    definition_yaml: str,
    roots: BuildPaths,
    legacy_runtime: LegacyBuildRuntime,
) -> Path:
    """Create a new build root with initialized metadata and state."""

    version = next_lineage_version(roots.build_root, definition)
    build_path = roots.build_root / build_root_name(definition, version)
    build_path.mkdir(parents=True, exist_ok=False)
    build_artifact_root(build_path, definition).mkdir(parents=True, exist_ok=True)
    (build_path / BUILD_LOG_FILENAME).touch()

    num_pixels = 12 * (4 ** definition.outer_level)
    state = np.zeros(num_pixels, dtype=STATE_DTYPE)
    state["outer_pix"] = np.arange(num_pixels, dtype=np.int64)
    state["work_status"] = WORK_STATUS_PENDING
    state["attempt_count"] = 0
    state["outer_file_state"] = ARTIFACT_STATE_PENDING
    state["inner_state"] = ARTIFACT_STATE_PENDING
    state["asterisms_state"] = ARTIFACT_STATE_PENDING

    if definition.min_galactic_latitude is not None:
        for outer_pix in range(num_pixels):
            coord = get_pixel_skycoord(definition.outer_level, outer_pix)
            if abs(coord.galactic.b.degree) < definition.min_galactic_latitude:
                state["asterisms_state"][outer_pix] = ARTIFACT_STATE_SKIPPED
                state["asterisms_skip_reason"][outer_pix] = (
                    _encode_fixed_bytes(
                        f"min_galactic_latitude<{definition.min_galactic_latitude:g}",
                        max_bytes=state.dtype["asterisms_skip_reason"].itemsize,
                        field_name="asterisms_skip_reason",
                    )
                )

    with h5py.File(build_path / BUILD_FILENAME, "w") as handle:
        metadata_group = handle.create_group("metadata")
        state_group = handle.create_group("state")
        metadata_group.create_dataset(
            "config_yaml",
            data=definition_yaml,
            dtype=h5py.string_dtype("utf-8"),
        )
        config_group = metadata_group.create_group("config")
        normalized = {
            "build_name": build_path.name,
            "ao_system_short_name": definition.ao_system_short_name,
            "config_short_name": definition.config_short_name,
            "lineage_version": version,
            "epoch": definition.epoch,
            "gaia_release": definition.gaia_release,
            "gaia_root": str(roots.gaia_root),
            "build_root": str(roots.build_root),
            "legacy_config_path": str(legacy_runtime.legacy_config_path),
            "outer_level": definition.outer_level,
            "inner_level": definition.inner_level,
            "min_galactic_latitude": (
                "" if definition.min_galactic_latitude is None else definition.min_galactic_latitude
            ),
            "layout_version": BUILD_LAYOUT_VERSION,
            "build_status": BUILD_STATUS_INITIALIZED,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in normalized.items():
            _write_scalar_dataset(config_group, key, value)
        state_group.create_dataset("outer_pixels", data=state)

    return build_path


def load_build_definition(build_path: Path) -> BuildDefinition:
    """Load the normalized build definition stored in one build root."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        return BuildDefinition(
            ao_system_short_name=str(_decode_bytes(config_group["ao_system_short_name"][()])),
            config_short_name=str(_decode_bytes(config_group["config_short_name"][()])),
            gaia_release=str(_decode_bytes(config_group["gaia_release"][()])),
            outer_level=int(config_group["outer_level"][()]),
            inner_level=int(config_group["inner_level"][()]),
            epoch=float(config_group["epoch"][()]),
            min_galactic_latitude=(
                None
                if str(_decode_bytes(config_group["min_galactic_latitude"][()])) == ""
                else float(config_group["min_galactic_latitude"][()])
            ),
        )


def load_build_roots(build_path: Path) -> BuildPaths:
    """Load resolved roots from one build root."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        return BuildPaths(
            gaia_root=Path(str(_decode_bytes(config_group["gaia_root"][()]))),
            build_root=Path(str(_decode_bytes(config_group["build_root"][()]))),
        )


def load_legacy_config_path(build_path: Path) -> Path:
    """Load the persisted legacy-config path for one build."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        value = handle["metadata"]["config"]["legacy_config_path"][()]
        return Path(str(_decode_bytes(value)))


def load_state(build_path: Path) -> np.ndarray:
    """Return the full outer-pixel state table."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        return handle["state"]["outer_pixels"][...]


def update_state_row(build_path: Path, outer_pix: int, **updates: object) -> None:
    """Update one row in the central outer-pixel state table."""

    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["state"]["outer_pixels"]
        row = dataset[int(outer_pix)]
        for key, value in updates.items():
            if key not in row.dtype.names:
                raise BuildError(f"Unknown state field {key!r}")
            if row.dtype[key].kind == "S":
                row[key] = _encode_fixed_bytes(
                    value,
                    max_bytes=row.dtype[key].itemsize,
                    field_name=key,
                )
            else:
                row[key] = value
        dataset[int(outer_pix)] = row


def set_build_status(build_path: Path, status: str) -> None:
    """Update the persisted build-level status."""

    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["metadata"]["config"]["build_status"]
        dataset[()] = np.asarray(status, dtype=h5py.string_dtype("utf-8"))


def summarize_build(build_path: Path) -> dict[str, object]:
    """Return a human-readable build summary payload."""

    state = load_state(build_path)
    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        build_status = str(_decode_bytes(config_group["build_status"][()]))

    work_counts = {
        "pending": int(np.count_nonzero(state["work_status"] == WORK_STATUS_PENDING)),
        "running": int(np.count_nonzero(state["work_status"] == 1)),
        "done": int(np.count_nonzero(state["work_status"] == WORK_STATUS_DONE)),
        "failed": int(np.count_nonzero(state["work_status"] == WORK_STATUS_FAILED)),
    }
    skip_reasons = [
        str(_decode_bytes(value)).strip()
        for value in state["asterisms_skip_reason"]
        if str(_decode_bytes(value)).strip()
    ]
    reason_counts: dict[str, int] = {}
    for reason in skip_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "build_path": str(build_path),
        "build_status": build_status,
        "work_counts": work_counts,
        "skip_reasons": reason_counts,
    }


def refresh_build_status(build_path: Path) -> str:
    """Derive and persist the current build-level status from outer-pixel rows."""

    state = load_state(build_path)
    if np.any(state["work_status"] == WORK_STATUS_FAILED):
        status = BUILD_STATUS_FAILED
    elif np.all(state["work_status"] == WORK_STATUS_DONE):
        status = BUILD_STATUS_COMPLETED
    else:
        status = BUILD_STATUS_INITIALIZED
    set_build_status(build_path, status)
    return status
