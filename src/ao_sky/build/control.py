"""Build root metadata and state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import h5py
import numpy as np
import yaml

from .._paths import get_outer_pixel_bucket_path
from ..survey import SurveyError, normalize_survey_extent_overlays
from ._constants import (
    BUILD_FILENAME,
    BUILD_LAYOUT_VERSION,
    BUILD_LOG_FILENAME,
    BUILD_PHASE_AGGREGATION,
    BUILD_PHASE_AUGMENTATION,
    BUILD_PHASE_GAIA_LOADING,
    BUILD_PHASE_TRAVERSAL,
    BUILD_STATUS_COMPLETED,
    BUILD_STATUS_FAILED,
    BUILD_STATUS_INITIALIZED,
    MAPS_FILENAME_TEMPLATE,
    STATE_DTYPE,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ._exceptions import BuildError
from ._models import BuildDefinition, BuildPaths


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


def append_build_log(build_path: Path, message: str) -> None:
    """Append one timestamped execution line to the build log."""

    with (build_path / BUILD_LOG_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def _serialize_survey_extent_overlays(definition: BuildDefinition) -> str:
    payload = [
        {
            "name": overlay.name,
            "moc_files": [str(filename) for filename in overlay.moc_files],
        }
        for overlay in definition.survey_extent_overlays
    ]
    return yaml.safe_dump(payload, sort_keys=False)


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


def maps_artifact_filename(build_path: Path, level: int) -> Path:
    """Return the persisted all-sky maps artifact filename for one level."""

    return build_path / MAPS_FILENAME_TEMPLATE.format(level=int(level))


def create_build_root(
    *,
    definition: BuildDefinition,
    definition_yaml: str,
    roots: BuildPaths,
    legacy_config_path: Path,
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
    state["gaia_loading_status"] = WORK_STATUS_DONE
    state["gaia_loading_attempt_count"] = 0
    state["traversal_status"] = WORK_STATUS_PENDING
    state["traversal_attempt_count"] = 0

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
            "dust_root": str(roots.dust_root),
            "model_root": str(roots.model_root),
            "legacy_config_path": str(Path(legacy_config_path).resolve()),
            "outer_level": definition.outer_level,
            "inner_level": definition.inner_level,
            "max_data_level": definition.max_data_level,
            "min_galactic_latitude": (
                "" if definition.min_galactic_latitude is None else definition.min_galactic_latitude
            ),
            "survey_extent_overlays_yaml": _serialize_survey_extent_overlays(definition),
            "layout_version": BUILD_LAYOUT_VERSION,
            "build_status": BUILD_STATUS_INITIALIZED,
            "current_phase": BUILD_PHASE_TRAVERSAL,
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
        overlays_yaml = str(_decode_bytes(config_group["survey_extent_overlays_yaml"][()]))
        try:
            overlays = normalize_survey_extent_overlays(
                yaml.safe_load(overlays_yaml) or [],
                base_dir=build_path,
            )
        except SurveyError as exc:
            raise BuildError(str(exc)) from exc
        return BuildDefinition(
            ao_system_short_name=str(_decode_bytes(config_group["ao_system_short_name"][()])),
            config_short_name=str(_decode_bytes(config_group["config_short_name"][()])),
            gaia_release=str(_decode_bytes(config_group["gaia_release"][()])),
            outer_level=int(config_group["outer_level"][()]),
            inner_level=int(config_group["inner_level"][()]),
            max_data_level=int(config_group["max_data_level"][()]),
            epoch=float(config_group["epoch"][()]),
            min_galactic_latitude=(
                None
                if str(_decode_bytes(config_group["min_galactic_latitude"][()])) == ""
                else float(config_group["min_galactic_latitude"][()])
            ),
            survey_extent_overlays=overlays,
        )


def load_build_roots(build_path: Path) -> BuildPaths:
    """Load resolved roots from one build root."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        return BuildPaths(
            gaia_root=Path(str(_decode_bytes(config_group["gaia_root"][()]))),
            build_root=Path(str(_decode_bytes(config_group["build_root"][()]))),
            dust_root=Path(str(_decode_bytes(config_group["dust_root"][()]))),
            model_root=Path(str(_decode_bytes(config_group["model_root"][()]))),
        )


def set_model_root(build_path: Path, model_root: Path) -> None:
    """Update the persisted model root for one build."""

    resolved = Path(model_root).expanduser().resolve()
    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["metadata"]["config"]["model_root"]
        dataset[()] = np.asarray(str(resolved), dtype=h5py.string_dtype("utf-8"))


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


def load_current_phase(build_path: Path) -> str:
    """Return the persisted current build phase."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        dataset = handle["metadata"]["config"]["current_phase"]
        return str(_decode_bytes(dataset[()]))


def set_current_phase(build_path: Path, phase: str) -> None:
    """Update the persisted current build phase."""

    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["metadata"]["config"]["current_phase"]
        dataset[()] = np.asarray(phase, dtype=h5py.string_dtype("utf-8"))


def phase_state_fields(phase: str) -> tuple[str, str, str] | None:
    """Return the state field names for one per-outer-pixel build phase."""

    if phase == BUILD_PHASE_GAIA_LOADING:
        return (
            "gaia_loading_status",
            "gaia_loading_attempt_count",
            "gaia_loading_last_error_message",
        )
    if phase == BUILD_PHASE_TRAVERSAL:
        return (
            "traversal_status",
            "traversal_attempt_count",
            "traversal_last_error_message",
        )
    return None


def summarize_build(build_path: Path) -> dict[str, object]:
    """Return a human-readable build summary payload."""

    state = load_state(build_path)
    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        build_status = str(_decode_bytes(config_group["build_status"][()]))
        current_phase = str(_decode_bytes(config_group["current_phase"][()]))

    phase_counts: dict[str, int] | None = None
    fields = phase_state_fields(current_phase)
    if fields is not None:
        status_field, _, _ = fields
        phase_counts = {
            "pending": int(np.count_nonzero(state[status_field] == WORK_STATUS_PENDING)),
            "running": int(np.count_nonzero(state[status_field] == WORK_STATUS_RUNNING)),
            "done": int(np.count_nonzero(state[status_field] == WORK_STATUS_DONE)),
            "failed": int(np.count_nonzero(state[status_field] == WORK_STATUS_FAILED)),
        }

    return {
        "build_path": str(build_path),
        "build_status": build_status,
        "current_phase": current_phase,
        "phase_counts": phase_counts,
    }


def refresh_build_status(build_path: Path) -> str:
    """Derive and persist the current build-level status from outer-pixel rows."""

    state = load_state(build_path)
    current_phase = load_current_phase(build_path)
    fields = phase_state_fields(current_phase)
    if fields is None:
        if current_phase in (BUILD_PHASE_AGGREGATION, BUILD_PHASE_AUGMENTATION):
            with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
                value = handle["metadata"]["config"]["build_status"][()]
            return str(_decode_bytes(value))
        raise BuildError(f"Unknown build phase {current_phase!r}")

    status_field, _, _ = fields
    if np.any(state[status_field] == WORK_STATUS_FAILED):
        status = BUILD_STATUS_FAILED
    else:
        status = BUILD_STATUS_INITIALIZED
    set_build_status(build_path, status)
    return status
