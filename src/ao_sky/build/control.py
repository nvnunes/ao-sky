"""Build root metadata and state persistence."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

from .._paths import get_outer_pixel_bucket_path
from ..survey import SurveyError, normalize_survey_extent_overlays
from ._constants import (
    BUILD_FILENAME,
    BUILD_LAYOUT_VERSION,
    BUILD_LOG_FILENAME,
    BUILD_PHASE_AUGMENTATION,
    BUILD_PHASE_GAIA_LOADING,
    BUILD_PHASE_TRAVERSAL,
    BUILD_STATUS_COMPLETED,
    BUILD_STATUS_INITIALIZED,
    MAPS_FILENAME_TEMPLATE,
    MODEL_SNAPSHOT_MANIFEST_FILENAME,
    RUNTIME_CONFIG_FILENAME,
    STATE_DTYPE,
    SURVEY_SNAPSHOT_DIRNAME,
    SURVEY_SNAPSHOT_MANIFEST_FILENAME,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ._exceptions import BuildError
from ._models import BuildDefinition, BuildInspection, BuildPaths
from ._schema_compat import (
    read_build_layout_version,
    require_current_build_layout,
    require_current_build_layout_if_present,
)


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

    require_current_build_layout_if_present(
        build_path,
        operation="append to the build log",
    )
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


def _build_relative_or_resolved_path(build_path: Path, path: Path) -> str:
    resolved_build_path = Path(build_path).resolve()
    resolved_path = Path(path).expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_build_path).as_posix()
    except ValueError:
        return str(resolved_path)


def _resolve_build_metadata_path(build_path: Path, value: object) -> Path:
    path = Path(str(_decode_bytes(value))).expanduser()
    if path.is_absolute():
        return path
    return (Path(build_path).resolve() / path).resolve()


def build_root_name(definition: BuildDefinition, version: int) -> str:
    """Return the canonical folder name for one build version."""

    return f"v{int(version)}"


def next_lineage_version(build_root: Path, definition: BuildDefinition) -> int:
    """Return the next free version number for one build lineage."""

    pattern = re.compile(r"^v(?P<version>\d+)$")
    max_version = 0
    if build_root.is_dir():
        for child in build_root.iterdir():
            match = pattern.match(child.name)
            if match is not None:
                max_version = max(max_version, int(match.group("version")))
    return max_version + 1


def latest_build_path(build_root: Path, lineage_name: str) -> Path:
    """Return the latest build directory for one lineage."""

    pattern = re.compile(r"^v(?P<version>\d+)$")
    candidates: list[tuple[int, Path]] = []
    if build_root.is_dir():
        for child in build_root.iterdir():
            match = pattern.match(child.name)
            if match is not None:
                candidates.append((int(match.group("version")), child))
    if not candidates:
        raise BuildError(f"No builds found under {build_root} for lineage {lineage_name}")
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
    runtime_config_source_path: Path,
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
            "lineage_name": definition.lineage_name,
            "lineage_version": version,
            "gaia_release": definition.gaia_release,
            "gaia_root": str(roots.gaia_root),
            "build_root": str(roots.build_root),
            "dust_root": str(roots.dust_root),
            "model_root": str(roots.model_root),
            "runtime_config_path": str((build_path / RUNTIME_CONFIG_FILENAME).resolve()),
            "runtime_config_source_path": str(Path(runtime_config_source_path).resolve()),
            "outer_level": definition.outer_level,
            "inner_level": definition.inner_level,
            "max_data_level": definition.max_data_level,
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
            lineage_name=str(_decode_bytes(config_group["lineage_name"][()])),
            gaia_release=str(_decode_bytes(config_group["gaia_release"][()])),
            outer_level=int(config_group["outer_level"][()]),
            inner_level=int(config_group["inner_level"][()]),
            max_data_level=int(config_group["max_data_level"][()]),
            survey_extent_overlays=overlays,
        )


def load_build_roots(build_path: Path) -> BuildPaths:
    """Load resolved roots from one build root."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        return BuildPaths(
            gaia_root=Path(str(_decode_bytes(config_group["gaia_root"][()]))),
            build_root=Path(str(_decode_bytes(config_group["build_root"][()]))),
            dust_root=_resolve_build_metadata_path(
                build_path,
                config_group["dust_root"][()],
            ),
            model_root=_resolve_build_metadata_path(
                build_path,
                config_group["model_root"][()],
            ),
        )


def set_model_root(build_path: Path, model_root: Path) -> None:
    """Update the persisted model root for one build."""

    require_current_build_layout(build_path, operation="update model metadata for")
    persisted = _build_relative_or_resolved_path(build_path, Path(model_root))
    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["metadata"]["config"]["model_root"]
        dataset[()] = np.asarray(persisted, dtype=h5py.string_dtype("utf-8"))


def set_dust_root(build_path: Path, dust_root: Path) -> None:
    """Update the persisted dust root for one build."""

    require_current_build_layout(build_path, operation="update dust metadata for")
    persisted = _build_relative_or_resolved_path(build_path, Path(dust_root))
    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["metadata"]["config"]["dust_root"]
        dataset[()] = np.asarray(persisted, dtype=h5py.string_dtype("utf-8"))


def load_survey_extent_overlay_sources(build_path: Path) -> tuple:
    """Load persisted survey overlays without resolving relative MOC paths."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        overlays_yaml = str(_decode_bytes(config_group["survey_extent_overlays_yaml"][()]))
    try:
        return normalize_survey_extent_overlays(
            yaml.safe_load(overlays_yaml) or [],
            base_dir=build_path,
            resolve_paths=False,
        )
    except SurveyError as exc:
        raise BuildError(str(exc)) from exc


def set_survey_extent_overlays(build_path: Path, overlays: tuple) -> None:
    """Update persisted survey overlay metadata for one build."""

    require_current_build_layout(build_path, operation="update survey metadata for")
    definition = load_build_definition(build_path)
    updated = BuildDefinition(
        lineage_name=definition.lineage_name,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
        inner_level=definition.inner_level,
        max_data_level=definition.max_data_level,
        survey_extent_overlays=overlays,
    )
    serialized = _serialize_survey_extent_overlays(updated)
    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["metadata"]["config"]["survey_extent_overlays_yaml"]
        dataset[()] = np.asarray(serialized, dtype=h5py.string_dtype("utf-8"))


def load_runtime_config_path(build_path: Path) -> Path:
    """Load the persisted native runtime-config path for one build."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        value = handle["metadata"]["config"]["runtime_config_path"][()]
        return Path(str(_decode_bytes(value)))


def load_runtime_config_source_path(build_path: Path) -> Path:
    """Load the source path used to create the native runtime config."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        value = handle["metadata"]["config"]["runtime_config_source_path"][()]
        return Path(str(_decode_bytes(value)))


def load_state(build_path: Path) -> np.ndarray:
    """Return the full outer-pixel state table."""

    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        return handle["state"]["outer_pixels"][...]


def update_state_row(build_path: Path, outer_pix: int, **updates: object) -> None:
    """Update one row in the central outer-pixel state table."""

    require_current_build_layout(build_path, operation="update state for")
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


def write_state_rows(
    build_path: Path,
    row_indexes: np.ndarray,
    state: np.ndarray,
) -> None:
    """Write selected rows from the in-memory outer-pixel state table."""

    if len(row_indexes) == 0:
        return
    require_current_build_layout(build_path, operation="write state for")
    with h5py.File(build_path / BUILD_FILENAME, "r+") as handle:
        dataset = handle["state"]["outer_pixels"]
        dataset[row_indexes] = state[row_indexes]


def set_build_status(build_path: Path, status: str) -> None:
    """Update the persisted build-level status."""

    require_current_build_layout(build_path, operation="update status for")
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

    require_current_build_layout(build_path, operation="update phase for")
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

    inspection = inspect_build(build_path)
    phase_counts = inspection.stage_counts
    return {
        "build_path": str(inspection.build_path),
        "build_status": inspection.build_status,
        "current_phase": inspection.current_stage,
        "phase_counts": phase_counts,
        "current_stage": inspection.current_stage,
        "stage_counts": phase_counts,
    }


def inspect_build(build_path: Path) -> BuildInspection:
    """Inspect one build root without repairing or mutating artifacts."""

    build_path = Path(build_path).expanduser().resolve()
    layout_version = read_build_layout_version(build_path)
    state = load_state(build_path)
    with h5py.File(build_path / BUILD_FILENAME, "r") as handle:
        config_group = handle["metadata"]["config"]
        lineage_name = str(_decode_bytes(config_group["lineage_name"][()]))
        lineage_version = int(config_group["lineage_version"][()])
        gaia_release = str(_decode_bytes(config_group["gaia_release"][()]))
        gaia_root = Path(str(_decode_bytes(config_group["gaia_root"][()])))
        dust_root = _resolve_build_metadata_path(build_path, config_group["dust_root"][()])
        model_root = _resolve_build_metadata_path(build_path, config_group["model_root"][()])
        runtime_config_path = Path(
            str(_decode_bytes(config_group["runtime_config_path"][()]))
        )
        runtime_config_source_path = Path(
            str(_decode_bytes(config_group["runtime_config_source_path"][()]))
        )
        outer_level = int(config_group["outer_level"][()])
        inner_level = int(config_group["inner_level"][()])
        max_data_level = int(config_group["max_data_level"][()])
        overlays_yaml = str(_decode_bytes(config_group["survey_extent_overlays_yaml"][()]))
        build_status = str(_decode_bytes(config_group["build_status"][()]))
        current_stage = str(_decode_bytes(config_group["current_phase"][()]))

    stage_counts: dict[str, int] | None = None
    fields = phase_state_fields(current_stage)
    if fields is not None:
        status_field, _, _ = fields
        stage_counts = {
            "pending": int(np.count_nonzero(state[status_field] == WORK_STATUS_PENDING)),
            "running": int(np.count_nonzero(state[status_field] == WORK_STATUS_RUNNING)),
            "done": int(np.count_nonzero(state[status_field] == WORK_STATUS_DONE)),
            "failed": int(np.count_nonzero(state[status_field] == WORK_STATUS_FAILED)),
        }

    try:
        overlays = normalize_survey_extent_overlays(
            yaml.safe_load(overlays_yaml) or [],
            base_dir=build_path,
            resolve_paths=False,
        )
    except SurveyError as exc:
        raise BuildError(str(exc)) from exc

    model_manifest_path = model_root / MODEL_SNAPSHOT_MANIFEST_FILENAME
    survey_manifest_path = (
        build_path / SURVEY_SNAPSHOT_DIRNAME / SURVEY_SNAPSHOT_MANIFEST_FILENAME
    )
    map_artifacts = {
        level: {
            "path": maps_artifact_filename(build_path, level),
            "exists": maps_artifact_filename(build_path, level).is_file(),
        }
        for level in range(outer_level, max_data_level + 1)
    }

    problems: list[str] = []
    _add_missing_path_problem(problems, runtime_config_path, "runtime config")
    _add_missing_path_problem(problems, runtime_config_source_path, "runtime config source")
    _add_missing_path_problem(problems, dust_root, "dust root")
    _add_missing_path_problem(problems, model_manifest_path, "model manifest")
    if overlays:
        _add_missing_path_problem(problems, survey_manifest_path, "survey manifest")
    if build_status == BUILD_STATUS_COMPLETED or current_stage == BUILD_PHASE_AUGMENTATION:
        for level, artifact in map_artifacts.items():
            if not bool(artifact["exists"]):
                problems.append(f"missing map artifact for level {level}: {artifact['path']}")

    return BuildInspection(
        build_path=build_path,
        layout_version=layout_version,
        build_status=build_status,
        current_stage=current_stage,
        stage_counts=stage_counts,
        lineage_name=lineage_name,
        lineage_version=lineage_version,
        gaia_release=gaia_release,
        outer_level=outer_level,
        inner_level=inner_level,
        max_data_level=max_data_level,
        runtime_config_path=runtime_config_path,
        runtime_config_source_path=runtime_config_source_path,
        gaia_root=gaia_root,
        dust_root=dust_root,
        model_root=model_root,
        model_manifest_path=model_manifest_path,
        model_manifest_exists=model_manifest_path.is_file(),
        survey_manifest_path=survey_manifest_path,
        survey_manifest_exists=survey_manifest_path.is_file(),
        survey_overlay_names=tuple(overlay.name for overlay in overlays),
        map_artifacts=map_artifacts,
        problems=tuple(problems),
    )


def _add_missing_path_problem(problems: list[str], path: Path, label: str) -> None:
    if not path.exists():
        problems.append(f"missing {label}: {path}")
