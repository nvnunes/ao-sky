"""Private schema-version compatibility for completed AO Sky builds."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.table import Table

from ._constants import (
    ARTIFACT_LAYOUT_VERSION_ATTRIBUTE,
    ASTERISMS_DTYPE,
    BUILD_FILENAME,
    BUILD_LAYOUT_VERSION,
    INNER_DTYPE,
    MAPS_DATASET,
    MAPS_DTYPE,
    OUTER_DATASET_ASTERISMS,
    OUTER_DATASET_INNER,
    RUNTIME_CONFIG_SCHEMA_VERSION,
)
from ._exceptions import BuildError

LEGACY_BUILD_LAYOUT_VERSION = 2
LEGACY_RUNTIME_CONFIG_SCHEMA_VERSION = 2

_LEGACY_RESULT_FIELDS = {
    "winner_ee_resolved": "on_axis_winner_ee",
    "winner_ee_averaged": "field_averaged_winner_ee",
    "coverage_resolved": "on_axis_coverage",
    "coverage_averaged": "field_averaged_coverage",
}
_LEGACY_RUNTIME_FIELDS = {
    "prediction": frozenset(
        {
            "resolved_models",
            "averaged_models",
            "resolved_device",
            "averaged_device",
        }
    ),
    "coverage": frozenset(
        {
            "resolved_ee_threshold",
            "averaged_ee_threshold",
        }
    ),
}
_LEGACY_INNER_DTYPE = np.dtype(
    [
        ("pix", "<i8"),
        ("gaia_A0", "<f8"),
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("best_ee", "<f8"),
        ("best_sr", "<f8"),
        ("best_fwhm", "<f8"),
        ("winner_asterism_id", "<i8"),
        ("winner_ee_resolved", "<f8"),
        ("winner_ee_averaged", "<f8"),
        ("coverage_resolved", "?"),
        ("coverage_averaged", "?"),
    ]
)
_LEGACY_MAPS_DTYPE = np.dtype(
    [
        ("pix", "<i8"),
        ("gaia_A0", "<f8"),
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("winner_asterism_count", "<i8"),
        ("best_sr", "<f8"),
        ("best_ee", "<f8"),
        ("best_fwhm", "<f8"),
        ("winner_ee_resolved", "<f8"),
        ("winner_ee_averaged", "<f8"),
        ("coverage_resolved", "<f8"),
        ("coverage_averaged", "<f8"),
    ]
)


def normalize_runtime_config_payload(
    payload: dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Return one runtime payload using only the current schema names."""

    schema_version = _parse_version(
        payload.get("schema_version", 0),
        field_name="runtime config schema_version",
    )

    if schema_version == RUNTIME_CONFIG_SCHEMA_VERSION:
        _reject_legacy_runtime_fields(payload)
        return deepcopy(payload)
    if schema_version != LEGACY_RUNTIME_CONFIG_SCHEMA_VERSION:
        raise BuildError(
            "Unsupported runtime config schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    if not allow_legacy:
        raise BuildError(
            "Schema-version-2 runtime configurations are read-only and cannot "
            "initialize a new build"
        )

    normalized = deepcopy(payload)
    normalized["schema_version"] = RUNTIME_CONFIG_SCHEMA_VERSION
    prediction = normalized.get("prediction")
    if isinstance(prediction, dict):
        _rename_mapping_key(prediction, "resolved_models", "models")
        _rename_mapping_key(
            prediction,
            "averaged_models",
            "legacy_field_averaged_models",
        )
        prediction.pop("resolved_device", None)
        prediction.pop("averaged_device", None)
    coverage = normalized.get("coverage")
    if isinstance(coverage, dict):
        _rename_mapping_key(
            coverage,
            "resolved_ee_threshold",
            "on_axis_ee_threshold",
        )
        _rename_mapping_key(
            coverage,
            "averaged_ee_threshold",
            "field_averaged_ee_threshold",
        )
    return normalized


def detect_artifact_layout_version(handle: h5py.File) -> int:
    """Return the declared or recognized layout version for an open artifact."""

    declared = handle.attrs.get(ARTIFACT_LAYOUT_VERSION_ATTRIBUTE)
    if declared is not None:
        layout_version = _parse_version(
            declared,
            field_name="artifact layout_version",
        )
        if layout_version not in {LEGACY_BUILD_LAYOUT_VERSION, BUILD_LAYOUT_VERSION}:
            raise BuildError(f"Unsupported artifact layout_version: {layout_version}")
        if layout_version == LEGACY_BUILD_LAYOUT_VERSION:
            _require_legacy_artifact_contract(handle)
        else:
            _require_current_artifact_contract(handle)
        return layout_version

    _require_legacy_artifact_contract(handle)
    return LEGACY_BUILD_LAYOUT_VERSION


def read_build_layout_version(build_path: Path) -> int:
    """Return the supported layout version declared by one build root."""

    filename = Path(build_path) / BUILD_FILENAME
    with h5py.File(filename, "r") as handle:
        try:
            value = handle["metadata"]["config"]["layout_version"][()]
        except KeyError as exc:
            raise BuildError(
                f"Build does not declare a layout_version: {filename}"
            ) from exc
    layout_version = _parse_version(value, field_name="build layout_version")
    if layout_version not in {LEGACY_BUILD_LAYOUT_VERSION, BUILD_LAYOUT_VERSION}:
        raise BuildError(f"Unsupported build layout_version: {layout_version}")
    return layout_version


def require_current_build_layout(build_path: Path, *, operation: str) -> None:
    """Reject a mutating operation against a completed legacy build."""

    layout_version = read_build_layout_version(build_path)
    if layout_version != BUILD_LAYOUT_VERSION:
        raise BuildError(
            f"Cannot {operation} a schema-version-2 build; legacy builds are "
            "read-only"
        )


def require_current_build_layout_if_present(
    build_path: Path,
    *,
    operation: str,
) -> None:
    """Reject legacy mutation when the path contains build metadata."""

    build_path = Path(build_path)
    if (build_path / BUILD_FILENAME).is_file():
        require_current_build_layout(build_path, operation=operation)


def require_current_artifact_target(filename: Path, *, operation: str) -> None:
    """Reject a write target contained by a legacy build root."""

    resolved = Path(filename).expanduser().resolve()
    for parent in resolved.parents:
        if (parent / BUILD_FILENAME).is_file():
            require_current_build_layout(parent, operation=operation)
            return


def normalize_artifact_table(table: Table, *, layout_version: int) -> Table:
    """Return an artifact table using only canonical in-memory field names."""

    parsed_layout_version = _parse_version(
        layout_version,
        field_name="artifact layout_version",
    )
    if parsed_layout_version == BUILD_LAYOUT_VERSION:
        return table
    if parsed_layout_version != LEGACY_BUILD_LAYOUT_VERSION:
        raise BuildError(f"Unsupported artifact layout_version: {layout_version}")

    normalized = table.copy(copy_data=False)
    for legacy_name, canonical_name in _LEGACY_RESULT_FIELDS.items():
        if legacy_name in normalized.colnames:
            normalized.rename_column(legacy_name, canonical_name)
    return normalized


def require_current_artifact_layout(handle: h5py.File) -> None:
    """Reject mutation of an artifact not written with the current layout."""

    layout_version = detect_artifact_layout_version(handle)
    if layout_version != BUILD_LAYOUT_VERSION:
        raise BuildError(
            "Schema-version-2 artifacts are read-only; create a schema-version-3 "
            "build for continued computation"
        )


def _rename_mapping_key(mapping: dict[str, Any], old: str, new: str) -> None:
    if old not in mapping:
        return
    if new in mapping:
        raise BuildError(
            f"Runtime config cannot contain both legacy {old!r} and current {new!r}"
        )
    mapping[new] = mapping.pop(old)


def _parse_version(value: object, *, field_name: str) -> int:
    """Return an exact integer version without lossy coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise BuildError(f"Invalid {field_name}: {value!r}; expected an integer")
    return int(value)


def _reject_legacy_runtime_fields(payload: dict[str, Any]) -> None:
    legacy_paths: list[str] = []
    for section_name, legacy_fields in _LEGACY_RUNTIME_FIELDS.items():
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        legacy_paths.extend(
            f"{section_name}.{field_name}"
            for field_name in sorted(legacy_fields & set(section))
        )
    if legacy_paths:
        raise BuildError(
            "Schema-version-3 runtime config contains legacy fields: "
            + ", ".join(legacy_paths)
        )


def _require_legacy_artifact_contract(handle: h5py.File) -> None:
    _require_artifact_contract(
        handle,
        inner_dtype=_LEGACY_INNER_DTYPE,
        maps_dtype=_LEGACY_MAPS_DTYPE,
        contract_name="Schema-version-2",
    )


def _require_current_artifact_contract(handle: h5py.File) -> None:
    _require_artifact_contract(
        handle,
        inner_dtype=INNER_DTYPE,
        maps_dtype=MAPS_DTYPE,
        contract_name="Layout-version-3",
    )


def _require_artifact_contract(
    handle: h5py.File,
    *,
    inner_dtype: np.dtype,
    maps_dtype: np.dtype,
    contract_name: str,
) -> None:
    has_outer_dataset = (
        OUTER_DATASET_INNER in handle or OUTER_DATASET_ASTERISMS in handle
    )
    has_maps_dataset = MAPS_DATASET in handle
    if has_outer_dataset and has_maps_dataset:
        raise BuildError(
            f"{contract_name} artifact cannot contain both outer and maps datasets"
        )
    if has_outer_dataset:
        if OUTER_DATASET_INNER not in handle or OUTER_DATASET_ASTERISMS not in handle:
            raise BuildError(
                f"{contract_name} outer artifact must contain inner and asterisms datasets"
            )
        _require_dataset_dtype(
            handle,
            OUTER_DATASET_INNER,
            inner_dtype,
            contract_name=contract_name,
        )
        _require_dataset_dtype(
            handle,
            OUTER_DATASET_ASTERISMS,
            ASTERISMS_DTYPE,
            contract_name=contract_name,
        )
        return
    if has_maps_dataset:
        _require_dataset_dtype(
            handle,
            MAPS_DATASET,
            maps_dtype,
            contract_name=contract_name,
        )
        return
    raise BuildError(
        f"Artifact does not match the supported {contract_name} artifact contract"
    )


def _require_dataset_dtype(
    handle: h5py.File,
    dataset_name: str,
    expected_dtype: np.dtype,
    *,
    contract_name: str,
) -> None:
    actual_dtype = handle[dataset_name].dtype
    if actual_dtype != expected_dtype:
        raise BuildError(
            f"{contract_name} dataset {dataset_name!r} has dtype {actual_dtype!r}; "
            f"expected {expected_dtype!r}"
        )
