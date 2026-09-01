"""Build-local native Traversal runtime configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import astropy.units as u
import yaml

from ..predict import AOSystemRuntime, PredictRuntime
from ._constants import RUNTIME_CONFIG_FILENAME, RUNTIME_CONFIG_SCHEMA_VERSION
from ._exceptions import BuildError
from ._schema_compat import (
    normalize_runtime_config_payload,
    require_current_build_layout_if_present,
)

DEFAULT_WINNER_TOP_K = 3
MAX_WINNER_TOP_K = 32


def runtime_config_filename(build_path: Path) -> Path:
    """Return the build-local native runtime config filename."""

    return Path(build_path) / RUNTIME_CONFIG_FILENAME


def write_runtime_config(
    build_path: Path,
    runtime: PredictRuntime,
) -> Path:
    """Persist one native runtime config inside a build root."""

    filename = runtime_config_filename(build_path)
    require_current_build_layout_if_present(
        build_path,
        operation="write runtime config for",
    )
    payload = runtime_to_config(runtime)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temp = filename.with_name(f".{filename.name}.tmp")
    try:
        temp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        temp.replace(filename)
    finally:
        if temp.exists():
            temp.unlink()
    return filename


def load_runtime_config(
    filename: Path,
    *,
    model_root: Path,
    allow_legacy: bool = False,
) -> PredictRuntime:
    """Load a current build-local runtime config.

    Set ``allow_legacy=True`` only for read-only access to a completed
    schema-version-2 build.
    """

    resolved = Path(filename).expanduser().resolve()
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise BuildError(f"Runtime config not found: {resolved}") from exc

    if not isinstance(payload, dict):
        raise BuildError("Runtime config must contain a mapping")
    payload = normalize_runtime_config_payload(payload, allow_legacy=allow_legacy)

    ao_system_raw = _required_mapping(payload, "ao_system", "Runtime config")
    traversal_raw = _required_mapping(payload, "traversal", "Runtime config")
    gaia_raw = _required_mapping(payload, "gaia", "Runtime config")
    asterism_raw = _required_mapping(payload, "asterism", "Runtime config")
    prediction_raw = _required_mapping(payload, "prediction", "Runtime config")
    best_raw = _required_mapping(payload, "best", "Runtime config")
    coverage_raw = _required_mapping(payload, "coverage", "Runtime config")
    seeing_raw = best_raw.get("seeing_baseline")
    if not isinstance(seeing_raw, dict):
        raise BuildError("Runtime config is missing required section: best.seeing_baseline")

    fov_arcsec = _required_float(ao_system_raw, "fov_arcsec", "ao_system")
    min_wfs = _required_int(ao_system_raw, "min_wfs", "ao_system")
    max_wfs = _required_int(ao_system_raw, "max_wfs", "ao_system")
    min_mag = _required_float(ao_system_raw, "min_mag", "ao_system")
    max_mag = _required_float(ao_system_raw, "max_mag", "ao_system")
    min_sep_arcsec = _required_float(ao_system_raw, "min_sep_arcsec", "ao_system")
    lgs = _load_lgs(ao_system_raw)

    outer_level = _required_int(traversal_raw, "outer_level", "traversal")
    inner_level = _required_int(traversal_raw, "inner_level", "traversal")

    epoch = _required_float(gaia_raw, "epoch", "gaia")
    max_bright_star_mag = _optional_float(
        gaia_raw.get("max_bright_star_mag"),
        field_name="gaia.max_bright_star_mag",
    )
    max_bright_star_exclusion_arcsec = _optional_float(
        gaia_raw.get("max_bright_star_exclusion_arcsec"),
        field_name="gaia.max_bright_star_exclusion_arcsec",
    )
    if max_bright_star_exclusion_arcsec is None:
        max_bright_star_exclusion_arcsec = 2.0 * fov_arcsec
    if "max_overlap" in asterism_raw:
        raise BuildError(
            "Runtime config asterism.max_overlap is legacy-only and is not "
            "supported by schema_version 3"
        )
    if "max_candidate_asterisms" in asterism_raw:
        raise BuildError(
            "Runtime config asterism.max_candidate_asterisms was removed and is not "
            "supported by schema_version 3"
        )
    winner_ee_epsilon = _optional_float(
        asterism_raw.get("winner_ee_epsilon"),
        field_name="asterism.winner_ee_epsilon",
    )
    if winner_ee_epsilon is None:
        winner_ee_epsilon = 0.01
    winner_top_k = _optional_int(
        asterism_raw.get("winner_top_k"),
        field_name="asterism.winner_top_k",
    )
    if winner_top_k is None:
        winner_top_k = DEFAULT_WINNER_TOP_K
    prediction_wavelength = _required_float(
        prediction_raw,
        "wavelength_micron",
        "prediction",
    )
    seeing_wavelength = _required_float(
        seeing_raw,
        "wavelength_micron",
        "best.seeing_baseline",
    )
    seeing_sr = _required_float(seeing_raw, "sr", "best.seeing_baseline")
    seeing_ee = _required_float(seeing_raw, "ee", "best.seeing_baseline")
    seeing_fwhm = _required_float(seeing_raw, "fwhm_mas", "best.seeing_baseline")
    on_axis_ee_threshold = _required_float(
        coverage_raw,
        "on_axis_ee_threshold",
        "coverage",
    )
    field_averaged_ee_threshold = _required_float(
        coverage_raw,
        "field_averaged_ee_threshold",
        "coverage",
    )

    _validate_runtime_values(
        fov_arcsec=fov_arcsec,
        min_wfs=min_wfs,
        max_wfs=max_wfs,
        min_mag=min_mag,
        max_mag=max_mag,
        min_sep_arcsec=min_sep_arcsec,
        outer_level=outer_level,
        inner_level=inner_level,
        max_bright_star_exclusion_arcsec=max_bright_star_exclusion_arcsec,
        winner_ee_epsilon=winner_ee_epsilon,
        winner_top_k=winner_top_k,
        prediction_wavelength=prediction_wavelength,
        seeing_wavelength=seeing_wavelength,
        seeing_fwhm=seeing_fwhm,
        on_axis_ee_threshold=on_axis_ee_threshold,
        field_averaged_ee_threshold=field_averaged_ee_threshold,
    )

    ao_system = AOSystemRuntime(
        band=_required_str(ao_system_raw, "band", "ao_system"),
        fov=fov_arcsec * u.arcsec,
        lgs=lgs,
        min_wfs=min_wfs,
        max_wfs=max_wfs,
        min_mag=min_mag,
        max_mag=max_mag,
        min_sep=min_sep_arcsec * u.arcsec,
    )
    models = _required_model_mapping(
        prediction_raw,
        "models",
        ao_system=ao_system,
    )
    legacy_field_averaged_models = _required_model_mapping(
        prediction_raw,
        "legacy_field_averaged_models",
        ao_system=ao_system,
    )
    return PredictRuntime(
        ao_system=ao_system,
        outer_level=outer_level,
        inner_level=inner_level,
        epoch=epoch,
        max_bright_star_mag=max_bright_star_mag,
        max_bright_star_exclusion=max_bright_star_exclusion_arcsec * u.arcsec,
        winner_ee_epsilon=winner_ee_epsilon,
        winner_top_k=winner_top_k,
        prediction_wavelength=prediction_wavelength * u.micron,
        models=models,
        legacy_field_averaged_models=legacy_field_averaged_models,
        seeing_reference_wavelength=seeing_wavelength * u.micron,
        seeing_reference_sr=seeing_sr,
        seeing_reference_ee=seeing_ee,
        seeing_reference_fwhm=seeing_fwhm,
        on_axis_ee_threshold=on_axis_ee_threshold,
        field_averaged_ee_threshold=field_averaged_ee_threshold,
        model_root=Path(model_root).expanduser().resolve(),
    )


def runtime_to_config(runtime: PredictRuntime) -> dict[str, Any]:
    """Return the persisted native runtime-config payload for one runtime."""

    ao_system = runtime.ao_system
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
    }
    payload.update(
        {
            "ao_system": {
                "band": ao_system.band,
                "fov_arcsec": float(ao_system.fov.to_value(u.arcsec)),
                "lgs": [
                    {"zd": float(star["zd"]), "az": float(star["az"])}
                    for star in ao_system.lgs
                ],
                "min_wfs": int(ao_system.min_wfs),
                "max_wfs": int(ao_system.max_wfs),
                "min_mag": float(ao_system.min_mag),
                "max_mag": float(ao_system.max_mag),
                "min_sep_arcsec": float(ao_system.min_sep.to_value(u.arcsec)),
            },
            "prediction": {
                "wavelength_micron": float(
                    runtime.prediction_wavelength.to_value(u.micron)
                ),
                "models": dict(runtime.models),
                "legacy_field_averaged_models": dict(runtime.legacy_field_averaged_models),
            },
            "traversal": {
                "outer_level": int(runtime.outer_level),
                "inner_level": int(runtime.inner_level),
            },
            "gaia": {
                "epoch": float(runtime.epoch),
                "max_bright_star_mag": runtime.max_bright_star_mag,
                "max_bright_star_exclusion_arcsec": float(
                    runtime.max_bright_star_exclusion.to_value(u.arcsec)
                ),
            },
            "asterism": {
                "winner_ee_epsilon": float(runtime.winner_ee_epsilon),
                "winner_top_k": int(runtime.winner_top_k),
            },
            "best": {
                "seeing_baseline": {
                    "wavelength_micron": float(
                        runtime.seeing_reference_wavelength.to_value(u.micron)
                    ),
                    "sr": float(runtime.seeing_reference_sr),
                    "ee": float(runtime.seeing_reference_ee),
                    "fwhm_mas": float(runtime.seeing_reference_fwhm),
                },
            },
            "coverage": {
                "on_axis_ee_threshold": float(runtime.on_axis_ee_threshold),
                "field_averaged_ee_threshold": float(runtime.field_averaged_ee_threshold),
            },
        }
    )
    return payload


def _required_model_mapping(
    payload: dict[str, object],
    name: str,
    *,
    ao_system: AOSystemRuntime,
) -> dict[str, str]:
    raw = payload.get(name)
    if not isinstance(raw, dict):
        raise BuildError(f"Runtime config prediction.{name} is required")
    result = {}
    for key, value in raw.items():
        normalized_key = str(key).strip()
        normalized_value = str(value).strip() if value is not None else ""
        if not normalized_key or not normalized_value:
            raise BuildError(f"Runtime config prediction.{name} entries must be non-empty")
        result[normalized_key] = normalized_value
    missing = [
        f"{count}star"
        for count in range(ao_system.min_wfs, ao_system.max_wfs + 1)
        if f"{count}star" not in result
    ]
    if missing:
        raise BuildError(
            f"Runtime config prediction.{name} is missing model keys: "
            + ", ".join(missing)
        )
    return result


def _required_mapping(
    payload: dict[str, object],
    name: str,
    context: str,
) -> dict[str, object]:
    raw = payload.get(name)
    if not isinstance(raw, dict):
        raise BuildError(f"{context} is missing required section: {name}")
    return raw


def _required_value(
    payload: dict[str, object],
    name: str,
    context: str,
) -> object:
    if name not in payload:
        raise BuildError(f"Runtime config is missing required value: {context}.{name}")
    return payload[name]


def _required_str(
    payload: dict[str, object],
    name: str,
    context: str,
) -> str:
    raw = _required_value(payload, name, context)
    if raw is None:
        raise BuildError(f"Runtime config {context}.{name} must be non-empty")
    value = str(raw).strip()
    if not value:
        raise BuildError(f"Runtime config {context}.{name} must be non-empty")
    return value


def _required_float(
    payload: dict[str, object],
    name: str,
    context: str,
) -> float:
    raw = _required_value(payload, name, context)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"Runtime config {context}.{name} must be a number") from exc
    if not math.isfinite(value):
        raise BuildError(f"Runtime config {context}.{name} must be finite")
    return value


def _required_int(
    payload: dict[str, object],
    name: str,
    context: str,
) -> int:
    raw = _required_value(payload, name, context)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"Runtime config {context}.{name} must be an integer") from exc
    return value


def _optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"Runtime config {field_name} must be an integer") from exc


def _optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"Runtime config {field_name} must be a number") from exc
    if not math.isfinite(result):
        raise BuildError(f"Runtime config {field_name} must be finite")
    return result


def _load_lgs(payload: dict[str, object]) -> tuple[dict[str, float], ...]:
    raw = payload.get("lgs", [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise BuildError("Runtime config ao_system.lgs must be a list")

    lgs: list[dict[str, float]] = []
    for index, star in enumerate(raw):
        if not isinstance(star, dict):
            raise BuildError(f"Runtime config ao_system.lgs[{index}] must be a mapping")
        lgs.append(
            {
                "zd": _required_float(star, "zd", f"ao_system.lgs[{index}]"),
                "az": _required_float(star, "az", f"ao_system.lgs[{index}]"),
            }
        )
    return tuple(lgs)


def _validate_runtime_values(
    *,
    fov_arcsec: float,
    min_wfs: int,
    max_wfs: int,
    min_mag: float,
    max_mag: float,
    min_sep_arcsec: float,
    outer_level: int,
    inner_level: int,
    max_bright_star_exclusion_arcsec: float,
    winner_ee_epsilon: float,
    winner_top_k: int,
    prediction_wavelength: float,
    seeing_wavelength: float,
    seeing_fwhm: float,
    on_axis_ee_threshold: float,
    field_averaged_ee_threshold: float,
) -> None:
    if fov_arcsec <= 0:
        raise BuildError("Runtime config ao_system.fov_arcsec must be positive")
    if min_wfs < 1 or max_wfs > 3 or min_wfs > max_wfs:
        raise BuildError("Runtime config ao_system WFS range must be within 1..3")
    if min_mag >= max_mag:
        raise BuildError("Runtime config ao_system.min_mag must be less than max_mag")
    if min_sep_arcsec < 0:
        raise BuildError("Runtime config ao_system.min_sep_arcsec must be non-negative")
    if min_sep_arcsec > fov_arcsec:
        raise BuildError("Runtime config ao_system.min_sep_arcsec cannot exceed fov_arcsec")
    if outer_level < 0:
        raise BuildError("Runtime config traversal.outer_level must be non-negative")
    if inner_level <= outer_level:
        raise BuildError("Runtime config traversal.inner_level must be larger than outer_level")
    if max_bright_star_exclusion_arcsec <= 0:
        raise BuildError(
            "Runtime config gaia.max_bright_star_exclusion_arcsec must be positive"
        )
    if not 0 <= winner_ee_epsilon < 1:
        raise BuildError(
            "Runtime config asterism.winner_ee_epsilon must be at least 0 and less than 1"
        )
    if not 1 <= winner_top_k <= MAX_WINNER_TOP_K:
        raise BuildError(
            "Runtime config asterism.winner_top_k must be between "
            f"1 and {MAX_WINNER_TOP_K}"
        )
    if prediction_wavelength <= 0:
        raise BuildError("Runtime config prediction.wavelength_micron must be positive")
    if seeing_wavelength <= 0:
        raise BuildError(
            "Runtime config best.seeing_baseline.wavelength_micron must be positive"
        )
    if seeing_fwhm < 0:
        raise BuildError("Runtime config best.seeing_baseline.fwhm_mas must be non-negative")
    if on_axis_ee_threshold < 0:
        raise BuildError("Runtime config coverage.on_axis_ee_threshold must be non-negative")
    if field_averaged_ee_threshold < 0:
        raise BuildError("Runtime config coverage.field_averaged_ee_threshold must be non-negative")
