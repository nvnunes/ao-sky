"""Thin adapter from legacy `survey_tools` YAML into native runtime records."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import yaml

from ..predict import AOSystemRuntime, PredictRuntime
from ._exceptions import BuildError
from ._models import BuildDefinition

_DEFAULT_LGS = (
    {"zd": 30.0, "az": 45.0},
    {"zd": 30.0, "az": 135.0},
    {"zd": 30.0, "az": 225.0},
    {"zd": 30.0, "az": 315.0},
)


def _normalize_rot_range(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    if len(values) != 2:
        raise BuildError("legacy rot_range must contain exactly two values when provided")
    return values


def load_native_runtime(
    definition: BuildDefinition,
    *,
    legacy_config_path: Path,
    model_root: Path,
) -> PredictRuntime:
    """Load Phase 9 native runtime policy from the temporary legacy YAML."""

    with Path(legacy_config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    systems = raw.get("ao_systems", [])
    ao_system_raw = next(
        (
            item
            for item in systems
            if str(item.get("name", "")).strip() == definition.ao_system_short_name
        ),
        None,
    )
    if ao_system_raw is None:
        raise BuildError(
            f"AO system {definition.ao_system_short_name!r} not found in {legacy_config_path}"
        )

    if "band" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'band'")
    if "fov" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'fov'")
    if "min_wfs" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'min_wfs'")
    if "max_wfs" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'max_wfs'")
    if "min_mag" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'min_mag'")
    if "max_mag" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'max_mag'")
    if "min_sep" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'min_sep'")
    if "max_sep" not in ao_system_raw:
        raise BuildError("legacy AO system is missing required field 'max_sep'")

    fov_1ngs = ao_system_raw.get("fov_1ngs", ao_system_raw["fov"])
    lgs_raw = ao_system_raw.get("lgs", _DEFAULT_LGS)
    ao_system = AOSystemRuntime(
        name=str(ao_system_raw["name"]).strip(),
        band=str(ao_system_raw["band"]).strip(),
        fov=float(ao_system_raw["fov"]) * u.arcsec,
        fov_1ngs=float(fov_1ngs) * u.arcsec,
        lgs=tuple(
            {"zd": float(star["zd"]), "az": float(star["az"])}
            for star in lgs_raw
        ),
        min_wfs=int(ao_system_raw["min_wfs"]),
        max_wfs=int(ao_system_raw["max_wfs"]),
        min_mag=float(ao_system_raw["min_mag"]),
        nom_mag=float(ao_system_raw.get("nom_mag", ao_system_raw["max_mag"])),
        max_mag=float(ao_system_raw["max_mag"]),
        min_sep=float(ao_system_raw["min_sep"]) * u.arcsec,
        max_sep=float(ao_system_raw["max_sep"]) * u.arcsec,
        min_rel_sep=float(ao_system_raw.get("min_rel_sep", 0.0)),
        max_rel_sep=float(ao_system_raw.get("max_rel_sep", 0.0)),
        min_rel_area=float(ao_system_raw.get("min_rel_area", 0.0)),
        max_rel_area=float(ao_system_raw.get("max_rel_area", 0.0)),
        point_models={
            str(key): str(value)
            for key, value in (ao_system_raw.get("point_models") or {}).items()
        },
        mean_models={
            str(key): str(value)
            for key, value in (ao_system_raw.get("models") or {}).items()
        },
        rot_range=_normalize_rot_range(ao_system_raw.get("rot_range")),
        rot_step=(
            None
            if ao_system_raw.get("rot_step") is None
            else float(ao_system_raw["rot_step"])
        ),
    )

    return PredictRuntime(
        ao_system=ao_system,
        outer_level=definition.outer_level,
        inner_level=definition.inner_level,
        epoch=definition.epoch,
        min_galactic_latitude=definition.min_galactic_latitude,
        max_star_density=(
            float(raw.get("asterisms_max_star_density", 2.0))
        ),
        max_bright_star_mag=(
            None
            if raw.get("asterisms_max_bright_star_mag") is None
            else float(raw["asterisms_max_bright_star_mag"])
        ),
        max_overlap=(
            None
            if raw.get("asterisms_max_overlap") is None
            else float(raw["asterisms_max_overlap"])
        ),
        prediction_wavelength=float(raw.get("prediction_wavelength", 1.654)) * u.micron,
        seeing_reference_wavelength=float(raw.get("seeing_reference_wavelength", 0.5)) * u.micron,
        seeing_reference_sr=float(raw.get("seeing_reference_sr", 0.0)),
        seeing_reference_ee=float(raw.get("seeing_reference_ee", 0.02)),
        seeing_reference_fwhm=float(raw.get("seeing_reference_fwhm", 650.0)),
        coverage_ee_threshold_resolved=float(
            raw.get("coverage_ee_threshold_resolved", 0.25)
        ),
        coverage_ee_threshold_mean=float(raw.get("coverage_ee_threshold_mean", 0.25)),
        model_root=Path(model_root).expanduser().resolve(),
        legacy_config_path=Path(legacy_config_path).expanduser().resolve(),
    )
