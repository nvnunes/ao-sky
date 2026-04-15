"""Typed build-definition and runtime records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u

from ..survey import SurveyExtentOverlaySpec


@dataclass(frozen=True, slots=True)
class BuildDefinition:
    """Minimal human-authored build definition."""

    ao_system_short_name: str
    config_short_name: str
    gaia_release: str
    outer_level: int
    inner_level: int
    max_data_level: int
    epoch: float
    min_galactic_latitude: float | None = None
    survey_extent_overlays: tuple[SurveyExtentOverlaySpec, ...] = ()


@dataclass(frozen=True, slots=True)
class BuildPaths:
    """Resolved filesystem roots used by build commands."""

    gaia_root: Path
    build_root: Path
    dust_root: Path


@dataclass(frozen=True, slots=True)
class LegacyAOSystemRuntime:
    """Temporary legacy AO-system runtime parameters used by Phase 3 builds."""

    name: str
    band: str
    fov: u.Quantity
    fov_1ngs: u.Quantity
    min_wfs: int
    max_wfs: int
    min_mag: float
    nom_mag: float
    max_mag: float
    min_sep: u.Quantity
    max_sep: u.Quantity
    min_rel_sep: float
    max_rel_sep: float
    min_rel_area: float
    max_rel_area: float


@dataclass(frozen=True, slots=True)
class LegacyBuildRuntime:
    """Temporary legacy-derived runtime policy used during Phase 3."""

    ao_system: LegacyAOSystemRuntime
    outer_level: int
    inner_level: int
    epoch: float
    min_galactic_latitude: float | None
    max_star_density: float | None
    max_bright_star_mag: float | None
    max_overlap: float | None
    legacy_config_path: Path
