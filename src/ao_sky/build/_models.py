"""Typed build-definition, filesystem, and execution records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    model_root: Path


@dataclass(frozen=True, slots=True)
class TraversalTaskContext:
    """Pickle-safe worker context for one Traversal run."""

    build_path: Path
    definition: BuildDefinition
    roots: BuildPaths
    legacy_config_path: Path


@dataclass(frozen=True, slots=True)
class TraversalTaskResult:
    """Serialized result for one worker-owned Traversal outer-pixel task."""

    outer_pix: int
    success: bool
    error_message: str = ""
