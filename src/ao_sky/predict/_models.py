"""Typed native runtime records for AO prediction and traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u


@dataclass(frozen=True, slots=True)
class AOSystemRuntime:
    """AO-system policy and model metadata needed by native traversal."""

    band: str
    fov: u.Quantity
    lgs: tuple[dict[str, float], ...]
    min_wfs: int
    max_wfs: int
    min_mag: float
    max_mag: float
    min_sep: u.Quantity


@dataclass(frozen=True, slots=True)
class PredictRuntime:
    """Native runtime contract passed into the real Traversal path."""

    ao_system: AOSystemRuntime
    outer_level: int
    inner_level: int
    epoch: float
    min_galactic_latitude: float | None
    max_star_density: float | None
    max_bright_star_mag: float | None
    max_overlap: float | None
    prediction_wavelength: u.Quantity
    resolved_models: dict[str, str]
    averaged_models: dict[str, str]
    seeing_reference_wavelength: u.Quantity
    seeing_reference_sr: float
    seeing_reference_ee: float
    seeing_reference_fwhm: float
    coverage_ee_threshold_resolved: float
    coverage_ee_threshold_averaged: float
    model_root: Path


@dataclass(frozen=True, slots=True)
class PointPredictionBatch:
    """Resolved point-performance predictions for one batch of pairs."""

    sr: object
    ee: object
    fwhm: object
    ee_angle: object


@dataclass(frozen=True, slots=True)
class SeeingBaselinePerformance:
    """Seeing-limited fallback metrics at the prediction wavelength."""

    sr: float | None
    ee: float | None
    fwhm: float | None
