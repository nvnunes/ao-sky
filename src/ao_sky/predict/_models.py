"""Typed native runtime records for AO prediction and traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np


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
    max_bright_star_mag: float | None
    max_bright_star_exclusion: u.Quantity
    winner_ee_epsilon: float
    winner_top_k: int
    prediction_wavelength: u.Quantity
    models: dict[str, str]
    legacy_field_averaged_models: dict[str, str]
    seeing_reference_wavelength: u.Quantity
    seeing_reference_sr: float
    seeing_reference_ee: float
    seeing_reference_fwhm: float
    on_axis_ee_threshold: float
    field_averaged_ee_threshold: float
    model_root: Path


@dataclass(frozen=True, slots=True)
class PredictionBatch:
    """Position-dependent outputs for one homogeneous prediction batch.

    Each array has shape `(rows,)` and preserves the input row order.

    Attributes:
        sr: Dimensionless Strehl-ratio predictions.
        ee: Dimensionless ensquared-energy predictions.
        fwhm: FWHM predictions in milliarcseconds.
        ee_angle: Reserved EE-orientation output, currently filled with zeros.
    """

    sr: np.ndarray
    ee: np.ndarray
    fwhm: np.ndarray
    ee_angle: np.ndarray


@dataclass(frozen=True, slots=True)
class SeeingBaselinePerformance:
    """Seeing-limited fallback metrics at the prediction wavelength."""

    sr: float | None
    ee: float | None
    fwhm: float | None
