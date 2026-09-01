"""Typed native runtime records for AO prediction and traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np


@dataclass(frozen=True, slots=True)
class AOSystemRuntime:
    """AO-system constraints used to construct prediction inputs.

    Attributes:
        band: Guide-star magnitude column used by Traversal.
        fov: Full angular diameter of the circular field of regard.
        lgs: Laser-guide-star records with zenith-distance and azimuth values.
        min_wfs: Minimum number of natural guide stars in a candidate.
        max_wfs: Maximum number of natural guide stars in a candidate.
        min_mag: Bright guide-star magnitude limit, inclusive.
        max_mag: Faint guide-star magnitude limit, inclusive.
        min_sep: Minimum angular separation between guide stars.
    """

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
    """Validated prediction and Traversal policy for one build.

    Attributes:
        ao_system: AO-system constraints used for candidate construction.
        outer_level: HEALPix level of one restartable outer-pixel task.
        inner_level: HEALPix level of evaluated science-point centers.
        epoch: Decimal year used for Gaia proper-motion propagation.
        max_bright_star_mag: Magnitude below which a star can mask nearby
            science points, or `None` to disable bright-star masking.
        max_bright_star_exclusion: Angular masking radius around bright stars.
        winner_ee_epsilon: Relative EE tolerance used by winner regularization.
        winner_top_k: Maximum candidate labels retained per inner pixel.
        prediction_wavelength: Wavelength at which model outputs are evaluated.
        models: Production position-dependent model names keyed by `<N>star`.
        legacy_field_averaged_models: Transitional field-averaged model names
            keyed by `<N>star`.
        seeing_reference_wavelength: Wavelength of the seeing fallback values.
        seeing_reference_sr: Reference seeing-limited Strehl ratio.
        seeing_reference_ee: Reference seeing-limited ensquared energy.
        seeing_reference_fwhm: Reference seeing-limited FWHM in milliarcseconds.
        on_axis_ee_threshold: EE threshold used for on-axis coverage.
        field_averaged_ee_threshold: EE threshold used for field-averaged
            coverage.
        model_root: Directory containing the configured model snapshots.
    """

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
    """Seeing-limited fallback metrics at the prediction wavelength.

    Attributes:
        sr: Dimensionless Strehl ratio, or `None` when unavailable.
        ee: Dimensionless ensquared energy, or `None` when unavailable.
        fwhm: FWHM in milliarcseconds, or `None` when unavailable.
    """

    sr: float | None
    ee: float | None
    fwhm: float | None
