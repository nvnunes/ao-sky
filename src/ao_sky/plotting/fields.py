"""Field-aware plotting conventions for `ao-sky` map artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._exceptions import PlottingError


@dataclass(frozen=True)
class FieldConvention:
    """Formatting defaults for one native map artifact field."""

    title: str
    unit: str = ""
    cmap: str = "viridis"
    norm: str | None = "linear"
    vmin: float | None = None
    vmax: float | None = None
    cbar_format: str = "%g"
    cbar_ticks: tuple[float, ...] | None = None
    mask_nonpositive_for_log: bool = False


FIELD_CONVENTIONS: dict[str, FieldConvention] = {
    "gaia_A0": FieldConvention(
        title="Gaia A0",
        unit="mag",
        cmap="inferno",
        norm="linear",
        vmin=0.0,
        cbar_format="%.2f",
    ),
    "star_count": FieldConvention(
        title="Star Count",
        unit="count",
        cmap="viridis",
        norm="log",
        vmin=1.0,
        cbar_format="%g",
        mask_nonpositive_for_log=True,
    ),
    "stellar_density": FieldConvention(
        title="Stellar Density",
        unit="stars/arcmin^2",
        cmap="gray",
        norm="symlog",
        cbar_format="%g",
    ),
    "ngs_density": FieldConvention(
        title="NGS Density",
        unit="NGS/arcmin^2",
        cmap="gray",
        norm="symlog",
        cbar_format="%g",
    ),
    "ngs_count": FieldConvention(
        title="NGS Count",
        unit="count",
        cmap="viridis",
        norm="log",
        vmin=1.0,
        cbar_format="%g",
        mask_nonpositive_for_log=True,
    ),
    "winner_asterism_count": FieldConvention(
        title="Winner Asterism Count",
        unit="count",
        cmap="viridis",
        norm="log",
        vmin=1.0,
        cbar_format="%g",
        mask_nonpositive_for_log=True,
    ),
    "best_sr": FieldConvention(
        title="Best Strehl Ratio",
        unit="SR",
        cmap="viridis",
        norm="linear",
        vmin=0.0,
        vmax=0.4,
        cbar_format="%.2f",
    ),
    "best_ee": FieldConvention(
        title="Best EE",
        unit="EE",
        cmap="viridis",
        norm="linear",
        vmin=0.1,
        vmax=0.5,
        cbar_format="%.2f",
    ),
    "best_fwhm": FieldConvention(
        title="Best FWHM",
        unit="FWHM",
        cmap="magma_r",
        norm="linear",
        vmin=50.0,
        vmax=350.0,
        cbar_format="%.3g",
    ),
    "winner_ee_resolved": FieldConvention(
        title="Winner EE Resolved",
        unit="EE",
        cmap="viridis",
        norm="linear",
        vmin=0.1,
        vmax=0.5,
        cbar_format="%.2f",
    ),
    "winner_ee_averaged": FieldConvention(
        title="Winner EE Averaged",
        unit="EE",
        cmap="viridis",
        norm="linear",
        vmin=0.1,
        vmax=0.5,
        cbar_format="%.2f",
    ),
    "coverage_resolved": FieldConvention(
        title="Coverage Resolved",
        unit="fraction",
        cmap="viridis",
        norm="linear",
        vmin=0.0,
        vmax=1.0,
        cbar_format="%.2f",
    ),
    "coverage_mean": FieldConvention(
        title="Coverage Mean",
        unit="fraction",
        cmap="viridis",
        norm="linear",
        vmin=0.0,
        vmax=1.0,
        cbar_format="%.2f",
    ),
    "coverage_averaged": FieldConvention(
        title="Coverage Averaged",
        unit="fraction",
        cmap="viridis",
        norm="linear",
        vmin=0.0,
        vmax=1.0,
        cbar_format="%.2f",
    ),
}


def get_field_convention(field: str, **overrides: Any) -> FieldConvention:
    """Return plotting conventions for a native map field."""

    try:
        convention = FIELD_CONVENTIONS[field]
    except KeyError as exc:
        raise PlottingError(f"No plotting convention is defined for map field '{field}'") from exc
    if overrides:
        return replace(convention, **{key: value for key, value in overrides.items() if value is not None})
    return convention


def prepare_field_values(values: np.ndarray, convention: FieldConvention) -> np.ndarray:
    """Return a plotting copy of values with convention-specific masking."""

    prepared = np.asarray(values, dtype=np.float64).copy()
    if convention.mask_nonpositive_for_log or convention.norm == "log":
        prepared[prepared <= 0.0] = np.nan
    return prepared
