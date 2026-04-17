"""Gaia-domain in-memory table transforms."""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from ._schema import coerce_table_to_canonical_gaia_schema
from ._exceptions import GaiaError


def _get_pm_component(values: np.ndarray) -> np.ndarray:
    data = np.ma.asarray(values, dtype=np.float64)
    if np.ma.isMaskedArray(data):
        return np.asarray(data.filled(0.0), dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def compute_r_magnitude(table: Table) -> np.ndarray:
    """Return the empirical Gaia-to-`R` magnitude estimate.

    This helper applies the same Gaia DR3 polynomial that the legacy
    `survey_tools` guide-star path used. It does not mutate the caller's input
    table and does not persist the derived values into the canonical Gaia store.

    Args:
        table: Gaia table exposing `G`, `BP`, and `RP` columns.

    Returns:
        NumPy array containing the derived `R` magnitude for each row. Rows with
        missing `BP` or `RP` propagate `NaN`.

    Raises:
        GaiaError: If the input table does not expose the canonical Gaia schema.
    """

    canonical = coerce_table_to_canonical_gaia_schema(table)
    mag_diff = np.asarray(canonical["BP"], dtype=np.float64) - np.asarray(
        canonical["RP"], dtype=np.float64
    )
    g_minus_r_mag = (
        -0.02275
        + 0.3961 * mag_diff
        - 0.1243 * np.power(mag_diff, 2)
        - 0.01396 * np.power(mag_diff, 3)
        + 0.003775 * np.power(mag_diff, 4)
    )
    return np.asarray(canonical["G"], dtype=np.float64) - g_minus_r_mag


def apply_proper_motion(
    table: Table,
    *,
    epoch: float | None = None,
    dt_years: float | None = None,
) -> Table:
    """Return a canonical Gaia table shifted by proper motion.

    This helper preserves the canonical Gaia schema and updates the returned
    ``ra``, ``dec``, and ``ref_epoch`` columns in-memory. It does not mutate
    the caller's input table.

    Args:
        table: Canonical Gaia table to shift.
        epoch: Target epoch for the returned coordinates.
        dt_years: Time shift in Julian years to apply to every row.

    Returns:
        Canonical Gaia `Table` with epoch-adjusted coordinates.

    Raises:
        GaiaError: If neither or both of ``epoch`` and ``dt_years`` are
            provided, or if the table does not match the canonical schema.
    """

    if epoch is None and dt_years is None:
        raise GaiaError("either epoch or dt_years must be provided")
    if epoch is not None and dt_years is not None:
        raise GaiaError("epoch and dt_years are mutually exclusive")

    shifted = coerce_table_to_canonical_gaia_schema(table)

    if epoch is not None:
        dt = float(epoch) - np.asarray(shifted["ref_epoch"], dtype=np.float64)
        new_ref_epoch = np.full(len(shifted), float(epoch), dtype=np.float64)
    else:
        dt = np.full(len(shifted), float(dt_years), dtype=np.float64)
        new_ref_epoch = np.asarray(shifted["ref_epoch"], dtype=np.float64) + dt

    pm_ra = _get_pm_component(table["pmra"]) / np.cos(
        np.asarray(shifted["dec"], dtype=np.float64) / 180.0 * np.pi
    )
    pm_dec = _get_pm_component(table["pmdec"])

    shifted["ra"] = np.asarray(shifted["ra"], dtype=np.float64) + dt * pm_ra / 1000.0 / 3600.0
    shifted["dec"] = np.asarray(shifted["dec"], dtype=np.float64) + dt * pm_dec / 1000.0 / 3600.0
    shifted["ref_epoch"] = new_ref_epoch
    return shifted
