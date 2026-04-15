"""Gaia proper-motion tests."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from ao_sky.gaia import (
    GAIA_SCHEMA_COLUMNS,
    GaiaError,
    apply_proper_motion,
    compute_legacy_r_magnitude,
)


def _make_table() -> Table:
    return Table(
        [
            np.asarray([101, 202], dtype=np.int64),
            np.asarray([10.0, 20.0]),
            np.asarray([0.0, 10.0]),
            np.asarray([12.1, 13.2]),
            np.asarray([12.4, 13.5]),
            np.asarray([11.8, 12.9]),
            np.asarray([2016.0, 2016.0]),
            MaskedColumn([100.0, 0.0], mask=[False, True]),
            MaskedColumn([50.0, 0.0], mask=[False, True]),
            np.asarray([False, True], dtype=np.bool_),
            np.asarray([1.01, 1.11]),
        ],
        names=GAIA_SCHEMA_COLUMNS,
    )


def test_apply_proper_motion_with_epoch_preserves_schema() -> None:
    shifted = apply_proper_motion(_make_table(), epoch=2017.0)

    assert shifted.colnames == list(GAIA_SCHEMA_COLUMNS)
    assert shifted["ref_epoch"].tolist() == [2017.0, 2017.0]
    assert np.isclose(shifted["ra"][0], 10.0 + 100.0 / 1000.0 / 3600.0)
    assert np.isclose(shifted["dec"][0], 50.0 / 1000.0 / 3600.0)


def test_apply_proper_motion_with_dt_years_updates_ref_epoch() -> None:
    shifted = apply_proper_motion(_make_table(), dt_years=2.0)

    assert shifted["ref_epoch"].tolist() == [2018.0, 2018.0]
    assert np.isclose(shifted["ra"][0], 10.0 + 2.0 * 100.0 / 1000.0 / 3600.0)
    assert np.isclose(shifted["dec"][0], 2.0 * 50.0 / 1000.0 / 3600.0)


def test_apply_proper_motion_treats_masked_pm_components_as_zero() -> None:
    shifted = apply_proper_motion(_make_table(), dt_years=3.0)

    assert shifted["ra"][1] == 20.0
    assert shifted["dec"][1] == 10.0


def test_apply_proper_motion_preserves_nan_pm_components() -> None:
    table = _make_table()
    table["pmra"] = np.asarray([100.0, np.nan], dtype=np.float64)
    table["pmdec"] = np.asarray([50.0, np.nan], dtype=np.float64)

    shifted = apply_proper_motion(table, dt_years=1.0)

    assert np.isnan(shifted["ra"][1])
    assert np.isnan(shifted["dec"][1])


def test_apply_proper_motion_requires_epoch_or_dt_years() -> None:
    with pytest.raises(GaiaError, match="either epoch or dt_years"):
        apply_proper_motion(_make_table())


def test_apply_proper_motion_rejects_epoch_and_dt_years_together() -> None:
    with pytest.raises(GaiaError, match="mutually exclusive"):
        apply_proper_motion(_make_table(), epoch=2017.0, dt_years=1.0)


def test_compute_legacy_r_magnitude_matches_legacy_polynomial() -> None:
    table = _make_table()
    table["G"] = np.asarray([12.5, 13.0])
    table["BP"] = np.asarray([12.8, 13.6])
    table["RP"] = np.asarray([12.1, 12.9])

    result = compute_legacy_r_magnitude(table)

    mag_diff = table["BP"] - table["RP"]
    expected = table["G"] - (
        -0.02275
        + 0.3961 * mag_diff
        - 0.1243 * np.power(mag_diff, 2)
        - 0.01396 * np.power(mag_diff, 3)
        + 0.003775 * np.power(mag_diff, 4)
    )
    assert np.allclose(result, expected)
