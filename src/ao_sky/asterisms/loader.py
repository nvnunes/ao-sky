"""Asterism-side star assembly helpers."""

from __future__ import annotations

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
import numpy as np

from ..gaia import GaiaHealpixStore, apply_proper_motion, compute_legacy_r_magnitude
from ..gaia._constants import GAIA_SCHEMA_COLUMNS
from ..spatial import get_pixel_from_skycoord, get_pixel_neighbours, get_subpixels
from ._exceptions import AsterismError


# Validation helpers

def _empty_gaia_table() -> Table:
    table = Table()
    table["source_id"] = np.array([], dtype=np.int64)
    for name in ("ra", "dec", "G", "BP", "RP", "ref_epoch", "pmra", "pmdec", "ruwe"):
        table[name] = np.array([], dtype=np.float64)
    table["non_single_star"] = np.array([], dtype=np.bool_)
    return table


def _require_gaia_table(table: Table) -> Table:
    missing = [name for name in GAIA_SCHEMA_COLUMNS if name not in table.colnames]
    if missing:
        raise AsterismError(
            "load_asterism_stars requires canonical Gaia rows with columns: "
            + ", ".join(GAIA_SCHEMA_COLUMNS)
        )
    return table[list(GAIA_SCHEMA_COLUMNS)]


def _get_bordering_subpixels(
    outer_level: int,
    outer_pix: int,
    neighbour_level: int,
) -> np.ndarray:
    subpixels = get_subpixels(outer_level, outer_pix, neighbour_level)
    neighbour_sets = [get_pixel_neighbours(neighbour_level, int(pixel)) for pixel in subpixels]
    if not neighbour_sets:
        return np.array([], dtype=np.int64)
    bordering = np.unique(np.concatenate(neighbour_sets))
    return np.setdiff1d(bordering, subpixels)


# Public loader

def load_asterism_stars(
    store: GaiaHealpixStore,
    outer_pix: int,
    *,
    neighbour_level: int | None = None,
    epoch: float | None = None,
    dt_years: float | None = None,
    include_locality: bool = False,
) -> Table:
    """Load the Gaia rows needed to search one outer pixel for asterisms.

    This helper assembles the local outer-pixel Gaia table and, when requested,
    the border-trimmed neighbour rows needed for edge-complete asterism search.
    It can also apply Gaia proper motion in-memory before returning the search
    table.

    Args:
        store: Canonical raw Gaia store to read from.
        outer_pix: Target outer-pixel id at ``store.config.healpix_level``.
        neighbour_level: Finer HEALPix level used to decide which neighbour
            rows are needed for border completeness.
        epoch: Optional target epoch for returned coordinates.
        dt_years: Optional direct year offset for returned coordinates.
        include_locality: When true, include ``is_local`` and
            ``source_outer_pix`` columns in the returned table.

    Returns:
        Gaia-schema `Table` containing the rows to search for one outer pixel.

    Raises:
        AsterismError: If the neighbour-level contract is invalid or the loaded
            tables do not expose the canonical Gaia schema.
    """

    outer_level = store.config.healpix_level
    if neighbour_level is not None and neighbour_level <= outer_level:
        raise AsterismError("neighbour_level must be larger than the store healpix level")

    local = _require_gaia_table(store.load_healpix(outer_pix)).copy(copy_data=True)
    local["is_local"] = np.ones(len(local), dtype=np.bool_)
    local["source_outer_pix"] = np.full(len(local), int(outer_pix), dtype=np.int64)
    tables = [local]

    if neighbour_level is not None:
        bordering_subpixels = _get_bordering_subpixels(outer_level, outer_pix, neighbour_level)
        for neighbour_pix in get_pixel_neighbours(outer_level, outer_pix):
            neighbour = _require_gaia_table(store.load_healpix(int(neighbour_pix))).copy(copy_data=True)
            if len(neighbour) == 0:
                continue
            pixels = get_pixel_from_skycoord(
                neighbour_level,
                SkyCoord(ra=neighbour["ra"], dec=neighbour["dec"], unit=(u.degree, u.degree)),
            )
            keep = np.isin(pixels, bordering_subpixels)
            trimmed = neighbour[keep]
            if len(trimmed) == 0:
                continue
            trimmed["is_local"] = np.zeros(len(trimmed), dtype=np.bool_)
            trimmed["source_outer_pix"] = np.full(len(trimmed), int(neighbour_pix), dtype=np.int64)
            tables.append(trimmed)

    stars = tables[0] if len(tables) == 1 else vstack(tables)

    if epoch is not None or dt_years is not None:
        locality_cols = {}
        for name in ("is_local", "source_outer_pix"):
            locality_cols[name] = np.asarray(stars[name]).copy()
        stars = apply_proper_motion(stars[list(GAIA_SCHEMA_COLUMNS)], epoch=epoch, dt_years=dt_years)
        for name, values in locality_cols.items():
            stars[name] = values

    stars["R"] = compute_legacy_r_magnitude(stars[list(GAIA_SCHEMA_COLUMNS)])

    stars = stars[~np.isnan(stars["ra"]) & ~np.isnan(stars["dec"])]
    if not include_locality:
        stars.remove_columns(["is_local", "source_outer_pix"])
    return stars
