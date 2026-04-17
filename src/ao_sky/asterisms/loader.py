"""Asterism-side star assembly helpers."""

from __future__ import annotations

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u
import numpy as np

from ..gaia import GaiaHealpixStore, apply_proper_motion, compute_r_magnitude
from ..gaia._constants import GAIA_SCHEMA_COLUMNS
from ..spatial import get_parent_pixel, get_pixel_from_skycoord, get_pixel_neighbours, get_subpixels
from ._exceptions import AsterismError

DERIVED_GAIA_COLUMNS = ("R", "hpx14")


# Validation helpers

def _empty_gaia_table() -> Table:
    table = Table()
    table["source_id"] = np.array([], dtype=np.int64)
    for name in ("ra", "dec", "G", "BP", "RP", "ref_epoch", "pmra", "pmdec", "ruwe"):
        table[name] = np.array([], dtype=np.float64)
    table["non_single_star"] = np.array([], dtype=np.bool_)
    return table


def _require_gaia_columns(table: Table) -> None:
    missing = [name for name in GAIA_SCHEMA_COLUMNS if name not in table.colnames]
    if missing:
        raise AsterismError(
            "load_asterism_stars requires canonical Gaia rows with columns: "
            + ", ".join(GAIA_SCHEMA_COLUMNS)
        )


def _require_gaia_table(table: Table) -> Table:
    _require_gaia_columns(table)
    columns = list(GAIA_SCHEMA_COLUMNS)
    columns.extend(name for name in DERIVED_GAIA_COLUMNS if name in table.colnames)
    return table[columns]


def _get_row_pixels(table: Table, level: int) -> np.ndarray:
    if "hpx14" in table.colnames and level <= 14:
        hpx = np.asarray(table["hpx14"], dtype=np.int64)
        pixels = np.full(hpx.shape, -1, dtype=np.int64)
        valid = hpx >= 0
        if np.any(valid):
            pixels[valid] = np.asarray(
                get_parent_pixel(14, hpx[valid], level),
                dtype=np.int64,
            )
        return pixels
    return np.asarray(
        get_pixel_from_skycoord(
            level,
            SkyCoord(ra=table["ra"], dec=table["dec"], unit=(u.degree, u.degree)),
        ),
        dtype=np.int64,
    )


def _get_bordering_subpixels(
    outer_level: int,
    outer_pix: int,
    neighbour_level: int,
    boundary_rings: int,
) -> np.ndarray:
    if boundary_rings < 1:
        raise AsterismError("boundary_rings must be at least 1")

    subpixels = get_subpixels(outer_level, outer_pix, neighbour_level)
    interior = set(int(pixel) for pixel in subpixels)
    visited = set(interior)
    frontier = set(interior)
    bordering: set[int] = set()
    for _ in range(int(boundary_rings)):
        next_frontier: set[int] = set()
        for pixel in frontier:
            for neighbour in get_pixel_neighbours(neighbour_level, int(pixel)):
                value = int(neighbour)
                if value in visited:
                    continue
                visited.add(value)
                bordering.add(value)
                next_frontier.add(value)
        frontier = next_frontier
        if not frontier:
            break
    return np.asarray(sorted(bordering), dtype=np.int64)


# Public loader

def load_asterism_stars(
    store: GaiaHealpixStore,
    outer_pix: int,
    *,
    neighbour_level: int | None = None,
    boundary_rings: int = 2,
    epoch: float | None = None,
    dt_years: float | None = None,
    include_locality: bool = False,
) -> Table:
    """Load the Gaia rows needed to search one outer pixel for asterisms.

    This helper assembles the local outer-pixel Gaia table and, when requested,
    the border-trimmed neighbour rows needed for edge-complete asterism search.
    The default two-ring boundary also provides headroom before proper-motion
    shifting, so stars just outside the raw boundary can still contribute after
    epoch shifting. It can apply Gaia proper motion in-memory before returning
    the search table.

    Args:
        store: Canonical raw Gaia store to read from.
        outer_pix: Target outer-pixel id at ``store.config.healpix_level``.
        neighbour_level: Finer HEALPix level used to decide which neighbour
            rows are needed for border completeness.
        boundary_rings: Number of fine-HEALPix rings outside the target outer
            pixel to include when ``neighbour_level`` is set. The default of
            two is the fixed Traversal rule because one ring was insufficient
            near boundaries and gives less buffer for epoch-shifted stars.
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
    if boundary_rings < 1:
        raise AsterismError("boundary_rings must be at least 1")

    local = _require_gaia_table(store.load_healpix(outer_pix, read_only=True)).copy(
        copy_data=True
    )
    local["is_local"] = np.ones(len(local), dtype=np.bool_)
    local["source_outer_pix"] = np.full(len(local), int(outer_pix), dtype=np.int64)
    tables = [local]

    if neighbour_level is not None:
        bordering_subpixels = _get_bordering_subpixels(
            outer_level,
            outer_pix,
            neighbour_level,
            boundary_rings,
        )
        for neighbour_pix in get_pixel_neighbours(outer_level, outer_pix):
            neighbour = store.load_healpix(int(neighbour_pix), read_only=True)
            _require_gaia_columns(neighbour)
            if len(neighbour) == 0:
                continue
            pixels = _get_row_pixels(neighbour, neighbour_level)
            keep = np.isin(pixels, bordering_subpixels)
            trimmed = _require_gaia_table(neighbour[keep]).copy(copy_data=True)
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

    if "R" not in stars.colnames:
        stars["R"] = compute_r_magnitude(stars[list(GAIA_SCHEMA_COLUMNS)])

    stars = stars[~np.isnan(stars["ra"]) & ~np.isnan(stars["dec"])]
    if not include_locality:
        stars.remove_columns(["is_local", "source_outer_pix"])
    return stars
