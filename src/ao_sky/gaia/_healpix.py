"""Minimal HEALPix helpers for the Gaia store.

This module owns the HEALPix primitives needed by the raw Gaia store for
pixel-centre lookup and query radius estimation. Broader neighbour and
subpixel traversal remain outside this module.
"""

from __future__ import annotations

import math

import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy_healpix import HEALPix

from ._exceptions import GaiaError


# Validation and construction

def _validate_level(level: int) -> int:
    if level < 0:
        raise GaiaError(f"healpix_level must be non-negative, got {level}")
    return level


def _get_healpix(level: int) -> HEALPix:
    return HEALPix(nside=2 ** _validate_level(level), order="nested", frame=ICRS())


def get_resolution(level: int) -> u.Quantity:
    """Return the characteristic pixel width for one nested HEALPix level.

    The returned quantity is the square root of the pixel area and is used by
    the Gaia archive query seam to build the retained region prefilter around a
    target outer pixel. The quantity is returned in angular units and callers
    may convert it as needed for persisted path or query construction logic.

    Args:
        level: Nested HEALPix level whose characteristic pixel width is
            required.

    Returns:
        An angular ``Quantity`` representing the square root of the HEALPix
        pixel area at ``level``.

    Raises:
        GaiaError: If ``level`` is negative.
    """

    level = _validate_level(level)
    area_deg2 = 4.0 * math.pi * (180.0 / math.pi) ** 2 / (12 * 4**level)
    return math.sqrt(area_deg2) * u.degree


def get_pixel_skycoord(level: int, pix: int) -> SkyCoord:
    """Return the ICRS centre coordinate for one nested-order HEALPix pixel.

    This helper is the canonical owner of the Gaia store's pixel-centre lookup.
    It validates both the requested level and the pixel index against the
    corresponding nested HEALPix grid before returning the ICRS sky coordinate
    used by the path-layout and archive-query owners.

    Args:
        level: Nested HEALPix level that defines the outer pixel grid.
        pix: Pixel index within the nested grid at ``level``.

    Returns:
        The ICRS centre ``SkyCoord`` for the requested nested HEALPix pixel.

    Raises:
        GaiaError: If ``level`` is negative or ``pix`` falls outside the valid
            range for that level.
    """

    healpix = _get_healpix(level)
    if pix < 0 or pix >= healpix.npix:
        raise GaiaError(
            f"outer_pix must be between 0 and {healpix.npix - 1}, got {pix}"
        )
    return healpix.healpix_to_skycoord(pix)
