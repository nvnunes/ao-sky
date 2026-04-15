"""Minimal HEALPix helpers for the public spatial surface.

This module owns the reusable nested-order HEALPix primitives needed by Gaia
loading, asterism assembly, and later pass/build orchestration. It intentionally
excludes plotting helpers and higher-level policy.
"""

from __future__ import annotations

import math
from typing import overload

import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
import numpy as np
from astropy_healpix import HEALPix
import astropy_healpix

from ._exceptions import SpatialError


# Validation and construction

def _validate_level(level: int) -> int:
    level = int(level)
    if level < 0:
        raise SpatialError(f"level must be non-negative, got {level}")
    return level


def _get_healpix(level: int) -> HEALPix:
    return HEALPix(nside=2 ** _validate_level(level), order="nested", frame=ICRS())


def _as_pixel_array(pix: int | np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    array = np.asarray(pix, dtype=np.int64)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _validate_pixels(level: int, pix: int | np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
    pixels = _as_pixel_array(pix)
    hp = _get_healpix(level)
    if np.any(pixels < 0) or np.any(pixels >= hp.npix):
        raise SpatialError(
            f"pixel values must be between 0 and {hp.npix - 1}, got {pixels.tolist()}"
        )
    return pixels


# Public primitives

def get_pixel_area(level: int) -> u.Quantity:
    """Return the area of one nested HEALPix pixel.

    Args:
        level: Nested HEALPix level whose pixel area is required.

    Returns:
        Angular area as an `astropy.units.Quantity` in ``arcsec**2`` units.

    Raises:
        SpatialError: If ``level`` is negative.
    """

    level = _validate_level(level)
    area_arcsec2 = 4.0 * math.pi * (180.0 / math.pi) ** 2 / (12 * 4**level) * 3600.0**2
    return area_arcsec2 * u.arcsec**2


def get_pixel_resolution(level: int) -> u.Quantity:
    """Return the characteristic width of one nested HEALPix pixel.

    The returned quantity is the square root of the pixel area and matches the
    legacy helper used by the Gaia query and asterism neighbour logic.

    Args:
        level: Nested HEALPix level whose characteristic width is required.

    Returns:
        Angular width as an `astropy.units.Quantity`.

    Raises:
        SpatialError: If ``level`` is negative.
    """

    return np.sqrt(get_pixel_area(level))


@overload
def get_pixel_from_skycoord(level: int, skycoords: SkyCoord) -> int | np.ndarray: ...


def get_pixel_from_skycoord(level: int, skycoords: SkyCoord) -> int | np.ndarray:
    """Return nested HEALPix pixel ids for one or more sky coordinates.

    Args:
        level: Nested HEALPix level that defines the target pixel grid.
        skycoords: One scalar or vector `SkyCoord`.

    Returns:
        Integer pixel id for scalar input or a NumPy integer array for vector
        input.

    Raises:
        SpatialError: If ``level`` is negative.
    """

    hp = _get_healpix(level)
    pixels = hp.skycoord_to_healpix(skycoords)
    array = np.asarray(pixels, dtype=np.int64)
    if array.ndim == 0:
        return int(array)
    return array


def get_pixel_skycoord(level: int, pix: int | np.ndarray | list[int] | tuple[int, ...]) -> SkyCoord:
    """Return the ICRS centre coordinate for one or more nested HEALPix pixels.

    Args:
        level: Nested HEALPix level that defines the pixel grid.
        pix: One pixel id or a sequence of pixel ids.

    Returns:
        Scalar or vector `SkyCoord` with the centre of each requested pixel.

    Raises:
        SpatialError: If ``level`` is negative or any pixel id is out of range.
    """

    pixels = _validate_pixels(level, pix)
    coords = _get_healpix(level).healpix_to_skycoord(pixels)
    if np.asarray(pix).ndim == 0:
        return coords[0]
    return coords


def get_parent_pixel(level: int, pix: int | np.ndarray | list[int] | tuple[int, ...], outer_level: int) -> int | np.ndarray:
    """Return the parent nested HEALPix pixel at a coarser level.

    Args:
        level: Current nested HEALPix level of ``pix``.
        pix: One pixel id or a sequence of pixel ids at ``level``.
        outer_level: Coarser nested HEALPix level to project into.

    Returns:
        Parent pixel id for scalar input or a NumPy integer array for vector
        input.

    Raises:
        SpatialError: If levels are invalid or ``outer_level`` is finer than
            ``level``.
    """

    level = _validate_level(level)
    outer_level = _validate_level(outer_level)
    if outer_level > level:
        raise SpatialError("outer_level must be less than or equal to level")

    pixels = _validate_pixels(level, pix)
    parents = pixels // 4 ** (level - outer_level)
    if np.asarray(pix).ndim == 0:
        return int(parents[0])
    return parents


def get_pixel_neighbours(level: int, pix: int) -> np.ndarray:
    """Return the valid neighbouring pixels for one nested HEALPix pixel.

    This public helper normalizes the Astropy neighbour output by removing the
    ``-1`` sentinel values used at map boundaries.

    Args:
        level: Nested HEALPix level that defines the pixel grid.
        pix: Pixel id at ``level``.

    Returns:
        One-dimensional NumPy integer array of valid neighbour pixel ids in the
        order reported by Astropy after dropping invalid sentinels.

    Raises:
        SpatialError: If ``level`` is negative or ``pix`` is out of range.
    """

    pixel = int(_validate_pixels(level, pix)[0])
    with np.errstate(invalid="ignore"):
        neighbours = astropy_healpix.neighbours(pixel, 2 ** level, order="nested")
    values = np.asarray(neighbours, dtype=np.int64)
    return values[values >= 0]


def get_subpixels(
    outer_level: int,
    outer_pix: int | np.ndarray | list[int] | tuple[int, ...],
    inner_level: int,
) -> np.ndarray:
    """Return all nested subpixels contained by one or more outer pixels.

    Args:
        outer_level: Coarser nested HEALPix level of the parent pixels.
        outer_pix: One parent pixel id or a sequence of parent pixel ids.
        inner_level: Finer nested HEALPix level to expand into.

    Returns:
        Flat NumPy integer array of the contained subpixel ids in nested-order
        block layout.

    Raises:
        SpatialError: If levels are invalid, pixel ids are out of range, or
            ``inner_level`` is not finer than ``outer_level``.
    """

    outer_level = _validate_level(outer_level)
    inner_level = _validate_level(inner_level)
    if inner_level <= outer_level:
        raise SpatialError("inner_level must be larger than outer_level")

    outer_pixels = _validate_pixels(outer_level, outer_pix)
    pixels_per_outer = 4 ** (inner_level - outer_level)
    all_pixels: list[np.ndarray] = []
    for pixel in outer_pixels:
        start_pixel = int(pixel) * pixels_per_outer
        stop_pixel = start_pixel + pixels_per_outer
        all_pixels.append(np.arange(start_pixel, stop_pixel, dtype=np.int64))
    return np.concatenate(all_pixels) if all_pixels else np.array([], dtype=np.int64)


def get_subpixel_indexes(
    level: int,
    pix: int | np.ndarray | list[int] | tuple[int, ...],
    inner_level: int,
    outer_level: int,
) -> np.ndarray:
    """Return local subpixel indexes inside parent outer-pixel blocks.

    Args:
        level: Nested HEALPix level of ``pix``.
        pix: One or more pixel ids at ``level``.
        inner_level: Finer nested HEALPix level whose local indexes are
            required.
        outer_level: Outer nested HEALPix level that defines each local block.

    Returns:
        Flat NumPy integer array of subpixel indexes relative to the start of
        each parent ``outer_level`` pixel's ``inner_level`` block.

    Raises:
        SpatialError: If the level ordering is invalid or any pixel id is out
            of range.
    """

    level = _validate_level(level)
    inner_level = _validate_level(inner_level)
    outer_level = _validate_level(outer_level)
    if level <= outer_level:
        raise SpatialError("level must be larger than outer_level")
    if level > inner_level:
        raise SpatialError("level must be less than or equal to inner_level")

    pixels = _validate_pixels(level, pix)
    block_size = 4 ** (inner_level - outer_level)
    subpixel_size = 4 ** (inner_level - level)
    all_indexes: list[np.ndarray] = []
    for pixel in pixels:
        parent_outer = int(get_parent_pixel(level, int(pixel), outer_level))
        outer_start = parent_outer * block_size
        start_pixel = int(pixel) * subpixel_size
        stop_pixel = start_pixel + subpixel_size
        all_indexes.append(
            np.arange(start_pixel - outer_start, stop_pixel - outer_start, dtype=np.int64)
        )
    return np.concatenate(all_indexes) if all_indexes else np.array([], dtype=np.int64)
