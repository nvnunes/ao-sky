"""Spatial helper tests."""

from __future__ import annotations

import numpy as np

from ao_sky.spatial import (
    get_parent_pixel,
    get_pixel_from_skycoord,
    get_pixel_neighbours,
    get_pixel_skycoord,
    get_subpixel_indexes,
    get_subpixels,
)


def test_pixel_roundtrip_from_skycoord() -> None:
    coord = get_pixel_skycoord(3, 10)

    assert get_pixel_from_skycoord(3, coord) == 10


def test_get_parent_pixel_and_subpixels_match_nested_blocks() -> None:
    subpixels = get_subpixels(1, 0, 2)

    assert subpixels.tolist() == [0, 1, 2, 3]
    assert np.array_equal(get_parent_pixel(2, subpixels, 1), np.zeros(4, dtype=np.int64))


def test_get_subpixel_indexes_are_local_to_parent_block() -> None:
    indexes = get_subpixel_indexes(2, [0, 1], 3, 1)

    assert indexes.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_get_pixel_neighbours_drops_invalid_sentinels() -> None:
    neighbours = get_pixel_neighbours(1, 0)

    assert np.all(neighbours >= 0)
    assert len(neighbours) <= 8
