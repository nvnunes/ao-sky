"""Asterism loader and search tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from ao_sky.asterisms import (
    ASTERISM_TABLE_COLUMNS,
    AsterismError,
    AsterismSearchOptions,
    find_asterisms,
    load_asterism_stars,
)
import ao_sky.asterisms.search as search_module
from ao_sky.gaia import (
    GAIA_SCHEMA_COLUMNS,
    GaiaHealpixStore,
    GaiaStoreConfig,
    compute_legacy_r_magnitude,
)
from ao_sky.spatial import get_parent_pixel, get_pixel_skycoord, get_pixel_neighbours, get_subpixels


def _empty_gaia_table() -> Table:
    table = Table()
    table["source_id"] = np.array([], dtype=np.int64)
    for name in ("ra", "dec", "G", "BP", "RP", "ref_epoch", "pmra", "pmdec", "ruwe"):
        table[name] = np.array([], dtype=np.float64)
    table["non_single_star"] = np.array([], dtype=np.bool_)
    return table


def _gaia_table(rows: list[tuple[int, float, float, float, float, float]]) -> Table:
    return Table(
        [
            np.asarray([row[0] for row in rows], dtype=np.int64),
            np.asarray([row[1] for row in rows], dtype=np.float64),
            np.asarray([row[2] for row in rows], dtype=np.float64),
            np.asarray([row[3] for row in rows], dtype=np.float64),
            np.asarray([row[3] + 0.2 for row in rows], dtype=np.float64),
            np.asarray([row[3] - 0.2 for row in rows], dtype=np.float64),
            np.asarray([2016.0 for _ in rows], dtype=np.float64),
            np.asarray([row[4] for row in rows], dtype=np.float64),
            np.asarray([row[5] for row in rows], dtype=np.float64),
            np.zeros(len(rows), dtype=np.bool_),
            np.ones(len(rows), dtype=np.float64),
        ],
        names=GAIA_SCHEMA_COLUMNS,
    )


def test_load_asterism_stars_returns_local_rows_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = GaiaHealpixStore(GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=1))
    local = _gaia_table([(101, 10.0, 0.0, 12.0, 0.0, 0.0)])

    monkeypatch.setattr(
        GaiaHealpixStore,
        "load_healpix",
        lambda self, pix, force_reload=False: local if pix == 0 else _empty_gaia_table(),
    )

    stars = load_asterism_stars(store, 0)

    assert stars.colnames == [*GAIA_SCHEMA_COLUMNS, "R"]
    assert stars["source_id"].tolist() == [101]
    assert np.isfinite(stars["R"][0])


def test_load_asterism_stars_border_trims_neighbour_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_level = 1
    neighbour_level = 2
    outer_pix = 0
    store = GaiaHealpixStore(GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=outer_level))

    local_subpixels = get_subpixels(outer_level, outer_pix, neighbour_level)
    bordering = np.setdiff1d(
        np.unique(np.concatenate([get_pixel_neighbours(neighbour_level, int(pixel)) for pixel in local_subpixels])),
        local_subpixels,
    )
    neighbour_outer_pix = int(get_parent_pixel(neighbour_level, int(bordering[0]), outer_level))
    neighbour_subpixels = get_subpixels(outer_level, neighbour_outer_pix, neighbour_level)
    interior_subpixel = int(np.setdiff1d(neighbour_subpixels, bordering)[0])

    local_coord = get_pixel_skycoord(neighbour_level, int(local_subpixels[0]))
    bordering_coord = get_pixel_skycoord(neighbour_level, int(bordering[0]))
    interior_coord = get_pixel_skycoord(neighbour_level, interior_subpixel)

    local = _gaia_table([(101, float(local_coord.ra.degree), float(local_coord.dec.degree), 12.0, 0.0, 0.0)])
    neighbour = _gaia_table(
        [
            (202, float(bordering_coord.ra.degree), float(bordering_coord.dec.degree), 13.0, 0.0, 0.0),
            (303, float(interior_coord.ra.degree), float(interior_coord.dec.degree), 14.0, 0.0, 0.0),
        ]
    )

    def fake_load(self: GaiaHealpixStore, pix: int, force_reload: bool = False) -> Table:
        if pix == outer_pix:
            return local
        if pix == neighbour_outer_pix:
            return neighbour
        return _empty_gaia_table()

    monkeypatch.setattr(GaiaHealpixStore, "load_healpix", fake_load)

    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=neighbour_level,
        include_locality=True,
    )

    assert stars["source_id"].tolist() == [101, 202]
    assert stars["is_local"].tolist() == [True, False]
    assert stars["source_outer_pix"].tolist() == [outer_pix, neighbour_outer_pix]


def test_load_asterism_stars_applies_proper_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GaiaHealpixStore(GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=1))
    local = _gaia_table([(101, 10.0, 0.0, 12.0, 100.0, 50.0)])
    monkeypatch.setattr(
        GaiaHealpixStore,
        "load_healpix",
        lambda self, pix, force_reload=False: local if pix == 0 else _empty_gaia_table(),
    )

    stars = load_asterism_stars(store, 0, epoch=2017.0)

    assert np.isclose(stars["ra"][0], 10.0 + 100.0 / 1000.0 / 3600.0)
    assert np.isclose(stars["dec"][0], 50.0 / 1000.0 / 3600.0)
    assert stars["ref_epoch"][0] == 2017.0


def _search_table(rows: list[tuple[int, float, float, float]]) -> Table:
    table = _gaia_table([(row[0], row[1], row[2], row[3], 0.0, 0.0) for row in rows])
    table["R"] = np.asarray([row[3] for row in rows], dtype=np.float64)
    return table


def test_find_asterisms_returns_single_star_output() -> None:
    stars = _search_table([(101, 10.0, 0.0, 12.0)])
    options = AsterismSearchOptions(min_stars=1, max_stars=1)

    result = find_asterisms(stars, options)

    assert result.colnames == list(ASTERISM_TABLE_COLUMNS)
    assert result["asterism_id"].tolist() == [1]
    assert result["num_stars"].tolist() == [1]
    assert result["star1_source_id"].tolist() == [101]
    assert result["star1_mag"].tolist() == [12.0]
    assert result["radius_arcsec"].tolist() == [30.0]
    assert result["separation_arcsec"].tolist() == [60.0]


def test_find_asterisms_returns_deterministic_two_and_three_star_results() -> None:
    arcsec = 1.0 / 3600.0
    stars = _search_table(
        [
            (101, 10.0, 0.0, 12.0),
            (202, 10.0 + 10.0 * arcsec, 0.0, 12.5),
            (303, 10.0 + 5.0 * arcsec, 8.0 * arcsec, 12.8),
        ]
    )

    pairs = find_asterisms(stars, AsterismSearchOptions(min_stars=2, max_stars=2))
    triplets = find_asterisms(stars, AsterismSearchOptions(min_stars=3, max_stars=3))
    triplets_repeat = find_asterisms(stars, AsterismSearchOptions(min_stars=3, max_stars=3))

    assert len(pairs) == 3
    assert len(triplets) == 1
    assert triplets["asterism_id"].tolist() == [1]
    assert triplets["star1_source_id"].tolist() == [101]
    assert triplets["star2_source_id"].tolist() == [202]
    assert triplets["star3_source_id"].tolist() == [303]
    assert triplets["star1_mag"].tolist() == [12.0]
    assert triplets["area_arcsec2"][0] > 0.0
    assert np.array_equal(triplets.as_array(), triplets_repeat.as_array())


def test_find_asterisms_handles_multiple_buffer_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arcsec = 1.0 / 3600.0
    stars = _search_table(
        [
            (101, 10.0, 0.0, 12.0),
            (202, 10.0 + 10.0 * arcsec, 0.0, 12.2),
            (303, 10.0 + 5.0 * arcsec, 8.0 * arcsec, 12.4),
        ]
    )

    original_init = search_module._AsterismBuffer.__init__

    def tiny_buffer_init(self: search_module._AsterismBuffer) -> None:
        original_init(self)
        self.max_size = 2

    monkeypatch.setattr(search_module._AsterismBuffer, "__init__", tiny_buffer_init)

    result = find_asterisms(stars, AsterismSearchOptions(min_stars=1, max_stars=3))

    assert len(result) == 7
    assert result["asterism_id"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_find_asterisms_uses_derived_r_magnitude_for_triplet_scoring() -> None:
    arcsec = 1.0 / 3600.0
    stars = _gaia_table(
        [
            (101, 10.0, 0.0, 12.0, 0.0, 0.0),
            (202, 10.0 + 10.0 * arcsec, 0.0, 12.0, 0.0, 0.0),
            (303, 10.0 + 5.0 * arcsec, 8.0 * arcsec, 12.0, 0.0, 0.0),
        ]
    )
    stars["G"] = np.asarray([12.0, 12.0, 12.0], dtype=np.float64)
    stars["BP"] = np.asarray([12.1, 12.2, 14.0], dtype=np.float64)
    stars["RP"] = np.asarray([11.9, 12.0, 13.8], dtype=np.float64)
    stars["R"] = compute_legacy_r_magnitude(stars)

    triplets = find_asterisms(stars, AsterismSearchOptions(min_stars=3, max_stars=3))

    assert len(triplets) == 1
    assert triplets["star1_mag"].tolist() == [float(stars["R"][0])]
    assert np.isclose(triplets["separation_arcsec"][0], 10.0, atol=0.1)


def test_asterism_search_options_enforce_phase_two_contract() -> None:
    with pytest.raises(AsterismError, match="supports only 1-3 stars"):
        AsterismSearchOptions(max_stars=4)

    with pytest.raises(AsterismError, match="cannot exceed"):
        AsterismSearchOptions(min_separation_arcsec=10.0, max_separation_arcsec=5.0)
