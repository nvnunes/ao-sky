"""Asterism loader and search tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from mocpy import MOC

from ao_sky._paths import get_outer_pixel_bucket_path
from ao_sky.asterisms import (
    ASTERISM_TABLE_COLUMNS,
    AsterismError,
    AsterismLookupFilters,
    AsterismSearchOptions,
    export_asterisms,
    find_asterisms,
    load_asterism_stars,
)
from ao_sky.build._constants import (
    ASTERISMS_DTYPE,
    BUILD_FILENAME,
    INNER_DTYPE,
    STATE_DTYPE,
    WORK_STATUS_DONE,
    WORK_STATUS_PENDING,
)
from ao_sky.build.artifacts import write_outer_artifact
import ao_sky.asterisms.search as search_module
from ao_sky.gaia import (
    GAIA_SCHEMA_COLUMNS,
    GaiaHealpixStore,
    GaiaStoreConfig,
    compute_r_magnitude,
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
        lambda self, pix, force_reload=False, read_only=False: local
        if pix == 0
        else _empty_gaia_table(),
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

    local = _gaia_table(
        [
            (
                101,
                float(local_coord.ra.degree),
                float(local_coord.dec.degree),
                12.0,
                0.0,
                0.0,
            )
        ]
    )
    neighbour = _gaia_table(
        [
            (202, float(bordering_coord.ra.degree), float(bordering_coord.dec.degree), 13.0, 0.0, 0.0),
            (303, float(interior_coord.ra.degree), float(interior_coord.dec.degree), 14.0, 0.0, 0.0),
        ]
    )

    def fake_load(
        self: GaiaHealpixStore,
        pix: int,
        force_reload: bool = False,
        read_only: bool = False,
    ) -> Table:
        if pix == outer_pix:
            return local
        if pix == neighbour_outer_pix:
            return neighbour
        return _empty_gaia_table()

    monkeypatch.setattr(GaiaHealpixStore, "load_healpix", fake_load)
    copy_lengths: list[int] = []
    real_copy = Table.copy

    def tracking_copy(self: Table, *args: object, **kwargs: object) -> Table:
        copy_lengths.append(len(self))
        return real_copy(self, *args, **kwargs)

    monkeypatch.setattr(Table, "copy", tracking_copy)

    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=neighbour_level,
        boundary_rings=1,
        include_locality=True,
    )

    assert stars["source_id"].tolist() == [101, 202]
    assert stars["is_local"].tolist() == [True, False]
    assert stars["source_outer_pix"].tolist() == [outer_pix, neighbour_outer_pix]
    assert len(neighbour) not in copy_lengths


def test_load_asterism_stars_uses_runtime_hpx14_for_neighbour_trim(
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
    local_coord = get_pixel_skycoord(neighbour_level, int(local_subpixels[0]))
    bordering_coord = get_pixel_skycoord(neighbour_level, int(bordering[0]))
    local = _gaia_table(
        [(101, float(local_coord.ra.degree), float(local_coord.dec.degree), 12.0, 0.0, 0.0)]
    )
    neighbour = _gaia_table(
        [
            (202, float(bordering_coord.ra.degree), float(bordering_coord.dec.degree), 13.0, 0.0, 0.0),
            (303, np.nan, float(bordering_coord.dec.degree), 14.0, 0.0, 0.0),
        ]
    )
    local["hpx14"] = np.asarray([int(local_subpixels[0]) * 4 ** (14 - neighbour_level)])
    neighbour["hpx14"] = np.asarray([int(bordering[0]) * 4 ** (14 - neighbour_level), -1])

    def fake_load(
        self: GaiaHealpixStore,
        pix: int,
        force_reload: bool = False,
        read_only: bool = False,
    ) -> Table:
        if pix == outer_pix:
            return local
        if pix == neighbour_outer_pix:
            return neighbour
        return _empty_gaia_table()

    monkeypatch.setattr(GaiaHealpixStore, "load_healpix", fake_load)
    monkeypatch.setattr(
        "ao_sky.asterisms.loader.get_pixel_from_skycoord",
        lambda *args, **kwargs: pytest.fail("hpx14 was not used for neighbour trim"),
    )

    stars = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=neighbour_level,
        boundary_rings=1,
        include_locality=True,
    )

    assert stars["source_id"].tolist() == [101, 202]


def test_load_asterism_stars_defaults_to_two_boundary_rings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer_level = 1
    neighbour_level = 3
    outer_pix = 0
    store = GaiaHealpixStore(GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=outer_level))

    local_subpixels = set(
        int(pixel)
        for pixel in get_subpixels(outer_level, outer_pix, neighbour_level)
    )
    first_ring = set(
        int(neighbour)
        for pixel in local_subpixels
        for neighbour in get_pixel_neighbours(neighbour_level, int(pixel))
        if int(neighbour) not in local_subpixels
    )
    second_ring = set(
        int(neighbour)
        for pixel in first_ring
        for neighbour in get_pixel_neighbours(neighbour_level, int(pixel))
        if int(neighbour) not in local_subpixels and int(neighbour) not in first_ring
    )
    second_ring_subpixel = min(second_ring)
    second_ring_outer_pix = int(
        get_parent_pixel(neighbour_level, second_ring_subpixel, outer_level)
    )

    local_coord = get_pixel_skycoord(neighbour_level, min(local_subpixels))
    second_ring_coord = get_pixel_skycoord(neighbour_level, second_ring_subpixel)
    local = _gaia_table([(101, float(local_coord.ra.degree), float(local_coord.dec.degree), 12.0, 0.0, 0.0)])
    neighbour = _gaia_table(
        [
            (
                202,
                float(second_ring_coord.ra.degree),
                float(second_ring_coord.dec.degree),
                13.0,
                0.0,
                0.0,
            )
        ]
    )

    def fake_load(
        self: GaiaHealpixStore,
        pix: int,
        force_reload: bool = False,
        read_only: bool = False,
    ) -> Table:
        if pix == outer_pix:
            return local
        if pix == second_ring_outer_pix:
            return neighbour
        return _empty_gaia_table()

    monkeypatch.setattr(GaiaHealpixStore, "load_healpix", fake_load)

    one_ring = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=neighbour_level,
        boundary_rings=1,
        include_locality=True,
    )
    two_rings = load_asterism_stars(
        store,
        outer_pix,
        neighbour_level=neighbour_level,
        include_locality=True,
    )

    assert one_ring["source_id"].tolist() == [101]
    assert two_rings["source_id"].tolist() == [101, 202]
    assert two_rings["source_outer_pix"].tolist() == [outer_pix, second_ring_outer_pix]


def test_load_asterism_stars_applies_proper_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GaiaHealpixStore(GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=1))
    local = _gaia_table([(101, 10.0, 0.0, 12.0, 100.0, 50.0)])
    monkeypatch.setattr(
        GaiaHealpixStore,
        "load_healpix",
        lambda self, pix, force_reload=False, read_only=False: local
        if pix == 0
        else _empty_gaia_table(),
    )

    stars = load_asterism_stars(store, 0, epoch=2017.0)

    assert np.isclose(stars["ra"][0], 10.0 + 100.0 / 1000.0 / 3600.0)
    assert np.isclose(stars["dec"][0], 50.0 / 1000.0 / 3600.0)
    assert stars["ref_epoch"][0] == 2017.0


def _search_table(rows: list[tuple[int, float, float, float]]) -> Table:
    table = _gaia_table([(row[0], row[1], row[2], row[3], 0.0, 0.0) for row in rows])
    table["R"] = np.asarray([row[3] for row in rows], dtype=np.float64)
    return table


def _write_lookup_build_root(
    build_path: Path,
    *,
    done_pixels: tuple[int, ...] = (0,),
) -> None:
    build_path.mkdir(parents=True, exist_ok=True)
    state = np.zeros(12, dtype=STATE_DTYPE)
    state["outer_pix"] = np.arange(12, dtype=np.int64)
    state["gaia_loading_status"] = WORK_STATUS_DONE
    state["traversal_status"] = WORK_STATUS_PENDING
    for outer_pix in done_pixels:
        state["traversal_status"][int(outer_pix)] = WORK_STATUS_DONE

    with h5py.File(build_path / BUILD_FILENAME, "w") as handle:
        metadata = handle.create_group("metadata")
        config = metadata.create_group("config")
        config.create_dataset("lineage_name", data="lookup", dtype=h5py.string_dtype("utf-8"))
        config.create_dataset("gaia_release", data="dr3", dtype=h5py.string_dtype("utf-8"))
        config.create_dataset("outer_level", data=0)
        config.create_dataset("inner_level", data=1)
        config.create_dataset("max_data_level", data=1)
        config.create_dataset("survey_extent_overlays_yaml", data="[]", dtype=h5py.string_dtype("utf-8"))
        state_group = handle.create_group("state")
        state_group.create_dataset("outer_pixels", data=state)


def _write_lookup_outer(
    build_path: Path,
    *,
    outer_pix: int,
    local_ids: tuple[int, ...] = (7,),
    source_ids: tuple[tuple[int, int, int], ...] = ((101, 202, -1),),
    asterism_pixs: tuple[int, ...] | None = None,
    member_mags: tuple[tuple[float, float, float], ...] | None = None,
    winner_ids: tuple[int, ...] = (7, 7, -1, -1),
    winner_ee_resolved: tuple[float, ...] = (0.2, 0.4, np.nan, np.nan),
    winner_ee_averaged: tuple[float, ...] = (0.3, 0.5, np.nan, np.nan),
    coverage_resolved: tuple[bool, ...] = (False, True, False, False),
    coverage_averaged: tuple[bool, ...] = (True, True, False, False),
) -> None:
    inner = np.zeros(4, dtype=INNER_DTYPE)
    inner["pix"] = np.arange(4, dtype=np.int64)
    inner["gaia_A0"] = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    inner["best_ee"] = np.asarray([0.4, 0.6, 0.1, 0.2], dtype=np.float64)
    inner["best_sr"] = np.asarray([0.04, 0.06, 0.01, 0.02], dtype=np.float64)
    inner["best_fwhm"] = np.asarray([100.0, 80.0, 200.0, 180.0], dtype=np.float64)
    inner["winner_asterism_id"] = np.asarray(winner_ids, dtype=np.int64)
    inner["winner_ee_resolved"] = np.asarray(winner_ee_resolved, dtype=np.float64)
    inner["winner_ee_averaged"] = np.asarray(winner_ee_averaged, dtype=np.float64)
    inner["coverage_resolved"] = np.asarray(coverage_resolved, dtype=np.bool_)
    inner["coverage_averaged"] = np.asarray(coverage_averaged, dtype=np.bool_)

    asterisms = np.zeros(len(local_ids), dtype=ASTERISMS_DTYPE)
    asterisms["asterism_id"] = np.asarray(local_ids, dtype=np.int64)
    asterisms["ra"] = np.arange(len(local_ids), dtype=np.float64) + 10.0
    asterisms["dec"] = np.arange(len(local_ids), dtype=np.float64) + 20.0
    asterisms["num_stars"] = np.asarray(
        [sum(1 for source_id in row if source_id >= 0) for row in source_ids],
        dtype=np.int64,
    )
    asterisms["pix"] = (
        np.arange(len(local_ids), dtype=np.int64)
        if asterism_pixs is None
        else np.asarray(asterism_pixs, dtype=np.int64)
    )
    for row_index, row_source_ids in enumerate(source_ids):
        for slot, source_id in enumerate(row_source_ids, start=1):
            asterisms[f"star{slot}_source_id"][row_index] = int(source_id)
            asterisms[f"star{slot}_ra"][row_index] = 10.0 + slot
            asterisms[f"star{slot}_dec"][row_index] = 20.0 + slot
            asterisms[f"star{slot}_mag"][row_index] = (
                11.0 + slot
                if member_mags is None
                else float(member_mags[row_index][slot - 1])
            )

    filename = (
        build_path
        / "hpx0-1"
        / get_outer_pixel_bucket_path(0, outer_pix)
        / "outer.h5"
    )
    write_outer_artifact(filename, inner=Table(inner), asterisms=Table(asterisms))


def _write_lookup_moc(path: Path, *, level: int, pixs: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    moc = MOC.from_healpix_cells(
        np.asarray(pixs, dtype=np.uint64),
        level,
        max_depth=level,
    )
    moc.save(str(path), format="fits", overwrite=True)
    return path


def _read_hdf5_asterism_export(path: Path) -> Table:
    arrays = []
    with h5py.File(path, "r") as handle:
        for chunk_name in sorted(handle["chunks"]):
            arrays.append(handle["chunks"][chunk_name]["asterisms"][()])
    if not arrays:
        return Table()
    return Table(np.concatenate(arrays))


def _read_fits_asterism_export(path: Path) -> Table:
    arrays = []
    with fits.open(path) as handle:
        for hdu in handle[1:]:
            arrays.append(np.asarray(hdu.data))
    if not arrays:
        return Table()
    return Table(np.concatenate(arrays))


def test_legacy_enumerate_asterism_candidates_returns_single_star_output() -> None:
    stars = _search_table([(101, 10.0, 0.0, 12.0)])
    options = AsterismSearchOptions(min_stars=1, max_stars=1)

    result = search_module._enumerate_asterism_candidates(stars, options)

    assert result.colnames == list(ASTERISM_TABLE_COLUMNS)
    assert result["asterism_id"].tolist() == [1]
    assert result["num_stars"].tolist() == [1]
    assert result["star1_source_id"].tolist() == [101]
    assert result["star1_mag"].tolist() == [12.0]
    assert result["radius_arcsec"].tolist() == [30.0]
    assert result["separation_arcsec"].tolist() == [60.0]


def test_legacy_enumerate_asterism_candidates_returns_deterministic_two_and_three_star_results() -> None:
    arcsec = 1.0 / 3600.0
    stars = _search_table(
        [
            (101, 10.0, 0.0, 12.0),
            (202, 10.0 + 10.0 * arcsec, 0.0, 12.5),
            (303, 10.0 + 5.0 * arcsec, 8.0 * arcsec, 12.8),
        ]
    )

    pairs = search_module._enumerate_asterism_candidates(stars, AsterismSearchOptions(min_stars=2, max_stars=2))
    triplets = search_module._enumerate_asterism_candidates(stars, AsterismSearchOptions(min_stars=3, max_stars=3))
    triplets_repeat = search_module._enumerate_asterism_candidates(stars, AsterismSearchOptions(min_stars=3, max_stars=3))

    assert len(pairs) == 3
    assert len(triplets) == 1
    assert triplets["asterism_id"].tolist() == [1]
    assert triplets["star1_source_id"].tolist() == [101]
    assert triplets["star2_source_id"].tolist() == [202]
    assert triplets["star3_source_id"].tolist() == [303]
    assert triplets["star1_mag"].tolist() == [12.0]
    assert triplets["area_arcsec2"][0] > 0.0
    assert np.array_equal(triplets.as_array(), triplets_repeat.as_array())


def test_legacy_enumerate_asterism_candidates_handles_multiple_buffer_chunks(
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

    result = search_module._enumerate_asterism_candidates(stars, AsterismSearchOptions(min_stars=1, max_stars=3))

    assert len(result) == 7
    assert result["asterism_id"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_legacy_enumerate_asterism_candidates_uses_derived_r_magnitude_for_triplet_scoring() -> None:
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
    stars["R"] = compute_r_magnitude(stars)

    triplets = search_module._enumerate_asterism_candidates(stars, AsterismSearchOptions(min_stars=3, max_stars=3))

    assert len(triplets) == 1
    assert triplets["star1_mag"].tolist() == [float(stars["R"][0])]
    assert np.isclose(triplets["separation_arcsec"][0], 10.0, atol=0.1)


def test_asterism_search_options_enforce_phase_two_contract() -> None:
    with pytest.raises(AsterismError, match="supports only 1-3 stars"):
        AsterismSearchOptions(max_stars=4)

    with pytest.raises(AsterismError, match="cannot exceed"):
        AsterismSearchOptions(min_separation_arcsec=10.0, max_separation_arcsec=5.0)


def test_find_asterisms_reads_supported_winners_from_done_outer_pixel(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        winner_ids=(7, 7, -1, -1),
    )

    result = find_asterisms(build_path, outer_pixels=0)

    assert len(result) == 1
    assert result.colnames[-7:] == [
        "global_asterism_id",
        "representative_outer_pix",
        "representative_asterism_id",
        "inner_pixel_count",
        "winner_ee_resolved",
        "winner_ee_averaged",
        "gaia_A0",
    ]
    assert int(result["global_asterism_id"][0]) == 1
    assert int(result["representative_outer_pix"][0]) == 0
    assert int(result["representative_asterism_id"][0]) == 7
    assert int(result["inner_pixel_count"][0]) == 2
    assert int(result["star1_source_id"][0]) == 101
    assert np.isclose(float(result["winner_ee_resolved"][0]), 0.3)
    assert np.isclose(float(result["winner_ee_averaged"][0]), 0.4)
    assert np.isclose(float(result["gaia_A0"][0]), 0.15)


def test_find_asterisms_deduplicates_physical_asterisms_across_outer_pixels(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0, 1))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7,),
        source_ids=((101, 202, -1),),
        winner_ids=(7, 7, -1, -1),
    )
    _write_lookup_outer(
        build_path,
        outer_pix=1,
        local_ids=(99,),
        source_ids=((202, 101, -1),),
        winner_ids=(99, -1, -1, -1),
        winner_ee_resolved=(0.8, np.nan, np.nan, np.nan),
        winner_ee_averaged=(0.7, np.nan, np.nan, np.nan),
    )

    result = find_asterisms(build_path, outer_pixels=[0, 1])

    assert len(result) == 1
    assert int(result["inner_pixel_count"][0]) == 3
    assert int(result["representative_outer_pix"][0]) == 0
    assert int(result["representative_asterism_id"][0]) == 7
    assert np.isclose(float(result["winner_ee_resolved"][0]), (0.2 + 0.4 + 0.8) / 3.0)


def test_find_asterisms_rejects_incomplete_outer_pixels(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=())

    with pytest.raises(AsterismError, match="traversal is not done"):
        find_asterisms(build_path, outer_pixels=0)


def test_find_asterisms_rejects_missing_done_artifact(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))

    with pytest.raises(AsterismError, match="outer artifact is missing"):
        find_asterisms(build_path, outer_pixels=0)


def test_find_asterisms_enforces_max_rows(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        winner_ids=(7, 8, -1, -1),
    )

    with pytest.raises(AsterismError, match="exceeding max_rows=1"):
        find_asterisms(build_path, outer_pixels=0, max_rows=1)


def test_find_asterisms_moc_selects_inner_winner_support(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        asterism_pixs=(3, 1),
        winner_ids=(7, 8, -1, -1),
    )
    moc = _write_lookup_moc(tmp_path / "region.fits", level=1, pixs=[0])

    result = find_asterisms(build_path, moc_file=moc)

    assert len(result) == 1
    assert int(result["representative_asterism_id"][0]) == 7
    assert int(result["pix"][0]) == 3
    assert int(result["inner_pixel_count"][0]) == 1
    assert int(result["star1_source_id"][0]) == 101


def test_find_asterisms_moc_aggregates_selected_support_only(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8, 9),
        source_ids=((101, 202, -1), (202, 101, -1), (303, 404, -1)),
        asterism_pixs=(3, 0, 1),
        winner_ids=(7, 8, 9, 7),
        winner_ee_resolved=(0.2, 0.4, 0.8, 1.0),
        winner_ee_averaged=(0.3, 0.5, 0.9, 1.1),
    )
    moc = _write_lookup_moc(tmp_path / "region.fits", level=1, pixs=[0, 1])

    result = find_asterisms(build_path, moc_file=moc)

    assert len(result) == 1
    assert int(result["representative_asterism_id"][0]) == 7
    assert int(result["pix"][0]) == 3
    assert int(result["inner_pixel_count"][0]) == 2
    assert int(result["star1_source_id"][0]) == 101
    assert int(result["star2_source_id"][0]) == 202
    assert float(result["winner_ee_resolved"][0]) == pytest.approx(0.3)
    assert float(result["winner_ee_averaged"][0]) == pytest.approx(0.4)
    assert float(result["gaia_A0"][0]) == pytest.approx(0.15)


def test_find_asterisms_moc_matches_equivalent_outer_pixel_region(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        winner_ids=(7, 8, -1, -1),
    )
    moc = _write_lookup_moc(tmp_path / "outer.fits", level=0, pixs=[0])

    outer_result = find_asterisms(build_path, outer_pixels=0)
    moc_result = find_asterisms(build_path, moc_file=moc)

    assert set(
        tuple(row)
        for row in outer_result["global_asterism_id", "inner_pixel_count"].as_array()
    ) == set(
        tuple(row)
        for row in moc_result["global_asterism_id", "inner_pixel_count"].as_array()
    )


def test_find_asterisms_moc_rejects_incomplete_intersecting_outer_pixels(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=())
    moc = _write_lookup_moc(tmp_path / "outer.fits", level=0, pixs=[0])

    with pytest.raises(AsterismError, match="traversal is not done"):
        find_asterisms(build_path, moc_file=moc)


def test_find_asterisms_filters_by_star_count(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, -1, -1), (202, 303, -1)),
        winner_ids=(7, 8, -1, -1),
    )

    result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(min_num_stars=2),
    )

    assert len(result) == 1
    assert int(result["representative_asterism_id"][0]) == 8
    assert int(result["num_stars"][0]) == 2


def test_find_asterisms_filters_by_member_magnitude(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        member_mags=((12.0, 13.0, -1.0), (12.0, 16.0, -1.0)),
        winner_ids=(7, 8, -1, -1),
    )

    result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(max_member_mag=15.0),
    )

    assert len(result) == 1
    assert int(result["representative_asterism_id"][0]) == 7


def test_find_asterisms_filters_by_lookup_summary_fields(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        winner_ids=(7, 7, 8, -1),
        winner_ee_resolved=(0.2, 0.4, 0.8, np.nan),
        winner_ee_averaged=(0.3, 0.5, 0.9, np.nan),
    )

    count_result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(min_inner_pixel_count=2),
    )
    resolved_result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(min_winner_ee_resolved=0.7),
    )
    averaged_result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(min_winner_ee_averaged=0.8),
    )
    dust_result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(max_gaia_A0=0.2),
    )

    assert count_result["representative_asterism_id"].tolist() == [7]
    assert resolved_result["representative_asterism_id"].tolist() == [8]
    assert averaged_result["representative_asterism_id"].tolist() == [8]
    assert dust_result["representative_asterism_id"].tolist() == [7]


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        (AsterismLookupFilters(min_num_stars=3), [9]),
        (AsterismLookupFilters(max_num_stars=1), [7]),
        (AsterismLookupFilters(min_member_mag=15.0), [9]),
        (AsterismLookupFilters(max_member_mag=14.0), [7, 8]),
        (AsterismLookupFilters(min_inner_pixel_count=2), [8]),
        (AsterismLookupFilters(max_inner_pixel_count=1), [7, 9]),
        (AsterismLookupFilters(min_winner_ee_resolved=0.7), [9]),
        (AsterismLookupFilters(max_winner_ee_resolved=0.3), [7]),
        (AsterismLookupFilters(min_winner_ee_averaged=0.8), [9]),
        (AsterismLookupFilters(max_winner_ee_averaged=0.4), [7]),
        (AsterismLookupFilters(min_gaia_A0=0.3), [9]),
        (AsterismLookupFilters(max_gaia_A0=0.15), [7]),
    ],
)
def test_find_asterisms_filter_options_cover_each_bound(
    tmp_path: Path,
    filters: AsterismLookupFilters,
    expected_ids: list[int],
) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8, 9),
        source_ids=((101, -1, -1), (202, 303, -1), (404, 505, 606)),
        member_mags=((12.0, 99.0, 99.0), (13.0, 14.0, 99.0), (15.0, 16.0, 17.0)),
        winner_ids=(7, 8, 8, 9),
        winner_ee_resolved=(0.2, 0.4, 0.6, 0.8),
        winner_ee_averaged=(0.3, 0.5, 0.7, 0.9),
    )

    result = find_asterisms(build_path, outer_pixels=0, filters=filters)

    assert result["representative_asterism_id"].tolist() == expected_ids


def test_find_asterisms_member_magnitude_filters_ignore_empty_slots(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8, 9),
        source_ids=((101, -1, -1), (202, 303, -1), (404, -1, -1)),
        member_mags=((12.0, 99.0, 99.0), (13.0, 16.0, 99.0), (16.0, 0.0, 0.0)),
        winner_ids=(7, 8, 9, -1),
    )

    bright_result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(max_member_mag=15.0),
    )
    faint_result = find_asterisms(
        build_path,
        outer_pixels=0,
        filters=AsterismLookupFilters(min_member_mag=15.0),
    )

    assert bright_result["representative_asterism_id"].tolist() == [7]
    assert faint_result["representative_asterism_id"].tolist() == [9]


def test_find_asterisms_applies_filters_to_moc_lookup(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, -1, -1), (202, 303, -1)),
        winner_ids=(7, 8, -1, -1),
    )
    moc = _write_lookup_moc(tmp_path / "outer.fits", level=0, pixs=[0])

    result = find_asterisms(
        build_path,
        moc_file=moc,
        filters=AsterismLookupFilters(max_num_stars=1),
    )

    assert len(result) == 1
    assert int(result["representative_asterism_id"][0]) == 7
    assert int(result["num_stars"][0]) == 1


def test_asterism_lookup_filters_validate_bounds() -> None:
    with pytest.raises(AsterismError, match="min_num_stars must be at least 1"):
        AsterismLookupFilters(min_num_stars=0)

    with pytest.raises(AsterismError, match="max_num_stars must be at least 1"):
        AsterismLookupFilters(max_num_stars=0)

    with pytest.raises(AsterismError, match="min_inner_pixel_count must be at least 1"):
        AsterismLookupFilters(min_inner_pixel_count=0)

    with pytest.raises(AsterismError, match="max_inner_pixel_count must be at least 1"):
        AsterismLookupFilters(max_inner_pixel_count=0)

    with pytest.raises(AsterismError, match="min_num_stars cannot exceed"):
        AsterismLookupFilters(min_num_stars=3, max_num_stars=2)

    with pytest.raises(AsterismError, match="min_member_mag cannot exceed"):
        AsterismLookupFilters(min_member_mag=15.0, max_member_mag=12.0)

    with pytest.raises(AsterismError, match="min_inner_pixel_count cannot exceed"):
        AsterismLookupFilters(min_inner_pixel_count=3, max_inner_pixel_count=2)

    with pytest.raises(AsterismError, match="min_winner_ee_resolved cannot exceed"):
        AsterismLookupFilters(min_winner_ee_resolved=0.8, max_winner_ee_resolved=0.7)

    with pytest.raises(AsterismError, match="min_winner_ee_averaged cannot exceed"):
        AsterismLookupFilters(min_winner_ee_averaged=0.8, max_winner_ee_averaged=0.7)

    with pytest.raises(AsterismError, match="min_gaia_A0 cannot exceed"):
        AsterismLookupFilters(min_gaia_A0=0.3, max_gaia_A0=0.2)


def test_export_asterisms_hdf5_exports_done_pixels_and_catalog_chunks(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0, 1, 2))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, -1, -1)),
        winner_ids=(7, 7, 8, -1),
        winner_ee_resolved=(0.2, 0.4, 0.6, np.nan),
        winner_ee_averaged=(0.3, 0.5, 0.7, np.nan),
    )
    _write_lookup_outer(
        build_path,
        outer_pix=1,
        local_ids=(99,),
        source_ids=((202, 101, -1),),
        winner_ids=(99, -1, -1, -1),
        winner_ee_resolved=(0.8, np.nan, np.nan, np.nan),
        winner_ee_averaged=(0.9, np.nan, np.nan, np.nan),
    )
    _write_lookup_outer(
        build_path,
        outer_pix=2,
        local_ids=(5,),
        source_ids=((404, 505, 606),),
        winner_ids=(5, -1, -1, -1),
        winner_ee_resolved=(1.0, np.nan, np.nan, np.nan),
        winner_ee_averaged=(1.1, np.nan, np.nan, np.nan),
    )
    output = tmp_path / "asterisms.h5"

    summary = export_asterisms(build_path, output, chunk_count=2)
    result = _read_hdf5_asterism_export(output)

    assert summary.exported_asterism_count == 3
    assert summary.selected_outer_pixel_count == 3
    assert summary.chunk_count == 2
    assert result.colnames == [
        "asterism_id",
        "outer_pix",
        "ra",
        "dec",
        "num_stars",
        "star1_source_id",
        "star1_ra",
        "star1_dec",
        "star1_mag",
        "star2_source_id",
        "star2_ra",
        "star2_dec",
        "star2_mag",
        "star3_source_id",
        "star3_ra",
        "star3_dec",
        "star3_mag",
        "inner_pixel_count",
        "winner_ee_resolved",
        "winner_ee_averaged",
        "gaia_A0",
    ]
    assert result["asterism_id"].tolist() == [1, 2, 3]
    assert "pix" not in result.colnames
    assert "global_asterism_id" not in result.colnames
    assert int(result["inner_pixel_count"][0]) == 3
    assert float(result["winner_ee_resolved"][0]) == pytest.approx((0.2 + 0.4 + 0.8) / 3.0)
    assert float(result["winner_ee_averaged"][0]) == pytest.approx((0.3 + 0.5 + 0.9) / 3.0)
    assert float(result["gaia_A0"][0]) == pytest.approx((0.1 + 0.2 + 0.1) / 3.0)
    with h5py.File(output, "r") as handle:
        assert list(sorted(handle["chunks"])) == ["chunk_000001", "chunk_000002"]
        assert handle["chunks/chunk_000001"].attrs["start_asterism_id"] == 1
        assert handle["chunks/chunk_000001"].attrs["end_asterism_id"] == 2
        assert handle["chunks/chunk_000002"].attrs["start_asterism_id"] == 3
        assert handle["chunks/chunk_000002"].attrs["end_asterism_id"] == 3


def test_export_asterisms_fits_round_trips_chunks(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, -1, -1), (202, 303, -1)),
        winner_ids=(7, 8, -1, -1),
    )
    output = tmp_path / "asterisms.fits"

    summary = export_asterisms(build_path, output, format="fits", chunk_count=4)
    result = _read_fits_asterism_export(output)

    assert summary.exported_asterism_count == 2
    assert summary.chunk_count == 2
    assert result["asterism_id"].tolist() == [1, 2]
    with fits.open(output) as handle:
        assert handle[0].header["NCHUNKS"] == 2
        assert handle[1].header["STARTID"] == 1
        assert handle[2].header["STARTID"] == 2


def test_export_asterisms_filters_match_find_asterisms(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8, 9),
        source_ids=((101, -1, -1), (202, 303, -1), (404, 505, 606)),
        winner_ids=(7, 8, 8, 9),
        winner_ee_resolved=(0.2, 0.4, 0.6, 0.8),
        winner_ee_averaged=(0.3, 0.5, 0.7, 0.9),
    )
    filters = AsterismLookupFilters(min_inner_pixel_count=2)
    output = tmp_path / "filtered.h5"

    export_asterisms(build_path, output, outer_pixels=0, filters=filters)
    exported = _read_hdf5_asterism_export(output)
    found = find_asterisms(build_path, outer_pixels=0, filters=filters)

    assert exported["star1_source_id"].tolist() == found["star1_source_id"].tolist()
    assert exported["inner_pixel_count"].tolist() == found["inner_pixel_count"].tolist()
    assert exported["asterism_id"].tolist() == [1]


def test_export_asterisms_moc_matches_find_asterisms_support(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7, 8),
        source_ids=((101, 202, -1), (303, 404, -1)),
        winner_ids=(7, 8, -1, -1),
    )
    moc = _write_lookup_moc(tmp_path / "region.fits", level=1, pixs=[0])
    output = tmp_path / "region.h5"

    export_asterisms(build_path, output, moc_file=moc)
    exported = _read_hdf5_asterism_export(output)
    found = find_asterisms(build_path, moc_file=moc)

    assert exported["star1_source_id"].tolist() == found["star1_source_id"].tolist()
    assert exported["inner_pixel_count"].tolist() == found["inner_pixel_count"].tolist()


def test_export_asterisms_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(build_path, outer_pix=0)
    output = tmp_path / "asterisms.h5"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(AsterismError, match="output already exists"):
        export_asterisms(build_path, output)

    summary = export_asterisms(build_path, output, overwrite=True)

    assert summary.exported_asterism_count == 1


def test_export_asterisms_rejects_invalid_format_and_chunk_count(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))

    with pytest.raises(AsterismError, match="format must be one of"):
        export_asterisms(build_path, tmp_path / "asterisms.bad", format="bad")

    with pytest.raises(AsterismError, match="chunk_count must be at least 1"):
        export_asterisms(build_path, tmp_path / "asterisms.h5", chunk_count=0)


def test_export_asterisms_writes_empty_export(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7,),
        source_ids=((101, -1, -1),),
        winner_ids=(-1, -1, -1, -1),
    )
    output = tmp_path / "empty.h5"

    summary = export_asterisms(build_path, output, chunk_count=3)

    assert summary.exported_asterism_count == 0
    assert summary.chunk_count == 0
    with h5py.File(output, "r") as handle:
        assert list(handle["chunks"].keys()) == []
        assert handle["metadata"].attrs["chunk_count"] == 0


def test_export_asterisms_outer_pixels_selects_only_requested_pixels(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0, 1))
    _write_lookup_outer(
        build_path,
        outer_pix=0,
        local_ids=(7,),
        source_ids=((101, -1, -1),),
        winner_ids=(7, -1, -1, -1),
    )
    _write_lookup_outer(
        build_path,
        outer_pix=1,
        local_ids=(8,),
        source_ids=((202, -1, -1),),
        winner_ids=(8, -1, -1, -1),
    )
    output = tmp_path / "outer1.h5"

    summary = export_asterisms(build_path, output, outer_pixels=1)
    exported = _read_hdf5_asterism_export(output)

    assert summary.selected_outer_pixel_count == 1
    assert exported["star1_source_id"].tolist() == [202]
    assert exported["outer_pix"].tolist() == [1]


def test_export_asterisms_rejects_incomplete_selected_outer_pixel(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=())

    with pytest.raises(AsterismError, match="traversal is not done"):
        export_asterisms(build_path, tmp_path / "asterisms.h5", outer_pixels=0)


def test_export_asterisms_rejects_missing_done_artifact(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))

    with pytest.raises(AsterismError, match="outer artifact is missing"):
        export_asterisms(build_path, tmp_path / "asterisms.h5", outer_pixels=0)


def test_export_asterisms_cleans_temporary_files(tmp_path: Path) -> None:
    build_path = tmp_path / "build"
    _write_lookup_build_root(build_path, done_pixels=(0,))
    _write_lookup_outer(build_path, outer_pix=0)
    output = tmp_path / "nested" / "asterisms.h5"

    export_asterisms(build_path, output)

    assert output.is_file()
    assert list(output.parent.glob(".asterisms.h5.*")) == []
