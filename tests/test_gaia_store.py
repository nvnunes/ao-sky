"""Phase 1 Gaia store tests."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from ao_sky.gaia import (
    GAIA_SUMMARY_DATASET_NAME,
    GAIA_SCHEMA_COLUMNS,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_OPTS,
    HDF5_DATASET_NAME,
    HDF5_SHUFFLE,
    GaiaError,
    GaiaHealpixStore,
    GaiaStoreConfig,
    GaiaSummaryStore,
    fetch_gaia_store,
)
from ao_sky.gaia._query import build_healpix_query


def _make_table(*, source_ids: tuple[int, ...] = (101, 202)) -> Table:
    count = len(source_ids)
    non_single_star = np.zeros(count, dtype=np.bool_)
    if count:
        non_single_star[min(1, count - 1)] = True
    return Table(
        [
            np.asarray(source_ids, dtype=np.int64),
            np.linspace(10.0, 20.0, count),
            np.linspace(-5.0, 5.0, count),
            np.linspace(12.1, 13.2, count),
            np.linspace(12.4, 13.5, count),
            np.linspace(11.8, 12.9, count),
            np.full(count, 2016.0),
            np.linspace(0.5, 1.5, count),
            np.linspace(-1.5, -0.5, count),
            non_single_star,
            np.linspace(1.01, 1.11, count),
        ],
        names=GAIA_SCHEMA_COLUMNS,
    )


def _write_hdf5(filename: Path, table: Table) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as handle:
        handle.create_dataset(
            HDF5_DATASET_NAME,
            data=table.as_array(),
            compression=HDF5_COMPRESSION,
            compression_opts=HDF5_COMPRESSION_OPTS,
            shuffle=HDF5_SHUFFLE,
        )


def test_healpix_path_uses_release_level_and_legacy_hour_override(tmp_path: Path) -> None:
    store = GaiaHealpixStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=6)
    )

    assert store.healpix_filename(0) == (
        tmp_path / "gaia-dr3-hpx6" / "3h" / "+00" / "0" / "gaia.h5"
    )
    assert store.healpix_filename(8960) == (
        tmp_path / "gaia-dr3-hpx6" / "14h" / "+20" / "8960" / "gaia.h5"
    )


def test_build_healpix_query_uses_release_filters_and_level() -> None:
    query = build_healpix_query("dr3", 6, 0)

    assert "FROM gaiadr3.gaia_source" in query
    assert "gaia_healpix_index(6, SOURCE_ID) = 0" in query
    assert "in_qso_candidates = '0'" in query
    assert "in_galaxy_candidates = '0'" in query
    assert "SOURCE_ID AS source_id" in query
    assert "ruwe AS ruwe" in query
    assert " AS R\n" not in query


def test_load_healpix_reads_existing_canonical_file(tmp_path: Path) -> None:
    store = GaiaHealpixStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=6)
    )
    expected = _make_table()
    filename = store.healpix_filename(0)
    _write_hdf5(filename, expected)

    loaded = store.load_healpix(0)

    assert loaded.colnames == list(GAIA_SCHEMA_COLUMNS)
    assert np.array_equal(loaded["source_id"], expected["source_id"])
    assert np.allclose(loaded["ra"], expected["ra"])


def test_load_healpix_materializes_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = GaiaHealpixStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=6)
    )
    raw = Table(
        [
            np.asarray([501], dtype=np.int64),
            np.asarray([33.0]),
            np.asarray([-4.0]),
            np.asarray([14.2]),
            np.asarray([14.8]),
            np.asarray([13.5]),
            np.asarray([2016.0]),
            MaskedColumn([0.0], mask=[True]),
            np.asarray([1.25]),
            np.asarray([1]),
            MaskedColumn([0.0], mask=[True]),
        ],
        names=GAIA_SCHEMA_COLUMNS,
    )

    def fake_query(release: str, healpix_level: int, outer_pix: int) -> Table:
        assert release == "dr3"
        assert healpix_level == 6
        assert outer_pix == 0
        return raw

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fake_query)

    loaded = store.load_healpix(0)

    assert loaded.colnames == list(GAIA_SCHEMA_COLUMNS)
    assert loaded["source_id"][0] == 501
    assert np.isnan(loaded["pmra"][0])
    assert np.isnan(loaded["ruwe"][0])
    assert loaded["non_single_star"][0]
    assert "R" not in loaded.colnames
    assert store.healpix_filename(0).is_file()


def test_force_reload_rewrites_canonical_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = GaiaHealpixStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=6)
    )
    filename = store.healpix_filename(0)
    _write_hdf5(filename, _make_table(source_ids=(1,)))

    reloaded = _make_table(source_ids=(9,))

    def fake_query(_: str, __: int, ___: int) -> Table:
        return reloaded

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fake_query)

    loaded = store.load_healpix(0, force_reload=True)

    assert loaded["source_id"].tolist() == [9]
    persisted = store.load_healpix(0)
    assert persisted["source_id"].tolist() == [9]


def test_materialized_hdf5_uses_expected_dataset_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GaiaHealpixStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=6)
    )

    monkeypatch.setattr(
        "ao_sky.gaia.store.query_healpix_table",
        lambda *_: _make_table(source_ids=(808,)),
    )

    store.load_healpix(0)

    with h5py.File(store.healpix_filename(0), "r") as handle:
        dataset = handle[HDF5_DATASET_NAME]
        assert dataset.compression == HDF5_COMPRESSION
        assert dataset.compression_opts == HDF5_COMPRESSION_OPTS
        assert dataset.shuffle == HDF5_SHUFFLE


def test_summary_filename_uses_release_and_level_root(tmp_path: Path) -> None:
    summary_store = GaiaSummaryStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=6)
    )

    assert summary_store.summary_filename() == (
        tmp_path / "gaia-dr3-hpx6" / "summary.h5"
    )


def test_fetch_gaia_store_writes_dense_summary_and_skips_existing_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    store = GaiaHealpixStore(config)
    _write_hdf5(store.healpix_filename(0), _make_table(source_ids=(1, 2, 3)))
    queried: list[int] = []

    def fake_query(_: str, __: int, outer_pix: int) -> Table:
        queried.append(outer_pix)
        return _make_table(source_ids=(outer_pix + 10,))

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fake_query)

    summary_filename = fetch_gaia_store(config)
    summary = GaiaSummaryStore(config).load_summary()

    assert summary_filename == GaiaSummaryStore(config).summary_filename()
    assert queried == list(range(1, 12))
    assert summary["outer_pix"].tolist() == list(range(12))
    assert summary["star_count"][0] == 3
    assert summary["star_count"][1] == 1
    assert np.all(np.asarray(summary["loaded"], dtype=np.bool_))
    with h5py.File(summary_filename, "r") as handle:
        dataset = handle[GAIA_SUMMARY_DATASET_NAME]
        assert dataset.compression == HDF5_COMPRESSION
        assert dataset.compression_opts == HDF5_COMPRESSION_OPTS
        assert dataset.shuffle == HDF5_SHUFFLE


def test_fetch_gaia_store_force_rewrites_files_one_pixel_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    store = GaiaHealpixStore(config)
    for outer_pix in range(12):
        _write_hdf5(store.healpix_filename(outer_pix), _make_table(source_ids=(outer_pix,)))
    queried: list[int] = []

    def fake_query(_: str, __: int, outer_pix: int) -> Table:
        queried.append(outer_pix)
        return _make_table(source_ids=(outer_pix + 100, outer_pix + 200))

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fake_query)

    fetch_gaia_store(config, force_reload=True)

    assert queried == list(range(12))
    summary = GaiaSummaryStore(config).load_summary()
    assert np.all(np.asarray(summary["star_count"], dtype=np.int64) == 2)
    assert store.load_healpix(0)["source_id"].tolist() == [100, 200]


def test_fetch_gaia_store_resume_skips_rows_already_loaded_in_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    store = GaiaHealpixStore(config)
    summary_store = GaiaSummaryStore(config)
    for outer_pix in range(12):
        _write_hdf5(store.healpix_filename(outer_pix), _make_table(source_ids=(outer_pix,)))

    summary = np.zeros(12, dtype=[("outer_pix", "<i8"), ("star_count", "<i8"), ("loaded", "?")])
    summary["outer_pix"] = np.arange(12, dtype=np.int64)
    summary["star_count"] = 1
    summary["loaded"] = True
    summary_store.write_summary(Table(summary))

    def fail_query(*_args, **_kwargs):
        raise AssertionError("resume should not requery already loaded rows")

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fail_query)

    fetch_gaia_store(config)

    reloaded = summary_store.load_summary()
    assert np.all(np.asarray(reloaded["star_count"], dtype=np.int64) == 1)
    assert np.all(np.asarray(reloaded["loaded"], dtype=np.bool_))


def test_fetch_gaia_store_resets_missing_loaded_row_before_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    store = GaiaHealpixStore(config)
    summary_store = GaiaSummaryStore(config)
    summary = np.zeros(12, dtype=[("outer_pix", "<i8"), ("star_count", "<i8"), ("loaded", "?")])
    summary["outer_pix"] = np.arange(12, dtype=np.int64)
    summary["star_count"] = 99
    summary["loaded"] = True
    summary_store.write_summary(Table(summary))
    for outer_pix in range(1, 12):
        _write_hdf5(store.healpix_filename(outer_pix), _make_table(source_ids=(outer_pix,)))

    queried: list[int] = []

    def fake_query(_: str, __: int, outer_pix: int) -> Table:
        queried.append(outer_pix)
        return _make_table(source_ids=(outer_pix + 100, outer_pix + 200))

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fake_query)

    fetch_gaia_store(config)

    assert queried == [0]
    reloaded = summary_store.load_summary()
    assert reloaded["star_count"][0] == 2
    assert reloaded["loaded"][0]
    assert np.all(np.asarray(reloaded["loaded"][1:], dtype=np.bool_))


def test_fetch_gaia_store_reports_progress_when_output_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    output = io.StringIO()

    monkeypatch.setattr(
        "ao_sky.gaia.store.query_healpix_table",
        lambda *_: _make_table(source_ids=(808,)),
    )

    fetch_gaia_store(config, output=output)

    rendered = output.getvalue()
    assert "Loading Gaia outer pixels for release dr3 at level 0:" in rendered
    assert "1/12" in rendered
    assert "loaded, " in rendered
    assert "done: 12px in " in rendered


def test_fetch_gaia_store_persists_partial_summary_progress_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    queried: list[int] = []

    def fake_query(_: str, __: int, outer_pix: int) -> Table:
        queried.append(outer_pix)
        if outer_pix == 2:
            raise GaiaError("boom")
        return _make_table(source_ids=(outer_pix + 10, outer_pix + 20))

    monkeypatch.setattr("ao_sky.gaia.store.query_healpix_table", fake_query)

    with pytest.raises(GaiaError, match="boom"):
        fetch_gaia_store(config)

    assert queried == [0, 1, 2]
    summary = GaiaSummaryStore(config).load_summary()
    assert summary["outer_pix"].tolist() == list(range(12))
    assert summary["star_count"][0] == 2
    assert summary["star_count"][1] == 2
    assert summary["star_count"][2] == 0
    assert summary["loaded"].tolist()[:3] == [True, True, False]
    assert not np.any(np.asarray(summary["loaded"][2:], dtype=np.bool_))


def test_load_summary_requires_expected_dataset(tmp_path: Path) -> None:
    summary_store = GaiaSummaryStore(
        GaiaStoreConfig(root=tmp_path, release="dr3", healpix_level=0)
    )
    filename = summary_store.summary_filename()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, "w") as handle:
        handle.create_dataset("wrong", data=np.zeros(1, dtype=np.int64))

    with pytest.raises(GaiaError, match=GAIA_SUMMARY_DATASET_NAME):
        summary_store.load_summary()
