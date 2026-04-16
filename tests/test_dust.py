"""Dust-field tests."""

from __future__ import annotations

from gzip import open as gzip_open
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from ao_sky.dust import (
    DustError,
    add_gaia_a0_to_inner,
    fetch_gaia_tge_dataset,
    gaia_tge_map_filename,
    sample_gaia_a0_for_outer_pixel,
)
from ao_sky.dust.gaia_tge import GAIA_TGE_RELATIVE_FILENAME


def _write_gaia_tge_map(
    dust_root: Path,
    rows: list[tuple[int, int, float]],
) -> Path:
    filename = dust_root / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz"
    filename.parent.mkdir(parents=True, exist_ok=True)
    with gzip_open(filename, "wt", encoding="utf-8") as handle:
        handle.write(
            "solution_id,healpix_id,healpix_level,a0,a0_uncertainty,a0_min,a0_max,"
            "num_tracers_used,optimum_hpx_flag,status\n"
        )
        for healpix_id, healpix_level, a0 in rows:
            handle.write(
                f"1,{healpix_id},{healpix_level},{a0},0.1,0.0,1.0,10,\"True\",0\n"
            )
    return filename


def test_gaia_tge_map_filename_requires_existing_map(tmp_path: Path) -> None:
    with pytest.raises(DustError, match="Gaia TGE map not found"):
        gaia_tge_map_filename(tmp_path / "dust")


def test_sample_gaia_a0_for_outer_pixel_repeats_from_max_data_level(tmp_path: Path) -> None:
    dust_root = tmp_path / "dust"
    _write_gaia_tge_map(
        dust_root,
        [(healpix_id, 1, healpix_id + 0.5) for healpix_id in range(48)],
    )

    values = sample_gaia_a0_for_outer_pixel(
        dust_root=dust_root,
        outer_level=0,
        outer_pix=0,
        inner_level=2,
        max_data_level=1,
    )

    assert values.tolist() == [0.5] * 4 + [1.5] * 4 + [2.5] * 4 + [3.5] * 4


def test_sample_gaia_a0_for_outer_pixel_preserves_missing_values_as_nan(tmp_path: Path) -> None:
    dust_root = tmp_path / "dust"
    _write_gaia_tge_map(
        dust_root,
        [
            (0, 1, 0.25),
            (1, 1, 1.25),
        ],
    )

    values = sample_gaia_a0_for_outer_pixel(
        dust_root=dust_root,
        outer_level=0,
        outer_pix=0,
        inner_level=2,
        max_data_level=1,
    )

    assert values[:4].tolist() == [0.25] * 4
    assert values[4:8].tolist() == [1.25] * 4
    assert np.all(np.isnan(values[8:]))


def test_add_gaia_a0_to_inner_inserts_field_after_pix(tmp_path: Path) -> None:
    dust_root = tmp_path / "dust"
    _write_gaia_tge_map(
        dust_root,
        [(healpix_id, 1, healpix_id + 0.5) for healpix_id in range(48)],
    )
    inner = Table(
        [
            np.arange(16, dtype=np.int64),
            np.zeros(16, dtype=np.int64),
            np.zeros(16, dtype=np.int64),
            np.zeros(16, dtype=np.int64),
        ],
        names=("pix", "star_count", "ngs_count", "asterism_count"),
    )

    result = add_gaia_a0_to_inner(
        inner,
        dust_root=dust_root,
        outer_level=0,
        outer_pix=0,
        inner_level=2,
        max_data_level=1,
    )

    assert result.colnames[:5] == [
        "pix",
        "gaia_A0",
        "star_count",
        "ngs_count",
        "asterism_count",
    ]
    assert result["gaia_A0"].tolist() == [0.5] * 4 + [1.5] * 4 + [2.5] * 4 + [3.5] * 4


def test_fetch_gaia_tge_dataset_fetches_and_restores_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig(dict):
        def get(self, key: str, default: object | None = None) -> object | None:
            return super().get(key, default)

        def remove(self, key: str) -> None:
            self.pop(key, None)

    fake_config = FakeConfig({"data_dir": "/previous"})

    class FakeGaiaTGE:
        @staticmethod
        def fetch() -> None:
            filename = Path(str(fake_config["data_dir"])) / GAIA_TGE_RELATIVE_FILENAME
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_text("fetched", encoding="utf-8")

    monkeypatch.setattr(
        "ao_sky.dust.gaia_tge._load_dustmaps_fetch_dependencies",
        lambda: (fake_config, FakeGaiaTGE),
    )

    filename = fetch_gaia_tge_dataset(tmp_path / "dust")

    assert filename == (tmp_path / "dust" / GAIA_TGE_RELATIVE_FILENAME).resolve()
    assert filename.is_file()
    assert fake_config["data_dir"] == "/previous"


def test_fetch_gaia_tge_dataset_is_noop_when_dataset_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dust_root = tmp_path / "dust"
    expected = dust_root / GAIA_TGE_RELATIVE_FILENAME
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("existing", encoding="utf-8")
    called = False

    def _unexpected() -> tuple[object, object]:
        nonlocal called
        called = True
        raise AssertionError("fetch dependencies should not be loaded")

    monkeypatch.setattr("ao_sky.dust.gaia_tge._load_dustmaps_fetch_dependencies", _unexpected)

    assert fetch_gaia_tge_dataset(dust_root) == expected.resolve()
    assert not called


def test_fetch_gaia_tge_dataset_raises_clear_error_without_dustmaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ao_sky.dust.gaia_tge._load_dustmaps_fetch_dependencies",
        lambda: (_ for _ in ()).throw(DustError("dustmaps must be installed to fetch Gaia TGE dust fields")),
    )

    with pytest.raises(DustError, match="dustmaps must be installed"):
        fetch_gaia_tge_dataset(tmp_path / "dust")
