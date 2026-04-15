"""Dust-field tests."""

from __future__ import annotations

from gzip import open as gzip_open
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from ao_sky.dust import DustError, add_gaia_a0_to_inner, gaia_tge_map_filename, sample_gaia_a0_for_outer_pixel


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

