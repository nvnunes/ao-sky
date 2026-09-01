from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ao_sky._paths import get_outer_pixel_bucket_path
from ao_sky.artifacts import AoSkyArtifactStore
from ao_sky.build._constants import ASTERISMS_DTYPE, INNER_DTYPE, MAPS_DTYPE
from ao_sky.build.artifacts import write_maps_artifact, write_outer_artifact
from ao_sky.gaia._constants import GAIA_SCHEMA_COLUMNS


def test_artifact_store_reads_runtime_maps_outer_and_gaia(tmp_path: Path) -> None:
    _write_runtime_config(tmp_path / "build.yaml")
    _write_maps(tmp_path / "maps-hpx1.h5")
    _write_outer(tmp_path / "outer-root", outer_pix=0)
    _write_gaia(tmp_path / "gaia-root", outer_pix=0)

    store = AoSkyArtifactStore(
        tmp_path,
        outer_root=tmp_path / "outer-root",
        gaia_root=tmp_path / "gaia-root",
    )

    runtime = store.runtime()
    assert runtime.outer_level == 0
    assert runtime.inner_level == 1
    assert runtime.ao_system.band == "R"

    map_data = store.map_data("coverage_resolved", level=1, nan_below=0.25)
    assert map_data.level == 1
    assert map_data.field == "coverage_resolved"
    assert len(map_data.coords) == len(map_data.values)
    assert map_data.values[0] != map_data.values[0]

    inner = store.inner(0, outer_level=0, inner_level=1)
    assert inner["pix"].tolist() == [0, 1, 2, 3]

    asterisms = store.asterisms(0, outer_level=0, inner_level=1)
    assert len(asterisms) == 1
    assert int(asterisms["asterism_id"][0]) == 7

    stars = store.gaia_stars(0, outer_level=0, epoch=2028.0)
    assert "R" in stars.colnames
    assert len(stars) == 1


def _write_runtime_config(filename: Path) -> None:
    filename.write_text(
        """
schema_version: 2
ao_system:
  band: R
  fov_arcsec: 120.0
  lgs: []
  min_wfs: 1
  max_wfs: 3
  min_mag: 8.0
  max_mag: 18.5
  min_sep_arcsec: 5.0
prediction:
  wavelength_micron: 1.654
  resolved_models:
    1star: resolved-1
    2star: resolved-2
    3star: resolved-3
  averaged_models:
    1star: averaged-1
    2star: averaged-2
    3star: averaged-3
traversal:
  outer_level: 0
  inner_level: 1
gaia:
  epoch: 2028.0
asterism:
  winner_ee_epsilon: 0.01
best:
  seeing_baseline:
    wavelength_micron: 0.5
    sr: 0.0
    ee: 0.02
    fwhm_mas: 650.0
coverage:
  resolved_ee_threshold: 0.4
  averaged_ee_threshold: 0.3
""".strip()
    )


def _write_maps(filename: Path) -> None:
    data = np.zeros(48, dtype=MAPS_DTYPE)
    data["pix"] = np.arange(48, dtype=np.int64)
    data["coverage_resolved"] = np.linspace(0.0, 1.0, 48)
    write_maps_artifact(filename, maps=data)


def _write_outer(root: Path, *, outer_pix: int) -> None:
    outer_file = root / "hpx0-1" / get_outer_pixel_bucket_path(0, outer_pix) / "outer.h5"
    inner = np.zeros(4, dtype=INNER_DTYPE)
    inner["pix"] = np.arange(4, dtype=np.int64)
    inner["best_ee"] = np.linspace(0.1, 0.4, 4)
    asterisms = np.zeros(1, dtype=ASTERISMS_DTYPE)
    asterisms["asterism_id"] = 7
    write_outer_artifact(outer_file, inner=inner, asterisms=asterisms)


def _write_gaia(root: Path, *, outer_pix: int) -> None:
    gaia_file = root / "hpx0" / get_outer_pixel_bucket_path(0, outer_pix) / "gaia.h5"
    gaia_file.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("source_id", "<i8"),
            ("ra", "<f8"),
            ("dec", "<f8"),
            ("G", "<f8"),
            ("BP", "<f8"),
            ("RP", "<f8"),
            ("ref_epoch", "<f8"),
            ("pmra", "<f8"),
            ("pmdec", "<f8"),
            ("non_single_star", "?"),
            ("ruwe", "<f8"),
        ]
    )
    data = np.zeros(1, dtype=dtype)
    data["source_id"] = 123
    data["ra"] = 0.0
    data["dec"] = 45.0
    data["G"] = 12.0
    data["BP"] = 12.5
    data["RP"] = 11.5
    data["ref_epoch"] = 2016.0
    data["ruwe"] = 1.0
    with h5py.File(gaia_file, "w") as handle:
        handle.create_dataset("gaia", data=data[list(GAIA_SCHEMA_COLUMNS)])
