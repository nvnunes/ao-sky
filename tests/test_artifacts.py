from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ao_sky._paths import get_outer_pixel_bucket_path
from ao_sky.artifacts import AoSkyArtifactStore
from ao_sky.build._constants import (
    ARTIFACT_LAYOUT_VERSION_ATTRIBUTE,
    ASTERISMS_DTYPE,
    BUILD_LAYOUT_VERSION,
    INNER_DTYPE,
    MAPS_DTYPE,
)
from ao_sky.build._exceptions import BuildError
from ao_sky.build.artifacts import (
    read_maps_dataset,
    read_outer_dataset,
    write_maps_artifact,
    write_maps_family_dataset,
    write_outer_artifact,
)
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

    map_data = store.map_data("on_axis_coverage", level=1, nan_below=0.25)
    assert map_data.level == 1
    assert map_data.field == "on_axis_coverage"
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


def test_schema_v2_build_reads_as_canonical_without_mutation(
    schema_v2_build: Path,
) -> None:
    runtime_source = schema_v2_build / "build.yaml"
    maps_source = schema_v2_build / "maps-hpx0.h5"
    outer_source = next((schema_v2_build / "hpx0-1").rglob("outer.h5"))
    original = {
        path: path.read_bytes()
        for path in (runtime_source, maps_source, outer_source)
    }
    store = AoSkyArtifactStore(schema_v2_build)

    runtime = store.runtime()
    maps = store.maps_table(0)
    inner = store.inner(0, outer_level=0, inner_level=1)

    assert runtime.models == {"1star": "point-one"}
    assert runtime.legacy_field_averaged_models == {"1star": "mean-one"}
    assert runtime.on_axis_ee_threshold == pytest.approx(0.4)
    assert runtime.field_averaged_ee_threshold == pytest.approx(0.3)
    assert np.allclose(maps["on_axis_winner_ee"], np.linspace(0.4, 0.51, 12))
    assert np.allclose(
        maps["field_averaged_winner_ee"],
        np.linspace(0.3, 0.41, 12),
    )
    assert inner["on_axis_winner_ee"].tolist() == [0.41, 0.42, 0.43, 0.44]
    assert inner["field_averaged_winner_ee"].tolist() == [0.31, 0.32, 0.33, 0.34]
    assert "winner_ee_resolved" not in maps.colnames
    assert "winner_ee_averaged" not in inner.colnames
    assert all(path.read_bytes() == content for path, content in original.items())


def test_schema_v3_artifact_writers_declare_layout_version(tmp_path: Path) -> None:
    outer = tmp_path / "outer.h5"
    maps = tmp_path / "maps.h5"
    write_outer_artifact(
        outer,
        inner=np.zeros(1, dtype=INNER_DTYPE),
        asterisms=np.zeros(0, dtype=ASTERISMS_DTYPE),
    )
    write_maps_artifact(maps, maps=np.zeros(12, dtype=MAPS_DTYPE))

    for filename in (outer, maps):
        with h5py.File(filename, "r") as handle:
            assert int(handle.attrs[ARTIFACT_LAYOUT_VERSION_ATTRIBUTE]) == (
                BUILD_LAYOUT_VERSION
            )


def test_schema_v2_artifacts_reject_direct_mutation(
    schema_v2_build: Path,
) -> None:
    maps = schema_v2_build / "maps-hpx0.h5"
    outer = next((schema_v2_build / "hpx0-1").rglob("outer.h5"))
    original_maps = maps.read_bytes()
    original_outer = outer.read_bytes()
    stale_temp = maps.with_suffix(maps.suffix + ".tmp")
    stale_temp.write_bytes(b"preserve until layout validation")

    with pytest.raises(BuildError, match="read-only"):
        write_maps_artifact(maps, maps=np.zeros(12, dtype=MAPS_DTYPE))
    with pytest.raises(BuildError, match="read-only"):
        write_maps_family_dataset(
            maps,
            dataset_name="replacement",
            data=np.zeros(12),
        )
    with pytest.raises(BuildError, match="read-only"):
        write_outer_artifact(
            outer,
            inner=np.zeros(1, dtype=INNER_DTYPE),
            asterisms=np.zeros(0, dtype=ASTERISMS_DTYPE),
        )

    assert maps.read_bytes() == original_maps
    assert outer.read_bytes() == original_outer
    assert stale_temp.read_bytes() == b"preserve until layout validation"


def test_schema_v2_build_rejects_creation_of_missing_artifacts(
    schema_v2_build: Path,
) -> None:
    missing_maps = schema_v2_build / "maps-hpx1.h5"
    missing_outer = (
        schema_v2_build
        / "hpx0-1"
        / get_outer_pixel_bucket_path(0, 1)
        / "outer.h5"
    )

    with pytest.raises(BuildError, match="read-only"):
        write_maps_artifact(missing_maps, maps=np.zeros(48, dtype=MAPS_DTYPE))
    with pytest.raises(BuildError, match="read-only"):
        write_outer_artifact(
            missing_outer,
            inner=np.zeros(1, dtype=INNER_DTYPE),
            asterisms=np.zeros(0, dtype=ASTERISMS_DTYPE),
        )

    assert not missing_maps.exists()
    assert not missing_outer.exists()


def test_schema_v2_detection_requires_complete_outer_contract(
    schema_v2_build: Path,
    tmp_path: Path,
) -> None:
    source = next((schema_v2_build / "hpx0-1").rglob("outer.h5"))
    malformed = tmp_path / "outer.h5"
    with h5py.File(source, "r") as source_handle, h5py.File(malformed, "w") as target:
        target.create_dataset("inner", data=source_handle["inner"][...])

    with pytest.raises(BuildError, match="inner and asterisms"):
        read_outer_dataset(malformed, "inner")


def test_schema_v2_detection_requires_exact_maps_dtype(
    schema_v2_build: Path,
    tmp_path: Path,
) -> None:
    source = schema_v2_build / "maps-hpx0.h5"
    with h5py.File(source, "r") as handle:
        dtype_description = list(handle["maps"].dtype.descr)

    malformed_dtypes = {
        "missing": np.dtype(
            [field for field in dtype_description if field[0] != "coverage_averaged"]
        ),
        "extra": np.dtype([*dtype_description, ("unexpected", "<f8")]),
        "wrong": np.dtype(
            [
                (name, "<i8" if name == "coverage_resolved" else dtype)
                for name, dtype in dtype_description
            ]
        ),
    }
    for label, dtype in malformed_dtypes.items():
        malformed = tmp_path / f"maps-{label}.h5"
        with h5py.File(malformed, "w") as handle:
            handle.create_dataset("maps", data=np.zeros(12, dtype=dtype))

        with pytest.raises(BuildError, match="has dtype"):
            read_maps_dataset(malformed)


def test_schema_v3_declaration_requires_exact_current_maps_dtype(
    schema_v2_build: Path,
    tmp_path: Path,
) -> None:
    source = schema_v2_build / "maps-hpx0.h5"
    malformed = tmp_path / "maps-layout-3-with-layout-2-dtype.h5"
    with h5py.File(source, "r") as source_handle, h5py.File(malformed, "w") as target:
        target.attrs[ARTIFACT_LAYOUT_VERSION_ATTRIBUTE] = BUILD_LAYOUT_VERSION
        target.create_dataset("maps", data=source_handle["maps"][...])

    with pytest.raises(BuildError, match="Layout-version-3 dataset 'maps' has dtype"):
        read_maps_dataset(malformed)


@pytest.mark.parametrize("layout_version", [3.9, True, "3"])
def test_artifact_layout_version_must_be_an_exact_integer(
    tmp_path: Path,
    layout_version: object,
) -> None:
    filename = tmp_path / "maps.h5"
    with h5py.File(filename, "w") as handle:
        handle.attrs[ARTIFACT_LAYOUT_VERSION_ATTRIBUTE] = layout_version
        handle.create_dataset("maps", data=np.zeros(12, dtype=MAPS_DTYPE))

    with pytest.raises(BuildError, match="Invalid artifact layout_version"):
        read_maps_dataset(filename)


def _write_runtime_config(filename: Path) -> None:
    filename.write_text(
        """
schema_version: 3
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
  models:
    1star: resolved-1
    2star: resolved-2
    3star: resolved-3
  legacy_field_averaged_models:
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
  on_axis_ee_threshold: 0.4
  field_averaged_ee_threshold: 0.3
""".strip()
    )


def _write_maps(filename: Path) -> None:
    data = np.zeros(48, dtype=MAPS_DTYPE)
    data["pix"] = np.arange(48, dtype=np.int64)
    data["on_axis_coverage"] = np.linspace(0.0, 1.0, 48)
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
