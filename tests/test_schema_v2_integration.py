from __future__ import annotations

import hashlib
import os
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest
import yaml
from matplotlib import pyplot as plt

from ao_sky.artifacts import AoSkyArtifactStore
from ao_sky.asterisms import export_asterisms, find_asterisms
from ao_sky.build import inspect_build, show_build
from ao_sky.build.aggregation import aggregate_maps
from ao_sky.plotting import plot_build_winner_ee, read_build_winner_ee
from ao_sky.plotting.maps import plot_map_artifact_field, read_map_layer
from ao_sky.spatial import get_pixel_skycoord

SCHEMA_V2_BUILD_ENV = "AOSKY_SCHEMA_V2_BUILD"
SCHEMA_V2_OUTER_PIX_ENV = "AOSKY_SCHEMA_V2_OUTER_PIX"


@pytest.mark.skipif(
    not os.environ.get(SCHEMA_V2_BUILD_ENV),
    reason=f"set {SCHEMA_V2_BUILD_ENV} to run the live schema-v2 audit",
)
def test_live_schema_v2_build_read_contract(tmp_path: Path) -> None:
    """Audit the complete read-only surface against an accepted legacy build."""

    build_path = Path(os.environ[SCHEMA_V2_BUILD_ENV]).expanduser().resolve()
    outer_pix = int(os.environ.get(SCHEMA_V2_OUTER_PIX_ENV, "41430"))
    store = AoSkyArtifactStore(build_path)
    outer_path = store.outer_path(
        outer_pix,
        outer_level=6,
        inner_level=14,
    )
    maps_path = store.maps_filename(6)
    tracked = (
        build_path / "build.h5",
        build_path / "build.yaml",
        build_path / "models" / "manifest.json",
        outer_path,
        maps_path,
    )
    before = {filename: _sha256(filename) for filename in tracked}

    inspection = inspect_build(build_path)
    assert inspection.layout_version == 2
    assert inspection.build_status == "completed"
    assert "layout version: 2" in show_build(build_path)

    runtime = store.runtime()
    runtime_raw = yaml.safe_load(
        (build_path / "build.yaml").read_text(encoding="utf-8")
    )
    assert runtime.models == runtime_raw["prediction"]["resolved_models"]
    assert (
        runtime.legacy_field_averaged_models
        == runtime_raw["prediction"]["averaged_models"]
    )
    assert runtime.on_axis_ee_threshold == (
        runtime_raw["coverage"]["resolved_ee_threshold"]
    )
    assert runtime.field_averaged_ee_threshold == (
        runtime_raw["coverage"]["averaged_ee_threshold"]
    )

    maps = store.maps_table(6)
    inner = store.inner(outer_pix, outer_level=6, inner_level=14)
    with h5py.File(maps_path, "r") as handle:
        raw_maps = handle["maps"]
        assert np.array_equal(
            maps["on_axis_winner_ee"],
            raw_maps["winner_ee_resolved"],
        )
        assert np.array_equal(
            maps["field_averaged_winner_ee"],
            raw_maps["winner_ee_averaged"],
        )
        assert np.array_equal(
            maps["on_axis_coverage"],
            raw_maps["coverage_resolved"],
        )
        assert np.array_equal(
            maps["field_averaged_coverage"],
            raw_maps["coverage_averaged"],
        )
    with h5py.File(outer_path, "r") as handle:
        raw_inner = handle["inner"]
        assert np.array_equal(
            inner["on_axis_winner_ee"],
            raw_inner["winner_ee_resolved"],
            equal_nan=True,
        )
        assert np.array_equal(
            inner["field_averaged_winner_ee"],
            raw_inner["winner_ee_averaged"],
            equal_nan=True,
        )
        assert np.array_equal(
            inner["on_axis_coverage"],
            raw_inner["coverage_resolved"],
        )
        assert np.array_equal(
            inner["field_averaged_coverage"],
            raw_inner["coverage_averaged"],
        )

    layer = read_map_layer(
        build_path,
        level=6,
        field="field_averaged_coverage",
    )
    assert np.array_equal(
        layer.values,
        maps["field_averaged_coverage"],
        equal_nan=True,
    )
    figure = plot_map_artifact_field(
        build_path,
        level=6,
        field="on_axis_coverage",
    )
    plt.close(figure)

    center = get_pixel_skycoord(6, outer_pix)
    local_inner = read_build_winner_ee(
        build_path,
        center=center,
        width=0.5 * u.deg,
    )
    assert len(local_inner) > 0
    figure = plot_build_winner_ee(
        build_path,
        center=center,
        width=0.5 * u.deg,
        ee_kind="field_averaged_winner_ee",
        add_colorbar=False,
    )
    plt.close(figure)

    found = find_asterisms(build_path, outer_pixels=outer_pix)
    assert "on_axis_winner_ee" in found.colnames
    assert "field_averaged_winner_ee" in found.colnames
    export_path = tmp_path / "asterisms.h5"
    export_asterisms(build_path, export_path, outer_pixels=outer_pix)
    with h5py.File(export_path, "r") as handle:
        names = {
            name
            for group in handle["chunks"].values()
            for name in group["asterisms"].dtype.names or ()
        }
    assert "on_axis_winner_ee" in names
    assert "field_averaged_winner_ee" in names

    aggregated = aggregate_maps(build_path, outer_pixs=[outer_pix])
    assert sorted(aggregated) == [6, 7, 8, 9]
    assert "on_axis_winner_ee" in (aggregated[9].dtype.names or ())
    assert "field_averaged_coverage" in (aggregated[9].dtype.names or ())
    assert {filename: _sha256(filename) for filename in tracked} == before


def _sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
