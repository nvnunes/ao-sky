from __future__ import annotations

import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from matplotlib import pyplot as plt

from ao_sky._paths import get_outer_pixel_bucket_path
from ao_sky.build._constants import ASTERISMS_DTYPE, INNER_DTYPE, MAPS_DTYPE
from ao_sky.build.artifacts import write_maps_artifact, write_outer_artifact
from ao_sky.gaia._constants import HDF5_DATASET_NAME
from ao_sky.plotting import (
    PlottingError,
    configure_matplotlib_cache,
    plot_asterism,
    plot_asterisms,
    plot_build_asterisms,
    plot_build_winner_ee,
    plot_winner_ee,
    read_build_asterisms,
    read_build_winner_ee,
)
from ao_sky.plotting.fields import FIELD_CONVENTIONS, prepare_field_values
from ao_sky.plotting.healpix import (
    _remap_dense_values_to_mapcoord,
    get_healpix_from_skycoord,
    get_pixel_skycoord,
)
from ao_sky.plotting.maps import (
    MapPlotOptions,
    maps_artifact_path,
    plot_map_artifact_field,
    read_map_layer,
)
from ao_sky.spatial import get_pixel_area


def test_field_conventions_cover_all_map_data_fields() -> None:
    fields = set(MAPS_DTYPE.names or ()) - {"pix"}
    assert fields <= set(FIELD_CONVENTIONS)
    assert "stellar_density" in FIELD_CONVENTIONS
    assert "ngs_density" in FIELD_CONVENTIONS
    assert "winner_asterism_count" in FIELD_CONVENTIONS


def test_best_metric_field_conventions_use_validation_ranges() -> None:
    assert FIELD_CONVENTIONS["best_ee"].vmin == 0.1
    assert FIELD_CONVENTIONS["best_ee"].vmax == 0.5
    assert FIELD_CONVENTIONS["best_sr"].vmin == 0.0
    assert FIELD_CONVENTIONS["best_sr"].vmax == 0.4
    assert FIELD_CONVENTIONS["best_fwhm"].vmin == 50.0
    assert FIELD_CONVENTIONS["best_fwhm"].vmax == 350.0
    assert FIELD_CONVENTIONS["best_fwhm"].unit == "FWHM [mas]"


def test_winner_ee_field_conventions_match_best_ee_range() -> None:
    assert FIELD_CONVENTIONS["field_averaged_winner_ee"].vmin == FIELD_CONVENTIONS["best_ee"].vmin
    assert FIELD_CONVENTIONS["field_averaged_winner_ee"].vmax == FIELD_CONVENTIONS["best_ee"].vmax
    assert FIELD_CONVENTIONS["on_axis_winner_ee"].vmin == FIELD_CONVENTIONS["best_ee"].vmin
    assert FIELD_CONVENTIONS["on_axis_winner_ee"].vmax == FIELD_CONVENTIONS["best_ee"].vmax


def test_stellar_density_uses_symlog_norm() -> None:
    assert FIELD_CONVENTIONS["stellar_density"].norm == "symlog"


def test_ngs_density_uses_symlog_norm() -> None:
    assert FIELD_CONVENTIONS["ngs_density"].norm == "symlog"


def test_configure_matplotlib_cache_respects_existing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configured = tmp_path / "mpl"
    monkeypatch.setenv("MPLCONFIGDIR", str(configured))

    result = configure_matplotlib_cache(tmp_path / "fallback")

    assert result == configured
    assert configured.is_dir()
    assert Path(os.environ["MPLCONFIGDIR"]) == configured


def test_read_map_layer_selects_native_field(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)

    layer = read_map_layer(artifact, field="field_averaged_winner_ee")

    assert layer.filename == artifact
    assert layer.level == 0
    assert layer.field == "field_averaged_winner_ee"
    assert np.allclose(layer.values, np.linspace(0.1, 0.9, 12))


def test_read_map_layer_derives_stellar_density(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)

    layer = read_map_layer(artifact, field="stellar_density")

    area_arcmin2 = get_pixel_area(0).to_value("arcmin2")
    expected = np.arange(1, 13, dtype=np.float64) / area_arcmin2
    assert layer.field == "stellar_density"
    assert np.allclose(layer.values, expected)


def test_read_map_layer_derives_ngs_density(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)

    layer = read_map_layer(artifact, field="ngs_density")

    area_arcmin2 = get_pixel_area(0).to_value("arcmin2")
    expected = np.arange(1, 13, dtype=np.float64) / area_arcmin2
    assert layer.field == "ngs_density"
    assert np.allclose(layer.values, expected)


def test_read_map_layer_selects_field_averaged_coverage(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)

    layer = read_map_layer(artifact, field="field_averaged_coverage")

    assert layer.field == "field_averaged_coverage"
    assert np.allclose(layer.values, np.linspace(0.0, 1.0, 12))


def test_schema_v2_map_and_winner_plot_readers_use_canonical_fields(
    schema_v2_build: Path,
) -> None:
    layer = read_map_layer(
        schema_v2_build,
        level=0,
        field="field_averaged_winner_ee",
    )
    center = get_pixel_skycoord(0, 0)
    inner = read_build_winner_ee(
        schema_v2_build,
        center=center,
        width=120.0 * u.deg,
    )

    assert np.allclose(layer.values, np.linspace(0.3, 0.41, 12))
    assert "on_axis_winner_ee" in inner.colnames
    assert "field_averaged_winner_ee" in inner.colnames
    assert "winner_ee_resolved" not in inner.colnames


def test_read_map_layer_raises_for_missing_field(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)

    with pytest.raises(PlottingError, match="Map field 'not_a_field' not found"):
        read_map_layer(artifact, field="not_a_field")


def test_maps_artifact_path_selects_highest_or_requested_level(tmp_path: Path) -> None:
    level0 = _write_synthetic_maps(tmp_path, level=0)
    level1 = _write_synthetic_maps(tmp_path, level=1)

    assert maps_artifact_path(tmp_path) == level1
    assert maps_artifact_path(tmp_path, level=0) == level0


def test_prepare_field_values_masks_log_values_without_mutating_source() -> None:
    convention = FIELD_CONVENTIONS["star_count"]
    values = np.array([0, 1, 10], dtype=np.int64)

    prepared = prepare_field_values(values, convention)

    assert np.isnan(prepared[0])
    assert prepared[1:].tolist() == [1.0, 10.0]
    assert values.tolist() == [0, 1, 10]


def test_galactic_remap_samples_icrs_values_at_galactic_pixel_coordinates() -> None:
    level = 1
    values = np.arange(12 * (4**level), dtype=np.float64)
    skycoords = get_pixel_skycoord(level)
    coords = SkyCoord(
        l=skycoords.ra.degree * u.degree,
        b=skycoords.dec.degree * u.degree,
        frame="galactic",
    ).icrs
    expected_pixs = get_healpix_from_skycoord(
        level,
        SkyCoord(coords.ra.degree, coords.dec.degree, unit=(u.degree, u.degree), frame="icrs"),
    )

    remapped = _remap_dense_values_to_mapcoord(
        values,
        level=level,
        skycoords=skycoords,
        mapcoord="G",
    )

    assert np.array_equal(remapped, values[expected_pixs])


def test_plot_map_artifact_field_renders_png(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)
    output_path = tmp_path / "plots" / "field_averaged_winner_ee-hpx0.png"

    plot_map_artifact_field(
        artifact,
        field="field_averaged_winner_ee",
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_map_artifact_field_adds_build_wavelength_to_ao_metric_title(tmp_path: Path) -> None:
    _write_synthetic_maps(tmp_path, level=0)
    (tmp_path / "build.yaml").write_text(
        "prediction:\n  wavelength_micron: 1.654\n",
        encoding="utf-8",
    )

    fig = plot_map_artifact_field(tmp_path, field="best_ee", level=0)

    try:
        assert fig.axes[0].get_title() == r"Best EE [$\lambda = 1.654\,\mu\mathrm{m}$]"
    finally:
        plt.close(fig)


def test_plot_map_artifact_field_reads_wavelength_from_build_h5_metadata(tmp_path: Path) -> None:
    _write_synthetic_maps(tmp_path, level=0)
    with h5py.File(tmp_path / "build.h5", "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.create_dataset(
            "config_yaml",
            data="prediction:\n  wavelength_micron: 1.234\n",
        )

    fig = plot_map_artifact_field(tmp_path, field="best_sr", level=0)

    try:
        assert fig.axes[0].get_title() == (
            r"Best Strehl Ratio [$\lambda = 1.234\,\mu\mathrm{m}$]"
        )
    finally:
        plt.close(fig)


def test_plot_map_artifact_field_renders_contours(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=1)
    output_path = tmp_path / "plots" / "field-averaged-coverage-dust-contours-hpx1.png"

    plot_map_artifact_field(
        artifact,
        field="field_averaged_coverage",
        contour_field="gaia_A0",
        options=MapPlotOptions(projection="cartesian", contour_levels=(0.5, 1.0, 1.5)),
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_map_artifact_field_renders_surveys_and_points(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=1)
    survey_path = tmp_path / "survey-poly.txt"
    survey_path.write_text(
        "\n".join(
            [
                "0.0 0.0 1",
                "1.0 0.0 1",
                "1.0 1.0 1",
                "0.0 1.0 1",
                "0.0 0.0 1",
            ]
        )
    )
    output_path = tmp_path / "plots" / "field-averaged-coverage-overlays-hpx1.png"

    plot_map_artifact_field(
        artifact,
        field="field_averaged_coverage",
        options=MapPlotOptions(
            projection="cartesian",
            surveys=[[str(survey_path), {"edgecolor": "red"}, "TEST"]],
            points=[
                [
                    np.array([[0.5, 0.5, "P", 0.2, 0.0]], dtype=object),
                    {"marker": "s", "edgecolor": "black", "facecolor": "none"},
                ]
            ],
        ),
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_asterisms_renders_stars_fov_and_connections(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)

    fig = plot_build_asterisms(tmp_path, gaia_root=tmp_path, center=center, width=1.0 * u.deg)

    try:
        ax = fig.axes[0]
        assert ax.collections
        assert ax.patches
        assert ax.lines
    finally:
        plt.close(fig)


def test_plot_asterism_renders_centered_fov_and_feasible_regions() -> None:
    center = SkyCoord(150.0 * u.deg, 2.0 * u.deg, frame="icrs")
    asterism = _single_asterism_table(center)
    stars = _single_asterism_background_stars(center)

    fig = plot_asterism(asterism[0], fov=120.0 * u.arcsec, stars=stars)

    try:
        ax = fig.axes[0]
        assert len(ax.collections) >= 2
        assert ax.patches
        assert ax.lines
    finally:
        plt.close(fig)


def test_plot_asterism_supports_hide_options() -> None:
    center = SkyCoord(150.0 * u.deg, 2.0 * u.deg, frame="icrs")
    asterism = _single_asterism_table(center)

    fig = plot_asterism(
        asterism,
        fov=120.0 * u.arcsec,
        hide_stars=True,
        hide_centered_fov=True,
        hide_connections=True,
        hide_valid_fov_centers=True,
        hide_coverable_region=True,
    )

    try:
        ax = fig.axes[0]
        assert not ax.collections
        assert not ax.patches
        assert not ax.lines
    finally:
        plt.close(fig)


def test_plot_asterism_validates_single_row_and_fov() -> None:
    center = SkyCoord(150.0 * u.deg, 2.0 * u.deg, frame="icrs")
    asterism = _single_asterism_table(center)

    with pytest.raises(PlottingError, match="exactly one row"):
        plot_asterism(vstack([asterism, asterism]), fov=120.0 * u.arcsec)

    with pytest.raises(PlottingError, match="fov must be a positive"):
        plot_asterism(asterism, fov=0.0 * u.arcsec)


def test_plot_asterisms_supports_hide_options(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)

    fig = plot_build_asterisms(
        tmp_path,
        center=center,
        width=1.0 * u.deg,
        hide_stars=True,
        hide_fov=True,
        hide_connections=True,
    )

    try:
        ax = fig.axes[0]
        assert not ax.collections
        assert not ax.patches
        assert not ax.lines
    finally:
        plt.close(fig)


def test_plot_asterisms_handles_empty_artifacts(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path, empty=True)

    fig = plot_build_asterisms(tmp_path, gaia_root=tmp_path, center=center, width=1.0 * u.deg)

    try:
        ax = fig.axes[0]
        assert not ax.patches
        assert not ax.lines
    finally:
        plt.close(fig)


def test_plot_asterisms_can_draw_on_existing_axis(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)
    fig, ax = plt.subplots()

    result = plot_build_asterisms(
        tmp_path,
        gaia_root=tmp_path,
        center=center,
        width=1.0 * u.deg,
        ax=ax,
    )

    try:
        assert result is fig
        assert fig.axes[0] is ax
    finally:
        plt.close(fig)


def test_plot_asterisms_validates_location_inputs(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)

    with pytest.raises(PlottingError, match="center must be an astropy"):
        plot_build_asterisms(
            tmp_path,
            gaia_root=tmp_path,
            center=(0.0, 0.0),
            width=1.0 * u.deg,
        )  # type: ignore[arg-type]

    with pytest.raises(PlottingError, match="width must be a positive"):
        plot_build_asterisms(tmp_path, gaia_root=tmp_path, center=center, width=0.0 * u.deg)

    with pytest.raises(PlottingError, match="gaia_root is required"):
        plot_build_asterisms(tmp_path, center=center, width=1.0 * u.deg)


def test_plot_asterisms_requires_center_outer_artifact(tmp_path: Path) -> None:
    center = get_pixel_skycoord(0, [0])[0]
    _write_synthetic_asterism_config(tmp_path)

    with pytest.raises(PlottingError, match="Center outer-pixel artifact is missing"):
        plot_build_asterisms(tmp_path, gaia_root=tmp_path, center=center, width=1.0 * u.deg)


def test_plot_asterisms_renders_normalized_tables(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)
    asterisms = Table(np.zeros(1, dtype=ASTERISMS_DTYPE))
    asterisms["asterism_id"] = [1]
    asterisms["num_stars"] = [1]
    asterisms["pix"] = [0]
    asterisms["ra"], asterisms["dec"] = _offset_position(center, 0.0, 0.0)
    asterisms["star1_source_id"] = [10]
    asterisms["star1_ra"], asterisms["star1_dec"] = _offset_position(center, 0.0, 0.0)
    asterisms["star1_mag"] = [12.0]
    stars = Table()
    stars["source_id"] = [10]
    stars["ra"] = [asterisms["star1_ra"][0]]
    stars["dec"] = [asterisms["star1_dec"][0]]
    stars["R"] = [12.0]

    fig = plot_asterisms(
        asterisms,
        center=center,
        width=1.0 * u.deg,
        stars=stars,
        fov=120.0 * u.arcsec,
    )

    try:
        ax = fig.axes[0]
        assert ax.collections
        assert ax.patches
    finally:
        plt.close(fig)


def test_plot_asterisms_can_draw_coverable_region_mask(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)
    asterisms = Table(np.zeros(1, dtype=ASTERISMS_DTYPE))
    asterisms["asterism_id"] = [1]
    asterisms["num_stars"] = [2]
    asterisms["pix"] = [0]
    asterisms["ra"], asterisms["dec"] = _offset_position(center, 0.0, 0.0)
    asterisms["star1_source_id"] = [10]
    asterisms["star1_ra"], asterisms["star1_dec"] = _offset_position(center, -0.01, 0.0)
    asterisms["star1_mag"] = [12.0]
    asterisms["star2_source_id"] = [20]
    asterisms["star2_ra"], asterisms["star2_dec"] = _offset_position(center, 0.01, 0.0)
    asterisms["star2_mag"] = [13.0]

    fig = plot_asterisms(
        asterisms,
        center=center,
        width=1.0 * u.deg,
        fov=120.0 * u.arcsec,
        hide_stars=True,
        hide_fov=True,
        hide_connections=True,
        coverable_region_mask=True,
    )

    try:
        ax = fig.axes[0]
        assert ax.patches
        assert not ax.lines
    finally:
        plt.close(fig)


def test_read_build_asterisms_supports_field_margin(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)

    field_asterisms = read_build_asterisms(tmp_path, center=center, width=1.0 * u.deg)
    expanded_asterisms = read_build_asterisms(
        tmp_path,
        center=center,
        width=1.0 * u.deg,
        margin=0.1 * u.deg,
    )

    assert len(field_asterisms) == 3
    assert len(expanded_asterisms) == 4


def test_plot_asterisms_can_use_expanded_coverable_region_table(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)
    field_asterisms = read_build_asterisms(tmp_path, center=center, width=1.0 * u.deg)
    expanded_asterisms = read_build_asterisms(
        tmp_path,
        center=center,
        width=1.0 * u.deg,
        margin=0.1 * u.deg,
    )

    fig = plot_asterisms(
        field_asterisms,
        center=center,
        width=1.0 * u.deg,
        fov=120.0 * u.arcsec,
        hide_stars=True,
        hide_fov=True,
        hide_connections=True,
        coverable_region_mask=True,
        coverable_region_asterisms=expanded_asterisms,
    )

    try:
        ax = fig.axes[0]
        assert ax.patches
    finally:
        plt.close(fig)


def test_plot_winner_ee_renders_smoothed_field(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)
    inner_pixels = read_build_winner_ee(tmp_path, center=center, width=120.0 * u.deg)

    fig = plot_winner_ee(inner_pixels, center=center, width=120.0 * u.deg, add_colorbar=False)

    try:
        ax = fig.axes[0]
        assert ax.images
    finally:
        plt.close(fig)


def test_plot_build_winner_ee_renders_smoothed_field(tmp_path: Path) -> None:
    center = _write_synthetic_asterism_build(tmp_path)

    fig = plot_build_winner_ee(
        tmp_path,
        center=center,
        width=120.0 * u.deg,
        ee_kind="field_averaged_winner_ee",
    )

    try:
        ax = fig.axes[0]
        assert ax.images
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_plot_winner_ee_validates_kind_and_columns() -> None:
    center = SkyCoord(150.0 * u.deg, 2.0 * u.deg, frame="icrs")
    inner_pixels = Table()
    inner_pixels["ra"] = [150.0]
    inner_pixels["dec"] = [2.0]
    inner_pixels["on_axis_winner_ee"] = [0.2]

    with pytest.raises(PlottingError, match="ee_kind"):
        plot_winner_ee(inner_pixels, center=center, width=1.0 * u.deg, ee_kind="bad")

    with pytest.raises(PlottingError, match="missing required columns"):
        plot_winner_ee(
            inner_pixels,
            center=center,
            width=1.0 * u.deg,
            ee_kind="field_averaged_winner_ee",
        )


def _single_asterism_table(center: SkyCoord) -> Table:
    asterism = Table(np.zeros(1, dtype=ASTERISMS_DTYPE))
    asterism["asterism_id"] = [1]
    asterism["num_stars"] = [2]
    asterism["pix"] = [0]
    asterism["star1_source_id"] = [10]
    asterism["star2_source_id"] = [20]
    asterism["star1_ra"], asterism["star1_dec"] = _offset_position(
        center,
        -20.0 / 3600.0,
        0.0,
    )
    asterism["star2_ra"], asterism["star2_dec"] = _offset_position(
        center,
        20.0 / 3600.0,
        0.0,
    )
    asterism["star1_mag"] = [12.0]
    asterism["star2_mag"] = [13.0]
    asterism["ra"], asterism["dec"] = _offset_position(center, 0.0, 0.0)
    return asterism


def _single_asterism_background_stars(center: SkyCoord) -> Table:
    stars = Table()
    stars["source_id"] = [1, 2, 3]
    positions = [
        _offset_position(center, -0.01, 0.0),
        _offset_position(center, 0.01, 0.0),
        _offset_position(center, 0.0, 0.012),
    ]
    stars["ra"] = [position[0] for position in positions]
    stars["dec"] = [position[1] for position in positions]
    stars["R"] = [12.0, 13.0, 14.0]
    return stars


def _write_synthetic_maps(root: Path, *, level: int) -> Path:
    npix = 12 * (4**level)
    maps = np.zeros(npix, dtype=MAPS_DTYPE)
    maps["pix"] = np.arange(npix, dtype=np.int64)
    maps["gaia_A0"] = np.linspace(0.0, 2.0, npix)
    maps["star_count"] = np.arange(1, npix + 1, dtype=np.int64)
    maps["ngs_count"] = np.arange(1, npix + 1, dtype=np.int64)
    maps["best_sr"] = np.linspace(0.05, 0.5, npix)
    maps["best_ee"] = np.linspace(0.1, 0.8, npix)
    maps["best_fwhm"] = np.linspace(0.5, 0.05, npix)
    maps["on_axis_winner_ee"] = np.linspace(0.1, 0.7, npix)
    maps["field_averaged_winner_ee"] = np.linspace(0.1, 0.9, npix)
    maps["on_axis_coverage"] = np.linspace(0.0, 1.0, npix)
    maps["field_averaged_coverage"] = np.linspace(0.0, 1.0, npix)

    filename = root / f"maps-hpx{level}.h5"
    write_maps_artifact(filename, maps=maps)
    return filename


def _write_synthetic_asterism_build(
    root: Path,
    *,
    empty: bool = False,
) -> SkyCoord:
    center = get_pixel_skycoord(0, [0])[0]
    _write_synthetic_asterism_config(root)
    _write_synthetic_asterism_outer(root, outer_pix=0, center=center, empty=empty)
    _write_synthetic_asterism_gaia(root, outer_pix=0, center=center, empty=empty)
    return center


def _write_synthetic_asterism_config(root: Path) -> None:
    (root / "build.yaml").write_text(
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
traversal:
  outer_level: 0
  inner_level: 1
gaia:
  release: dr3
  epoch: 2028.0
asterism:
  winner_ee_epsilon: 0.01
""".strip(),
        encoding="utf-8",
    )


def _write_synthetic_asterism_outer(
    root: Path,
    *,
    outer_pix: int,
    center: SkyCoord,
    empty: bool,
) -> None:
    outer_file = root / "hpx0-1" / get_outer_pixel_bucket_path(0, outer_pix) / "outer.h5"
    inner = Table(np.zeros(4, dtype=INNER_DTYPE))
    inner["pix"] = np.arange(4, dtype=np.int64)
    inner["winner_asterism_id"] = [1, 1, 2, 3]
    inner["on_axis_winner_ee"] = [0.22, 0.30, 0.38, 0.44]
    inner["field_averaged_winner_ee"] = [0.20, 0.28, 0.34, 0.40]
    if empty:
        asterisms = Table(np.zeros(0, dtype=ASTERISMS_DTYPE))
    else:
        asterisms = Table(np.zeros(4, dtype=ASTERISMS_DTYPE))
        asterisms["asterism_id"] = [1, 2, 3, 4]
        asterisms["num_stars"] = [1, 2, 3, 1]
        asterisms["pix"] = [0, 0, 0, 0]
        for row, offsets in enumerate(
            (
                ((0.02, 0.00),),
                ((-0.10, -0.04), (-0.04, -0.01)),
                ((0.05, 0.05), (0.12, 0.06), (0.08, 0.12)),
                ((0.56, 0.00),),
            ),
            start=0,
        ):
            star_points = list(offsets)
            center_x = float(np.mean([point[0] for point in star_points]))
            center_y = float(np.mean([point[1] for point in star_points]))
            ra, dec = _offset_position(center, center_x, center_y)
            asterisms["ra"][row] = ra
            asterisms["dec"][row] = dec
            for index, (x_offset, y_offset) in enumerate(star_points, start=1):
                ra, dec = _offset_position(center, x_offset, y_offset)
                asterisms[f"star{index}_source_id"][row] = 100 * (row + 1) + index
                asterisms[f"star{index}_ra"][row] = ra
                asterisms[f"star{index}_dec"][row] = dec
                asterisms[f"star{index}_mag"][row] = 12.0 + index
    write_outer_artifact(outer_file, inner=inner, asterisms=asterisms)


def _write_synthetic_asterism_gaia(
    root: Path,
    *,
    outer_pix: int,
    center: SkyCoord,
    empty: bool,
) -> None:
    gaia_file = root / "gaia-dr3-hpx0" / get_outer_pixel_bucket_path(0, outer_pix) / "gaia.h5"
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
    data = np.zeros(0 if empty else 4, dtype=dtype)
    for index in range(len(data)):
        ra, dec = _offset_position(center, -0.15 + 0.1 * index, -0.1 + 0.05 * index)
        data["source_id"][index] = 10 + index
        data["ra"][index] = ra
        data["dec"][index] = dec
        data["G"][index] = 12.0 + index
        data["BP"][index] = 12.5 + index
        data["RP"][index] = 11.5 + index
        data["ref_epoch"][index] = 2028.0
        data["ruwe"][index] = 1.0
    with h5py.File(gaia_file, "w") as handle:
        handle.create_dataset(HDF5_DATASET_NAME, data=data)


def _offset_position(center: SkyCoord, x_deg: float, y_deg: float) -> tuple[float, float]:
    ra = center.ra.deg + x_deg / np.cos(center.dec.to_value(u.rad))
    dec = center.dec.deg + y_deg
    return ra, dec
