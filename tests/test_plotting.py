from __future__ import annotations

import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

from astropy.table import Table
from matplotlib import pyplot as plt
import numpy as np
import pytest

from ao_sky._paths import get_outer_pixel_bucket_path
from ao_sky.build._constants import ASTERISMS_DTYPE, INNER_DTYPE, MAPS_DTYPE
from ao_sky.build.artifacts import write_maps_artifact, write_outer_artifact
from ao_sky.gaia._constants import HDF5_DATASET_NAME
from ao_sky.plotting import (
    PlottingError,
    configure_matplotlib_cache,
    plot_asterisms,
    plot_build_asterisms,
)
from ao_sky.plotting.fields import FIELD_CONVENTIONS, prepare_field_values
from ao_sky.plotting.healpix import _remap_dense_values_to_mapcoord, get_healpix_from_skycoord, get_pixel_skycoord
from ao_sky.plotting.maps import MapPlotOptions, maps_artifact_path, plot_map_artifact_field, read_map_layer
from ao_sky.spatial import get_pixel_area
from astropy.coordinates import SkyCoord
import astropy.units as u



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


def test_winner_ee_field_conventions_match_best_ee_range() -> None:
    assert FIELD_CONVENTIONS["winner_ee_averaged"].vmin == FIELD_CONVENTIONS["best_ee"].vmin
    assert FIELD_CONVENTIONS["winner_ee_averaged"].vmax == FIELD_CONVENTIONS["best_ee"].vmax
    assert FIELD_CONVENTIONS["winner_ee_resolved"].vmin == FIELD_CONVENTIONS["best_ee"].vmin
    assert FIELD_CONVENTIONS["winner_ee_resolved"].vmax == FIELD_CONVENTIONS["best_ee"].vmax


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

    layer = read_map_layer(artifact, field="winner_ee_averaged")

    assert layer.filename == artifact
    assert layer.level == 0
    assert layer.field == "winner_ee_averaged"
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


def test_read_map_layer_supports_coverage_mean_alias(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=0)

    layer = read_map_layer(artifact, field="coverage_mean")

    assert layer.field == "coverage_mean"
    assert np.allclose(layer.values, np.linspace(0.0, 1.0, 12))


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
    output_path = tmp_path / "plots" / "winner_ee_averaged-hpx0.png"

    plot_map_artifact_field(
        artifact,
        field="winner_ee_averaged",
        output_path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_map_artifact_field_renders_contours(tmp_path: Path) -> None:
    artifact = _write_synthetic_maps(tmp_path, level=1)
    output_path = tmp_path / "plots" / "coverage_mean-dust-contours-hpx1.png"

    plot_map_artifact_field(
        artifact,
        field="coverage_mean",
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
    output_path = tmp_path / "plots" / "coverage_mean-overlays-hpx1.png"

    plot_map_artifact_field(
        artifact,
        field="coverage_mean",
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
    maps["winner_ee_resolved"] = np.linspace(0.1, 0.7, npix)
    maps["winner_ee_averaged"] = np.linspace(0.1, 0.9, npix)
    maps["coverage_resolved"] = np.linspace(0.0, 1.0, npix)
    maps["coverage_averaged"] = np.linspace(0.0, 1.0, npix)

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
    if empty:
        asterisms = Table(np.zeros(0, dtype=ASTERISMS_DTYPE))
    else:
        asterisms = Table(np.zeros(3, dtype=ASTERISMS_DTYPE))
        asterisms["asterism_id"] = [1, 2, 3]
        asterisms["num_stars"] = [1, 2, 3]
        asterisms["pix"] = [0, 0, 0]
        for row, offsets in enumerate(
            (
                ((0.02, 0.00),),
                ((-0.10, -0.04), (-0.04, -0.01)),
                ((0.05, 0.05), (0.12, 0.06), (0.08, 0.12)),
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
