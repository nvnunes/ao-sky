from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ao_sky.build._constants import MAPS_DTYPE
from ao_sky.build.artifacts import write_maps_artifact
from ao_sky.plotting import PlottingError, configure_matplotlib_cache
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
