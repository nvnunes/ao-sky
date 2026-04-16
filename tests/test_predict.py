"""Native prediction and Phase 9 traversal-contract tests."""

from __future__ import annotations

from gzip import open as gzip_open
from pathlib import Path

import numpy as np
import pytest
import yaml
import astropy.units as u
from astropy.table import Table

from ao_sky.build import init_build as real_init_build
from ao_sky.build._models import BuildDefinition
from ao_sky.build.config import load_build_definition as load_build_definition_yaml
from ao_sky.build.control import load_build_roots
from ao_sky.build.legacy_config import load_native_runtime
from ao_sky.build.traversal import (
    _update_inner_pixel_asterism_performance,
    build_traversal_products,
)
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore
from ao_sky.gaia._constants import GAIA_SCHEMA_COLUMNS
from ao_sky.predict import PredictError
from ao_sky.predict import backend as predict_backend
from ao_sky.predict import service as predict_service
from ao_sky.predict._models import PointPredictionBatch


def _write_legacy_config(path: Path) -> Path:
    path.write_text(
        """
ao_systems:
  - name: GNAO
    band: R
    fov: 120.0
    min_wfs: 2
    max_wfs: 3
    min_mag: 8.0
    max_mag: 18.5
    min_sep: 5.0
    max_sep: 120.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_build_definition(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "ao_system_short_name": "GNAO",
                "config_short_name": "baseline",
                "gaia_release": "dr3",
                "outer_level": 0,
                "inner_level": 1,
                "max_data_level": 1,
                "epoch": 2028.0,
                "min_galactic_latitude": 90.0,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


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


def _write_gaia_summary(
    gaia_root: Path,
    *,
    release: str,
    outer_level: int,
) -> Path:
    num_pixels = 12 * (4 ** outer_level)
    summary = np.zeros(
        num_pixels,
        dtype=[("outer_pix", "<i8"), ("star_count", "<i8"), ("loaded", "?")],
    )
    summary["outer_pix"] = np.arange(num_pixels, dtype=np.int64)
    summary["loaded"] = True
    return GaiaSummaryStore(
        GaiaStoreConfig(root=gaia_root, release=release, healpix_level=outer_level)
    ).write_summary(Table(summary))


def init_build(
    *,
    definition_filename: Path,
    gaia_root: Path | None,
    build_root: Path | None,
    dust_root: Path | None,
    legacy_config_path: Path,
    model_root: Path | None = None,
    aosky_conf: Path | None = None,
) -> Path:
    definition, _ = load_build_definition_yaml(definition_filename)
    if gaia_root is not None:
        _write_gaia_summary(
            Path(gaia_root),
            release=definition.gaia_release,
            outer_level=definition.outer_level,
        )
    return real_init_build(
        definition_filename=definition_filename,
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=dust_root,
        legacy_config_path=legacy_config_path,
        model_root=model_root,
        aosky_conf=aosky_conf,
    )


def _empty_gaia_table() -> Table:
    table = Table()
    table["source_id"] = np.array([], dtype=np.int64)
    for name in ("ra", "dec", "G", "BP", "RP", "ref_epoch", "pmra", "pmdec", "ruwe"):
        table[name] = np.array([], dtype=np.float64)
    table["non_single_star"] = np.array([], dtype=np.bool_)
    return table[list(GAIA_SCHEMA_COLUMNS)]


class _FakeStore:
    def load_healpix(self, outer_pix: int) -> Table:
        return _empty_gaia_table()


def test_load_native_runtime_applies_defaults_and_model_root(tmp_path: Path) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    definition = BuildDefinition(
        ao_system_short_name="GNAO",
        config_short_name="baseline",
        gaia_release="dr3",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
        epoch=2028.0,
        min_galactic_latitude=10.0,
    )

    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy,
        model_root=tmp_path / "models",
    )

    assert runtime.model_root == (tmp_path / "models").resolve()
    assert runtime.prediction_wavelength.to_value() == pytest.approx(1.654)
    assert runtime.seeing_reference_wavelength.to_value() == pytest.approx(0.5)
    assert runtime.coverage_ee_threshold_resolved == pytest.approx(0.25)
    assert runtime.coverage_ee_threshold_mean == pytest.approx(0.25)
    assert runtime.max_star_density == pytest.approx(2.0)
    assert runtime.ao_system.fov_1ngs.to(u.arcsec).value == pytest.approx(120.0)
    assert len(runtime.ao_system.lgs) == 4
    assert runtime.ao_system.point_models == {}
    assert runtime.ao_system.mean_models == {}


def test_init_build_persists_model_root(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    model_root = tmp_path / "models"

    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        model_root=model_root,
        legacy_config_path=legacy,
    )

    roots = load_build_roots(build_path)
    assert roots.model_root == model_root.resolve()


def test_backend_missing_dependency_raises_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predict_backend, "__file__", str(tmp_path / "backend.py"))

    def _missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(predict_backend, "import_module", _missing)
    with pytest.raises(PredictError, match="ao_tools.training"):
        predict_backend._get_training_module()


def test_clear_backend_cache_releases_loaded_backend_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()
    point_model = object()
    mean_model = object()
    predict_service._POINT_MODEL_CACHE["point:test:2star"] = point_model
    predict_service._MEAN_MODEL_CACHE["mean:test:3star"] = mean_model
    cleared: list[object] = []
    monkeypatch.setattr(predict_service.backend, "clear_cache", lambda model: cleared.append(model))

    predict_service.clear_backend_cache()

    assert cleared == [point_model, mean_model]
    assert predict_service._POINT_MODEL_CACHE == {}
    assert predict_service._MEAN_MODEL_CACHE == {}


def test_winner_fields_remain_local_when_neighbour_has_better_best_ee(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    definition = BuildDefinition(
        ao_system_short_name="GNAO",
        config_short_name="baseline",
        gaia_release="dr3",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
        epoch=2028.0,
    )
    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy,
        model_root=tmp_path / "models",
    )
    inner = Table(
        [
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([2], dtype=np.int64),
            np.array([np.nan], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
            np.array([-1], dtype=np.int64),
            np.array([np.nan], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
            np.array([False], dtype=np.bool_),
            np.array([False], dtype=np.bool_),
        ],
        names=(
            "pix",
            "star_count",
            "ngs_count",
            "asterism_count",
            "best_ee",
            "best_sr",
            "best_fwhm",
            "winner_asterism_id",
            "winner_distance_arcsec",
            "winner_ee_resolved",
            "winner_ee_averaged",
            "coverage_resolved",
            "coverage_averaged",
        ),
    )
    context = type("Context", (), {})()
    context.pixel_idxs = np.array([0, 0], dtype=np.int64)
    context.asterism_idxs = np.array([0, 1], dtype=np.int64)
    context.local_asterism_mask = np.array([True, False], dtype=np.bool_)
    context.inner_x = np.array([0.0], dtype=np.float64)
    context.inner_y = np.array([0.0], dtype=np.float64)
    context.asterism_x = np.array([1.0, 2.0], dtype=np.float64)
    context.asterism_y = np.array([0.0, 0.0], dtype=np.float64)
    context.asterisms = Table(
        [
            np.array([7, 8], dtype=np.int64),
            np.array([2, 2], dtype=np.int64),
            np.array([10.0, 10.0], dtype=np.float64),
            np.array([11.0, 11.0], dtype=np.float64),
            np.array([12.0, 12.0], dtype=np.float64),
            np.array([13.0, 13.0], dtype=np.float64),
        ],
        names=(
            "asterism_id",
            "num_stars",
            "star1_mag",
            "star2_mag",
            "star1_ra",
            "star2_ra",
        ),
    )
    monkeypatch.setattr(
        "ao_sky.build.traversal._get_valid_ngs_from_context_pair",
        lambda *args, **kwargs: [
            {"zd": 1.0, "az": 0.0, "mag": 10.0},
            {"zd": 2.0, "az": 90.0, "mag": 11.0},
        ],
    )
    monkeypatch.setattr("ao_sky.build.traversal.get_point_model", lambda runtime, surviving_stars: object())
    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_point_batch",
        lambda runtime, num_stars, model, ngs: PointPredictionBatch(
            sr=np.array([0.4, 0.6], dtype=np.float64),
            ee=np.array([0.7, 0.8], dtype=np.float64),
            fwhm=np.array([0.5, 0.3], dtype=np.float64),
            ee_angle=np.array([0.0, 15.0], dtype=np.float64),
        ),
    )
    monkeypatch.setattr("ao_sky.build.traversal.clear_backend_cache", lambda: None)

    winner_idxs, winner_angles = _update_inner_pixel_asterism_performance(runtime, inner, context)

    assert float(inner["best_ee"][0]) == pytest.approx(0.8)
    assert float(inner["winner_ee_resolved"][0]) == pytest.approx(0.7)
    assert int(inner["winner_asterism_id"][0]) == 7
    assert float(inner["winner_distance_arcsec"][0]) == pytest.approx(1.0)
    assert int(winner_idxs[0]) == 0
    assert float(winner_angles[0]) == pytest.approx(0.0)


def test_build_traversal_products_skip_path_still_injects_dust(tmp_path: Path) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    definition = BuildDefinition(
        ao_system_short_name="GNAO",
        config_short_name="baseline",
        gaia_release="dr3",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
        epoch=2028.0,
        min_galactic_latitude=90.0,
    )
    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy,
        model_root=tmp_path / "models",
    )
    _write_gaia_tge_map(tmp_path / "dust", [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)])

    asterisms, inner = build_traversal_products(
        _FakeStore(),
        runtime,
        0,
        dust_root=tmp_path / "dust",
        max_data_level=1,
    )

    assert len(asterisms) == 0
    assert "gaia_A0" in inner.colnames
    assert inner.colnames[1] == "gaia_A0"
    assert np.all(np.isfinite(np.asarray(inner["gaia_A0"], dtype=np.float64)))
