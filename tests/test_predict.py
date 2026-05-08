"""Native prediction and Phase 9 traversal-contract tests."""

from __future__ import annotations

from dataclasses import replace
from gzip import open as gzip_open
from pathlib import Path

import numpy as np
import pytest
import yaml
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ao_sky.build import init_build as real_init_build
from ao_sky.build._models import BuildDefinition, TraversalExecutionConfig
from ao_sky.build._exceptions import BuildError
from ao_sky.build.config import load_build_definition as load_build_definition_yaml
from ao_sky.build.control import load_build_roots
from ao_sky.build.runtime_gaia import RUNTIME_HPX_COLUMN, RuntimeGaiaHealpixStore
from ao_sky.build.runtime_config import (
    load_runtime_config,
    runtime_config_filename,
    runtime_to_config,
    write_runtime_config,
)
from ao_sky.build.traversal import (
    DEFAULT_BACKEND_BUCKETS,
    TraversalGeometry,
    TraversalStructureProfile,
    _backend_buffer_row_count,
    _backend_row_count,
    _build_candidate_graph_from_ngs,
    _build_candidate_set,
    _build_retained_asterism_table,
    _build_regional_candidate_graph,
    _candidate_enclosing_fov_center,
    _fill_regularized_winner_fields,
    _MulticoverSelection,
    _nearest_feasible_pointing,
    _nearest_feasible_pointing_generic,
    _StarForCoverage,
    _local_neighbor_index_matrix,
    _max_regional_combination_work,
    _regularize_winner_labels,
    _regional_selector_max_depth,
    _select_ngs_for_multicover,
    _stream_recovered_candidate_predictions,
    _stream_batch_size,
    _update_regularized_winner_averaged_ee,
    build_base_inner_table,
    prepare_search_inputs,
    build_traversal_products,
)
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore
from ao_sky.gaia._constants import GAIA_SCHEMA_COLUMNS
from ao_sky.predict import PredictError
from ao_sky.predict import backend as predict_backend
from ao_sky.predict import service as predict_service
from ao_sky.predict.service import build_model_x_from_ngs_arrays
from ao_sky.predict._models import AOSystemRuntime, PointPredictionBatch, PredictRuntime
from ao_sky.spatial import get_parent_pixel, get_pixel_from_skycoord, get_pixel_skycoord


def _write_legacy_config(
    path: Path,
) -> Path:
    text = """
ao_systems:
  - name: GNAO
    band: R
    fov: 120.0
    min_wfs: 2
    max_wfs: 3
    min_mag: 8.0
    max_mag: 18.5
    min_sep: 5.0
    point_models:
      2star: point_two
      3star: point_three
    models:
      2star: mean_two
      3star: mean_three
coverage_ee_threshold_resolved: 0.4
coverage_ee_threshold_mean: 0.3
""".strip()
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _write_build_definition(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "build": {},
                "ao_system": {
                    "band": "R",
                    "fov_arcsec": 120.0,
                    "lgs": [],
                    "min_wfs": 1,
                    "max_wfs": 3,
                    "min_mag": 8.0,
                    "max_mag": 18.5,
                    "min_sep_arcsec": 5.0,
                },
                "prediction": {
                    "wavelength_micron": 1.654,
                    "resolved_models": {
                        "1star": "point_one",
                        "2star": "point_two",
                        "3star": "point_three",
                    },
                    "averaged_models": {
                        "1star": "mean_one",
                        "2star": "mean_two",
                        "3star": "mean_three",
                    },
                },
                "traversal": {
                    "outer_level": 0,
                    "inner_level": 1,
                },
                "gaia": {
                    "release": "dr3",
                    "epoch": 2028.0,
                    "max_bright_star_mag": 8.0,
                    "max_bright_star_exclusion_arcsec": 240.0,
                },
                "maps": {"max_level": 1},
                "asterism": {
                    "winner_ee_epsilon": 0.01,
                },
                "best": {
                    "seeing_baseline": {
                        "wavelength_micron": 0.5,
                        "sr": 0.0,
                        "ee": 0.02,
                        "fwhm_mas": 650.0,
                    },
                },
                "coverage": {
                    "resolved_ee_threshold": 0.4,
                    "averaged_ee_threshold": 0.3,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def load_native_runtime(
    definition: BuildDefinition,
    *,
    legacy_config_path: Path,
    model_root: Path,
) -> PredictRuntime:
    with Path(legacy_config_path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    system = raw["ao_systems"][0]
    ao_system = AOSystemRuntime(
        band=str(system["band"]),
        fov=float(system["fov"]) * u.arcsec,
        lgs=(
            {"zd": 30.0, "az": 45.0},
            {"zd": 30.0, "az": 135.0},
            {"zd": 30.0, "az": 225.0},
            {"zd": 30.0, "az": 315.0},
        ),
        min_wfs=int(system["min_wfs"]),
        max_wfs=int(system["max_wfs"]),
        min_mag=float(system["min_mag"]),
        max_mag=float(system["max_mag"]),
        min_sep=float(system["min_sep"]) * u.arcsec,
    )
    return PredictRuntime(
        ao_system=ao_system,
        outer_level=definition.outer_level,
        inner_level=definition.inner_level,
        epoch=float(raw.get("asterism_epoch", 2028.0)),
        max_bright_star_mag=(
            None
            if raw.get("asterisms_max_bright_star_mag") is None
            else float(raw["asterisms_max_bright_star_mag"])
        ),
        max_bright_star_exclusion=2.0 * ao_system.fov,
        winner_ee_epsilon=0.01,
        winner_top_k=3,
        prediction_wavelength=1.654 * u.micron,
        resolved_models={
            str(key): str(value)
            for key, value in (system.get("point_models") or {}).items()
        },
        averaged_models={
            str(key): str(value) for key, value in (system.get("models") or {}).items()
        },
        seeing_reference_wavelength=0.5 * u.micron,
        seeing_reference_sr=0.0,
        seeing_reference_ee=0.02,
        seeing_reference_fwhm=650.0,
        coverage_ee_threshold_resolved=float(
            raw.get("coverage_ee_threshold_resolved", 0.25)
        ),
        coverage_ee_threshold_averaged=float(raw.get("coverage_ee_threshold_mean", 0.25)),
        model_root=Path(model_root).resolve(),
    )


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


def _write_required_model_files(model_root: Path, model_name: str) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    pt_file = model_root / f"{model_name}.pt"
    metadata_file = model_root / f"{model_name}_metadata.pkl"
    if not pt_file.exists():
        pt_file.write_bytes(model_name.encode("utf-8") + b":pt")
    if not metadata_file.exists():
        metadata_file.write_bytes(model_name.encode("utf-8") + b":metadata")


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
    aosky_yaml: Path | None = None,
) -> Path:
    definition, _ = load_build_definition_yaml(definition_filename)
    effective_model_root = model_root or Path(legacy_config_path).parent / "models"
    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy_config_path,
        model_root=effective_model_root,
    )
    for model_name in set(runtime.resolved_models.values()) | set(
        runtime.averaged_models.values()
    ):
        _write_required_model_files(effective_model_root, model_name)
    build_config = yaml.safe_load(Path(definition_filename).read_text(encoding="utf-8"))
    payload = runtime_to_config(runtime)
    payload["build"] = {**payload.get("build", {}), **build_config.get("build", {})}
    payload["gaia"]["release"] = build_config["gaia"]["release"]
    payload["maps"] = build_config["maps"]
    Path(definition_filename).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    if gaia_root is not None:
        _write_gaia_summary(
            Path(gaia_root),
            release=definition.gaia_release,
            outer_level=definition.outer_level,
        )
        _write_gaia_tge_map(
            Path(gaia_root),
            [
                (healpix_id, definition.max_data_level, healpix_id + 0.5)
                for healpix_id in range(12 * (4 ** definition.max_data_level))
            ],
        )
    return real_init_build(
        config_filename=definition_filename,
        gaia_root=gaia_root,
        build_root=build_root,
        model_root=effective_model_root,
        aosky_yaml=aosky_yaml,
    )


def _empty_gaia_table() -> Table:
    table = Table()
    table["source_id"] = np.array([], dtype=np.int64)
    for name in ("ra", "dec", "G", "BP", "RP", "ref_epoch", "pmra", "pmdec", "ruwe"):
        table[name] = np.array([], dtype=np.float64)
    table["non_single_star"] = np.array([], dtype=np.bool_)
    return table[list(GAIA_SCHEMA_COLUMNS)]


def _gaia_table(
    rows: list[tuple[int, float, float, float, float, float]],
) -> Table:
    return Table(
        [
            np.asarray([row[0] for row in rows], dtype=np.int64),
            np.asarray([row[1] for row in rows], dtype=np.float64),
            np.asarray([row[2] for row in rows], dtype=np.float64),
            np.asarray([row[3] for row in rows], dtype=np.float64),
            np.asarray([row[3] + 0.2 for row in rows], dtype=np.float64),
            np.asarray([row[3] - 0.2 for row in rows], dtype=np.float64),
            np.asarray([2016.0 for _ in rows], dtype=np.float64),
            np.asarray([row[4] for row in rows], dtype=np.float64),
            np.asarray([row[5] for row in rows], dtype=np.float64),
            np.zeros(len(rows), dtype=np.bool_),
            np.ones(len(rows), dtype=np.float64),
        ],
        names=GAIA_SCHEMA_COLUMNS,
    )


class _FakeStore:
    def load_healpix(
        self,
        outer_pix: int,
        *,
        force_reload: bool = False,
        read_only: bool = False,
    ) -> Table:
        table = _empty_gaia_table()
        table["R"] = np.array([], dtype=np.float64)
        table[RUNTIME_HPX_COLUMN] = np.array([], dtype=np.int64)
        return table


class _RawGaiaStore:
    def __init__(self, root: Path, tables: dict[int, Table]) -> None:
        self.config = GaiaStoreConfig(root=root, release="dr3", healpix_level=0)
        self.tables = tables
        self.loads: list[int] = []

    def healpix_filename(self, outer_pix: int) -> Path:
        return Path(f"{outer_pix}/gaia.h5")

    def load_healpix(
        self,
        outer_pix: int,
        *,
        force_reload: bool = False,
        read_only: bool = False,
    ) -> Table:
        self.loads.append(int(outer_pix))
        table = self.tables.get(int(outer_pix), _empty_gaia_table()).copy(copy_data=True)
        if read_only:
            for name in table.colnames:
                table[name].flags.writeable = False
        return table


def test_load_native_runtime_applies_defaults_and_model_root(tmp_path: Path) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    definition = BuildDefinition(
        lineage_name="baseline",
        gaia_release="dr3",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
    )

    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy,
        model_root=tmp_path / "models",
    )

    assert runtime.model_root == (tmp_path / "models").resolve()
    assert runtime.prediction_wavelength.to_value() == pytest.approx(1.654)
    assert runtime.seeing_reference_wavelength.to_value() == pytest.approx(0.5)
    assert runtime.coverage_ee_threshold_resolved == pytest.approx(0.4)
    assert runtime.coverage_ee_threshold_averaged == pytest.approx(0.3)
    assert len(runtime.ao_system.lgs) == 4
    assert runtime.resolved_models == {"2star": "point_two", "3star": "point_three"}
    assert runtime.averaged_models == {"2star": "mean_two", "3star": "mean_three"}


def test_runtime_config_round_trips_native_policy(tmp_path: Path) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    definition = BuildDefinition(
        lineage_name="baseline",
        gaia_release="dr3",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
    )
    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy,
        model_root=tmp_path / "models-a",
    )

    filename = write_runtime_config(tmp_path / "build", runtime)
    payload = yaml.safe_load(filename.read_text(encoding="utf-8"))
    loaded = load_runtime_config(filename, model_root=tmp_path / "models-b")

    assert filename == runtime_config_filename(tmp_path / "build")
    assert "source" not in payload
    assert "build" not in payload
    assert loaded.model_root == (tmp_path / "models-b").resolve()
    assert loaded.ao_system.fov.to_value(u.arcsec) == pytest.approx(120.0)
    assert loaded.resolved_models == runtime.resolved_models
    assert loaded.averaged_models == runtime.averaged_models
    assert loaded.outer_level == runtime.outer_level
    assert loaded.inner_level == runtime.inner_level
    assert loaded.max_bright_star_exclusion.to_value(u.arcsec) == pytest.approx(
        runtime.max_bright_star_exclusion.to_value(u.arcsec)
    )
    assert loaded.winner_ee_epsilon == pytest.approx(runtime.winner_ee_epsilon)
    assert loaded.winner_top_k == runtime.winner_top_k
    assert loaded.prediction_wavelength.to_value(u.micron) == pytest.approx(1.654)


def test_load_runtime_config_rejects_missing_nested_values(tmp_path: Path) -> None:
    filename = write_runtime_config(
        tmp_path / "build",
        _make_predict_runtime(model_root=tmp_path / "models"),
    )
    payload = yaml.safe_load(filename.read_text(encoding="utf-8"))
    del payload["ao_system"]["band"]
    filename.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(BuildError, match="ao_system.band"):
        load_runtime_config(filename, model_root=tmp_path / "models")


def test_load_runtime_config_rejects_invalid_runtime_values(tmp_path: Path) -> None:
    filename = write_runtime_config(
        tmp_path / "build",
        _make_predict_runtime(model_root=tmp_path / "models"),
    )
    payload = yaml.safe_load(filename.read_text(encoding="utf-8"))
    payload["ao_system"]["fov_arcsec"] = -1.0
    filename.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(BuildError, match="fov_arcsec must be positive"):
        load_runtime_config(filename, model_root=tmp_path / "models")


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("asterism", "max_overlap", 0.66, "max_overlap is legacy-only"),
        ("asterism", "max_candidate_asterisms", 64, "max_candidate_asterisms was removed"),
        ("asterism", "winner_ee_epsilon", 1.0, "winner_ee_epsilon must be at least 0"),
        ("asterism", "winner_top_k", 0, "winner_top_k must be between"),
        (
            "gaia",
            "max_bright_star_exclusion_arcsec",
            0.0,
            "max_bright_star_exclusion_arcsec must be positive",
        ),
    ],
)
def test_load_runtime_config_rejects_invalid_phase14_values(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    match: str,
) -> None:
    filename = write_runtime_config(
        tmp_path / "build",
        _make_predict_runtime(model_root=tmp_path / "models"),
    )
    payload = yaml.safe_load(filename.read_text(encoding="utf-8"))
    payload[section][key] = value
    filename.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(BuildError, match=match):
        load_runtime_config(filename, model_root=tmp_path / "models")


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
    assert roots.model_root == (build_path / "models").resolve()
    assert (build_path / "models" / "manifest.json").is_file()


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
    assert predict_service._POINT_MODEL_CACHE == {"point:test:2star": point_model}
    assert predict_service._MEAN_MODEL_CACHE == {"mean:test:3star": mean_model}


def _make_predict_runtime(
    *,
    model_root: Path,
    point_model: str = "point-a.pt",
    mean_model: str = "mean-a.pt",
    fov: u.Quantity = 120.0 * u.arcsec,
    min_wfs: int = 2,
    max_wfs: int = 2,
) -> PredictRuntime:
    ao_system = AOSystemRuntime(
        band="R",
        fov=fov,
        lgs=(),
        min_wfs=min_wfs,
        max_wfs=max_wfs,
        min_mag=8.0,
        max_mag=18.5,
        min_sep=5.0 * u.arcsec,
    )
    return PredictRuntime(
        ao_system=ao_system,
        outer_level=0,
        inner_level=1,
        epoch=2028.0,
        max_bright_star_mag=None,
        max_bright_star_exclusion=2.0 * fov,
        winner_ee_epsilon=0.01,
        winner_top_k=3,
        prediction_wavelength=1.654 * u.micron,
        resolved_models={
            f"{count}star": point_model
            for count in range(min_wfs, max_wfs + 1)
        },
        averaged_models={
            f"{count}star": mean_model
            for count in range(min_wfs, max_wfs + 1)
        },
        seeing_reference_wavelength=0.5 * u.micron,
        seeing_reference_sr=0.0,
        seeing_reference_ee=0.0,
        seeing_reference_fwhm=0.0,
        coverage_ee_threshold_resolved=0.25,
        coverage_ee_threshold_averaged=0.25,
        model_root=model_root,
    )


def test_candidate_enclosing_fov_center_handles_single_and_pair() -> None:
    single = _candidate_enclosing_fov_center(
        np.asarray([12.0], dtype=np.float64),
        np.asarray([-3.0], dtype=np.float64),
    )
    assert single.ra_deg == pytest.approx(12.0)
    assert single.dec_deg == pytest.approx(-3.0)
    assert single.radius_arcsec == pytest.approx(0.0)

    pair = _candidate_enclosing_fov_center(
        np.asarray([0.0, 10.0 / 3600.0], dtype=np.float64),
        np.asarray([0.0, 0.0], dtype=np.float64),
    )
    assert pair.ra_deg == pytest.approx(5.0 / 3600.0)
    assert pair.dec_deg == pytest.approx(0.0)
    assert pair.radius_arcsec == pytest.approx(5.0)


def test_candidate_enclosing_fov_center_uses_circumcenter_for_acute_triangle() -> None:
    center = _candidate_enclosing_fov_center(
        np.asarray([0.0, 100.0 / 3600.0, 10.0 / 3600.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 80.0 / 3600.0], dtype=np.float64),
    )

    assert center.ra_deg == pytest.approx(50.0 / 3600.0)
    assert center.dec_deg == pytest.approx(34.375 / 3600.0)
    assert center.radius_arcsec == pytest.approx(np.hypot(50.0, 34.375))


def test_candidate_enclosing_fov_center_uses_longest_side_for_obtuse_triangle() -> None:
    center = _candidate_enclosing_fov_center(
        np.asarray([-10.0 / 3600.0, 10.0 / 3600.0, 5.0 / 3600.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0 / 3600.0], dtype=np.float64),
    )

    assert center.ra_deg == pytest.approx(0.0)
    assert center.dec_deg == pytest.approx(0.0)
    assert center.radius_arcsec == pytest.approx(10.0)


def test_nearest_feasible_pointing_keeps_feasible_pixel_center() -> None:
    pointing_x, pointing_y, feasible = _nearest_feasible_pointing(
        np.asarray([5.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([[0.0, 10.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        10.0,
    )

    assert bool(feasible[0])
    assert float(pointing_x[0]) == pytest.approx(5.0)
    assert float(pointing_y[0]) == pytest.approx(0.0)


def test_nearest_feasible_pointing_projects_to_two_star_lens() -> None:
    pointing_x, pointing_y, feasible = _nearest_feasible_pointing(
        np.asarray([0.0], dtype=np.float64),
        np.asarray([30.0], dtype=np.float64),
        np.asarray([[-58.0, 58.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        60.0,
    )

    assert bool(feasible[0])
    assert float(pointing_x[0]) == pytest.approx(0.0)
    assert float(pointing_y[0]) == pytest.approx(np.sqrt(60.0**2 - 58.0**2))


def test_nearest_feasible_pointing_projects_to_three_star_intersection() -> None:
    pointing_x, pointing_y, feasible = _nearest_feasible_pointing(
        np.asarray([5.0], dtype=np.float64),
        np.asarray([14.0], dtype=np.float64),
        np.asarray([[0.0, 10.0, 5.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0, 8.0]], dtype=np.float64),
        10.0,
    )

    assert bool(feasible[0])
    assert float(pointing_x[0]) == pytest.approx(5.0)
    assert float(pointing_y[0]) == pytest.approx(np.sqrt(10.0**2 - 5.0**2))


def test_nearest_feasible_pointing_can_fail_to_cover_science_pixel() -> None:
    pointing_x, pointing_y, feasible = _nearest_feasible_pointing(
        np.asarray([0.0], dtype=np.float64),
        np.asarray([80.0], dtype=np.float64),
        np.asarray([[-58.0, 58.0]], dtype=np.float64),
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        60.0,
    )
    science_distance = np.hypot(pointing_x[0], pointing_y[0] - 80.0)

    assert bool(feasible[0])
    assert float(science_distance) > 60.0


@pytest.mark.parametrize("star_count", [1, 2, 3])
def test_nearest_feasible_pointing_matches_generic_solver(star_count: int) -> None:
    rng = np.random.default_rng(1024 + star_count)
    row_count = 64
    radius = 60.0
    target_x = rng.uniform(-75.0, 75.0, size=row_count)
    target_y = rng.uniform(-75.0, 75.0, size=row_count)
    disk_x = np.zeros((row_count, star_count), dtype=np.float64)
    disk_y = np.zeros((row_count, star_count), dtype=np.float64)

    disk_x[:, 0] = rng.uniform(-15.0, 15.0, size=row_count)
    disk_y[:, 0] = rng.uniform(-15.0, 15.0, size=row_count)
    for disk_index in range(1, star_count):
        angle = rng.uniform(0.0, 2.0 * np.pi, size=row_count)
        separation = rng.uniform(0.0, 1.8 * radius, size=row_count)
        disk_x[:, disk_index] = disk_x[:, 0] + separation * np.cos(angle)
        disk_y[:, disk_index] = disk_y[:, 0] + separation * np.sin(angle)

    pointing_x, pointing_y, feasible = _nearest_feasible_pointing(
        target_x,
        target_y,
        disk_x,
        disk_y,
        radius,
    )
    expected_x, expected_y, expected_feasible = _nearest_feasible_pointing_generic(
        target_x,
        target_y,
        disk_x,
        disk_y,
        radius,
    )

    np.testing.assert_array_equal(feasible, expected_feasible)
    np.testing.assert_allclose(pointing_x, expected_x, equal_nan=True)
    np.testing.assert_allclose(pointing_y, expected_y, equal_nan=True)


def test_candidate_members_are_emitted_in_sensing_magnitude_order(
    tmp_path: Path,
) -> None:
    runtime = replace(
        _make_predict_runtime(
            model_root=tmp_path / "models",
            min_wfs=1,
            max_wfs=3,
        ),
        outer_level=0,
        inner_level=3,
    )
    stars = Table()
    stars["source_id"] = np.array([30, 20, 10], dtype=np.int64)
    stars["ra"] = np.array([0.006, 0.003, 0.0], dtype=np.float64)
    stars["dec"] = np.zeros(3, dtype=np.float64)
    stars["R"] = np.array([12.0, 10.0, 10.0], dtype=np.float64)

    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)

    magnitudes = np.asarray(graph.sorted_ngs["R"], dtype=np.float64)
    source_ids = np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64)
    assert source_ids.tolist() == [10, 20, 30]
    for members in candidate_set.members:
        valid = members[members >= 0]
        member_magnitudes = magnitudes[valid]
        member_source_ids = source_ids[valid]
        assert np.all(member_magnitudes[:-1] <= member_magnitudes[1:])
        for first, second in zip(
            range(len(valid) - 1),
            range(1, len(valid)),
            strict=False,
        ):
            if member_magnitudes[first] == member_magnitudes[second]:
                assert member_source_ids[first] < member_source_ids[second]


def test_unbounded_candidate_graph_counts_geometry_limited_candidates(
    tmp_path: Path,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=1,
        max_wfs=3,
    )
    stars = Table()
    stars["source_id"] = np.array([30, 20, 10, 40], dtype=np.int64)
    stars["ra"] = np.array([0.006, 0.003, 0.0, 1.0], dtype=np.float64)
    stars["dec"] = np.zeros(4, dtype=np.float64)
    stars["R"] = np.array([12.0, 10.0, 10.0, 13.0], dtype=np.float64)

    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)

    assert graph.final_count == 4
    assert len(graph.edges) == 3
    assert len(graph.triangles) == 1
    assert graph.candidate_count == 8
    assert len(candidate_set.members) == 8
    assert np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64).tolist() == [
        10,
        20,
        30,
        40,
    ]


def test_candidate_graph_retains_pairwise_valid_fov_invalid_triangle_for_late_skip(
    tmp_path: Path,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=20.0 * u.arcsec,
        min_wfs=1,
        max_wfs=3,
    )
    side_arcsec = 19.0
    stars = Table()
    stars["source_id"] = np.array([10, 20, 30], dtype=np.int64)
    stars["ra"] = np.array(
        [0.0, side_arcsec / 3600.0, side_arcsec / 2.0 / 3600.0],
        dtype=np.float64,
    )
    stars["dec"] = np.array(
        [0.0, 0.0, np.sqrt(3.0) / 2.0 * side_arcsec / 3600.0],
        dtype=np.float64,
    )
    stars["R"] = np.array([10.0, 11.0, 12.0], dtype=np.float64)

    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)

    assert len(graph.edges) == 3
    assert len(graph.triangles) == 1
    assert graph.candidate_count == 7
    assert len(candidate_set.members) == 7


def test_candidate_graph_keeps_near_boundary_valid_triangle(
    tmp_path: Path,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=20.0 * u.arcsec,
        min_wfs=1,
        max_wfs=3,
    )
    side_arcsec = np.sqrt(3.0) * 10.0
    stars = Table()
    stars["source_id"] = np.array([10, 20, 30], dtype=np.int64)
    stars["ra"] = np.array(
        [0.0, side_arcsec / 3600.0, side_arcsec / 2.0 / 3600.0],
        dtype=np.float64,
    )
    stars["dec"] = np.array(
        [0.0, 0.0, np.sqrt(3.0) / 2.0 * side_arcsec / 3600.0],
        dtype=np.float64,
    )
    stars["R"] = np.array([10.0, 11.0, 12.0], dtype=np.float64)

    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)

    assert len(graph.edges) == 3
    assert len(graph.triangles) == 1
    assert graph.candidate_count == 7
    assert len(candidate_set.members) == 7


def test_retained_asterism_table_uses_enclosing_fov_center(
    tmp_path: Path,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=200.0 * u.arcsec,
        min_wfs=1,
        max_wfs=3,
    )
    stars = Table()
    stars["source_id"] = np.array([1, 2, 3], dtype=np.int64)
    stars["ra"] = np.array([0.0, 100.0 / 3600.0, 10.0 / 3600.0], dtype=np.float64)
    stars["dec"] = np.array([0.0, 0.0, 80.0 / 3600.0], dtype=np.float64)
    stars["R"] = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)
    candidate_id = np.flatnonzero(
        np.all(candidate_set.source_keys == np.asarray([1, 2, 3]), axis=1),
    )[0]
    expected = _candidate_enclosing_fov_center(
        np.asarray(stars["ra"], dtype=np.float64),
        np.asarray(stars["dec"], dtype=np.float64),
    )

    retained = _build_retained_asterism_table(
        candidate_set,
        np.asarray([candidate_id], dtype=np.int64),
        graph.sorted_ngs,
        runtime,
    )

    assert float(retained["ra"][0]) == pytest.approx(expected.ra_deg)
    assert float(retained["dec"][0]) == pytest.approx(expected.dec_deg)
    assert float(retained["ra"][0]) != pytest.approx(float(np.mean(stars["ra"])))
    expected_pix = get_pixel_from_skycoord(
        runtime.inner_level,
        SkyCoord(
            ra=[expected.ra_deg],
            dec=[expected.dec_deg],
            unit=(u.degree, u.degree),
        ),
    )
    assert int(retained["pix"][0]) == int(np.asarray(expected_pix)[0])


def test_recovered_prediction_uses_nearest_valid_pointing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=120.0 * u.arcsec,
        min_wfs=2,
        max_wfs=2,
    )
    stars = Table()
    stars["source_id"] = np.array([10, 20], dtype=np.int64)
    stars["ra"] = np.array([-58.0 / 3600.0, 58.0 / 3600.0], dtype=np.float64)
    stars["dec"] = np.array([0.0, 0.0], dtype=np.float64)
    stars["R"] = np.array([10.0, 11.0], dtype=np.float64)
    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)
    seen: dict[str, np.ndarray] = {}

    monkeypatch.setattr(
        "ao_sky.build.traversal.get_point_model",
        lambda runtime, num_stars, **kwargs: object(),
    )

    def fake_predict_point_arrays(
        runtime,
        num_stars,
        model,
        ngs_zd,
        ngs_az_deg,
        ngs_mag,
        backend_row_count=None,
        feature_buffer_row_count=None,
        prediction_telemetry=None,
    ):
        seen["ngs_zd"] = np.asarray(ngs_zd, dtype=np.float64)
        if prediction_telemetry is not None:
            prediction_telemetry.record_feature_batch(
                np.zeros((len(ngs_zd), 1), dtype=np.float64),
                0.0,
                row_count=len(ngs_zd),
            )
        return PointPredictionBatch(
            sr=np.asarray([0.5], dtype=np.float64),
            ee=np.asarray([0.7], dtype=np.float64),
            fwhm=np.asarray([100.0], dtype=np.float64),
            ee_angle=np.asarray([0.0], dtype=np.float64),
        )

    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_point_arrays",
        fake_predict_point_arrays,
    )

    best_ee = np.full(2, np.nan, dtype=np.float64)
    best_sr = np.full(2, np.nan, dtype=np.float64)
    best_fwhm = np.full(2, np.nan, dtype=np.float64)
    top_refs = np.full((2, 3), -1, dtype=np.int64)
    top_ee = np.full((2, 3), -np.inf, dtype=np.float64)
    top_sr = np.full((2, 3), np.nan, dtype=np.float64)
    top_fwhm = np.full((2, 3), np.nan, dtype=np.float64)
    top_pointing_x = np.full((2, 3), np.nan, dtype=np.float64)
    top_pointing_y = np.full((2, 3), np.nan, dtype=np.float64)
    structure_profile = TraversalStructureProfile()

    recovered_rows = _stream_recovered_candidate_predictions(
        runtime,
        TraversalExecutionConfig(resolved_backend_buckets=()),
        candidate_set,
        np.asarray([[1], [1]], dtype=np.uint64),
        np.asarray([1], dtype=np.uint64),
        star_x=np.asarray([-58.0, 58.0], dtype=np.float64),
        star_y=np.asarray([0.0, 0.0], dtype=np.float64),
        inner_x=np.asarray([1000.0, 0.0], dtype=np.float64),
        inner_y=np.asarray([1000.0, 30.0], dtype=np.float64),
        star_mags=np.asarray([10.0, 11.0], dtype=np.float64),
        best_ee=best_ee,
        best_sr=best_sr,
        best_fwhm=best_fwhm,
        top_refs=top_refs,
        top_ee=top_ee,
        top_sr=top_sr,
        top_fwhm=top_fwhm,
        top_pointing_x=top_pointing_x,
        top_pointing_y=top_pointing_y,
        recovery_pixel_index_map=np.asarray([1], dtype=np.int64),
        structure_profile=structure_profile,
    )

    assert recovered_rows == 1
    assert structure_profile.to_stats().recovered_point_prediction_rows == 1
    assert int(top_refs[0, 0]) == -1
    assert int(top_refs[1, 0]) == 0
    assert float(top_pointing_x[1, 0]) == pytest.approx(0.0)
    assert float(top_pointing_y[1, 0]) == pytest.approx(np.sqrt(60.0**2 - 58.0**2))
    assert seen["ngs_zd"][0].tolist() == pytest.approx([60.0, 60.0])


def test_recovery_skips_candidates_failing_wide_fov_prefilter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=120.0 * u.arcsec,
        min_wfs=2,
        max_wfs=2,
    )
    stars = Table()
    stars["source_id"] = np.array([10, 20], dtype=np.int64)
    stars["ra"] = np.array([-58.0 / 3600.0, 58.0 / 3600.0], dtype=np.float64)
    stars["dec"] = np.array([0.0, 0.0], dtype=np.float64)
    stars["R"] = np.array([10.0, 11.0], dtype=np.float64)
    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)
    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_point_arrays",
        lambda *args, **kwargs: pytest.fail("recovery should not run prediction"),
    )

    recovered_rows = _stream_recovered_candidate_predictions(
        runtime,
        TraversalExecutionConfig(resolved_backend_buckets=()),
        candidate_set,
        np.asarray([[1], [0]], dtype=np.uint64),
        np.asarray([1], dtype=np.uint64),
        star_x=np.asarray([-58.0, 58.0], dtype=np.float64),
        star_y=np.asarray([0.0, 0.0], dtype=np.float64),
        inner_x=np.asarray([0.0], dtype=np.float64),
        inner_y=np.asarray([30.0], dtype=np.float64),
        star_mags=np.asarray([10.0, 11.0], dtype=np.float64),
        best_ee=np.full(1, np.nan, dtype=np.float64),
        best_sr=np.full(1, np.nan, dtype=np.float64),
        best_fwhm=np.full(1, np.nan, dtype=np.float64),
        top_refs=np.full((1, 3), -1, dtype=np.int64),
        top_ee=np.full((1, 3), -np.inf, dtype=np.float64),
        top_sr=np.full((1, 3), np.nan, dtype=np.float64),
        top_fwhm=np.full((1, 3), np.nan, dtype=np.float64),
        top_pointing_x=np.full((1, 3), np.nan, dtype=np.float64),
        top_pointing_y=np.full((1, 3), np.nan, dtype=np.float64),
    )

    assert recovered_rows == 0


def test_averaged_prediction_uses_selected_pointing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=120.0 * u.arcsec,
        min_wfs=2,
        max_wfs=2,
    )
    stars = Table()
    stars["source_id"] = np.array([10, 20], dtype=np.int64)
    stars["ra"] = np.array([-58.0 / 3600.0, 58.0 / 3600.0], dtype=np.float64)
    stars["dec"] = np.array([0.0, 0.0], dtype=np.float64)
    stars["R"] = np.array([10.0, 11.0], dtype=np.float64)
    graph = _build_candidate_graph_from_ngs(stars, runtime)
    candidate_set = _build_candidate_set(graph, runtime)
    inner = Table(
        [
            np.asarray([0], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.zeros(1, dtype=np.int64),
            np.full(1, np.nan, dtype=np.float64),
            np.full(1, np.nan, dtype=np.float64),
            np.full(1, np.nan, dtype=np.float64),
            np.asarray([1], dtype=np.int64),
            np.full(1, np.nan, dtype=np.float64),
            np.full(1, 0.7, dtype=np.float64),
            np.zeros(1, dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        ],
        names=(
            "pix",
            "star_count",
            "ngs_count",
            "best_ee",
            "best_sr",
            "best_fwhm",
            "winner_asterism_id",
            "winner_ee_resolved",
            "winner_ee_averaged",
            "coverage_resolved",
            "coverage_averaged",
        ),
    )
    seen: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        "ao_sky.build.traversal.get_mean_model",
        lambda runtime, num_stars, **kwargs: object(),
    )

    def fake_predict_field_mean_arrays(
        runtime,
        num_stars,
        model,
        ngs_zd,
        ngs_az_deg,
        ngs_mag,
        backend_row_count=None,
        feature_buffer_row_count=None,
        prediction_telemetry=None,
    ):
        seen["ngs_zd"] = np.asarray(ngs_zd, dtype=np.float64)
        if prediction_telemetry is not None:
            prediction_telemetry.record_feature_batch(
                np.zeros((len(ngs_zd), 1), dtype=np.float64),
                0.0,
                row_count=len(ngs_zd),
            )
        return np.asarray([0.6], dtype=np.float64)

    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_field_mean_arrays",
        fake_predict_field_mean_arrays,
    )
    pointing_y = np.sqrt(60.0**2 - 58.0**2)

    _update_regularized_winner_averaged_ee(
        runtime,
        TraversalExecutionConfig(averaged_backend_buckets=()),
        inner,
        np.asarray([0], dtype=np.int64),
        candidate_set,
        star_x=np.asarray([-58.0, 58.0], dtype=np.float64),
        star_y=np.asarray([0.0, 0.0], dtype=np.float64),
        inner_x=np.asarray([0.0], dtype=np.float64),
        inner_y=np.asarray([30.0], dtype=np.float64),
        winner_pointing_x=np.asarray([0.0], dtype=np.float64),
        winner_pointing_y=np.asarray([pointing_y], dtype=np.float64),
        star_mags=np.asarray([10.0, 11.0], dtype=np.float64),
    )

    assert seen["ngs_zd"][0].tolist() == pytest.approx([60.0, 60.0])
    assert float(inner["winner_ee_averaged"][0]) == pytest.approx(0.7)


def test_regional_candidate_graph_keeps_exact_tractable_region(
    tmp_path: Path,
) -> None:
    runtime = replace(
        _make_predict_runtime(
            model_root=tmp_path / "models",
            min_wfs=1,
            max_wfs=3,
        ),
        outer_level=0,
        inner_level=3,
    )
    stars = Table()
    stars["source_id"] = np.array([10, 20, 30, 40], dtype=np.int64)
    stars["ra"] = np.array([0.0, 0.003, 0.006, 0.009], dtype=np.float64)
    stars["dec"] = np.zeros(4, dtype=np.float64)
    stars["R"] = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    coverage = _StarForCoverage(
        pixel_indices=np.zeros(4, dtype=np.int64),
        starts=np.arange(5, dtype=np.int64),
        full_depth=np.array([4, 0, 0, 0], dtype=np.uint32),
        target_depth=np.array([3, 0, 0, 0], dtype=np.uint16),
    )

    graph, stats = _build_regional_candidate_graph(
        stars,
        coverage,
        runtime,
    )
    candidate_set = _build_candidate_set(graph, runtime)

    assert stats.exact_regions == 1
    assert stats.for_optimized_regions == 0
    assert stats.max_regional_combination_work == 64
    assert stats.exact_combination_work == 14
    assert stats.final_resolved_inferences == 14
    assert graph.final_count == 4
    assert graph.candidate_count == 14
    assert len(candidate_set.members) == 14


def test_regional_candidate_graph_uses_for_selection_at_dense_floor(
    tmp_path: Path,
) -> None:
    runtime = replace(
        _make_predict_runtime(
            model_root=tmp_path / "models",
            fov=1000.0 * u.arcsec,
            min_wfs=1,
            max_wfs=3,
        ),
    )
    count = 20
    stars = Table()
    stars["source_id"] = np.arange(100, 100 + count, dtype=np.int64)
    stars["ra"] = np.arange(count, dtype=np.float64) * 0.002
    stars["dec"] = np.zeros(count, dtype=np.float64)
    stars["R"] = np.arange(count, dtype=np.float64) + 10.0
    coverage = _StarForCoverage(
        pixel_indices=np.zeros(count, dtype=np.int64),
        starts=np.arange(count + 1, dtype=np.int64),
        full_depth=np.array([count, 0, 0, 0], dtype=np.uint32),
        target_depth=np.array([3, 0, 0, 0], dtype=np.uint16),
    )

    graph, stats = _build_regional_candidate_graph(
        stars,
        coverage,
        runtime,
    )
    candidate_set = _build_candidate_set(graph, runtime)

    assert stats.exact_regions == 0
    assert stats.for_optimized_regions == 1
    assert stats.exact_combination_work == 0
    assert stats.final_resolved_inferences == 7
    assert np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64).tolist() == [
        100,
        101,
        102,
    ]
    assert graph.candidate_count == 7
    assert len(candidate_set.members) == 7


def test_regional_candidate_graph_uses_unbounded_for_selection_under_budget(
    tmp_path: Path,
) -> None:
    runtime = replace(
        _make_predict_runtime(
            model_root=tmp_path / "models",
            fov=1000.0 * u.arcsec,
            min_wfs=1,
            max_wfs=3,
        ),
        outer_level=14,
        inner_level=14,
    )
    count = 12
    stars = Table()
    stars["source_id"] = np.arange(100, 100 + count, dtype=np.int64)
    stars["ra"] = np.arange(count, dtype=np.float64) * 0.002
    stars["dec"] = np.zeros(count, dtype=np.float64)
    stars["R"] = np.arange(count, dtype=np.float64) + 10.0
    pixel_rows = [
        [0],
        [0],
        [0, 1],
        [0, 1],
        [0, 1],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    ]
    starts = np.zeros((count + 1,), dtype=np.int64)
    starts[1:] = np.cumsum([len(rows) for rows in pixel_rows], dtype=np.int64)
    coverage = _StarForCoverage(
        pixel_indices=np.asarray(
            [pixel for rows in pixel_rows for pixel in rows],
            dtype=np.int64,
        ),
        starts=starts,
        full_depth=np.array([count, 3, 0, 0], dtype=np.uint32),
        target_depth=np.array([3, 3, 0, 0], dtype=np.uint16),
    )

    graph, stats = _build_regional_candidate_graph(
        stars,
        coverage,
        runtime,
    )

    assert stats.exact_regions == 0
    assert stats.for_optimized_regions == 1
    assert stats.incomplete_for_regions == 0
    assert stats.final_resolved_inferences == 32
    assert np.asarray(graph.sorted_ngs["source_id"], dtype=np.int64).tolist() == [
        100,
        101,
        102,
        103,
        104,
    ]


def test_regional_selector_max_depth_uses_level_above_for_size(
    tmp_path: Path,
) -> None:
    runtime = replace(
        _make_predict_runtime(
            model_root=tmp_path / "models",
            fov=120.0 * u.arcsec,
            min_wfs=1,
            max_wfs=3,
        ),
        outer_level=6,
        inner_level=14,
    )

    assert _regional_selector_max_depth(runtime) == 4
    assert _max_regional_combination_work(runtime) == 65536


def test_regularized_winner_labels_use_local_support_with_epsilon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = replace(
        _make_predict_runtime(model_root=tmp_path / "models"),
        winner_ee_epsilon=0.05,
    )
    top_refs = np.array(
        [
            [1, 2, -1],
            [2, 1, -1],
            [2, 1, -1],
            [3, 1, -1],
            [4, 1, -1],
        ],
        dtype=np.int64,
    )
    top_ee = np.array(
        [
            [1.00, 0.97, -np.inf],
            [1.00, 0.97, -np.inf],
            [1.00, 0.97, -np.inf],
            [1.00, 0.90, -np.inf],
            [1.00, 0.97, -np.inf],
        ],
        dtype=np.float64,
    )
    neighbour_matrix = np.array(
        [
            [1, 2, -1, -1, -1, -1, -1, -1],
            [0, 2, -1, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1, -1],
            [0, 1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int64,
    )
    monkeypatch.setattr(
        "ao_sky.build.traversal._local_neighbor_index_matrix",
        lambda inner_pixs, inner_level: neighbour_matrix,
    )

    labels = _regularize_winner_labels(
        runtime,
        np.arange(len(top_refs), dtype=np.int64),
        top_refs,
        top_ee,
    )

    assert labels.tolist() == [2, 2, 2, 3, 1]


def test_backend_bucket_helpers_cap_logical_batches_at_buffer_size() -> None:
    buckets = (1024, 2048, 4096)

    assert DEFAULT_BACKEND_BUCKETS == tuple(range(1000, 25001, 1000))
    assert _stream_batch_size(5000, buckets) == 4096
    assert _stream_batch_size(3000, buckets) == 3000
    assert _stream_batch_size(3000, ()) == 3000
    assert _backend_buffer_row_count(buckets) == 4096
    assert _backend_buffer_row_count(()) is None
    assert _backend_row_count(3000, buckets) == 4096
    assert _backend_row_count(3000, ()) is None

    with pytest.raises(BuildError, match="largest backend bucket"):
        _backend_row_count(4097, buckets)


def test_multicover_selection_keeps_fainter_stars_for_spatial_depth() -> None:
    ngs = Table()
    ngs["source_id"] = np.array([10, 20, 30, 40], dtype=np.int64)
    ngs["R"] = np.array([9.0, 10.0, 14.0, 15.0], dtype=np.float64)
    coverage = _StarForCoverage(
        pixel_indices=np.array([0, 1, 0, 1, 2, 0], dtype=np.int64),
        starts=np.array([0, 2, 4, 5, 6], dtype=np.int64),
        full_depth=np.array([3, 2, 1], dtype=np.uint32),
        target_depth=np.array([2, 2, 1], dtype=np.uint16),
    )

    selection = _select_ngs_for_multicover(ngs, coverage)

    assert isinstance(selection, _MulticoverSelection)
    assert selection.star_indices.tolist() == [0, 1, 2]
    assert selection.depth.tolist() == [2, 2, 1]


def test_multicover_selection_preserves_only_achievable_depth() -> None:
    ngs = Table()
    ngs["source_id"] = np.array([10, 20], dtype=np.int64)
    ngs["R"] = np.array([9.0, 10.0], dtype=np.float64)
    coverage = _StarForCoverage(
        pixel_indices=np.array([0, 1], dtype=np.int64),
        starts=np.array([0, 1, 2], dtype=np.int64),
        full_depth=np.array([1, 1, 0], dtype=np.uint32),
        target_depth=np.array([1, 1, 0], dtype=np.uint16),
    )

    selection = _select_ngs_for_multicover(ngs, coverage)

    assert selection.star_indices.tolist() == [0, 1]
    assert selection.depth.tolist() == [1, 1, 0]


def test_local_neighbor_index_matrix_requires_dense_inner_pixel_block() -> None:
    with pytest.raises(BuildError, match="dense contiguous nested inner-pixel block"):
        _local_neighbor_index_matrix(
            np.array([100, 102, 101], dtype=np.int64),
            inner_level=4,
        )


def test_vectorized_model_features_match_backend_layout(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )

    x = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=np.array([[10.0, 20.0]], dtype=np.float64),
        ngs_az_deg=np.array([[45.0, 190.0]], dtype=np.float64),
        ngs_mag=np.array([[11.0, 12.0]], dtype=np.float64),
    )
    mean_x = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=np.array([[10.0, 20.0]], dtype=np.float64),
        ngs_az_deg=np.array([[45.0, 190.0]], dtype=np.float64),
        ngs_mag=np.array([[11.0, 12.0]], dtype=np.float64),
        mean_only=True,
    )

    assert x.shape == (1, 9)
    assert x[0, :7] == pytest.approx(
        [
            1.654,
            10.0,
            np.deg2rad(45.0),
            11.0,
            20.0,
            np.deg2rad(-170.0),
            12.0,
        ]
    )
    assert x[0, 7:] == pytest.approx([0.0, 0.0])
    assert mean_x.shape == (1, 7)
    assert mean_x[0] == pytest.approx(x[0, :7])


def test_vectorized_model_features_preserve_tied_magnitude_order(
    tmp_path: Path,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )

    x = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=np.array([[20.0, 10.0]], dtype=np.float64),
        ngs_az_deg=np.array([[190.0, 45.0]], dtype=np.float64),
        ngs_mag=np.array([[12.0, 12.0]], dtype=np.float64),
        mean_only=True,
    )

    assert x[0] == pytest.approx(
        [
            1.654,
            20.0,
            np.deg2rad(-170.0),
            12.0,
            10.0,
            np.deg2rad(45.0),
            12.0,
        ]
    )


def test_vectorized_model_features_reuse_static_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=1,
        max_wfs=1,
    )
    runtime = replace(
        runtime,
        ao_system=replace(
            runtime.ao_system,
            lgs=(
                {"zd": 30.0, "az": 0.0},
                {"zd": 40.0, "az": 90.0},
            ),
        ),
    )
    predict_service._FEATURE_TEMPLATE_CACHE.clear()
    original_get_lgs_xy = predict_service._get_ao_lgs_xy
    calls = 0

    def fake_get_lgs_xy(lgs):
        nonlocal calls
        calls += 1
        return original_get_lgs_xy(lgs)

    monkeypatch.setattr(predict_service, "_get_ao_lgs_xy", fake_get_lgs_xy)

    first = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=np.array([[20.0], [10.0]], dtype=np.float64),
        ngs_az_deg=np.array([[190.0], [45.0]], dtype=np.float64),
        ngs_mag=np.array([[12.0], [11.0]], dtype=np.float64),
    )
    second = build_model_x_from_ngs_arrays(
        runtime,
        ngs_zd=np.array([[15.0]], dtype=np.float64),
        ngs_az_deg=np.array([[90.0]], dtype=np.float64),
        ngs_mag=np.array([[10.0]], dtype=np.float64),
    )

    assert calls == 1
    assert first.shape == (2, 8)
    assert first[:, 0] == pytest.approx([1.654, 1.654])
    assert first[:, 4:6] == pytest.approx(np.zeros((2, 2), dtype=np.float64))
    assert first[:, 6:] == pytest.approx(
        np.array([[30.0, 40.0], [30.0, 40.0]], dtype=np.float64)
    )
    assert second[0] == pytest.approx(
        [
            1.654,
            15.0,
            np.deg2rad(90.0),
            10.0,
            0.0,
            0.0,
            30.0,
            40.0,
        ]
    )


def test_vectorized_prediction_records_feature_and_backend_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )
    captured: dict[str, object] = {}

    def fake_get_prediction(x, model, skip_cache_clear=False):
        captured["shape"] = x.shape
        captured["skip_cache_clear"] = skip_cache_clear
        return np.ones((x.shape[0], 3), dtype=np.float64)

    monkeypatch.setattr(predict_service.backend, "get_prediction", fake_get_prediction)

    telemetry = predict_service.PredictionArrayTelemetry()
    result = predict_service.predict_field_mean_arrays(
        runtime,
        num_stars=2,
        model=object(),
        ngs_zd=np.array([[10.0, 20.0], [15.0, 30.0]], dtype=np.float64),
        ngs_az_deg=np.array([[45.0, 190.0], [20.0, 10.0]], dtype=np.float64),
        ngs_mag=np.array([[11.0, 12.0], [12.5, 13.0]], dtype=np.float64),
        prediction_telemetry=telemetry,
    )

    assert result.shape == (2,)
    assert captured == {
        "shape": (2, 7),
        "skip_cache_clear": True,
    }
    assert telemetry.batches == 1
    assert telemetry.rows == 2
    assert telemetry.batch_rows_peak == 2
    assert telemetry.feature_bytes_peak == 2 * 7 * np.dtype(np.float64).itemsize
    assert telemetry.feature_seconds >= 0.0
    assert telemetry.backend_seconds >= 0.0


def test_vectorized_prediction_requires_num_stars_to_match_arrays(
    tmp_path: Path,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=3,
    )

    with pytest.raises(PredictError, match="num_stars must match"):
        predict_service.predict_point_arrays(
            runtime,
            num_stars=3,
            model=object(),
            ngs_zd=np.array([[10.0, 20.0]], dtype=np.float64),
            ngs_az_deg=np.array([[45.0, 190.0]], dtype=np.float64),
            ngs_mag=np.array([[11.0, 12.0]], dtype=np.float64),
        )

    with pytest.raises(PredictError, match="num_stars must match"):
        predict_service.predict_field_mean_arrays(
            runtime,
            num_stars=3,
            model=object(),
            ngs_zd=np.array([[10.0, 20.0]], dtype=np.float64),
            ngs_az_deg=np.array([[45.0, 190.0]], dtype=np.float64),
            ngs_mag=np.array([[11.0, 12.0]], dtype=np.float64),
        )


def test_vectorized_prediction_can_use_fixed_backend_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )
    captured: dict[str, object] = {}
    predict_service._FEATURE_BUFFER_CACHE.clear()

    def fake_get_prediction(x, model, skip_cache_clear=False):
        captured["shape"] = x.shape
        captured["last_row"] = x[-1].copy()
        return np.column_stack(
            (
                np.arange(x.shape[0], dtype=np.float64),
                np.arange(x.shape[0], dtype=np.float64) + 10.0,
                np.arange(x.shape[0], dtype=np.float64) + 20.0,
            )
        )

    monkeypatch.setattr(predict_service.backend, "get_prediction", fake_get_prediction)

    telemetry = predict_service.PredictionArrayTelemetry()
    result = predict_service.predict_point_arrays(
        runtime,
        num_stars=2,
        model=object(),
        ngs_zd=np.array([[10.0, 20.0], [15.0, 30.0]], dtype=np.float64),
        ngs_az_deg=np.array([[45.0, 190.0], [20.0, 10.0]], dtype=np.float64),
        ngs_mag=np.array([[11.0, 12.0], [12.5, 13.0]], dtype=np.float64),
        backend_row_count=5,
        prediction_telemetry=telemetry,
    )

    assert captured["shape"] == (5, 9)
    assert captured["last_row"][[1, 3, 4, 6]] == pytest.approx(
        [0.0, 0.0, 0.0, 0.0]
    )
    assert result.sr.tolist() == [0.0, 1.0]
    assert result.ee.tolist() == [10.0, 11.0]
    assert result.fwhm.tolist() == [20.0, 21.0]
    assert telemetry.batches == 1
    assert telemetry.rows == 2
    assert telemetry.batch_rows_peak == 2
    assert telemetry.feature_bytes_peak == 5 * 9 * np.dtype(np.float64).itemsize


def test_vectorized_prediction_reuses_fixed_backend_feature_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )
    captured: list[np.ndarray] = []
    predict_service._FEATURE_BUFFER_CACHE.clear()

    def fake_get_prediction(x, model, skip_cache_clear=False):
        captured.append(x)
        return np.zeros((x.shape[0], 3), dtype=np.float64)

    monkeypatch.setattr(predict_service.backend, "get_prediction", fake_get_prediction)

    model = object()
    for scale in (1.0, 2.0):
        predict_service.predict_point_arrays(
            runtime,
            num_stars=2,
            model=model,
            ngs_zd=np.array(
                [[10.0 * scale, 20.0 * scale], [15.0 * scale, 30.0 * scale]],
                dtype=np.float64,
            ),
            ngs_az_deg=np.array([[45.0, 190.0], [20.0, 10.0]], dtype=np.float64),
            ngs_mag=np.array([[11.0, 12.0], [12.5, 13.0]], dtype=np.float64),
            backend_row_count=5,
        )

    assert len(captured) == 2
    assert captured[0] is captured[1]
    assert captured[1][[0, 1, 4], 0].tolist() == pytest.approx([1.654, 1.654, 1.654])
    assert captured[1][0, 1] == pytest.approx(20.0)
    assert captured[1][1, 1] == pytest.approx(30.0)
    assert captured[1][2, 1] == pytest.approx(0.0)


def test_vectorized_prediction_slices_one_larger_feature_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )
    captured: list[np.ndarray] = []
    predict_service._FEATURE_BUFFER_CACHE.clear()

    def fake_get_prediction(x, model, skip_cache_clear=False):
        captured.append(x)
        return np.zeros((x.shape[0], 3), dtype=np.float64)

    monkeypatch.setattr(predict_service.backend, "get_prediction", fake_get_prediction)
    model = object()

    predict_service.predict_point_arrays(
        runtime,
        num_stars=2,
        model=model,
        ngs_zd=np.array(
            [[10.0, 20.0], [15.0, 30.0], [25.0, 35.0]],
            dtype=np.float64,
        ),
        ngs_az_deg=np.array(
            [[45.0, 190.0], [20.0, 10.0], [80.0, 120.0]],
            dtype=np.float64,
        ),
        ngs_mag=np.array(
            [[11.0, 12.0], [12.5, 13.0], [14.0, 15.0]],
            dtype=np.float64,
        ),
        backend_row_count=5,
    )
    predict_service.predict_point_arrays(
        runtime,
        num_stars=2,
        model=model,
        ngs_zd=np.array([[5.0, 6.0]], dtype=np.float64),
        ngs_az_deg=np.array([[7.0, 8.0]], dtype=np.float64),
        ngs_mag=np.array([[9.0, 10.0]], dtype=np.float64),
        backend_row_count=3,
    )

    large, smaller = captured
    assert large.shape == (5, 9)
    assert smaller.shape == (3, 9)
    assert np.shares_memory(large, smaller)
    assert smaller[0, [1, 3, 4, 6]] == pytest.approx([5.0, 9.0, 6.0, 10.0])
    assert smaller[1, [1, 3, 4, 6]] == pytest.approx([15.0, 12.5, 30.0, 13.0])
    assert smaller[2, [1, 3, 4, 6]] == pytest.approx([25.0, 14.0, 35.0, 15.0])


def test_vectorized_prediction_records_mps_memory_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        min_wfs=2,
        max_wfs=2,
    )
    samples = iter(
        (
            (10, 20, 100),
            (30, 50, 100),
        )
    )

    monkeypatch.setattr(predict_service.backend, "get_model_device_type", lambda model: "mps")
    monkeypatch.setattr(
        predict_service.backend,
        "get_mps_memory_bytes",
        lambda: next(samples),
    )
    monkeypatch.setattr(
        predict_service.backend,
        "get_prediction",
        lambda x, model, skip_cache_clear=False: np.ones((x.shape[0], 3)),
    )

    telemetry = predict_service.PredictionArrayTelemetry()
    predict_service.predict_field_mean_arrays(
        runtime,
        num_stars=2,
        model=object(),
        ngs_zd=np.array([[10.0, 20.0]], dtype=np.float64),
        ngs_az_deg=np.array([[45.0, 190.0]], dtype=np.float64),
        ngs_mag=np.array([[11.0, 12.0]], dtype=np.float64),
        prediction_telemetry=telemetry,
    )

    assert telemetry.mps_current_bytes_peak == 30
    assert telemetry.mps_driver_bytes_peak == 50
    assert telemetry.mps_recommended_bytes == 100


def test_runtime_gaia_store_caches_epoch_shifted_read_only_rows(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    raw = _gaia_table([(101, 10.0, 0.0, 12.0, 100.0, 50.0)])
    base_store = _RawGaiaStore(tmp_path, {0: raw})
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )

    first = store.load_healpix(0)
    second = store.load_healpix(0)
    mutable = store.load_healpix(0, read_only=False)
    mutable["ra"][0] = 99.0

    assert first is second
    assert base_store.loads == [0]
    assert "R" in first.colnames
    assert RUNTIME_HPX_COLUMN in first.colnames
    assert not first["ra"].flags.writeable
    assert first["ref_epoch"][0] == pytest.approx(runtime.epoch)
    assert float(first["ra"][0]) != pytest.approx(float(raw["ra"][0]))
    assert float(second["ra"][0]) != pytest.approx(99.0)
    assert store.stats().hits == 2
    assert store.stats().misses == 1


def test_runtime_gaia_store_force_reload_replaces_cached_table(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    base_store = _RawGaiaStore(
        tmp_path,
        {0: _gaia_table([(101, 10.0, 0.0, 12.0, 100.0, 50.0)])},
    )
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )

    first = store.load_healpix(0)
    base_store.tables[0] = _gaia_table([(202, 20.0, 0.0, 13.0, 0.0, 0.0)])
    refreshed = store.load_healpix(0, force_reload=True)
    again = store.load_healpix(0)

    assert base_store.loads == [0, 0]
    assert first is not refreshed
    assert refreshed is again
    assert int(again["source_id"][0]) == 202


def test_runtime_gaia_store_does_not_cache_oversized_tables(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    raw = _gaia_table([(101, 10.0, 0.0, 12.0, 100.0, 50.0)])
    base_store = _RawGaiaStore(tmp_path, {0: raw})
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=4,
        max_bytes=1,
    )

    first = store.load_healpix(0)
    second = store.load_healpix(0)

    assert first is not second
    assert base_store.loads == [0, 0]
    stats = store.stats()
    assert stats.entries == 0
    assert stats.current_bytes == 0
    assert stats.oversized_skips == 2


def test_runtime_gaia_store_marks_invalid_coordinates_with_invalid_hpx(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    raw = _gaia_table(
        [
            (101, 10.0, 0.0, 12.0, 0.0, 0.0),
            (202, np.nan, 0.0, 13.0, 0.0, 0.0),
        ]
    )
    store = RuntimeGaiaHealpixStore(
        _RawGaiaStore(tmp_path, {0: raw}),
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )

    table = store.load_healpix(0)

    assert int(table[RUNTIME_HPX_COLUMN][0]) >= 0
    assert int(table[RUNTIME_HPX_COLUMN][1]) == -1


def test_prepare_search_inputs_uses_runtime_gaia_without_reapplying_epoch_shift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    raw = _gaia_table([(101, 10.0, 0.0, 12.0, 0.0, 0.0)])
    store = RuntimeGaiaHealpixStore(
        _RawGaiaStore(tmp_path, {0: raw}),
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )
    monkeypatch.setattr(
        "ao_sky.build.traversal.get_pixel_from_skycoord",
        lambda *args, **kwargs: pytest.fail("runtime hpx14 was not reused"),
    )

    stars, ngs = prepare_search_inputs(store, runtime, 0)

    assert stars["source_id"].tolist() == [101]
    assert ngs["source_id"].tolist() == [101]
    assert RUNTIME_HPX_COLUMN in stars.colnames


def test_prepare_search_inputs_ignores_invalid_runtime_gaia_rows(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    raw = _gaia_table(
        [
            (101, 10.0, 0.0, 12.0, 0.0, 0.0),
            (202, np.nan, 0.0, 13.0, 0.0, 0.0),
        ]
    )
    store = RuntimeGaiaHealpixStore(
        _RawGaiaStore(tmp_path, {0: raw}),
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )

    stars, ngs = prepare_search_inputs(store, runtime, 0)

    assert stars["source_id"].tolist() == [101]
    assert ngs["source_id"].tolist() == [101]
    assert np.all(np.asarray(stars[RUNTIME_HPX_COLUMN], dtype=np.int64) >= 0)


def test_build_base_inner_table_ignores_invalid_runtime_gaia_rows(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    coord = get_pixel_skycoord(runtime.outer_level, 0)
    raw = _gaia_table(
        [
            (101, float(coord.ra.degree), float(coord.dec.degree), 12.0, 0.0, 0.0),
            (202, np.nan, 0.0, 13.0, 0.0, 0.0),
        ]
    )
    store = RuntimeGaiaHealpixStore(
        _RawGaiaStore(tmp_path, {0: raw}),
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )

    inner = build_base_inner_table(store, runtime, 0)

    assert int(np.sum(inner["star_count"])) == 1
    assert int(np.sum(inner["ngs_count"])) == 1


def test_build_base_inner_table_initializes_winners_to_seeing_baseline(tmp_path: Path) -> None:
    runtime = replace(
        _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg),
        seeing_reference_ee=0.02,
    )

    inner = build_base_inner_table(_FakeStore(), runtime, 0)
    baseline = predict_service.get_seeing_baseline_performance(runtime)

    assert np.allclose(inner["best_ee"], baseline.ee)
    assert np.allclose(inner["winner_ee_resolved"], baseline.ee)
    assert np.allclose(inner["winner_ee_averaged"], baseline.ee)
    assert np.all(np.asarray(inner["winner_asterism_id"], dtype=np.int64) == -1)


def test_regularized_winner_resolved_ee_keeps_seeing_floor() -> None:
    inner = Table()
    inner["winner_asterism_id"] = np.asarray([-1, -1], dtype=np.int64)
    inner["winner_ee_resolved"] = np.asarray([0.5, 0.5], dtype=np.float64)

    _fill_regularized_winner_fields(
        inner,
        labels=np.asarray([10, 20], dtype=np.int64),
        top_refs=np.asarray([[10], [20]], dtype=np.int64),
        top_ee=np.asarray([[0.3], [0.8]], dtype=np.float64),
        top_pointing_x=np.asarray([[1.0], [2.0]], dtype=np.float64),
        top_pointing_y=np.asarray([[3.0], [4.0]], dtype=np.float64),
        retained_candidate_ids=np.asarray([10, 20], dtype=np.int64),
    )

    assert np.asarray(inner["winner_asterism_id"], dtype=np.int64).tolist() == [1, 2]
    assert np.asarray(inner["winner_ee_resolved"], dtype=np.float64).tolist() == pytest.approx(
        [0.5, 0.8]
    )


def test_build_base_inner_table_uses_runtime_gaia_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    raw = _gaia_table([(101, 10.0, 0.0, 12.0, 100.0, 50.0)])
    store = RuntimeGaiaHealpixStore(
        _RawGaiaStore(tmp_path, {0: raw}),
        runtime,
        max_entries=4,
        max_bytes=1024 * 1024,
    )

    def fake_runtime_get_pixel_from_skycoord(level, skycoord):
        assert level == 14
        return np.zeros(len(skycoord), dtype=np.int64)

    def fail_traversal_projection(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("inner counts should use runtime hpx14")

    monkeypatch.setattr(
        "ao_sky.build.runtime_gaia.get_pixel_from_skycoord",
        fake_runtime_get_pixel_from_skycoord,
    )
    monkeypatch.setattr(
        "ao_sky.build.traversal.get_pixel_from_skycoord",
        fail_traversal_projection,
    )

    inner = build_base_inner_table(store, runtime, 0)
    expected_pix = int(get_parent_pixel(14, np.asarray([0]), runtime.inner_level)[0])
    expected_row = int(np.flatnonzero(np.asarray(inner["pix"], dtype=np.int64) == expected_pix)[0])

    assert int(np.sum(inner["star_count"])) == 1
    assert int(inner["star_count"][expected_row]) == 1


def test_traversal_geometry_does_not_retain_inner_pixel_geometry(tmp_path: Path) -> None:
    runtime = _make_predict_runtime(model_root=tmp_path / "models", fov=30.0 * u.deg)
    geometry = TraversalGeometry.from_runtime(runtime)

    assert np.array_equal(geometry.inner_pixs(0), geometry.inner_pixs(0))
    assert geometry.inner_centres(0) is not geometry.inner_centres(0)


def test_model_cache_key_includes_model_root_and_model_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()
    loaded: list[tuple[Path, str, bool]] = []

    def fake_load_model(
        model_root: Path,
        model_name: str,
        force_cpu: bool = True,
    ) -> object:
        loaded.append((Path(model_root), model_name, bool(force_cpu)))
        return f"{Path(model_root).name}:{model_name}"

    monkeypatch.setattr(predict_service.backend, "load_model", fake_load_model)
    runtime_a = _make_predict_runtime(model_root=tmp_path / "models-a")
    runtime_b = _make_predict_runtime(model_root=tmp_path / "models-b")
    runtime_c = _make_predict_runtime(
        model_root=tmp_path / "models-a",
        point_model="point-c.pt",
    )

    assert predict_service.get_point_model(runtime_a, 2) == "models-a:point-a.pt"
    assert predict_service.get_point_model(runtime_b, 2) == "models-b:point-a.pt"
    assert predict_service.get_point_model(runtime_c, 2) == "models-a:point-c.pt"
    assert predict_service.get_point_model(runtime_a, 2) == "models-a:point-a.pt"

    assert loaded == [
        (tmp_path / "models-a", "point-a.pt", True),
        (tmp_path / "models-b", "point-a.pt", True),
        (tmp_path / "models-a", "point-c.pt", True),
    ]


def test_model_cache_key_includes_backend_device_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()
    loaded: list[bool] = []

    def fake_load_model(
        model_root: Path,
        model_name: str,
        force_cpu: bool = True,
    ) -> object:
        loaded.append(bool(force_cpu))
        return f"force_cpu={bool(force_cpu)}"

    monkeypatch.setattr(predict_service.backend, "load_model", fake_load_model)
    runtime = _make_predict_runtime(model_root=tmp_path / "models")

    assert predict_service.get_point_model(runtime, 2, device="cpu") == "force_cpu=True"

    assert predict_service.get_point_model(runtime, 2, device="gpu") == "force_cpu=False"

    assert loaded == [True, False]


def test_mean_model_stays_on_cpu_when_resolved_allows_auto_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()
    loaded: list[bool] = []

    def fake_load_model(
        model_root: Path,
        model_name: str,
        force_cpu: bool = True,
    ) -> object:
        loaded.append(bool(force_cpu))
        return f"force_cpu={bool(force_cpu)}"

    monkeypatch.setattr(predict_service.backend, "load_model", fake_load_model)
    runtime = _make_predict_runtime(model_root=tmp_path / "models")

    assert predict_service.get_mean_model(runtime, 2) == "force_cpu=True"
    assert loaded == [True]


def test_mean_model_can_use_auto_device_when_explicitly_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()
    loaded: list[bool] = []

    def fake_load_model(
        model_root: Path,
        model_name: str,
        force_cpu: bool = True,
    ) -> object:
        loaded.append(bool(force_cpu))
        return f"force_cpu={bool(force_cpu)}"

    monkeypatch.setattr(predict_service.backend, "load_model", fake_load_model)
    runtime = _make_predict_runtime(model_root=tmp_path / "models")

    assert predict_service.get_mean_model(runtime, 2, device="gpu") == "force_cpu=False"
    assert loaded == [False]


def test_model_device_policy_rejects_unknown_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()

    with pytest.raises(PredictError, match="device"):
        predict_service.get_point_model(
            _make_predict_runtime(model_root=tmp_path / "models"),
            2,
            device="mps",
        )


def test_mean_model_validates_averaged_device_policy_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()

    with pytest.raises(PredictError, match="device"):
        predict_service.get_mean_model(
            _make_predict_runtime(model_root=tmp_path / "models"),
            2,
            device="mps",
        )


def test_configure_inference_threads_sets_backend_thread_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        predict_service.backend,
        "configure_inference_threads",
        lambda num_threads: calls.append(num_threads),
    )

    predict_service.configure_inference_threads(1)

    assert calls == [1]


@pytest.mark.parametrize(
    ("cache_clear_every", "expected_clear_count"),
    [(1, 4), (0, 1)],
)
def test_build_traversal_products_retains_regularized_winners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_clear_every: int,
    expected_clear_count: int,
) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    definition = BuildDefinition(
        lineage_name="baseline",
        gaia_release="dr3",
        outer_level=1,
        inner_level=2,
        max_data_level=2,
    )
    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy,
        model_root=tmp_path / "models",
    )
    geometry = TraversalGeometry.from_runtime(runtime)
    inner_pixs = geometry.inner_pixs(0)
    first_centre = get_pixel_skycoord(runtime.inner_level, int(inner_pixs[0]))
    star1 = first_centre.directional_offset_by(90.0 * u.deg, 10.0 * u.arcsec)
    star2 = first_centre.directional_offset_by(270.0 * u.deg, 10.0 * u.arcsec)
    stars = Table()
    stars["source_id"] = np.array([101, 102], dtype=np.int64)
    stars["ra"] = np.array([star1.ra.deg, star2.ra.deg], dtype=np.float64)
    stars["dec"] = np.array([star1.dec.deg, star2.dec.deg], dtype=np.float64)
    stars["R"] = np.array([10.0, 10.1], dtype=np.float64)
    stars["pmra"] = np.zeros(2, dtype=np.float64)
    stars["pmdec"] = np.zeros(2, dtype=np.float64)
    stars["ref_epoch"] = np.full(2, 2016.0, dtype=np.float64)
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "ao_sky.build.traversal.prepare_search_inputs",
        lambda store, runtime, outer_pix, **kwargs: (stars, stars),
    )

    def fake_build_base_inner_table(
        store,
        runtime,
        outer_pix,
        *,
        geometry=None,
        asterisms=None,
    ):
        pixs = geometry.inner_pixs(outer_pix)
        size = len(pixs)
        return Table(
            [
                np.asarray(pixs, dtype=np.int64),
                np.zeros(size, dtype=np.int64),
                np.zeros(size, dtype=np.int64),
                np.full(size, np.nan, dtype=np.float64),
                np.full(size, np.nan, dtype=np.float64),
                np.full(size, np.nan, dtype=np.float64),
                np.full(size, -1, dtype=np.int64),
                np.full(size, np.nan, dtype=np.float64),
                np.full(size, np.nan, dtype=np.float64),
                np.zeros(size, dtype=np.bool_),
                np.zeros(size, dtype=np.bool_),
            ],
            names=(
                "pix",
                "star_count",
                "ngs_count",
                "best_ee",
                "best_sr",
                "best_fwhm",
                "winner_asterism_id",
                "winner_ee_resolved",
                "winner_ee_averaged",
                "coverage_resolved",
                "coverage_averaged",
            ),
        )

    monkeypatch.setattr("ao_sky.build.traversal.build_base_inner_table", fake_build_base_inner_table)
    monkeypatch.setattr(
        "ao_sky.build.traversal.get_point_model",
        lambda runtime, num_stars, **kwargs: object(),
    )
    monkeypatch.setattr(
        "ao_sky.build.traversal.get_mean_model",
        lambda runtime, num_stars, **kwargs: object(),
    )

    def fake_predict_point_arrays(
        runtime,
        num_stars,
        model,
        ngs_zd,
        ngs_az_deg,
        ngs_mag,
        backend_row_count=None,
        feature_buffer_row_count=None,
        prediction_telemetry=None,
    ):
        seen["resolved_num_stars"] = num_stars
        seen["resolved_payload_count"] = len(ngs_zd)
        return PointPredictionBatch(
            sr=np.full(len(ngs_zd), 0.5, dtype=np.float64),
            ee=np.full(len(ngs_zd), 0.7, dtype=np.float64),
            fwhm=np.full(len(ngs_zd), 100.0, dtype=np.float64),
            ee_angle=np.zeros(len(ngs_zd), dtype=np.float64),
        )

    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_point_arrays",
        fake_predict_point_arrays,
    )

    def fake_predict_field_mean_arrays(
        runtime,
        num_stars,
        model,
        ngs_zd,
        ngs_az_deg,
        ngs_mag,
        backend_row_count=None,
        feature_buffer_row_count=None,
        prediction_telemetry=None,
    ):
        seen["averaged_num_stars"] = num_stars
        seen["averaged_payload_count"] = len(ngs_zd)
        return np.full(len(ngs_zd), 0.6, dtype=np.float64)

    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_field_mean_arrays",
        fake_predict_field_mean_arrays,
    )
    monkeypatch.setattr("ao_sky.build.traversal.add_gaia_a0_to_inner", lambda inner, **kwargs: inner)
    cleared_backend_cache: list[None] = []
    monkeypatch.setattr(
        "ao_sky.build.traversal.clear_backend_cache",
        lambda: cleared_backend_cache.append(None),
    )

    structure_profile = TraversalStructureProfile()
    asterisms, result_inner = build_traversal_products(
        _FakeStore(),
        runtime,
        0,
        execution_config=TraversalExecutionConfig(
            resolved_cache_clear_every=cache_clear_every,
        ),
        dust_root=tmp_path / "dust",
        max_data_level=2,
        structure_profile=structure_profile,
    )
    structure_stats = structure_profile.to_stats()

    assert seen == {
        "resolved_num_stars": 2,
        "resolved_payload_count": 1,
        "averaged_num_stars": 2,
        "averaged_payload_count": 1,
    }
    assert len(asterisms) == 1
    assert int(asterisms["asterism_id"][0]) == 1
    assert int(asterisms["num_stars"][0]) == 2
    assert sorted(
        [
            int(asterisms["star1_source_id"][0]),
            int(asterisms["star2_source_id"][0]),
        ]
    ) == [101, 102]
    assert float(result_inner["best_ee"][0]) == pytest.approx(0.7)
    assert float(result_inner["winner_ee_resolved"][0]) == pytest.approx(0.7)
    assert float(result_inner["winner_ee_averaged"][0]) == pytest.approx(0.6)
    assert int(result_inner["winner_asterism_id"][0]) == 1
    assert bool(result_inner["coverage_resolved"][0])
    assert bool(result_inner["coverage_averaged"][0])
    assert np.all(np.asarray(result_inner["winner_asterism_id"][1:], dtype=np.int64) == -1)
    assert structure_stats.raw_asterism_rows_peak == 1
    assert structure_stats.winner_payload_rows_peak == 1
    assert len(cleared_backend_cache) == expected_clear_count
