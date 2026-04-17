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
    TraversalGeometry,
    _filter_neighbours_by_galactic_latitude,
    build_base_inner_table,
    prepare_search_inputs,
    _update_inner_pixel_asterism_field_mean,
    _update_inner_pixel_asterism_performance,
    build_traversal_products,
)
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore
from ao_sky.gaia._constants import GAIA_SCHEMA_COLUMNS
from ao_sky.predict import PredictError
from ao_sky.predict import backend as predict_backend
from ao_sky.predict import service as predict_service
from ao_sky.predict._models import AOSystemRuntime, PointPredictionBatch, PredictRuntime
from ao_sky.spatial import get_parent_pixel, get_pixel_skycoord


def _write_legacy_config(
    path: Path,
    *,
    min_galactic_latitude: float | None = None,
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
    if min_galactic_latitude is not None:
        text += f"\nasterisms_min_galactic_latitude: {float(min_galactic_latitude)}"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _write_build_definition(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "build": {},
                "ao_system": {
                    "band": "R",
                    "fov_arcsec": 120.0,
                    "lgs": [],
                    "min_wfs": 2,
                    "max_wfs": 3,
                    "min_mag": 8.0,
                    "max_mag": 18.5,
                    "min_sep_arcsec": 5.0,
                },
                "prediction": {
                    "wavelength_micron": 1.654,
                    "resolved_models": {
                        "2star": "point_two",
                        "3star": "point_three",
                    },
                    "averaged_models": {
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
                    "min_galactic_latitude_deg": None,
                    "max_star_density": 6.0,
                    "max_bright_star_mag": 8.0,
                },
                "maps": {"max_level": 1},
                "asterism": {"max_overlap": 0.66},
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
        min_galactic_latitude=(
            None
            if raw.get("asterisms_min_galactic_latitude") is None
            else float(raw["asterisms_min_galactic_latitude"])
        ),
        max_star_density=float(raw.get("asterisms_max_star_density", 2.0)),
        max_bright_star_mag=(
            None
            if raw.get("asterisms_max_bright_star_mag") is None
            else float(raw["asterisms_max_bright_star_mag"])
        ),
        max_overlap=(
            None
            if raw.get("asterisms_max_overlap") is None
            else float(raw["asterisms_max_overlap"])
        ),
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
    assert runtime.max_star_density == pytest.approx(2.0)
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
    assert loaded.max_star_density == pytest.approx(runtime.max_star_density)
    assert loaded.max_overlap == pytest.approx(runtime.max_overlap)
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
    min_galactic_latitude: float | None = None,
) -> PredictRuntime:
    ao_system = AOSystemRuntime(
        band="R",
        fov=fov,
        lgs=(),
        min_wfs=2,
        max_wfs=2,
        min_mag=8.0,
        max_mag=18.5,
        min_sep=5.0 * u.arcsec,
    )
    return PredictRuntime(
        ao_system=ao_system,
        outer_level=0,
        inner_level=1,
        epoch=2028.0,
        min_galactic_latitude=min_galactic_latitude,
        max_star_density=None,
        max_bright_star_mag=None,
        max_overlap=None,
        prediction_wavelength=1.654 * u.micron,
        resolved_models={"2star": point_model},
        averaged_models={"2star": mean_model},
        seeing_reference_wavelength=0.5 * u.micron,
        seeing_reference_sr=0.0,
        seeing_reference_ee=0.0,
        seeing_reference_fwhm=0.0,
        coverage_ee_threshold_resolved=0.25,
        coverage_ee_threshold_averaged=0.25,
        model_root=model_root,
    )


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


def test_neighbour_latitude_filter_uses_outer_pixel_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_predict_runtime(
        model_root=tmp_path / "models",
        fov=30.0 * u.deg,
        min_galactic_latitude=10.0,
    )
    geometry = TraversalGeometry.from_runtime(runtime)
    stars = Table(
        [
            np.array([101, 202], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        ],
        names=("source_id", "source_outer_pix"),
    )
    seen_levels: list[int] = []

    def fake_get_pixel_skycoord(level: int, pix: int):
        seen_levels.append(int(level))
        assert level == runtime.outer_level

        class _B:
            degree = 0.0 if pix == 1 else 90.0

        class _Galactic:
            b = _B()

        class _Coord:
            galactic = _Galactic()

        return _Coord()

    monkeypatch.setattr(
        "ao_sky.build.traversal.get_pixel_skycoord",
        fake_get_pixel_skycoord,
    )

    filtered = _filter_neighbours_by_galactic_latitude(
        stars,
        outer_pix=0,
        geometry=geometry,
    )

    assert filtered["source_id"].tolist() == [101]
    assert seen_levels == [runtime.outer_level]


def test_model_cache_key_includes_model_root_and_model_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict_service._POINT_MODEL_CACHE.clear()
    predict_service._MEAN_MODEL_CACHE.clear()
    loaded: list[tuple[Path, str]] = []

    def fake_load_model(model_root: Path, model_name: str, force_cpu: bool = True) -> object:
        loaded.append((Path(model_root), model_name))
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
        (tmp_path / "models-a", "point-a.pt"),
        (tmp_path / "models-b", "point-a.pt"),
        (tmp_path / "models-a", "point-c.pt"),
    ]


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


def test_winner_fields_remain_local_when_neighbour_has_better_best_ee(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    winner_idxs, winner_ngs_payloads = _update_inner_pixel_asterism_performance(
        runtime,
        inner,
        context,
    )

    assert float(inner["best_ee"][0]) == pytest.approx(0.8)
    assert float(inner["winner_ee_resolved"][0]) == pytest.approx(0.7)
    assert int(inner["winner_asterism_id"][0]) == 7
    assert float(inner["winner_distance_arcsec"][0]) == pytest.approx(1.0)
    assert int(winner_idxs[0]) == 0
    assert winner_ngs_payloads[0] == [
        {"zd": 1.0, "az": 0.0, "mag": 10.0},
        {"zd": 2.0, "az": 90.0, "mag": 11.0},
    ]


def test_field_mean_reuses_winner_ngs_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = load_native_runtime(
        BuildDefinition(
            lineage_name="baseline",
            gaia_release="dr3",
            outer_level=0,
            inner_level=1,
            max_data_level=1,
        ),
        legacy_config_path=_write_legacy_config(tmp_path / "legacy.yaml"),
        model_root=tmp_path / "models",
    )
    inner = Table(
        [
            np.array([0], dtype=np.int64),
            np.array([np.nan], dtype=np.float64),
        ],
        names=("pix", "winner_ee_averaged"),
    )
    payload = [
        {"zd": 1.0, "az": 0.0, "mag": 10.0},
        {"zd": 2.0, "az": 90.0, "mag": 11.0},
    ]
    winner_payloads = np.empty((1,), dtype=object)
    winner_payloads[0] = payload
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "ao_sky.build.traversal._get_valid_ngs_from_context_pair",
        lambda *args, **kwargs: pytest.fail("winner NGS payload was recomputed"),
    )
    monkeypatch.setattr("ao_sky.build.traversal.get_mean_model", lambda runtime, surviving_stars: object())

    def fake_predict_field_mean_batch(runtime, num_stars, model, ngs):
        seen["num_stars"] = num_stars
        seen["ngs"] = ngs
        return np.array([0.42], dtype=np.float64)

    monkeypatch.setattr(
        "ao_sky.build.traversal.predict_field_mean_batch",
        fake_predict_field_mean_batch,
    )
    monkeypatch.setattr("ao_sky.build.traversal.clear_backend_cache", lambda: None)

    _update_inner_pixel_asterism_field_mean(
        runtime,
        inner,
        context=object(),
        winner_asterism_idxs=np.array([0], dtype=np.int64),
        winner_ngs_payloads=winner_payloads,
    )

    assert float(inner["winner_ee_averaged"][0]) == pytest.approx(0.42)
    assert seen == {
        "num_stars": 2,
        "ngs": [payload],
    }


def test_build_traversal_products_uses_expanded_asterisms_for_inner_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    local_coord = get_pixel_skycoord(runtime.outer_level, 0)
    nonlocal_coord = get_pixel_skycoord(runtime.outer_level, 1)
    expanded_asterisms = Table()
    expanded_asterisms["asterism_id"] = np.array([10, 20], dtype=np.int64)
    expanded_asterisms["ra"] = np.array(
        [local_coord.ra.deg, nonlocal_coord.ra.deg],
        dtype=np.float64,
    )
    expanded_asterisms["dec"] = np.array(
        [local_coord.dec.deg, nonlocal_coord.dec.deg],
        dtype=np.float64,
    )
    expanded_asterisms["num_stars"] = np.array([2, 2], dtype=np.int64)
    expanded_asterisms["pix"] = np.array([0, 4], dtype=np.int64)
    for index in (1, 2, 3):
        expanded_asterisms[f"star{index}_source_id"] = np.array(
            [index, index + 10],
            dtype=np.int64,
        )
        expanded_asterisms[f"star{index}_ra"] = np.array(
            [local_coord.ra.deg, nonlocal_coord.ra.deg],
            dtype=np.float64,
        )
        expanded_asterisms[f"star{index}_dec"] = np.array(
            [local_coord.dec.deg, nonlocal_coord.dec.deg],
            dtype=np.float64,
        )
        expanded_asterisms[f"star{index}_mag"] = np.array([10.0, 10.5], dtype=np.float64)

    inner = Table(
        [
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
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
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        "ao_sky.build.traversal._build_expanded_outer_pixel_asterisms",
        lambda store, runtime, outer_pix, **kwargs: (
            _empty_gaia_table(),
            _empty_gaia_table(),
            expanded_asterisms,
            get_pixel_skycoord(runtime.inner_level, np.asarray(expanded_asterisms["pix"])),
        ),
    )

    def fake_build_base_inner_table(
        store,
        runtime,
        outer_pix,
        *,
        geometry=None,
        asterisms=None,
    ):
        seen["persisted_candidate_ids"] = np.asarray(
            asterisms["asterism_id"],
            dtype=np.int64,
        ).tolist()
        return inner.copy(copy_data=True)

    monkeypatch.setattr("ao_sky.build.traversal.build_base_inner_table", fake_build_base_inner_table)
    monkeypatch.setattr(
        "ao_sky.build.traversal.search_around_sky",
        lambda *args, **kwargs: (
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            None,
            None,
        ),
    )

    def fake_update_performance(runtime, inner, context):
        seen["context_candidate_ids"] = np.asarray(
            context.asterisms["asterism_id"],
            dtype=np.int64,
        ).tolist()
        seen["local_asterism_mask"] = context.local_asterism_mask.tolist()
        inner["best_ee"][0] = 0.8
        winner_payloads = np.empty((1,), dtype=object)
        winner_payloads[:] = None
        return (
            np.array([-1], dtype=np.int64),
            winner_payloads,
        )

    monkeypatch.setattr(
        "ao_sky.build.traversal._update_inner_pixel_asterism_performance",
        fake_update_performance,
    )
    monkeypatch.setattr("ao_sky.build.traversal.add_gaia_a0_to_inner", lambda inner, **kwargs: inner)

    asterisms, result_inner = build_traversal_products(
        _FakeStore(),
        runtime,
        0,
        dust_root=tmp_path / "dust",
        max_data_level=2,
    )

    assert seen["persisted_candidate_ids"] == [10]
    assert seen["context_candidate_ids"] == [10, 20]
    assert seen["local_asterism_mask"] == [True, False]
    assert asterisms["asterism_id"].tolist() == [10]
    assert float(result_inner["best_ee"][0]) == pytest.approx(0.8)


def test_build_traversal_products_skip_path_still_injects_dust(tmp_path: Path) -> None:
    legacy = _write_legacy_config(tmp_path / "legacy.yaml", min_galactic_latitude=90.0)
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
