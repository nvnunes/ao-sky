"""Build-model tests."""

from __future__ import annotations

import io
import json
from pathlib import Path
from gzip import open as gzip_open

import h5py
import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.table import Table
from mocpy import MOC

from ao_sky._hdf5 import HDF5_BLOSC_FILTER_ID, HDF5_BLOSC_LEVEL
from ao_sky.build import (
    check_runtime_roots,
    fetch_gaia_data,
    init_build as real_init_build,
    restart_build,
    run_build,
    run_build_outer_pixels,
    show_build,
)
import ao_sky.build.aggregation as aggregation_module
from ao_sky.build.augmentation import build_survey_extent_layers
from ao_sky.build.aggregation import aggregate_maps, build_maps
from ao_sky.build._constants import (
    BUILD_FILENAME,
    BUILD_PHASE_AGGREGATION,
    BUILD_PHASE_AUGMENTATION,
    BUILD_PHASE_TRAVERSAL,
    MAPS_DATASET,
    MAPS_DTYPE,
    OUTER_DATASET_ASTERISMS,
    OUTER_DATASET_INNER,
    STATE_DTYPE,
    SURVEY_EXTENT_DATASET,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ao_sky.build._exceptions import BuildError
from ao_sky.build.config import (
    load_build_definition as load_build_definition_yaml,
    resolve_gaia_root_only,
    resolve_runtime_root_candidates,
    resolve_traversal_execution_config,
)
from ao_sky.build.control import (
    load_build_definition,
    load_build_roots,
    load_runtime_config_path,
    load_runtime_config_source_path,
    load_state,
    maps_artifact_filename,
    outer_artifact_filename,
    set_current_phase,
    summarize_build,
    update_state_row,
)
from ao_sky.build.runtime_config import load_runtime_config, runtime_to_config, write_runtime_config
from ao_sky.build.model_snapshot import fetch_model_data
from ao_sky.build.survey_snapshot import fetch_survey_data
from ao_sky.build.scheduler import OuterPixelScheduler
from ao_sky.build.regional import (
    build_dynamic_work_batches,
    build_regional_worker_plans,
    order_region_outer_pixs,
)
from ao_sky.build.artifacts import (
    write_outer_artifact,
    write_outer_artifact_profiled,
)
from ao_sky.build._models import (
    BuildDefinition,
    TraversalExecutionConfig,
    TraversalTaskResult,
)
from ao_sky.dust import gaia_tge_a0_cache_filename, prepare_gaia_tge_a0_cache
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore
from ao_sky.spatial import get_pixel_skycoord
from ao_sky.predict import AOSystemRuntime, PredictRuntime


@pytest.fixture(autouse=True)
def _disable_runner_model_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ao_sky.build.runner.warm_model_cache",
        lambda runtime, **kwargs: None,
    )


def _write_build_definition(
    path: Path,
    *,
    outer_level: int = 0,
    inner_level: int = 1,
    max_data_level: int = 1,
    survey_overlays: list[dict[str, object]] | None = None,
) -> Path:
    payload: dict[str, object] = {
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
            "outer_level": outer_level,
            "inner_level": inner_level,
        },
        "gaia": {
            "release": "dr3",
            "epoch": 2028.0,
            "max_bright_star_mag": 8.0,
            "max_bright_star_exclusion_arcsec": 240.0,
        },
        "maps": {"max_level": max_data_level},
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
    }
    if survey_overlays is not None:
        payload["survey_overlays"] = survey_overlays
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


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
asterisms_max_bright_star_mag: 8.0
asterisms_max_overlap: 0.66
coverage_ee_threshold_resolved: 0.4
coverage_ee_threshold_mean: 0.3
""".strip()
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _write_legacy_config_with_models(path: Path) -> Path:
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
    point_models:
      2star: point_two
      3star: shared_three
    models:
      2star: mean_two
      3star: shared_three
asterisms_max_bright_star_mag: 8.0
asterisms_max_overlap: 0.66
coverage_ee_threshold_resolved: 0.4
coverage_ee_threshold_mean: 0.3
""".strip()
        + "\n",
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
        lgs=(),
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
            str(key): str(value)
            for key, value in (system.get("models") or {}).items()
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


def _write_model_bundle(
    model_root: Path,
    model_name: str,
    *,
    payload: bytes | None = None,
) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    content = payload if payload is not None else model_name.encode("utf-8")
    (model_root / f"{model_name}.pt").write_bytes(content + b":pt")
    (model_root / f"{model_name}_metadata.pkl").write_bytes(content + b":metadata")


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
    star_counts: list[int] | None = None,
    loaded: bool = True,
) -> Path:
    num_pixels = 12 * (4 ** outer_level)
    counts = np.zeros(num_pixels, dtype=np.int64)
    if star_counts is not None:
        counts[: len(star_counts)] = np.asarray(star_counts, dtype=np.int64)
    summary = np.zeros(
        num_pixels,
        dtype=[("outer_pix", "<i8"), ("star_count", "<i8"), ("loaded", "?")],
    )
    summary["outer_pix"] = np.arange(num_pixels, dtype=np.int64)
    summary["star_count"] = counts
    summary["loaded"] = loaded
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
    survey_root: Path | None = None,
    aosky_yaml: Path | None = None,
    ensure_model_files: bool = True,
) -> Path:
    definition, _ = load_build_definition_yaml(definition_filename)
    effective_model_root = model_root or Path(legacy_config_path).parent / "models"
    runtime = load_native_runtime(
        definition,
        legacy_config_path=legacy_config_path,
        model_root=effective_model_root,
    )
    if ensure_model_files:
        for model_name in set(runtime.resolved_models.values()) | set(
            runtime.averaged_models.values()
        ):
            _write_required_model_files(effective_model_root, model_name)
    build_config = yaml.safe_load(Path(definition_filename).read_text(encoding="utf-8"))
    payload = runtime_to_config(runtime)
    payload["build"] = {**payload.get("build", {}), **build_config.get("build", {})}
    payload["gaia"]["release"] = build_config["gaia"]["release"]
    payload["maps"] = build_config["maps"]
    if "survey_overlays" in build_config:
        payload["survey_overlays"] = build_config["survey_overlays"]
    Path(definition_filename).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    resolved_roots = resolve_runtime_root_candidates(
        gaia_root=gaia_root,
        build_root=build_root,
        model_root=effective_model_root,
        aosky_yaml=aosky_yaml,
    )
    resolved_gaia_root = resolved_roots["gaia_root"]
    if resolved_gaia_root is not None:
        _write_gaia_summary(
            resolved_gaia_root,
            release=definition.gaia_release,
            outer_level=definition.outer_level,
        )
        _write_gaia_tge_map(
            resolved_gaia_root,
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
        survey_root=survey_root,
        aosky_yaml=aosky_yaml,
    )


def _init_model_snapshot_build(
    tmp_path: Path,
    model_root: Path | None = None,
    *,
    ensure_model_files: bool = True,
) -> Path:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config_with_models(tmp_path / "legacy.yaml")
    return init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        model_root=model_root or tmp_path / "source-models",
        legacy_config_path=legacy,
        ensure_model_files=ensure_model_files,
    )


def _make_scheduler_state(statuses: list[int], *, field: str = "traversal_status") -> np.ndarray:
    state = np.zeros(len(statuses), dtype=STATE_DTYPE)
    state["outer_pix"] = np.arange(len(statuses), dtype=np.int64)
    state[field] = np.asarray(statuses, dtype=np.int16)
    return state


def _make_inner(
    *,
    pixs: list[int] | None = None,
    gaia_a0: float = np.nan,
    star_count: int = 0,
    ngs_count: int = 0,
    best_ee: float = np.nan,
    best_sr: float = np.nan,
    best_fwhm: float = np.nan,
    winner_asterism_id: int = -1,
    winner_ee_resolved: float = np.nan,
    winner_ee_averaged: float = np.nan,
    coverage_resolved: bool = False,
    coverage_averaged: bool = False,
) -> Table:
    pix_values = np.asarray([0, 1, 2, 3] if pixs is None else pixs, dtype=np.int64)
    nrows = len(pix_values)
    return Table(
        [
            pix_values,
            np.full(nrows, gaia_a0, dtype=np.float64),
            np.full(nrows, star_count, dtype=np.int64),
            np.full(nrows, ngs_count, dtype=np.int64),
            np.full(nrows, best_ee, dtype=np.float64),
            np.full(nrows, best_sr, dtype=np.float64),
            np.full(nrows, best_fwhm, dtype=np.float64),
            np.full(nrows, winner_asterism_id, dtype=np.int64),
            np.full(nrows, winner_ee_resolved, dtype=np.float64),
            np.full(nrows, winner_ee_averaged, dtype=np.float64),
            np.full(nrows, coverage_resolved, dtype=np.bool_),
            np.full(nrows, coverage_averaged, dtype=np.bool_),
        ],
        names=(
            "pix",
            "gaia_A0",
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


def _write_moc(path: Path, *, level: int, pixs: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    moc = MOC.from_healpix_cells(
        np.asarray(pixs, dtype=np.uint64),
        level,
        max_depth=level,
    )
    moc.save(str(path), format="fits", overwrite=True)
    return path


def _make_asterisms(
    *,
    empty: bool = False,
    centers: list[tuple[float, float]] | None = None,
    pixs: list[int] | None = None,
) -> Table:
    if empty:
        return Table(
            [
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            ],
            names=(
                "asterism_id",
                "ra",
                "dec",
                "num_stars",
                "pix",
                "star1_source_id",
                "star1_ra",
                "star1_dec",
                "star1_mag",
                "star2_source_id",
                "star2_ra",
                "star2_dec",
                "star2_mag",
                "star3_source_id",
                "star3_ra",
                "star3_dec",
                "star3_mag",
            ),
        )

    if centers is None:
        centers = [(10.0, 20.0)]
    nrows = len(centers)
    ra_values = np.asarray([ra for ra, _ in centers], dtype=np.float64)
    dec_values = np.asarray([dec for _, dec in centers], dtype=np.float64)
    pix_values = (
        np.arange(nrows, dtype=np.int64)
        if pixs is None
        else np.asarray(pixs, dtype=np.int64)
    )
    return Table(
        [
            np.arange(7, 7 + nrows, dtype=np.int64),
            ra_values,
            dec_values,
            np.full(nrows, 2, dtype=np.int64),
            pix_values,
            np.arange(101, 101 + nrows, dtype=np.int64),
            ra_values,
            dec_values,
            np.full(nrows, 12.0, dtype=np.float64),
            np.arange(202, 202 + nrows, dtype=np.int64),
            ra_values + 0.1,
            dec_values + 0.1,
            np.full(nrows, 13.0, dtype=np.float64),
            np.full(nrows, -1, dtype=np.int64),
            np.full(nrows, -1.0, dtype=np.float64),
            np.full(nrows, -1.0, dtype=np.float64),
            np.full(nrows, -1.0, dtype=np.float64),
        ],
        names=(
            "asterism_id",
            "ra",
            "dec",
            "num_stars",
            "pix",
            "star1_source_id",
            "star1_ra",
            "star1_dec",
            "star1_mag",
            "star2_source_id",
            "star2_ra",
            "star2_dec",
            "star2_mag",
            "star3_source_id",
            "star3_ra",
            "star3_dec",
            "star3_mag",
        ),
    )


def test_init_build_creates_root_and_full_sky_state(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    assert build_path.name == "v1"
    assert (build_path / BUILD_FILENAME).is_file()
    assert (build_path / "build.log").is_file()
    assert (build_path / "build.yaml").is_file()
    assert (build_path / "hpx0-1").is_dir()

    state = load_state(build_path)
    assert len(state) == 12
    assert state["outer_pix"].tolist() == list(range(12))
    assert np.all(state["gaia_loading_status"] == WORK_STATUS_DONE)
    assert np.all(state["traversal_status"] == WORK_STATUS_PENDING)
    assert load_build_definition(build_path).max_data_level == 1

    summary = summarize_build(build_path)
    assert summary["build_status"] == "initialized"
    assert summary["current_phase"] == BUILD_PHASE_TRAVERSAL

    runtime_config_path = load_runtime_config_path(build_path)
    runtime = load_runtime_config(
        runtime_config_path,
        model_root=load_build_roots(build_path).model_root,
    )
    runtime_payload = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
    assert runtime_config_path == (build_path / "build.yaml").resolve()
    assert load_runtime_config_source_path(build_path) == definition.resolve()
    assert "source" not in runtime_payload
    assert "build" not in runtime_payload
    assert runtime.ao_system.band == "R"


def test_init_build_uses_ao_sky_yaml_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "ao-sky.yaml").write_text(
        (
            "build:\n"
            "  roots:\n"
            f"    gaia: {tmp_path / 'gaia'}\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)

    build_path = init_build(
        definition_filename=definition,
        gaia_root=None,
        build_root=None,
        dust_root=None,
        legacy_config_path=legacy,
    )

    assert build_path.parent == project_root.resolve()
    roots = load_build_roots(build_path)
    assert roots.dust_root == (build_path / "dust").resolve()
    assert gaia_tge_a0_cache_filename(roots.dust_root, 1).is_file()


def test_resolve_gaia_root_only_uses_ao_sky_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    expected = tmp_path / "gaia"
    (project_root / "ao-sky.yaml").write_text(
        f"build:\n  roots:\n    gaia: {expected}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)

    assert resolve_gaia_root_only(gaia_root=None) == expected.resolve()


def test_resolve_gaia_root_only_requires_cli_or_conf(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="gaia_root must be provided"):
        resolve_gaia_root_only(gaia_root=None, cwd=tmp_path)


def test_check_runtime_roots_reports_ok_and_missing(tmp_path: Path) -> None:
    gaia_root = tmp_path / "gaia"
    build_root = tmp_path / "builds"
    model_root = tmp_path / "models"
    gaia_root.mkdir()
    build_root.mkdir()
    model_root.mkdir()
    _write_gaia_tge_map(gaia_root, [(healpix_id, 1, 0.5) for healpix_id in range(48)])

    ok, report = check_runtime_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        model_root=model_root,
    )

    assert ok
    assert report.splitlines() == [
        f"gaia_root: OK {gaia_root.resolve()}",
        f"build_root: OK {build_root.resolve()}",
        f"dust_root: OK {gaia_root.resolve()}",
        f"model_root: OK {model_root.resolve()}",
    ]

    missing_gaia_root = tmp_path / "missing-gaia"
    ok, report = check_runtime_roots(
        gaia_root=missing_gaia_root,
        build_root=build_root,
        model_root=None,
    )

    assert not ok
    assert f"gaia_root: MISSING {missing_gaia_root.resolve()}" in report
    assert f"dust_root: MISSING {missing_gaia_root.resolve()}" in report
    assert "model_root: MISSING <unset>" in report

    empty_dust_root = tmp_path / "empty-dust"
    empty_dust_root.mkdir()
    ok, report = check_runtime_roots(
        gaia_root=empty_dust_root,
        build_root=build_root,
        model_root=model_root,
    )
    assert not ok
    assert f"dust_root: MISSING {empty_dust_root.resolve()}" in report

    file_root = tmp_path / "not-a-directory"
    file_root.write_text("not a root\n", encoding="utf-8")
    ok, report = check_runtime_roots(
        gaia_root=file_root,
        build_root=build_root,
        model_root=model_root,
    )
    assert not ok
    assert f"gaia_root: MISSING {file_root.resolve()}" in report


def test_fetch_gaia_data_resolves_gaia_root_from_ao_sky_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    expected = tmp_path / "gaia" / "gaia-dr3-hpx0" / "summary.h5"
    (project_root / "ao-sky.yaml").write_text(
        "build:\n"
        "  roots:\n"
        f"    gaia: {tmp_path / 'gaia'}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(
        "ao_sky.build.environment.fetch_gaia_store",
        lambda config, force_reload=False, output=None: expected.resolve(),
    )
    monkeypatch.setattr(
        "ao_sky.build.environment.fetch_gaia_tge_dataset",
        lambda dust_root: tmp_path / "gaia" / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz",
    )

    assert (
        fetch_gaia_data(
            gaia_root=None,
            gaia_release="dr3",
            outer_level=0,
        )
        == expected.resolve()
    )


def test_fetch_gaia_data_propagates_force_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(config, force_reload: bool = False, output=None):
        captured["root"] = config.root
        captured["release"] = config.release
        captured["level"] = config.healpix_level
        captured["force_reload"] = force_reload
        captured["output"] = output
        return tmp_path / "gaia" / "gaia-dr3-hpx0" / "summary.h5"

    monkeypatch.setattr("ao_sky.build.environment.fetch_gaia_store", fake_fetch)
    monkeypatch.setattr(
        "ao_sky.build.environment.fetch_gaia_tge_dataset",
        lambda dust_root: tmp_path / "gaia" / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz",
    )

    output = io.StringIO()

    fetch_gaia_data(
        gaia_root=tmp_path / "gaia",
        gaia_release="dr3",
        outer_level=0,
        force=True,
        output=output,
    )

    assert captured == {
        "root": (tmp_path / "gaia").resolve(),
        "release": "dr3",
        "level": 0,
        "force_reload": True,
        "output": output,
    }


def test_resolve_traversal_execution_config_uses_defaults_and_ao_sky_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conf = tmp_path / "ao-sky.yaml"
    conf.write_text(
        "\n".join(
            (
                "build:",
                "  workers: 5",
                "  scheduler: static",
                "  low_latitude_workers: 3",
                "  memory_limit_mb: 12288",
                "gaia_cache_entries: 128",
                "gaia_cache_mb: 4096",
                "prediction:",
                "  resolved_device: gpu",
                "  averaged_device: gpu",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = resolve_traversal_execution_config(outer_level=6, aosky_yaml=conf)

    assert config == TraversalExecutionConfig(
        workers=5,
        scheduler="static",
        low_latitude_workers=3,
        gaia_cache_entries=128,
        gaia_cache_mb=4096,
        region_level=3,
        parent_memory_limit_mb=12288,
        prediction_device="gpu",
        averaged_prediction_device="gpu",
    )

    override = resolve_traversal_execution_config(
        outer_level=6,
        workers=3,
        low_latitude_workers=2,
        gaia_cache_entries=64,
        gaia_cache_mb=2048,
        parent_memory_limit_mb=8192,
        aosky_yaml=conf,
    )
    assert override == TraversalExecutionConfig(
        workers=3,
        scheduler="static",
        low_latitude_workers=2,
        gaia_cache_entries=64,
        gaia_cache_mb=2048,
        region_level=4,
        parent_memory_limit_mb=8192,
        prediction_device="gpu",
        averaged_prediction_device="gpu",
    )

    conf.write_text(
        "\n".join(("build:", "  workers: 2")) + "\n",
        encoding="utf-8",
    )
    without_device = resolve_traversal_execution_config(outer_level=6, aosky_yaml=conf)
    assert without_device.prediction_device == "cpu"
    assert without_device.averaged_prediction_device == "cpu"


def test_resolve_traversal_execution_config_validates_values(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="workers must be at least 1"):
        resolve_traversal_execution_config(outer_level=2, workers=0)
    with pytest.raises(BuildError, match="low_latitude_workers"):
        resolve_traversal_execution_config(
            outer_level=2,
            workers=3,
            low_latitude_workers=4,
        )
    with pytest.raises(BuildError, match="scheduler must be either"):
        resolve_traversal_execution_config(outer_level=2, scheduler="other")
    with pytest.raises(BuildError, match="only supported by the static scheduler"):
        resolve_traversal_execution_config(
            outer_level=2,
            scheduler="dynamic",
            low_latitude_workers=1,
        )
    with pytest.raises(BuildError, match="gaia_cache_entries must be non-negative"):
        resolve_traversal_execution_config(outer_level=2, gaia_cache_entries=-1)
    with pytest.raises(BuildError, match="memory_limit_mb must be non-negative"):
        resolve_traversal_execution_config(outer_level=2, parent_memory_limit_mb=-1)
    with pytest.raises(BuildError, match="telemetry must be either"):
        resolve_traversal_execution_config(outer_level=2, telemetry="verbose")

    conf = tmp_path / "ao-sky.yaml"
    conf.write_text(
        "\n".join(("prediction:", "  resolved_device: mps")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BuildError, match="prediction.resolved_device"):
        resolve_traversal_execution_config(outer_level=2, aosky_yaml=conf)

    conf.write_text(
        "\n".join(("build:", "  parent_memory_limit_mb: 12288")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BuildError, match="build.memory_limit_mb"):
        resolve_traversal_execution_config(outer_level=2, aosky_yaml=conf)


def test_regional_worker_plans_can_reserve_low_latitude_workers() -> None:
    state = _make_scheduler_state([WORK_STATUS_PENDING] * 64)
    star_counts = np.arange(64, dtype=np.int64) + 1

    plans = build_regional_worker_plans(
        state=state,
        outer_level=2,
        region_level=1,
        workers=3,
        status_field="traversal_status",
        star_counts=star_counts,
        low_latitude_deg=90.0,
        low_latitude_workers=2,
    )

    assert {plan.worker_id for plan in plans} == {0, 1, 2}
    assert sorted(pix for plan in plans for pix in plan.outer_pixs) == list(range(64))
    assert sum(plan.estimated_star_count for plan in plans) == int(np.sum(star_counts))


def test_dynamic_work_batches_classify_stress_and_compute_workers() -> None:
    state = _make_scheduler_state([WORK_STATUS_PENDING] * 64)
    star_counts = np.full(64, 1_000, dtype=np.int64)
    star_counts[32:40] = 500_000

    schedule = build_dynamic_work_batches(
        state=state,
        outer_level=2,
        workers=4,
        status_field="traversal_status",
        star_counts=star_counts,
        memory_limit_mb=8192,
        trim_fraction=0.85,
        worker_ram_overhead_mb=0.0,
    )

    assert schedule.stress_star_threshold > 0
    assert schedule.stress_worker_count >= 1
    assert any(batch.is_stress for batch in schedule.batches)
    assert any(not batch.is_stress for batch in schedule.batches)
    assert sorted(pix for batch in schedule.batches for pix in batch.outer_pixs) == list(range(64))
    assert all(batch.region_level == 1 for batch in schedule.batches if batch.is_stress)
    assert all(batch.region_level == 0 for batch in schedule.batches if not batch.is_stress)


def test_order_region_outer_pixs_prefers_neighbours(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = {
        3: np.asarray([2], dtype=np.int64),
        2: np.asarray([1], dtype=np.int64),
        1: np.asarray([0], dtype=np.int64),
        0: np.asarray([], dtype=np.int64),
    }
    monkeypatch.setattr(
        "ao_sky.build.regional.get_pixel_neighbours",
        lambda level, pix: graph[pix],
    )
    star_counts = np.asarray([1, 2, 3, 4], dtype=np.int64)

    assert order_region_outer_pixs([0, 1, 2, 3], outer_level=0, star_counts=star_counts) == (
        3,
        2,
        1,
        0,
    )


def test_order_region_outer_pixs_can_stagger_initial_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ao_sky.build.regional.get_pixel_neighbours",
        lambda level, pix: np.asarray([], dtype=np.int64),
    )
    star_counts = np.asarray([10, 20, 30, 40], dtype=np.int64)

    assert order_region_outer_pixs(
        [0, 1, 2, 3],
        outer_level=0,
        star_counts=star_counts,
        seed_rank=2,
    ) == (
        1,
        3,
        2,
        0,
    )


def test_fetch_model_data_copies_configured_model_bundle_and_updates_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-models"
    _write_model_bundle(source_root, "point_two")
    (source_root / "point_two_data.pkl").write_bytes(b"unused provenance")
    _write_model_bundle(source_root, "mean_two")
    _write_model_bundle(source_root, "shared_three")
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)

    manifest_path = fetch_model_data(build_path)

    destination_root = (build_path / "models").resolve()
    assert manifest_path == destination_root / "manifest.json"
    assert load_build_roots(build_path).model_root == destination_root
    assert (destination_root / "point_two.pt").is_file()
    assert (destination_root / "point_two_metadata.pkl").is_file()
    assert not (destination_root / "point_two_data.pkl").exists()
    assert (destination_root / "mean_two.pt").is_file()
    assert (destination_root / "mean_two_metadata.pkl").is_file()
    assert (destination_root / "shared_three.pt").is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_model_root"] == str(source_root.resolve())
    assert manifest["model_root"] == "models"
    assert manifest["runtime_config_path"] == str(load_runtime_config_path(build_path))
    models = {item["name"]: item for item in manifest["models"]}
    assert set(models) == {"point_two", "mean_two", "shared_three"}
    assert models["shared_three"]["roles"] == ["averaged:3star", "resolved:3star"]
    assert all(
        len(file_info["sha256"]) == 64
        for model_info in models.values()
        for file_info in model_info["files"]
    )


def test_fetch_model_data_uses_ao_sky_yaml_model_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persisted_root = tmp_path / "persisted-models"
    source_root = tmp_path / "conf-models"
    for model_name in ("point_two", "mean_two", "shared_three"):
        _write_model_bundle(source_root, model_name)
    build_path = _init_model_snapshot_build(tmp_path, model_root=persisted_root)
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "ao-sky.yaml").write_text(
        f"build:\n  roots:\n    model: {source_root}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)

    manifest_path = fetch_model_data(build_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_model_root"] == str(source_root.resolve())


def test_fetch_model_data_fails_when_required_model_file_is_missing(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-models"
    _write_model_bundle(source_root, "point_two")
    _write_model_bundle(source_root, "shared_three")
    (source_root / "mean_two.pt").write_bytes(b"missing metadata")
    with pytest.raises(BuildError, match="Required model file is missing"):
        _init_model_snapshot_build(
            tmp_path,
            model_root=source_root,
            ensure_model_files=False,
        )


def test_fetch_model_data_requires_force_for_differing_existing_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-models"
    for model_name in ("point_two", "mean_two", "shared_three"):
        _write_model_bundle(source_root, model_name)
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)
    fetch_model_data(build_path)
    destination_file = build_path / "models" / "point_two.pt"
    original_bytes = destination_file.read_bytes()

    _write_model_bundle(source_root, "point_two", payload=b"new")
    with pytest.raises(BuildError, match="rerun with --force"):
        fetch_model_data(build_path, model_root=source_root)
    assert destination_file.read_bytes() == original_bytes

    fetch_model_data(build_path, model_root=source_root, force=True)
    assert destination_file.read_bytes() == b"new:pt"


def test_fetch_model_data_rerun_against_local_snapshot_is_noop(tmp_path: Path) -> None:
    source_root = tmp_path / "source-models"
    for model_name in ("point_two", "mean_two", "shared_three"):
        _write_model_bundle(source_root, model_name)
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)
    manifest_path = fetch_model_data(build_path)
    original_manifest = manifest_path.read_text(encoding="utf-8")

    assert fetch_model_data(build_path) == manifest_path

    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_fetch_model_data_rejects_build_after_traversal_started(tmp_path: Path) -> None:
    source_root = tmp_path / "source-models"
    for model_name in ("point_two", "mean_two", "shared_three"):
        _write_model_bundle(source_root, model_name)
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)
    update_state_row(build_path, 0, traversal_status=WORK_STATUS_DONE)

    with pytest.raises(BuildError, match="before Traversal has started"):
        fetch_model_data(build_path)

    assert (build_path / "models" / "manifest.json").is_file()


def test_run_build_uses_build_local_model_root_after_fetch_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build.runtime_config import load_runtime_config as real_load_runtime_config

    source_root = tmp_path / "source-models"
    for model_name in ("point_two", "mean_two", "shared_three"):
        _write_model_bundle(source_root, model_name)
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)
    fetch_model_data(build_path)
    seen_model_roots: list[Path] = []

    def capture_runtime(runtime_config_path: Path, *, model_root: Path):
        seen_model_roots.append(Path(model_root).resolve())
        return real_load_runtime_config(
            runtime_config_path,
            model_root=model_root,
        )

    monkeypatch.setattr("ao_sky.build.runner.load_runtime_config", capture_runtime)
    monkeypatch.setattr(
        "ao_sky.build.runner.warm_model_cache",
        lambda runtime, **kwargs: None,
    )
    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (_make_asterisms(), _make_inner()),
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    assert seen_model_roots
    assert set(seen_model_roots) == {(build_path / "models").resolve()}


def test_load_build_definition_parses_overlay_specs_and_preserves_relative_paths(
    tmp_path: Path,
) -> None:
    moc = _write_moc(tmp_path / "mocs" / "ews.fits", level=1, pixs=[1])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[
            {
                "name": "EWS-Yr1",
                "moc_files": [str(moc.relative_to(tmp_path))],
            }
        ],
    )

    loaded, _ = load_build_definition_yaml(definition)

    assert len(loaded.survey_extent_overlays) == 1
    overlay = loaded.survey_extent_overlays[0]
    assert overlay.name == "ews_yr1"
    assert overlay.moc_files == (Path("mocs/ews.fits"),)


def test_fetch_survey_data_copies_overlay_files_and_updates_metadata(
    tmp_path: Path,
) -> None:
    survey_root = tmp_path / "source-surveys"
    _write_moc(survey_root / "year1.fits", level=1, pixs=[1])
    _write_moc(survey_root / "year2.fits", level=1, pixs=[2])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[
            {
                "name": "EWS",
                "moc_files": [
                    "year1.fits",
                    "year2.fits",
                ],
            }
        ],
    )
    legacy = _write_legacy_config_with_models(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        model_root=tmp_path / "models",
        survey_root=survey_root,
        legacy_config_path=legacy,
    )

    manifest_path = fetch_survey_data(build_path, survey_root=survey_root)

    assert manifest_path == build_path / "surveys" / "manifest.json"
    assert (build_path / "surveys" / "year1.fits").is_file()
    assert (build_path / "surveys" / "year2.fits").is_file()
    loaded = load_build_definition(build_path)
    assert loaded.survey_extent_overlays[0].moc_files == (
        (build_path / "surveys" / "year1.fits").resolve(),
        (build_path / "surveys" / "year2.fits").resolve(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_survey_root"] == str(survey_root.resolve())
    assert manifest["survey_root"] == "surveys"
    assert manifest["overlays"][0]["name"] == "ews"
    assert manifest["overlays"][0]["files"][0]["path"] == "surveys/year1.fits"


def test_fetch_survey_data_uses_ao_sky_yaml_survey_root(tmp_path: Path) -> None:
    survey_root = tmp_path / "source-surveys"
    _write_moc(survey_root / "ews.fits", level=1, pixs=[1])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[
            {"name": "ews", "moc_files": ["ews.fits"]},
        ],
    )
    legacy = _write_legacy_config_with_models(tmp_path / "legacy.yaml")
    conf = tmp_path / "ao-sky.yaml"
    conf.write_text(f"build:\n  roots:\n    survey: {survey_root}\n", encoding="utf-8")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        model_root=tmp_path / "models",
        legacy_config_path=legacy,
        aosky_yaml=conf,
    )

    manifest_path = fetch_survey_data(build_path, aosky_yaml=conf)

    assert manifest_path.is_file()
    assert (build_path / "surveys" / "ews.fits").is_file()


def test_fetch_survey_data_requires_force_for_differing_existing_files(
    tmp_path: Path,
) -> None:
    survey_root = tmp_path / "source-surveys"
    _write_moc(survey_root / "ews.fits", level=1, pixs=[1])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[
            {"name": "ews", "moc_files": ["ews.fits"]},
        ],
    )
    legacy = _write_legacy_config_with_models(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        model_root=tmp_path / "models",
        survey_root=survey_root,
        legacy_config_path=legacy,
    )
    fetch_survey_data(build_path, survey_root=survey_root)
    destination = build_path / "surveys" / "ews.fits"
    original = destination.read_bytes()
    destination.write_bytes(b"different")

    with pytest.raises(BuildError, match="rerun with --force"):
        fetch_survey_data(build_path, survey_root=survey_root)

    assert destination.read_bytes() == b"different"
    fetch_survey_data(build_path, survey_root=survey_root, force=True)
    assert destination.read_bytes() == original


def test_load_build_definition_rejects_duplicate_overlay_names(tmp_path: Path) -> None:
    moc_a = _write_moc(tmp_path / "mocs" / "a.fits", level=1, pixs=[1])
    moc_b = _write_moc(tmp_path / "mocs" / "b.fits", level=1, pixs=[2])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[
            {"name": "EWS-Yr1", "moc_files": [str(moc_a)]},
            {"name": "ews_yr1", "moc_files": [str(moc_b)]},
        ],
    )

    with pytest.raises(BuildError, match="duplicate survey overlay name"):
        load_build_definition_yaml(definition)


def test_load_build_definition_requires_non_empty_overlay_paths(tmp_path: Path) -> None:
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[{"name": "ews", "moc_files": []}],
    )

    with pytest.raises(BuildError, match="moc_files must not be empty"):
        load_build_definition_yaml(definition)


def test_load_build_definition_rejects_blank_gaia_release(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    payload = yaml.safe_load(definition.read_text(encoding="utf-8"))
    payload["gaia"]["release"] = "   "
    definition.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(BuildError, match="gaia.release must be a non-empty string"):
        load_build_definition_yaml(definition)


def test_load_build_definition_rejects_max_data_level_above_inner_level(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml", max_data_level=2)

    with pytest.raises(BuildError, match="maps.max_level must be less than or equal to traversal.inner_level"):
        load_build_definition_yaml(definition)


def test_scheduler_uses_highest_star_count_seed_when_no_anchor() -> None:
    scheduler = OuterPixelScheduler(
        outer_level=0,
        star_counts=np.array([1, 3, 7], dtype=np.int64),
    )
    state = _make_scheduler_state(
        [
            WORK_STATUS_DONE,
            WORK_STATUS_FAILED,
            WORK_STATUS_PENDING,
        ]
    )

    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=None) == 2


def test_scheduler_falls_back_to_lowest_unfinished_seed_without_star_counts() -> None:
    scheduler = OuterPixelScheduler(outer_level=0)
    state = _make_scheduler_state(
        [
            WORK_STATUS_DONE,
            WORK_STATUS_FAILED,
            WORK_STATUS_PENDING,
        ]
    )

    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=None) == 1


def test_scheduler_preserves_neighbour_order_and_depth_first_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        0: np.array([1, 2], dtype=np.int64),
        1: np.array([3, 4], dtype=np.int64),
        2: np.array([], dtype=np.int64),
        3: np.array([], dtype=np.int64),
        4: np.array([], dtype=np.int64),
    }
    monkeypatch.setattr(
        "ao_sky.build.scheduler.get_pixel_neighbours",
        lambda level, outer_pix: graph.get(outer_pix, np.array([], dtype=np.int64)),
    )

    scheduler = OuterPixelScheduler(outer_level=0)
    state = _make_scheduler_state([WORK_STATUS_PENDING] * 5)

    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=None) == 0
    state["traversal_status"][0] = WORK_STATUS_DONE
    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=0) == 1
    state["traversal_status"][1] = WORK_STATUS_DONE
    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=1) == 3
    state["traversal_status"][3] = WORK_STATUS_DONE
    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=3) == 4
    state["traversal_status"][4] = WORK_STATUS_DONE
    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=4) == 2


def test_scheduler_does_not_retry_failed_pixel_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = {
        0: np.array([1, 2], dtype=np.int64),
        1: np.array([3], dtype=np.int64),
        2: np.array([], dtype=np.int64),
        3: np.array([], dtype=np.int64),
    }
    monkeypatch.setattr(
        "ao_sky.build.scheduler.get_pixel_neighbours",
        lambda level, outer_pix: graph.get(outer_pix, np.array([], dtype=np.int64)),
    )

    scheduler = OuterPixelScheduler(outer_level=0)
    state = _make_scheduler_state([WORK_STATUS_PENDING] * 4)

    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=None) == 0
    state["traversal_status"][0] = WORK_STATUS_DONE
    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=0) == 1
    state["traversal_status"][1] = WORK_STATUS_FAILED
    assert scheduler.select_next_outer_pixel(state, last_completed_outer_pix=0) == 2


def test_init_build_requires_matching_gaia_summary(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    runtime = load_native_runtime(
        load_build_definition_yaml(definition)[0],
        legacy_config_path=legacy,
        model_root=tmp_path / "models",
    )
    for model_name in set(runtime.resolved_models.values()) | set(
        runtime.averaged_models.values()
    ):
        _write_required_model_files(tmp_path / "models", model_name)

    with pytest.raises(BuildError, match="fetch-gaia --gaia-release dr3 --outer-level 0"):
        real_init_build(
            config_filename=definition,
            gaia_root=tmp_path / "gaia",
            build_root=tmp_path / "builds",
            model_root=tmp_path / "models",
        )


def test_run_build_writes_outer_artifacts_and_updates_traversal_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (
            _make_asterisms(),
            _make_inner(
                gaia_a0=0.12,
                star_count=3,
                ngs_count=2,
                best_ee=0.5,
                best_sr=0.4,
                best_fwhm=0.3,
                winner_asterism_id=7,
                winner_ee_resolved=0.5,
                winner_ee_averaged=0.45,
                coverage_resolved=True,
                coverage_averaged=True,
            ),
        ),
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    state = load_state(build_path)
    assert np.all(state["traversal_status"] == WORK_STATUS_DONE)
    summary = summarize_build(build_path)
    assert summary["current_phase"] == BUILD_PHASE_AGGREGATION
    assert summary["build_status"] == "completed"

    outer_filename = outer_artifact_filename(
        build_path,
        load_build_definition(build_path),
        0,
    )
    assert outer_filename.is_file()
    with h5py.File(outer_filename, "r") as handle:
        assert OUTER_DATASET_INNER in handle
        assert OUTER_DATASET_ASTERISMS in handle
        assert set(handle[OUTER_DATASET_INNER].dtype.names) == {
            "pix",
            "gaia_A0",
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
        }


def test_run_build_uses_gaia_summary_star_counts_for_initial_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    gaia_root = tmp_path / "gaia"
    build_path = init_build(
        definition_filename=definition,
        gaia_root=gaia_root,
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    _write_gaia_summary(
        gaia_root,
        release="dr3",
        outer_level=0,
        star_counts=[0, 3, 1, 9] + [0] * 8,
    )
    visited: list[int] = []

    def fake_build(context, outer_pix: int, **kwargs: object):
        visited.append(int(outer_pix))
        return tmp_path / f"outer-{outer_pix}.h5", Table({"pix": [outer_pix]}), Table()

    monkeypatch.setattr(
        "ao_sky.build.runner._build_outer_pixel_products",
        fake_build,
    )
    monkeypatch.setattr(
        "ao_sky.build.runner._write_outer_pixel_products",
        lambda filename, *, inner, asterisms: 0.0,
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    assert sorted(visited) == list(range(12))
    assert 3 in visited


def test_run_build_rejects_invalid_worker_count(tmp_path: Path) -> None:
    build_path = tmp_path / "missing"

    with pytest.raises(BuildError, match="workers must be at least 1"):
        run_build(build_path, workers=0)


def test_run_build_outer_pixels_runs_selected_traversal_without_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    seen: list[int] = []

    def fake_run_outer_pixel(
        context,
        outer_pix: int,
        *,
        execution_config=None,
    ) -> TraversalTaskResult:
        seen.append(int(outer_pix))
        return TraversalTaskResult(outer_pix=int(outer_pix), success=True)

    monkeypatch.setattr(
        "ao_sky.build.runner._run_outer_pixel_traversal_task",
        fake_run_outer_pixel,
    )

    result = run_build_outer_pixels(build_path, [0, 1])

    state = load_state(build_path)
    assert result == build_path
    assert seen == [0, 1]
    assert state["traversal_status"][0] == WORK_STATUS_DONE
    assert state["traversal_status"][1] == WORK_STATUS_DONE
    assert state["traversal_status"][2] == WORK_STATUS_PENDING
    assert summarize_build(build_path)["current_phase"] == BUILD_PHASE_TRAVERSAL
    assert summarize_build(build_path)["build_status"] == "running"
    assert not maps_artifact_filename(build_path, 1).exists()


def test_run_build_parallel_workers_dispatch_distinct_pixels_and_update_parent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    seen_workers: list[int] = []
    seen_outer_pixs: list[int] = []

    def fake_regional_workers(
        *,
        build_path: Path,
        context,
        state,
        plans,
        execution_config,
        telemetry,
        state_writer,
        **kwargs,
    ) -> bool:
        seen_workers.extend(plan.worker_id for plan in plans)
        for plan in plans:
            for outer_pix in plan.outer_pixs:
                seen_outer_pixs.append(int(outer_pix))
                runner_module._handle_regional_worker_message(
                    build_path=build_path,
                    state=state,
                    message=runner_module.TraversalWorkerMessage(
                        worker_id=plan.worker_id,
                        kind="started",
                        outer_pix=int(outer_pix),
                    ),
                    telemetry=telemetry,
                    state_writer=state_writer,
                )
                runner_module._handle_regional_worker_message(
                    build_path=build_path,
                    state=state,
                    message=runner_module.TraversalWorkerMessage(
                        worker_id=plan.worker_id,
                        kind="completed",
                        result=TraversalTaskResult(outer_pix=int(outer_pix), success=True),
                    ),
                    telemetry=telemetry,
                    state_writer=state_writer,
                )
        return False

    monkeypatch.setattr(
        "ao_sky.build.runner._run_regional_traversal_workers",
        fake_regional_workers,
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})
    monkeypatch.setattr("ao_sky.build.runner.TRAVERSAL_PROGRESS_LOG_INTERVAL", 5)

    run_build(build_path, workers=3, gaia_cache_entries=0)

    assert seen_workers == [0, 1, 2]
    assert sorted(seen_outer_pixs) == list(range(12))
    state = load_state(build_path)
    assert np.all(state["traversal_status"] == WORK_STATUS_DONE)
    assert np.all(np.asarray(state["traversal_attempt_count"], dtype=np.int64) == 1)
    summary = summarize_build(build_path)
    assert summary["build_status"] == "completed"
    assert summary["current_phase"] == BUILD_PHASE_AGGREGATION
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "run start phase=traversal repaired_stale_running=0 workers=3" in log_text
    assert "phase=traversal progress completed=5" not in log_text
    assert "phase=traversal outer_pix=0 dispatched" not in log_text
    assert "phase=traversal outer_pix=0 completed" not in log_text


def test_run_build_uses_ao_sky_yaml_execution_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    config = tmp_path / "ao-sky.yaml"
    config.write_text(
        "\n".join(
            (
                "build:",
                "  workers: 5",
                "  memory_limit_mb: 12288",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, TraversalExecutionConfig] = {}

    def fake_run_traversal_phase(
        build_path: Path,
        *,
        execution_config: TraversalExecutionConfig,
    ) -> tuple[bool, dict[str, int]]:
        captured["execution_config"] = execution_config
        return False, {}

    monkeypatch.setattr(
        "ao_sky.build.runner._run_traversal_phase",
        fake_run_traversal_phase,
    )

    run_build(build_path, aosky_yaml=config)

    execution_config = captured["execution_config"]
    assert execution_config.workers == 5
    assert execution_config.parent_memory_limit_mb == 12288


def test_run_build_parallel_workers_continue_after_failed_outer_pixel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(3, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    def fake_regional_workers(
        *,
        build_path: Path,
        context,
        state,
        plans,
        execution_config,
        telemetry,
        state_writer,
        **kwargs,
    ) -> bool:
        failed = False
        for plan in plans:
            for outer_pix in plan.outer_pixs:
                runner_module._handle_regional_worker_message(
                    build_path=build_path,
                    state=state,
                    message=runner_module.TraversalWorkerMessage(
                        worker_id=plan.worker_id,
                        kind="started",
                        outer_pix=int(outer_pix),
                    ),
                    telemetry=telemetry,
                    state_writer=state_writer,
                )
                result = (
                    TraversalTaskResult(
                        outer_pix=int(outer_pix),
                        success=False,
                        error_message="boom",
                    )
                    if int(outer_pix) == 0
                    else TraversalTaskResult(outer_pix=int(outer_pix), success=True)
                )
                failed = runner_module._handle_regional_worker_message(
                    build_path=build_path,
                    state=state,
                    message=runner_module.TraversalWorkerMessage(
                        worker_id=plan.worker_id,
                        kind="failed" if not result.success else "completed",
                        result=result,
                    ),
                    telemetry=telemetry,
                    state_writer=state_writer,
                ) or failed
        return failed

    monkeypatch.setattr(
        "ao_sky.build.runner._run_regional_traversal_workers",
        fake_regional_workers,
    )

    run_build(build_path, workers=3, gaia_cache_entries=0)

    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_FAILED
    assert int(state["traversal_status"][1]) == WORK_STATUS_DONE
    assert int(state["traversal_status"][2]) == WORK_STATUS_DONE
    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal worker=0 outer_pixel_failed outer_pix=0" in log_text
    assert "error=boom" in log_text
    assert "run complete phase=traversal status=failed" in log_text


def test_run_build_truncates_overlong_traversal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(1, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    def fail_build(*args: object, **kwargs: object) -> tuple[Table, Table]:
        raise RuntimeError("x" * 2000)

    monkeypatch.setattr("ao_sky.build.runner.build_traversal_products", fail_build)

    run_build(build_path)

    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_FAILED
    message = state["traversal_last_error_message"][0].decode("utf-8")
    assert len(message.encode("utf-8")) <= 1024
    assert message.endswith("[truncated]")
    assert summarize_build(build_path)["build_status"] == "failed"


def test_run_build_regional_cache_path_updates_parent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(3, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)
    seen_configs: list[TraversalExecutionConfig] = []

    def fake_regional_workers(
        *,
        build_path: Path,
        context,
        state: np.ndarray,
        plans,
        execution_config: TraversalExecutionConfig,
        telemetry=None,
        state_writer=None,
        **kwargs,
    ) -> bool:
        seen_configs.append(execution_config)
        for plan in plans:
            for outer_pix in plan.outer_pixs:
                runner_module._handle_regional_worker_message(
                    build_path=build_path,
                    state=state,
                    message=runner_module.TraversalWorkerMessage(
                        worker_id=plan.worker_id,
                        kind="started",
                        outer_pix=outer_pix,
                    ),
                    telemetry=telemetry,
                    state_writer=state_writer,
                )
                runner_module._handle_regional_worker_message(
                    build_path=build_path,
                    state=state,
                    message=runner_module.TraversalWorkerMessage(
                        worker_id=plan.worker_id,
                        kind="completed",
                        outer_pix=outer_pix,
                        result=TraversalTaskResult(outer_pix=outer_pix, success=True),
                    ),
                    telemetry=telemetry,
                    state_writer=state_writer,
                )
        return False

    monkeypatch.setattr(
        "ao_sky.build.runner._run_regional_traversal_workers",
        fake_regional_workers,
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path, workers=3, gaia_cache_entries=64, gaia_cache_mb=2048)

    assert seen_configs == [
        TraversalExecutionConfig(
            workers=3,
            gaia_cache_entries=64,
            gaia_cache_mb=2048,
            region_level=0,
            parent_memory_limit_mb=0,
        )
    ]
    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_DONE
    assert int(state["traversal_attempt_count"][0]) == 1
    assert np.all(state["traversal_status"] == WORK_STATUS_DONE)
    assert summarize_build(build_path)["build_status"] == "completed"


def test_parent_memory_limit_uses_parent_plus_worker_current_rss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    class FakeProcess:
        pid = 123

    monkeypatch.setattr(runner_module, "_current_rss_mb", lambda: 4096.0)
    monkeypatch.setattr(runner_module, "_process_current_rss_mb", lambda pid: 9000.0)

    with pytest.raises(BuildError, match="parent memory limit exceeded"):
        runner_module._raise_if_parent_memory_limit_exceeded(
            TraversalExecutionConfig(parent_memory_limit_mb=12288),
            [FakeProcess()],
        )

    runner_module._raise_if_parent_memory_limit_exceeded(
        TraversalExecutionConfig(parent_memory_limit_mb=14000),
        [FakeProcess()],
    )


def test_parent_memory_limit_reserves_gpu_driver_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    class FakeProcess:
        pid = 123

    monkeypatch.setattr(runner_module.predict_backend, "mps_is_available", lambda: True)
    monkeypatch.setattr(runner_module, "_current_rss_mb", lambda: 4096.0)
    monkeypatch.setattr(runner_module, "_process_current_rss_mb", lambda pid: 9000.0)

    with pytest.raises(BuildError, match="current total RAM"):
        runner_module._raise_if_parent_memory_limit_exceeded(
            TraversalExecutionConfig(
                parent_memory_limit_mb=14000,
                prediction_device="gpu",
                parent_gpu_driver_reserve_mb=1000.0,
            ),
            [FakeProcess()],
        )


def test_parent_memory_limit_does_not_reserve_gpu_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    class FakeProcess:
        pid = 123

    monkeypatch.setattr(runner_module.predict_backend, "mps_is_available", lambda: False)
    monkeypatch.setattr(runner_module, "_current_rss_mb", lambda: 4096.0)
    monkeypatch.setattr(runner_module, "_process_current_rss_mb", lambda pid: 9000.0)

    runner_module._raise_if_parent_memory_limit_exceeded(
        TraversalExecutionConfig(
            parent_memory_limit_mb=14000,
            prediction_device="gpu",
            parent_gpu_driver_reserve_mb=1000.0,
        ),
        [FakeProcess()],
    )


def test_parent_memory_pressure_trims_heaviest_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    class FakeQueue:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def put(self, command: str) -> None:
            self.commands.append(command)

    worker_rss = {10: 1000.0, 11: 1500.0, 12: 5750.0}
    monkeypatch.setattr(runner_module, "_current_rss_mb", lambda: 500.0)
    monkeypatch.setattr(
        runner_module,
        "_process_current_rss_mb",
        lambda pid: worker_rss[int(pid)],
    )
    queues = {worker_id: FakeQueue() for worker_id in range(3)}
    states = {
        worker_id: runner_module.PARENT_MEMORY_PRESSURE_NORMAL
        for worker_id in range(3)
    }

    pressure_state, targets = runner_module._update_parent_memory_pressure(
        build_path=tmp_path,
        execution_config=TraversalExecutionConfig(parent_memory_limit_mb=10000),
        process_by_worker_id={
            0: FakeProcess(10),
            1: FakeProcess(11),
            2: FakeProcess(12),
        },
        control_queues=queues,
        active_worker_ids={0, 1, 2},
        worker_pressure_states=states,
        worker_pressure_command_times={},
        previous_state=runner_module.PARENT_MEMORY_PRESSURE_NORMAL,
        previous_targets=(),
    )

    assert pressure_state == runner_module.PARENT_MEMORY_PRESSURE_TRIM
    assert targets == (2,)
    assert queues[0].commands == []
    assert queues[1].commands == []
    assert queues[2].commands == [runner_module.PARENT_MEMORY_PRESSURE_TRIM]


def test_parent_memory_pressure_pauses_heaviest_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    class FakeQueue:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def put(self, command: str) -> None:
            self.commands.append(command)

    worker_rss = {10: 4000.0, 11: 3000.0, 12: 2500.0}
    monkeypatch.setattr(runner_module, "_current_rss_mb", lambda: 250.0)
    monkeypatch.setattr(
        runner_module,
        "_process_current_rss_mb",
        lambda pid: worker_rss[int(pid)],
    )
    queues = {worker_id: FakeQueue() for worker_id in range(3)}
    states = {
        worker_id: runner_module.PARENT_MEMORY_PRESSURE_NORMAL
        for worker_id in range(3)
    }

    pressure_state, targets = runner_module._update_parent_memory_pressure(
        build_path=tmp_path,
        execution_config=TraversalExecutionConfig(parent_memory_limit_mb=10000),
        process_by_worker_id={
            0: FakeProcess(10),
            1: FakeProcess(11),
            2: FakeProcess(12),
        },
        control_queues=queues,
        active_worker_ids={0, 1, 2},
        worker_pressure_states=states,
        worker_pressure_command_times={},
        previous_state=runner_module.PARENT_MEMORY_PRESSURE_NORMAL,
        previous_targets=(),
    )

    assert pressure_state == runner_module.PARENT_MEMORY_PRESSURE_PAUSE
    assert targets == (0, 1)
    assert queues[0].commands == [runner_module.PARENT_MEMORY_PRESSURE_PAUSE]
    assert queues[1].commands == [runner_module.PARENT_MEMORY_PRESSURE_PAUSE]
    assert queues[2].commands == []


def test_regional_profile_logs_artifact_write_and_cache_telemetry(tmp_path: Path) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)

    runner_module._handle_regional_worker_message(
        build_path=build_path,
        state=state,
        message=runner_module.TraversalWorkerMessage(
            worker_id=1,
            kind="profile",
            worker_stats=runner_module.TraversalWorkerStats(
                completed=2,
                failed=0,
                elapsed_seconds=5.0,
                pixel_seconds=4.0,
                artifact_write_seconds=0.5,
                artifact_inner_input_bytes=2 * 1024 * 1024,
                artifact_asterism_input_bytes=1024 * 1024,
                artifact_inner_structured_bytes=3 * 1024 * 1024,
                artifact_asterism_structured_bytes=4 * 1024 * 1024,
                stage_stats=runner_module.TraversalStageStats(
                    star_selection_seconds=0.1,
                    candidate_generation_seconds=0.2,
                    filtering_seconds=0.3,
                    bright_star_filter_seconds=0.01,
                    inner_assignment_seconds=0.04,
                    local_selection_seconds=0.05,
                    inner_table_seconds=0.4,
                    context_seconds=0.5,
                    point_prediction_seconds=0.6,
                    point_prediction_eligibility_seconds=0.06,
                    point_prediction_eligibility_intersection_seconds=0.04,
                    point_prediction_eligibility_extract_seconds=0.02,
                    point_prediction_buffer_seconds=0.07,
                    point_prediction_ngs_array_seconds=0.08,
                    point_prediction_model_seconds=0.09,
                    point_prediction_feature_seconds=0.11,
                    point_prediction_backend_seconds=0.22,
                    point_prediction_scatter_seconds=0.12,
                    point_prediction_scatter_filter_seconds=0.021,
                    point_prediction_scatter_merge_seconds=0.022,
                    point_prediction_scatter_sort_seconds=0.033,
                    point_prediction_scatter_write_seconds=0.044,
                    point_prediction_cache_clear_seconds=0.13,
                    field_mean_prediction_seconds=0.7,
                    field_mean_prediction_feature_seconds=0.33,
                    field_mean_prediction_backend_seconds=0.44,
                    coverage_seconds=0.8,
                    dust_seconds=0.9,
                    persisted_asterisms_seconds=1.0,
                ),
                structure_stats=runner_module.TraversalStructureStats(
                    search_star_rows=11,
                    ngs_rows=12,
                    close_pair_rows=13,
                    context_pair_rows=14,
                    winner_payload_rows=15,
                    point_prediction_batches=3,
                    point_prediction_rows=100,
                    point_prediction_batch_rows_peak=40,
                    point_feature_bytes_peak=2 * 1024 * 1024,
                    point_mps_current_bytes_peak=5 * 1024 * 1024,
                    point_mps_driver_bytes_peak=6 * 1024 * 1024,
                    point_mps_recommended_bytes=7 * 1024 * 1024,
                    field_mean_prediction_batches=4,
                    field_mean_prediction_rows=50,
                    field_mean_prediction_batch_rows_peak=20,
                    field_mean_feature_bytes_peak=1024 * 1024,
                    field_mean_mps_current_bytes_peak=8 * 1024 * 1024,
                    field_mean_mps_driver_bytes_peak=9 * 1024 * 1024,
                    field_mean_mps_recommended_bytes=10 * 1024 * 1024,
                    close_pair_rows_peak=16,
                    context_pair_rows_peak=17,
                    winner_payload_rows_peak=18,
                ),
                cache_stats=runner_module.TraversalCacheStats(
                    hits=1,
                    misses=2,
                    evictions=3,
                    oversized_skips=4,
                ),
            ),
        ),
    )

    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "traversal_profiled_s=5.550" in log_text
    assert "stage_star_selection_s=0.100" in log_text
    assert "stage_filtering_s=0.300" in log_text
    assert "stage_local_selection_s=0.050" in log_text
    assert "stage_point_prediction_s=0.600" in log_text
    assert "stage_point_prediction_eligibility_s=0.060" in log_text
    assert "stage_point_prediction_eligibility_intersection_s=0.040" in log_text
    assert "stage_point_prediction_eligibility_extract_s=0.020" in log_text
    assert "stage_point_prediction_buffer_s=0.070" in log_text
    assert "stage_point_prediction_ngs_array_s=0.080" in log_text
    assert "stage_point_prediction_model_s=0.090" in log_text
    assert "stage_point_prediction_feature_s=0.110" in log_text
    assert "stage_point_prediction_backend_s=0.220" in log_text
    assert "stage_point_prediction_scatter_s=0.120" in log_text
    assert "stage_point_prediction_scatter_filter_s=0.021" in log_text
    assert "stage_point_prediction_scatter_merge_s=0.022" in log_text
    assert "stage_point_prediction_scatter_sort_s=0.033" in log_text
    assert "stage_point_prediction_scatter_write_s=0.044" in log_text
    assert "stage_point_prediction_cache_clear_s=0.130" in log_text
    assert "stage_field_mean_prediction_feature_s=0.330" in log_text
    assert "stage_field_mean_prediction_backend_s=0.440" in log_text
    assert "stage_dust_s=0.900" in log_text
    assert "artifact_write_s=0.500" in log_text
    assert "avg_artifact_write_s=0.250" in log_text
    assert "artifact_inner_input_mib=2.0" in log_text
    assert "artifact_asterism_structured_mib=4.0" in log_text
    assert "search_star_rows=11" in log_text
    assert "point_prediction_batches=3" in log_text
    assert "point_prediction_rows=100" in log_text
    assert "point_prediction_batch_rows_peak=40" in log_text
    assert "point_feature_mib_peak=2.000" in log_text
    assert "point_mps_current_mib_peak=5.000" in log_text
    assert "point_mps_driver_mib_peak=6.000" in log_text
    assert "point_mps_recommended_mib=7.000" in log_text
    assert "field_mean_prediction_batches=4" in log_text
    assert "field_mean_prediction_rows=50" in log_text
    assert "field_mean_prediction_batch_rows_peak=20" in log_text
    assert "field_mean_feature_mib_peak=1.000" in log_text
    assert "field_mean_mps_current_mib_peak=8.000" in log_text
    assert "field_mean_mps_driver_mib_peak=9.000" in log_text
    assert "field_mean_mps_recommended_mib=10.000" in log_text
    assert "close_pair_rows_peak=16" in log_text
    assert "context_pair_rows_peak=17" in log_text
    assert "winner_payload_rows_peak=18" in log_text
    assert "cache_oversized_skips=4" in log_text


def test_detailed_traversal_telemetry_writes_memory_diagnostics(tmp_path: Path) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)

    with runner_module._TraversalDiagnosticsWriter(build_path) as writer:
        runner_module._handle_regional_worker_message(
            build_path=build_path,
            state=state,
            message=runner_module.TraversalWorkerMessage(
                worker_id=2,
                kind="memory_sample",
                memory_sample=runner_module.TraversalMemorySample(
                    outer_pix=7,
                    worker_id=2,
                    success=True,
                    rss_start_mb=100.0,
                    rss_after_context_mb=150.0,
                    context_pair_rows=123,
                        winner_payload_rows=45,
                        point_prediction_batches=2,
                        point_prediction_rows=77,
                        point_prediction_batch_rows_peak=40,
                        point_prediction_backend_rows=80,
                        point_prediction_backend_batch_rows_peak=50,
                        point_prediction_backend_bucket_counts="40:1;50:1",
                        point_prediction_backend_bucket_rows="40:30;50:47",
                        point_feature_mib_peak=3.5,
                        point_mps_driver_mib_peak=4.5,
                ),
            ),
            diagnostics_writer=writer,
        )

    diagnostics = build_path / "diagnostics" / "traversal-memory.csv"
    text = diagnostics.read_text(encoding="utf-8")
    assert "outer_pix,worker_id,success" in text
    assert "7,2,1" in text
    assert ",123,45," in text
    assert "point_prediction_batches" in text
    assert "point_prediction_batch_rows_peak" in text
    assert "point_prediction_backend_bucket_counts" in text
    assert "point_mps_driver_mib_peak" in text
    assert "4.500" in text
    assert ",2,77,40,80,50,40:1;50:1,40:30;50:47,3.500," in text


def test_state_update_telemetry_retries_hdf5_lock_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    calls = 0
    real_update = runner_module.update_state_row

    def flaky_update_state_row(build_path, outer_pix, **updates):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlockingIOError("unable to lock file, Resource temporarily unavailable")
        return real_update(build_path, outer_pix, **updates)

    monkeypatch.setattr(runner_module, "update_state_row", flaky_update_state_row)
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: None)
    telemetry = runner_module._StateUpdateTelemetry()

    runner_module._mark_outer_pixel_running(build_path, load_state(build_path), 0, telemetry)

    assert telemetry.updates == 1
    assert telemetry.flushes == 1
    assert telemetry.rows_written == 1
    assert telemetry.lock_retries == 1
    assert telemetry.lock_wait_seconds == pytest.approx(
        runner_module.STATE_UPDATE_RETRY_DELAY_SECONDS
    )


def test_buffered_state_writer_coalesces_rows_before_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    real_write_state_rows = runner_module.write_state_rows
    flushed_rows: list[tuple[int, ...]] = []

    def recording_write_state_rows(build_path, row_indexes, state):
        flushed_rows.append(tuple(int(row) for row in row_indexes))
        return real_write_state_rows(build_path, row_indexes, state)

    monkeypatch.setattr(runner_module, "write_state_rows", recording_write_state_rows)
    state = load_state(build_path)
    telemetry = runner_module._StateUpdateTelemetry()
    writer = runner_module._BufferedStateWriter(
        build_path=build_path,
        state=state,
        telemetry=telemetry,
        flush_interval=2,
    )

    writer.update(0, traversal_status=WORK_STATUS_RUNNING)
    writer.update(0, traversal_status=WORK_STATUS_DONE)
    assert flushed_rows == []
    assert load_state(build_path)["traversal_status"][0] == WORK_STATUS_PENDING

    writer.update(1, traversal_status=WORK_STATUS_RUNNING)

    persisted = load_state(build_path)
    assert flushed_rows == [(0, 1)]
    assert persisted["traversal_status"][0] == WORK_STATUS_DONE
    assert persisted["traversal_status"][1] == WORK_STATUS_RUNNING
    assert telemetry.updates == 3
    assert telemetry.flushes == 1
    assert telemetry.rows_written == 2
    assert telemetry.dirty_rows_peak == 2

    writer.update(1, traversal_status=WORK_STATUS_DONE)
    writer.flush()

    persisted = load_state(build_path)
    assert flushed_rows == [(0, 1), (1,)]
    assert persisted["traversal_status"][1] == WORK_STATUS_DONE
    assert telemetry.updates == 4
    assert telemetry.flushes == 2
    assert telemetry.rows_written == 3


def test_run_build_regional_infrastructure_failure_marks_build_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    def fail_regional_workers(**kwargs):
        raise RuntimeError("semaphore denied")

    monkeypatch.setattr(
        "ao_sky.build.runner._run_regional_traversal_workers",
        fail_regional_workers,
    )

    with pytest.raises(RuntimeError, match="semaphore denied"):
        run_build(build_path, workers=3, gaia_cache_entries=0)

    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    state = load_state(build_path)
    assert not np.any(state["traversal_status"] == WORK_STATUS_RUNNING)
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal infrastructure failed: semaphore denied" in log_text
    assert "run complete phase=traversal status=failed reset_running=0" in log_text


def test_run_build_regional_failure_resets_running_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(1, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    def fail_after_start(
        *,
        build_path: Path,
        context,
        state,
        plans,
        execution_config,
        telemetry,
        state_writer,
        **kwargs,
    ) -> bool:
        runner_module._handle_regional_worker_message(
            build_path=build_path,
            state=state,
            message=runner_module.TraversalWorkerMessage(
                worker_id=plans[0].worker_id,
                kind="started",
                outer_pix=0,
            ),
            telemetry=telemetry,
            state_writer=state_writer,
        )
        raise RuntimeError("regional worker broke")

    monkeypatch.setattr(
        "ao_sky.build.runner._run_regional_traversal_workers",
        fail_after_start,
    )

    with pytest.raises(RuntimeError, match="regional worker broke"):
        run_build(build_path, workers=3, gaia_cache_entries=0)

    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_PENDING
    assert int(state["traversal_attempt_count"][0]) == 1
    assert not np.any(state["traversal_status"] == WORK_STATUS_RUNNING)
    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal infrastructure failed: regional worker broke" in log_text
    assert "run complete phase=traversal status=failed reset_running=1" in log_text


def test_outer_pixel_worker_task_writes_artifact_without_mutating_build_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build import runner as runner_module

    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (_make_asterisms(empty=True), _make_inner()),
    )

    context = runner_module._load_traversal_task_context(build_path)
    result = runner_module._run_outer_pixel_traversal_task(context, 0)

    assert result == TraversalTaskResult(outer_pix=0, success=True)
    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_PENDING
    assert int(state["traversal_attempt_count"][0]) == 0
    assert outer_artifact_filename(
        build_path,
        load_build_definition(build_path),
        0,
    ).is_file()


def test_processed_empty_pixels_still_write_empty_asterisms_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (
            _make_asterisms(empty=True),
            _make_inner(star_count=4, ngs_count=0),
        ),
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    outer_filename = outer_artifact_filename(
        build_path,
        load_build_definition(build_path),
        0,
    )
    with h5py.File(outer_filename, "r") as handle:
        assert OUTER_DATASET_ASTERISMS in handle
        assert len(handle[OUTER_DATASET_ASTERISMS]) == 0


def test_outer_artifact_uses_blosc_zstd(tmp_path: Path) -> None:
    write_outer_artifact_profiled(
        tmp_path / "outer.h5",
        inner=_make_inner(),
        asterisms=_make_asterisms(empty=True),
    )

    with h5py.File(tmp_path / "outer.h5", "r") as handle:
        dataset = handle[OUTER_DATASET_INNER]
        filter_id, _, filter_values, filter_name = dataset.id.get_create_plist().get_filter(
            0
        )
        assert dataset.compression == "unknown"
        assert filter_id == HDF5_BLOSC_FILTER_ID
        assert filter_name == b"blosc"
        assert filter_values[4] == HDF5_BLOSC_LEVEL
        assert filter_values[5] == 1


def test_run_build_does_not_apply_density_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        outer_level=1,
        inner_level=2,
        max_data_level=2,
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    persisted_definition = load_build_definition(build_path)
    star_counts = [0] * (12 * (4 ** persisted_definition.outer_level))
    star_counts[0] = 10**9
    _write_gaia_summary(
        tmp_path / "gaia",
        release=persisted_definition.gaia_release,
        outer_level=persisted_definition.outer_level,
        star_counts=star_counts,
    )
    state = load_state(build_path)
    for outer_pix in range(1, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    def fake_build_traversal_products(store, runtime, outer_pix, **kwargs):
        assert "asterism_skip_reason" not in kwargs
        return _make_asterisms(empty=True), _make_inner()

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        fake_build_traversal_products,
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path, gaia_cache_entries=0)

    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "density_skipped" not in log_text


def test_restart_build_uses_latest_lineage_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "builds"
    (build_root / "v1").mkdir(parents=True)
    latest = build_root / "v2"
    latest.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_run_build(
        build_path: Path,
        *,
        workers: int | None = None,
        scheduler: str | None = None,
        low_latitude_workers: int | None = None,
        gaia_cache_entries: int | None = None,
        gaia_cache_mb: int | None = None,
        parent_memory_limit_mb: int | None = None,
        telemetry: str | None = None,
        aosky_yaml: Path | None = None,
    ) -> Path:
        captured["workers"] = workers
        captured["scheduler"] = scheduler
        captured["low_latitude_workers"] = low_latitude_workers
        captured["gaia_cache_entries"] = gaia_cache_entries
        captured["gaia_cache_mb"] = gaia_cache_mb
        captured["parent_memory_limit_mb"] = parent_memory_limit_mb
        captured["telemetry"] = telemetry
        captured["aosky_yaml"] = aosky_yaml
        return build_path

    monkeypatch.setattr("ao_sky.build.runner.run_build", fake_run_build)

    restarted = restart_build(
        lineage_name="baseline",
        build_root=build_root,
        workers=3,
        low_latitude_workers=2,
    )

    assert restarted == latest
    assert captured == {
        "workers": 3,
        "scheduler": None,
        "low_latitude_workers": 2,
        "gaia_cache_entries": None,
        "gaia_cache_mb": None,
        "parent_memory_limit_mb": None,
        "telemetry": None,
        "aosky_yaml": None,
    }


def test_run_build_repairs_stale_running_rows_and_logs_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(1, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)
    update_state_row(build_path, 0, traversal_status=WORK_STATUS_RUNNING)

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (_make_asterisms(empty=True), _make_inner()),
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_DONE
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "run start phase=traversal repaired_stale_running=1" in log_text


def test_run_build_is_successful_no_op_when_all_rows_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    def _unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("run_build should not process completed builds")

    monkeypatch.setattr("ao_sky.build.runner.build_traversal_products", _unexpected)
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    summary = summarize_build(build_path)
    assert summary["build_status"] == "completed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "run complete phase=traversal status=running" in log_text
    assert "run complete phase=aggregation status=completed" in log_text


def test_run_build_continues_after_failure_and_marks_build_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    state = load_state(build_path)
    for outer_pix in range(2, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    def _build_traversal_products(
        store: object,
        runtime: object,
        outer_pix: int,
        **kwargs: object,
    ) -> tuple[Table, Table]:
        assert kwargs["dust_root"] == (build_path / "dust").resolve()
        assert kwargs["max_data_level"] == 1
        if outer_pix == 0:
            raise RuntimeError("boom")
        return _make_asterisms(empty=True), _make_inner()

    monkeypatch.setattr("ao_sky.build.runner.build_traversal_products", _build_traversal_products)

    run_build(build_path)

    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_FAILED
    assert int(state["traversal_status"][1]) == WORK_STATUS_DONE
    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal worker=0 outer_pixel_failed outer_pix=0" in log_text
    assert "error=boom" in log_text
    assert "run complete phase=traversal status=failed" in log_text


def test_run_build_auto_advances_to_aggregation_and_writes_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    _write_gaia_tge_map(
        tmp_path / "gaia",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (
            _make_asterisms(empty=True),
            _make_inner(
                star_count=outer_pix + 1,
                ngs_count=outer_pix % 2,
                best_ee=0.5,
                best_sr=0.4,
                best_fwhm=0.3,
                winner_ee_resolved=0.5,
                winner_ee_averaged=0.45,
                coverage_resolved=True,
                coverage_averaged=False,
            ),
        ),
    )

    run_build(build_path)

    summary = summarize_build(build_path)
    assert summary["current_phase"] == BUILD_PHASE_AGGREGATION
    assert summary["build_status"] == "completed"
    assert maps_artifact_filename(build_path, 0).is_file()
    assert maps_artifact_filename(build_path, 1).is_file()
    with h5py.File(maps_artifact_filename(build_path, 1), "r") as handle:
        assert MAPS_DATASET in handle
        assert handle[MAPS_DATASET].dtype == MAPS_DTYPE
        assert len(handle[MAPS_DATASET]) == 48


def test_run_build_auto_advances_to_augmentation_when_overlays_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _write_moc(tmp_path / "mocs" / "ews.fits", level=1, pixs=[1, 3])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[{"name": "ews", "moc_files": [str(overlay)]}],
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    _write_gaia_tge_map(
        tmp_path / "gaia",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (
            _make_asterisms(empty=True),
            _make_inner(
                star_count=outer_pix + 1,
                ngs_count=outer_pix % 2,
                best_ee=0.5,
                best_sr=0.4,
                best_fwhm=0.3,
                winner_ee_resolved=0.5,
                winner_ee_averaged=0.45,
                coverage_resolved=True,
                coverage_averaged=False,
            ),
        ),
    )

    run_build(build_path)

    summary = summarize_build(build_path)
    assert summary["current_phase"] == BUILD_PHASE_AUGMENTATION
    assert summary["build_status"] == "completed"
    with h5py.File(maps_artifact_filename(build_path, 1), "r") as handle:
        assert MAPS_DATASET in handle
        assert SURVEY_EXTENT_DATASET in handle
        assert handle[SURVEY_EXTENT_DATASET].dtype.names == ("pix", "ews")
        assert bool(handle[SURVEY_EXTENT_DATASET]["ews"][1])
        assert bool(handle[SURVEY_EXTENT_DATASET]["ews"][3])


def test_run_build_restarts_from_aggregation_and_overwrites_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    _write_gaia_tge_map(
        tmp_path / "gaia",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )
    for outer_pix in range(12):
        write_outer_artifact(
            outer_artifact_filename(build_path, load_build_definition(build_path), outer_pix),
            inner=_make_inner(star_count=1, best_ee=0.5, best_sr=0.4, best_fwhm=0.3),
            asterisms=_make_asterisms(empty=True),
        )
    set_current_phase(build_path, BUILD_PHASE_AGGREGATION)
    stale = np.zeros(12, dtype=MAPS_DTYPE)
    stale["pix"] = np.arange(12, dtype=np.int64)
    stale["star_count"] = -1
    from ao_sky.build.artifacts import write_maps_artifact

    write_maps_artifact(maps_artifact_filename(build_path, 0), maps=stale)
    monkeypatch.setattr("ao_sky.build.runner.build_traversal_products", lambda *args, **kwargs: None)

    run_build(build_path)

    with h5py.File(maps_artifact_filename(build_path, 0), "r") as handle:
        assert np.all(handle[MAPS_DATASET]["star_count"][:] >= 0)


def test_run_build_restarts_from_augmentation_and_overwrites_overlay_dataset(
    tmp_path: Path,
) -> None:
    overlay = _write_moc(tmp_path / "mocs" / "ews.fits", level=1, pixs=[1, 3])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[{"name": "ews", "moc_files": [str(overlay)]}],
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    base_maps = np.zeros(48, dtype=MAPS_DTYPE)
    base_maps["pix"] = np.arange(48, dtype=np.int64)
    from ao_sky.build.artifacts import write_maps_artifact, write_maps_family_dataset

    write_maps_artifact(maps_artifact_filename(build_path, 0), maps=np.zeros(12, dtype=MAPS_DTYPE))
    write_maps_artifact(maps_artifact_filename(build_path, 1), maps=base_maps)
    stale = np.zeros(48, dtype=[("pix", "<i8"), ("ews", "?")])
    stale["pix"] = np.arange(48, dtype=np.int64)
    stale["ews"] = True
    write_maps_family_dataset(
        maps_artifact_filename(build_path, 1),
        dataset_name=SURVEY_EXTENT_DATASET,
        data=stale,
    )
    set_current_phase(build_path, BUILD_PHASE_AUGMENTATION)

    run_build(build_path)

    with h5py.File(maps_artifact_filename(build_path, 1), "r") as handle:
        assert np.array_equal(handle[MAPS_DATASET]["pix"][:], np.arange(48, dtype=np.int64))
        assert bool(handle[SURVEY_EXTENT_DATASET]["ews"][1])
        assert not bool(handle[SURVEY_EXTENT_DATASET]["ews"][0])


def test_run_build_marks_build_failed_when_aggregation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    for outer_pix in range(12):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)
    set_current_phase(build_path, BUILD_PHASE_AGGREGATION)
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: (_ for _ in ()).throw(RuntimeError("agg boom")))

    with pytest.raises(RuntimeError, match="agg boom"):
        run_build(build_path)

    summary = summarize_build(build_path)
    assert summary["current_phase"] == BUILD_PHASE_AGGREGATION
    assert summary["build_status"] == "failed"


def test_run_build_marks_build_failed_when_augmentation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay = _write_moc(tmp_path / "mocs" / "ews.fits", level=1, pixs=[1])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[{"name": "ews", "moc_files": [str(overlay)]}],
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    set_current_phase(build_path, BUILD_PHASE_AUGMENTATION)
    monkeypatch.setattr(
        "ao_sky.build.runner.build_survey_extent_layers",
        lambda build_path: (_ for _ in ()).throw(RuntimeError("aug boom")),
    )

    with pytest.raises(RuntimeError, match="aug boom"):
        run_build(build_path)

    summary = summarize_build(build_path)
    assert summary["current_phase"] == BUILD_PHASE_AUGMENTATION
    assert summary["build_status"] == "failed"


def test_show_build_reports_phase_and_phase_counts(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    update_state_row(build_path, 0, traversal_status=WORK_STATUS_DONE)
    update_state_row(build_path, 1, traversal_status=WORK_STATUS_FAILED)

    text = show_build(build_path)

    assert "phase: traversal" in text
    assert "pending=10" in text
    assert "done=1" in text
    assert "failed=1" in text


def test_show_build_omits_phase_counts_for_aggregation(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    set_current_phase(build_path, BUILD_PHASE_AGGREGATION)

    text = show_build(build_path)

    assert "phase: aggregation" in text
    assert "phase work:" not in text


def test_show_build_omits_phase_counts_for_augmentation(tmp_path: Path) -> None:
    overlay = _write_moc(tmp_path / "mocs" / "ews.fits", level=1, pixs=[1])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[{"name": "ews", "moc_files": [overlay.name]}],
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        survey_root=overlay.parent,
        legacy_config_path=legacy,
    )
    set_current_phase(build_path, BUILD_PHASE_AUGMENTATION)

    text = show_build(build_path)

    assert "phase: augmentation" in text
    assert "phase work:" not in text


def test_update_state_row_rejects_overlong_error_message(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    with pytest.raises(BuildError, match="traversal_last_error_message exceeds persisted limit"):
        update_state_row(build_path, 0, traversal_last_error_message="x" * 1025)


def test_aggregate_maps_recomputes_dust_and_reduces_fields_by_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    _write_gaia_tge_map(
        tmp_path / "gaia",
        [
            (0, 1, 0.5),
            (1, 1, 1.5),
            (2, 1, 2.5),
        ],
    )
    prepare_gaia_tge_a0_cache(
        source_dust_root=tmp_path / "gaia",
        destination_dust_root=load_build_roots(build_path).dust_root,
        level=1,
        force=True,
    )
    inner = Table(
        [
            np.arange(4, dtype=np.int64),
            np.zeros(4, dtype=np.float64),
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int64),
            np.array([0.1, 0.2, np.nan, 0.4], dtype=np.float64),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
            np.full(4, 7, dtype=np.int64),
            np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float64),
            np.array([0.7, 0.8, 0.9, 1.0], dtype=np.float64),
            np.array([True, False, True, False], dtype=np.bool_),
            np.array([False, False, True, True], dtype=np.bool_),
        ],
        names=(
            "pix",
            "gaia_A0",
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
    write_outer_artifact(
        outer_artifact_filename(build_path, load_build_definition(build_path), 0),
        inner=inner,
        asterisms=_make_asterisms(
            centers=[
                (
                    float(get_pixel_skycoord(1, 0).ra.deg),
                    float(get_pixel_skycoord(1, 0).dec.deg),
                ),
                (
                    float(get_pixel_skycoord(1, 0).ra.deg),
                    float(get_pixel_skycoord(1, 0).dec.deg),
                ),
                (
                    float(get_pixel_skycoord(1, 1).ra.deg),
                    float(get_pixel_skycoord(1, 1).dec.deg),
                ),
            ],
            pixs=[0, 0, 1],
        ),
    )
    read_product_filenames: list[Path] = []

    def counting_read_outer_aggregation_products(filename: Path):
        read_product_filenames.append(filename)
        return real_read_outer_aggregation_products(filename)

    real_read_outer_aggregation_products = aggregation_module.read_outer_aggregation_products
    monkeypatch.setattr(
        aggregation_module,
        "read_outer_aggregation_products",
        counting_read_outer_aggregation_products,
    )

    level_maps = aggregate_maps(build_path, outer_pixs=[0])

    assert len(read_product_filenames) == 1
    assert sorted(level_maps) == [0, 1]
    level1 = level_maps[1]
    assert level1["gaia_A0"][:3].tolist() == [0.5, 1.5, 2.5]
    assert np.isnan(level1["gaia_A0"][3])
    assert level1["star_count"][:4].tolist() == [1, 2, 3, 4]
    assert level1["coverage_resolved"][:4].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert level1["winner_asterism_count"][:4].tolist() == [2, 1, 0, 0]

    level0 = level_maps[0]
    assert int(level0["winner_asterism_count"][0]) == 3
    assert int(level0["star_count"][0]) == 10
    assert int(level0["ngs_count"][0]) == 2
    assert float(level0["gaia_A0"][0]) == pytest.approx(1.5)
    assert float(level0["best_ee"][0]) == pytest.approx((0.1 + 0.2 + 0.4) / 3.0)
    assert float(level0["best_sr"][0]) == pytest.approx(2.5)
    assert float(level0["best_fwhm"][0]) == pytest.approx(25.0)
    assert float(level0["winner_ee_resolved"][0]) == pytest.approx(0.45)
    assert float(level0["winner_ee_averaged"][0]) == pytest.approx(0.85)
    assert float(level0["coverage_resolved"][0]) == pytest.approx(0.5)
    assert float(level0["coverage_averaged"][0]) == pytest.approx(0.5)


def test_winner_asterism_count_accumulates_center_owner_pixels_globally() -> None:
    level_maps = {
        0: np.zeros(12, dtype=MAPS_DTYPE),
        1: np.zeros(48, dtype=MAPS_DTYPE),
    }

    aggregation_module._add_winner_asterism_counts(
        level_maps,
        outer_level=0,
        inner_level=1,
        max_data_level=1,
        asterism_pix=np.asarray([0, 4], dtype=np.int64),
    )

    assert level_maps[0]["winner_asterism_count"][:2].tolist() == [1, 1]
    assert int(level_maps[1]["winner_asterism_count"][0]) == 1
    assert int(level_maps[1]["winner_asterism_count"][4]) == 1


def test_build_maps_writes_dense_maps_artifacts_with_expected_contract(tmp_path: Path) -> None:
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        outer_level=0,
        inner_level=1,
        max_data_level=1,
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )
    _write_gaia_tge_map(
        tmp_path / "gaia",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )
    for outer_pix in range(12):
        write_outer_artifact(
            outer_artifact_filename(build_path, load_build_definition(build_path), outer_pix),
            inner=_make_inner(
                gaia_a0=-1.0,
                star_count=1,
                ngs_count=2,
                best_ee=0.5,
                best_sr=0.4,
                best_fwhm=0.3,
                winner_ee_resolved=0.5,
                winner_ee_averaged=0.45,
                coverage_resolved=True,
                coverage_averaged=False,
            ),
            asterisms=_make_asterisms(empty=True),
        )

    level_maps = build_maps(build_path)

    assert sorted(level_maps) == [0, 1]
    for level in (0, 1):
        filename = maps_artifact_filename(build_path, level)
        assert filename.is_file()
        with h5py.File(filename, "r") as handle:
            assert MAPS_DATASET in handle
            dataset = handle[MAPS_DATASET]
            assert dataset.dtype == MAPS_DTYPE
            assert len(dataset) == 12 * (4**level)


def test_build_survey_extent_layers_preserves_maps_and_writes_dense_dataset(
    tmp_path: Path,
) -> None:
    overlay_a = _write_moc(tmp_path / "mocs" / "a.fits", level=1, pixs=[1])
    overlay_b = _write_moc(tmp_path / "mocs" / "b.fits", level=1, pixs=[2])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_overlays=[
            {"name": "ews", "moc_files": [str(overlay_a), str(overlay_b)]},
            {"name": "edf-north", "moc_files": [str(overlay_a)]},
        ],
    )
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    base_maps = np.zeros(48, dtype=MAPS_DTYPE)
    base_maps["pix"] = np.arange(48, dtype=np.int64)
    base_maps["star_count"] = np.arange(48, dtype=np.int64)
    from ao_sky.build.artifacts import write_maps_artifact

    write_maps_artifact(maps_artifact_filename(build_path, 0), maps=np.zeros(12, dtype=MAPS_DTYPE))
    write_maps_artifact(maps_artifact_filename(build_path, 1), maps=base_maps)
    layers = build_survey_extent_layers(build_path)

    assert list(layers) == [0, 1]
    with h5py.File(maps_artifact_filename(build_path, 1), "r") as handle:
        assert np.array_equal(handle[MAPS_DATASET][...], base_maps)
        assert SURVEY_EXTENT_DATASET in handle
        layer = handle[SURVEY_EXTENT_DATASET][...]
        assert layer.dtype.names == ("pix", "ews", "edf_north")
        assert np.array_equal(layer["pix"], np.arange(48, dtype=np.int64))
        assert bool(layer["ews"][1])
        assert bool(layer["ews"][2])
        assert bool(layer["edf_north"][1])
        assert not bool(layer["edf_north"][2])
