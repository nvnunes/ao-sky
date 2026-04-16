"""Build-model tests."""

from __future__ import annotations

from concurrent.futures import Future
import io
import json
from pathlib import Path
from gzip import open as gzip_open

import h5py
import numpy as np
import pytest
import yaml
from astropy.table import Table
from mocpy import MOC

from ao_sky.build import (
    check_runtime_roots,
    fetch_dust_data,
    fetch_gaia_data,
    fetch_model_data,
    init_build as real_init_build,
    restart_build,
    run_build,
    show_build,
)
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
    resolve_dust_root_only,
    resolve_gaia_root_only,
    resolve_runtime_root_candidates,
)
from ao_sky.build.control import (
    load_build_definition,
    load_build_roots,
    load_state,
    maps_artifact_filename,
    outer_artifact_filename,
    set_current_phase,
    summarize_build,
    update_state_row,
)
from ao_sky.build.scheduler import OuterPixelScheduler
from ao_sky.build.artifacts import write_outer_artifact
from ao_sky.build._models import TraversalTaskResult
from ao_sky.gaia import GaiaStoreConfig, GaiaSummaryStore


class _ImmediateExecutor:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.submitted: list[int] = []

    def __enter__(self) -> "_ImmediateExecutor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def shutdown(self, *, wait: bool = True, kill_workers: bool = False) -> None:
        return None

    def submit(self, fn, context, outer_pix: int) -> Future:
        self.submitted.append(int(outer_pix))
        future: Future = Future()
        try:
            future.set_result(fn(context, outer_pix))
        except Exception as exc:
            future.set_exception(exc)
        return future


class _FailingSubmitExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool = True, kill_workers: bool = False) -> None:
        self.shutdown_calls.append((wait, kill_workers))

    def submit(self, fn, context, outer_pix: int) -> Future:
        raise RuntimeError("pool submit broke")


def _write_build_definition(
    path: Path,
    *,
    outer_level: int = 0,
    inner_level: int = 1,
    min_galactic_latitude: float | None = None,
    max_data_level: int = 1,
    survey_extent_overlays: list[dict[str, object]] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "ao_system_short_name": "GNAO",
        "config_short_name": "baseline",
        "gaia_release": "dr3",
        "outer_level": outer_level,
        "inner_level": inner_level,
        "max_data_level": max_data_level,
        "epoch": 2028.0,
    }
    if min_galactic_latitude is not None:
        payload["min_galactic_latitude"] = min_galactic_latitude
    if survey_extent_overlays is not None:
        payload["survey_extent_overlays"] = survey_extent_overlays
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_legacy_config(path: Path) -> Path:
    path.write_text(
        """
ao_systems:
  - name: GNAO
    band: R
    fov: 120.0
    fov_1ngs: 60.0
    min_wfs: 2
    max_wfs: 3
    min_mag: 8.0
    nom_mag: 16.0
    max_mag: 18.5
    min_sep: 5.0
    max_sep: 120.0
asterisms_max_star_density: 6.0
asterisms_max_bright_star_mag: 8.0
asterisms_max_overlap: 0.66
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_legacy_config_with_models(path: Path) -> Path:
    path.write_text(
        """
ao_systems:
  - name: GNAO
    band: R
    fov: 120.0
    fov_1ngs: 60.0
    min_wfs: 2
    max_wfs: 3
    min_mag: 8.0
    nom_mag: 16.0
    max_mag: 18.5
    min_sep: 5.0
    max_sep: 120.0
    point_models:
      2star: point_two
      3star: shared_three
    models:
      2star: mean_two
      3star: shared_three
asterisms_max_star_density: 6.0
asterisms_max_bright_star_mag: 8.0
asterisms_max_overlap: 0.66
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_model_bundle(
    model_root: Path,
    model_name: str,
    *,
    include_data: bool = True,
    payload: bytes | None = None,
) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    content = payload if payload is not None else model_name.encode("utf-8")
    (model_root / f"{model_name}.pt").write_bytes(content + b":pt")
    (model_root / f"{model_name}_metadata.pkl").write_bytes(content + b":metadata")
    if include_data:
        (model_root / f"{model_name}_data.pkl").write_bytes(content + b":data")


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
    aosky_conf: Path | None = None,
) -> Path:
    definition, _ = load_build_definition_yaml(definition_filename)
    resolved_roots = resolve_runtime_root_candidates(
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=dust_root,
        model_root=model_root,
        aosky_conf=aosky_conf,
    )
    resolved_gaia_root = resolved_roots["gaia_root"]
    if resolved_gaia_root is not None:
        _write_gaia_summary(
            resolved_gaia_root,
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


def _init_model_snapshot_build(tmp_path: Path, model_root: Path | None = None) -> Path:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config_with_models(tmp_path / "legacy.yaml")
    return init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        model_root=model_root or tmp_path / "source-models",
        legacy_config_path=legacy,
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
    asterism_count: int = 0,
    best_ee: float = np.nan,
    best_sr: float = np.nan,
    best_fwhm: float = np.nan,
    winner_asterism_id: int = -1,
    winner_distance_arcsec: float = np.nan,
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
            np.full(nrows, asterism_count, dtype=np.int64),
            np.full(nrows, best_ee, dtype=np.float64),
            np.full(nrows, best_sr, dtype=np.float64),
            np.full(nrows, best_fwhm, dtype=np.float64),
            np.full(nrows, winner_asterism_id, dtype=np.int64),
            np.full(nrows, winner_distance_arcsec, dtype=np.float64),
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


def _make_asterisms(*, empty: bool = False) -> Table:
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

    return Table(
        [
            np.array([7], dtype=np.int64),
            np.array([10.0], dtype=np.float64),
            np.array([20.0], dtype=np.float64),
            np.array([2], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([101], dtype=np.int64),
            np.array([10.0], dtype=np.float64),
            np.array([20.0], dtype=np.float64),
            np.array([12.0], dtype=np.float64),
            np.array([202], dtype=np.int64),
            np.array([10.1], dtype=np.float64),
            np.array([20.1], dtype=np.float64),
            np.array([13.0], dtype=np.float64),
            np.array([-1], dtype=np.int64),
            np.array([-1.0], dtype=np.float64),
            np.array([-1.0], dtype=np.float64),
            np.array([-1.0], dtype=np.float64),
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
    definition = _write_build_definition(tmp_path / "build.yaml", min_galactic_latitude=90.0)
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        dust_root=tmp_path / "dust",
        legacy_config_path=legacy,
    )

    assert build_path.name == "GNAO-baseline-v1"
    assert (build_path / BUILD_FILENAME).is_file()
    assert (build_path / "build.log").is_file()
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


def test_init_build_uses_aosky_conf_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "aosky.conf").write_text(
        (
            f"gaia_root: {tmp_path / 'gaia'}\n"
            f"build_root: {tmp_path / 'builds'}\n"
            f"dust_root: {tmp_path / 'dust'}\n"
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

    assert build_path.parent == (tmp_path / "builds").resolve()


def test_resolve_dust_root_only_uses_aosky_conf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    expected = tmp_path / "dust"
    (project_root / "aosky.conf").write_text(
        f"dust_root: {expected}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)

    assert resolve_dust_root_only(dust_root=None) == expected.resolve()


def test_resolve_dust_root_only_requires_cli_or_conf(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match="dust_root must be provided"):
        resolve_dust_root_only(dust_root=None, cwd=tmp_path)


def test_resolve_gaia_root_only_uses_aosky_conf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    expected = tmp_path / "gaia"
    (project_root / "aosky.conf").write_text(
        f"gaia_root: {expected}\n",
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
    dust_root = tmp_path / "dust"
    model_root = tmp_path / "models"
    gaia_root.mkdir()
    build_root.mkdir()
    model_root.mkdir()
    _write_gaia_tge_map(dust_root, [(healpix_id, 1, 0.5) for healpix_id in range(48)])

    ok, report = check_runtime_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=dust_root,
        model_root=model_root,
    )

    assert ok
    assert report.splitlines() == [
        f"gaia_root: OK {gaia_root.resolve()}",
        f"build_root: OK {build_root.resolve()}",
        f"dust_root: OK {dust_root.resolve()}",
        f"model_root: OK {model_root.resolve()}",
    ]

    ok, report = check_runtime_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=tmp_path / "missing-dust",
        model_root=None,
    )

    assert not ok
    assert f"dust_root: MISSING {(tmp_path / 'missing-dust').resolve()}" in report
    assert "model_root: MISSING <unset>" in report

    empty_dust_root = tmp_path / "empty-dust"
    empty_dust_root.mkdir()
    ok, report = check_runtime_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=empty_dust_root,
        model_root=model_root,
    )
    assert not ok
    assert f"dust_root: MISSING {empty_dust_root.resolve()}" in report


def test_fetch_dust_data_resolves_dust_root_from_aosky_conf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    expected = tmp_path / "dust" / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz"
    (project_root / "aosky.conf").write_text(
        f"dust_root: {tmp_path / 'dust'}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(
        "ao_sky.build.environment.fetch_gaia_tge_dataset",
        lambda dust_root: expected.resolve(),
    )

    assert fetch_dust_data(dust_root=None) == expected.resolve()


def test_fetch_gaia_data_resolves_gaia_root_from_aosky_conf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    expected = tmp_path / "gaia" / "gaia-dr3-hpx0" / "summary.h5"
    (project_root / "aosky.conf").write_text(
        f"gaia_root: {tmp_path / 'gaia'}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(
        "ao_sky.build.environment.fetch_gaia_store",
        lambda config, force_reload=False, output=None: expected.resolve(),
    )

    assert fetch_gaia_data(gaia_root=None, gaia_release="dr3", outer_level=0) == expected.resolve()


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


def test_fetch_model_data_copies_configured_model_bundle_and_updates_metadata(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source-models"
    _write_model_bundle(source_root, "point_two")
    _write_model_bundle(source_root, "mean_two", include_data=False)
    _write_model_bundle(source_root, "shared_three")
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)

    manifest_path = fetch_model_data(build_path)

    destination_root = (build_path / "models").resolve()
    assert manifest_path == destination_root / "manifest.json"
    assert load_build_roots(build_path).model_root == destination_root
    assert (destination_root / "point_two.pt").is_file()
    assert (destination_root / "point_two_metadata.pkl").is_file()
    assert (destination_root / "point_two_data.pkl").is_file()
    assert (destination_root / "mean_two.pt").is_file()
    assert (destination_root / "mean_two_metadata.pkl").is_file()
    assert not (destination_root / "mean_two_data.pkl").exists()
    assert (destination_root / "shared_three.pt").is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_model_root"] == str(source_root.resolve())
    assert manifest["model_root"] == str(destination_root)
    assert manifest["ao_system"] == "GNAO"
    models = {item["name"]: item for item in manifest["models"]}
    assert set(models) == {"point_two", "mean_two", "shared_three"}
    assert models["shared_three"]["roles"] == ["mean:3star", "point:3star"]
    assert all(
        len(file_info["sha256"]) == 64
        for model_info in models.values()
        for file_info in model_info["files"]
    )


def test_fetch_model_data_uses_aosky_conf_model_root(
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
    (project_root / "aosky.conf").write_text(
        f"model_root: {source_root}\n",
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
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)

    with pytest.raises(BuildError, match="Required model file is missing"):
        fetch_model_data(build_path)

    assert load_build_roots(build_path).model_root == source_root.resolve()


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

    assert not (build_path / "models").exists()


def test_run_build_uses_build_local_model_root_after_fetch_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ao_sky.build.legacy_config import load_native_runtime as real_load_native_runtime

    source_root = tmp_path / "source-models"
    for model_name in ("point_two", "mean_two", "shared_three"):
        _write_model_bundle(source_root, model_name)
    build_path = _init_model_snapshot_build(tmp_path, model_root=source_root)
    fetch_model_data(build_path)
    seen_model_roots: list[Path] = []

    def capture_runtime(definition, *, legacy_config_path: Path, model_root: Path):
        seen_model_roots.append(Path(model_root).resolve())
        return real_load_native_runtime(
            definition,
            legacy_config_path=legacy_config_path,
            model_root=model_root,
        )

    monkeypatch.setattr("ao_sky.build.runner.load_native_runtime", capture_runtime)
    monkeypatch.setattr("ao_sky.build.runner.warm_model_cache", lambda runtime: None)
    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (_make_asterisms(), _make_inner()),
    )
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    assert seen_model_roots
    assert set(seen_model_roots) == {(build_path / "models").resolve()}


def test_load_build_definition_parses_overlay_specs_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    moc = _write_moc(tmp_path / "mocs" / "ews.fits", level=1, pixs=[1])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_extent_overlays=[
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
    assert overlay.moc_files == (moc.resolve(),)


def test_load_build_definition_rejects_duplicate_overlay_names(tmp_path: Path) -> None:
    moc_a = _write_moc(tmp_path / "mocs" / "a.fits", level=1, pixs=[1])
    moc_b = _write_moc(tmp_path / "mocs" / "b.fits", level=1, pixs=[2])
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_extent_overlays=[
            {"name": "EWS-Yr1", "moc_files": [str(moc_a)]},
            {"name": "ews_yr1", "moc_files": [str(moc_b)]},
        ],
    )

    with pytest.raises(BuildError, match="duplicate survey overlay name"):
        load_build_definition_yaml(definition)


def test_load_build_definition_requires_non_empty_overlay_paths(tmp_path: Path) -> None:
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_extent_overlays=[{"name": "ews", "moc_files": []}],
    )

    with pytest.raises(BuildError, match="moc_files must not be empty"):
        load_build_definition_yaml(definition)


def test_load_build_definition_rejects_blank_gaia_release(tmp_path: Path) -> None:
    definition = tmp_path / "build.yaml"
    definition.write_text(
        "\n".join(
            (
                "ao_system_short_name: GNAO",
                "config_short_name: baseline",
                "gaia_release: '   '",
                "outer_level: 0",
                "inner_level: 1",
                "max_data_level: 1",
                "epoch: 2028.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildError, match="gaia_release must be a non-empty string"):
        load_build_definition_yaml(definition)


def test_load_build_definition_rejects_non_finite_epoch(tmp_path: Path) -> None:
    definition = tmp_path / "build.yaml"
    definition.write_text(
        "\n".join(
            (
                "ao_system_short_name: GNAO",
                "config_short_name: baseline",
                "gaia_release: dr3",
                "outer_level: 0",
                "inner_level: 1",
                "max_data_level: 1",
                "epoch: .nan",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildError, match="epoch must be a finite number"):
        load_build_definition_yaml(definition)


def test_load_build_definition_rejects_max_data_level_above_inner_level(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml", max_data_level=2)

    with pytest.raises(BuildError, match="max_data_level must be less than or equal to inner_level"):
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

    with pytest.raises(BuildError, match="fetch-gaia --gaia-release dr3 --outer-level 0"):
        real_init_build(
            definition_filename=definition,
            gaia_root=tmp_path / "gaia",
            build_root=tmp_path / "builds",
            dust_root=tmp_path / "dust",
            legacy_config_path=legacy,
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
                asterism_count=5,
                best_ee=0.5,
                best_sr=0.4,
                best_fwhm=0.3,
                winner_asterism_id=7,
                winner_distance_arcsec=12.0,
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

    def fake_build_outer(build_path: Path, outer_pix: int) -> None:
        visited.append(int(outer_pix))
        update_state_row(build_path, int(outer_pix), traversal_status=WORK_STATUS_DONE)

    monkeypatch.setattr("ao_sky.build.runner.build_outer_pixel_products", fake_build_outer)
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path)

    assert visited[0] == 3


def test_run_build_rejects_invalid_worker_count(tmp_path: Path) -> None:
    build_path = tmp_path / "missing"

    with pytest.raises(BuildError, match="workers must be at least 1"):
        run_build(build_path, workers=0)


def test_run_build_parallel_workers_dispatch_distinct_pixels_and_update_parent_state(
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
    executors: list[_ImmediateExecutor] = []

    def fake_executor(workers: int) -> _ImmediateExecutor:
        executor = _ImmediateExecutor(workers)
        executors.append(executor)
        return executor

    def fake_task(context, outer_pix: int) -> TraversalTaskResult:
        return TraversalTaskResult(outer_pix=int(outer_pix), success=True)

    monkeypatch.setattr("ao_sky.build.runner._create_traversal_executor", fake_executor)
    monkeypatch.setattr("ao_sky.build.runner._run_outer_pixel_traversal_task", fake_task)
    monkeypatch.setattr("ao_sky.build.runner.build_maps", lambda build_path: {})

    run_build(build_path, workers=3)

    assert executors[0].workers == 3
    assert sorted(executors[0].submitted) == list(range(12))
    state = load_state(build_path)
    assert np.all(state["traversal_status"] == WORK_STATUS_DONE)
    assert np.all(np.asarray(state["traversal_attempt_count"], dtype=np.int64) == 1)
    summary = summarize_build(build_path)
    assert summary["build_status"] == "completed"
    assert summary["current_phase"] == BUILD_PHASE_AGGREGATION
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "run start phase=traversal repaired_stale_running=0 workers=3" in log_text
    assert "phase=traversal outer_pix=0 dispatched" in log_text
    assert "phase=traversal outer_pix=0 completed" in log_text


def test_run_build_parallel_workers_continue_after_failed_outer_pixel(
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
    for outer_pix in range(3, len(state)):
        update_state_row(build_path, outer_pix, traversal_status=WORK_STATUS_DONE)

    monkeypatch.setattr(
        "ao_sky.build.runner._create_traversal_executor",
        lambda workers: _ImmediateExecutor(workers),
    )

    def fake_task(context, outer_pix: int) -> TraversalTaskResult:
        if outer_pix == 0:
            return TraversalTaskResult(outer_pix=outer_pix, success=False, error_message="boom")
        return TraversalTaskResult(outer_pix=outer_pix, success=True)

    monkeypatch.setattr("ao_sky.build.runner._run_outer_pixel_traversal_task", fake_task)

    run_build(build_path, workers=3)

    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_FAILED
    assert int(state["traversal_status"][1]) == WORK_STATUS_DONE
    assert int(state["traversal_status"][2]) == WORK_STATUS_DONE
    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal outer_pix=0 failed: boom" in log_text
    assert "run complete phase=traversal status=failed" in log_text


def test_run_build_parallel_executor_creation_failure_marks_build_failed(
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

    def fail_create(workers: int):
        raise RuntimeError("semaphore denied")

    monkeypatch.setattr("ao_sky.build.runner._create_traversal_executor", fail_create)

    with pytest.raises(RuntimeError, match="semaphore denied"):
        run_build(build_path, workers=3)

    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    state = load_state(build_path)
    assert not np.any(state["traversal_status"] == WORK_STATUS_RUNNING)
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal infrastructure failed: semaphore denied" in log_text
    assert "run complete phase=traversal status=failed reset_running=0" in log_text


def test_run_build_parallel_submit_failure_resets_running_rows(
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
    executor = _FailingSubmitExecutor()
    monkeypatch.setattr(
        "ao_sky.build.runner._create_traversal_executor",
        lambda workers: executor,
    )

    with pytest.raises(RuntimeError, match="pool submit broke"):
        run_build(build_path, workers=3)

    assert executor.shutdown_calls == [(False, True)]
    state = load_state(build_path)
    assert int(state["traversal_status"][0]) == WORK_STATUS_PENDING
    assert int(state["traversal_attempt_count"][0]) == 1
    assert not np.any(state["traversal_status"] == WORK_STATUS_RUNNING)
    summary = summarize_build(build_path)
    assert summary["build_status"] == "failed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "phase=traversal infrastructure failed: pool submit broke" in log_text
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
    definition = _write_build_definition(tmp_path / "build.yaml", min_galactic_latitude=90.0)
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
            _make_inner(star_count=4, ngs_count=0, asterism_count=0),
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


def test_restart_build_uses_latest_lineage_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "builds"
    (build_root / "GNAO-baseline-v1").mkdir(parents=True)
    latest = build_root / "GNAO-baseline-v2"
    latest.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_run_build(build_path: Path, *, workers: int = 1) -> Path:
        captured["workers"] = workers
        return build_path

    monkeypatch.setattr("ao_sky.build.runner.run_build", fake_run_build)

    restarted = restart_build(
        ao_system_short_name="GNAO",
        config_short_name="baseline",
        build_root=build_root,
        workers=3,
    )

    assert restarted == latest
    assert captured == {"workers": 3}


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
        assert kwargs["dust_root"] == tmp_path / "dust"
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
    assert "phase=traversal outer_pix=0 failed: boom" in log_text
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
        tmp_path / "dust",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (
            _make_asterisms(empty=True),
            _make_inner(
                star_count=outer_pix + 1,
                ngs_count=outer_pix % 2,
                asterism_count=outer_pix + 2,
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
        survey_extent_overlays=[{"name": "ews", "moc_files": [str(overlay)]}],
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
        tmp_path / "dust",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_traversal_products",
        lambda store, runtime, outer_pix, **kwargs: (
            _make_asterisms(empty=True),
            _make_inner(
                star_count=outer_pix + 1,
                ngs_count=outer_pix % 2,
                asterism_count=outer_pix + 2,
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
        tmp_path / "dust",
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
        survey_extent_overlays=[{"name": "ews", "moc_files": [str(overlay)]}],
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
        survey_extent_overlays=[{"name": "ews", "moc_files": [str(overlay)]}],
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
    definition = _write_build_definition(
        tmp_path / "build.yaml",
        survey_extent_overlays=[{"name": "ews", "moc_files": ["/tmp/ews.fits"]}],
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


def test_aggregate_maps_recomputes_dust_and_reduces_fields_by_type(tmp_path: Path) -> None:
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
        tmp_path / "dust",
        [
            (0, 1, 0.5),
            (1, 1, 1.5),
            (2, 1, 2.5),
        ],
    )
    inner = Table(
        [
            np.arange(4, dtype=np.int64),
            np.zeros(4, dtype=np.float64),
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int64),
            np.array([4, 5, 6, 7], dtype=np.int64),
            np.array([0.1, 0.2, np.nan, 0.4], dtype=np.float64),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
            np.full(4, -1, dtype=np.int64),
            np.full(4, np.nan, dtype=np.float64),
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
    write_outer_artifact(
        outer_artifact_filename(build_path, load_build_definition(build_path), 0),
        inner=inner,
        asterisms=_make_asterisms(empty=True),
    )

    level_maps = aggregate_maps(build_path, outer_pixs=[0])

    assert sorted(level_maps) == [0, 1]
    level1 = level_maps[1]
    assert level1["gaia_A0"][:3].tolist() == [0.5, 1.5, 2.5]
    assert np.isnan(level1["gaia_A0"][3])
    assert level1["star_count"][:4].tolist() == [1, 2, 3, 4]
    assert level1["coverage_resolved"][:4].tolist() == [1.0, 0.0, 1.0, 0.0]

    level0 = level_maps[0]
    assert int(level0["star_count"][0]) == 10
    assert int(level0["ngs_count"][0]) == 2
    assert int(level0["asterism_count"][0]) == 22
    assert np.isnan(level0["gaia_A0"][0])
    assert np.isnan(level0["best_ee"][0])
    assert float(level0["best_sr"][0]) == pytest.approx(2.5)
    assert float(level0["best_fwhm"][0]) == pytest.approx(25.0)
    assert float(level0["winner_ee_resolved"][0]) == pytest.approx(0.45)
    assert float(level0["winner_ee_averaged"][0]) == pytest.approx(0.85)
    assert float(level0["coverage_resolved"][0]) == pytest.approx(0.5)
    assert float(level0["coverage_averaged"][0]) == pytest.approx(0.5)


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
        tmp_path / "dust",
        [(healpix_id, 1, healpix_id + 0.25) for healpix_id in range(48)],
    )
    for outer_pix in range(12):
        write_outer_artifact(
            outer_artifact_filename(build_path, load_build_definition(build_path), outer_pix),
            inner=_make_inner(
                gaia_a0=-1.0,
                star_count=1,
                ngs_count=2,
                asterism_count=3,
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
        survey_extent_overlays=[
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
