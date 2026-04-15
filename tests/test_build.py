"""Build-model tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.table import Table

from ao_sky.build import init_build, restart_build, run_build, show_build
from ao_sky.build._constants import (
    BUILD_FILENAME,
    BUILD_PHASE_TRAVERSAL,
    OUTER_DATASET_ASTERISMS,
    OUTER_DATASET_INNER,
    STATE_DTYPE,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ao_sky.build._exceptions import BuildError
from ao_sky.build.config import load_build_definition as load_build_definition_yaml
from ao_sky.build.control import (
    load_build_definition,
    load_state,
    outer_artifact_filename,
    summarize_build,
    update_state_row,
)
from ao_sky.build.scheduler import OuterPixelScheduler


def _write_build_definition(
    path: Path,
    *,
    min_galactic_latitude: float | None = None,
    max_data_level: int = 1,
) -> Path:
    lines = [
        "ao_system_short_name: GNAO",
        "config_short_name: baseline",
        "gaia_release: dr3",
        "outer_level: 0",
        "inner_level: 1",
        f"max_data_level: {max_data_level}",
        "epoch: 2028.0",
    ]
    if min_galactic_latitude is not None:
        lines.append(f"min_galactic_latitude: {min_galactic_latitude}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def test_scheduler_uses_lowest_unfinished_seed_when_no_anchor() -> None:
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

    run_build(build_path)

    state = load_state(build_path)
    assert np.all(state["traversal_status"] == WORK_STATUS_DONE)

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

    monkeypatch.setattr("ao_sky.build.runner.run_build", lambda build_path: build_path)

    restarted = restart_build(
        ao_system_short_name="GNAO",
        config_short_name="baseline",
        build_root=build_root,
    )

    assert restarted == latest


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

    run_build(build_path)

    summary = summarize_build(build_path)
    assert summary["build_status"] == "completed"
    log_text = (build_path / "build.log").read_text(encoding="utf-8")
    assert "run complete phase=traversal status=completed" in log_text


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
