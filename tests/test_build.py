"""Build-model tests."""

from __future__ import annotations

from pathlib import Path

import h5py
from astropy.table import Table
import numpy as np
import pytest

from ao_sky.build import init_build, restart_build, run_build
from ao_sky.build._exceptions import BuildError
from ao_sky.build._constants import (
    ARTIFACT_STATE_DONE,
    ARTIFACT_STATE_SKIPPED,
    BUILD_FILENAME,
    OUTER_DATASET_ASTERISMS,
    OUTER_DATASET_INNER,
    WORK_STATUS_DONE,
)
from ao_sky.build.config import load_build_definition as load_build_definition_yaml
from ao_sky.build.control import (
    load_build_definition,
    load_state,
    outer_artifact_filename,
    update_state_row,
)


def _write_build_definition(path: Path, *, min_galactic_latitude: float | None = None) -> Path:
    lines = [
        "ao_system_short_name: GNAO",
        "config_short_name: baseline",
        "gaia_release: dr3",
        "outer_level: 0",
        "inner_level: 1",
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


def test_init_build_creates_root_and_full_sky_state(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml", min_galactic_latitude=90.0)
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        legacy_config_path=legacy,
    )

    assert build_path.name == "GNAO-baseline-v1"
    assert (build_path / BUILD_FILENAME).is_file()
    assert (build_path / "build.log").is_file()
    assert (build_path / "hpx0-1").is_dir()

    state = load_state(build_path)
    assert len(state) == 12
    assert state["outer_pix"].tolist() == list(range(12))
    assert np.all(state["asterisms_state"] == ARTIFACT_STATE_SKIPPED)
    assert np.all(state["work_status"] == 0)


def test_init_build_uses_aosky_conf_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "aosky.conf").write_text(
        f"gaia_root: {tmp_path / 'gaia'}\nbuild_root: {tmp_path / 'builds'}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project_root)

    build_path = init_build(
        definition_filename=definition,
        gaia_root=None,
        build_root=None,
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
                "epoch: .nan",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BuildError, match="epoch must be a finite number"):
        load_build_definition_yaml(definition)


def test_build_outer_artifact_is_written_and_state_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        legacy_config_path=legacy,
    )

    stars = Table()
    ngs = Table()
    asterisms = Table(
        [
            np.array([1], dtype=np.int64),
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
    inner = Table(
        [
            np.array([0, 1, 2, 3], dtype=np.int64),
            np.array([1, 0, 2, 0], dtype=np.int64),
            np.array([1, 0, 1, 0], dtype=np.int64),
        ],
        names=("pix", "ngs_count", "asterism_count"),
    )

    monkeypatch.setattr(
        "ao_sky.build.runner.build_outer_pixel_asterisms",
        lambda store, runtime, outer_pix: (stars, ngs, asterisms),
    )
    monkeypatch.setattr(
        "ao_sky.build.runner.build_inner_table",
        lambda store, runtime, outer_pix, asterisms=None: inner,
    )

    run_build(build_path)

    state = load_state(build_path)
    assert int(state["work_status"][0]) == WORK_STATUS_DONE
    assert int(state["outer_file_state"][0]) == ARTIFACT_STATE_DONE
    assert int(state["inner_state"][0]) == ARTIFACT_STATE_DONE
    assert int(state["asterisms_state"][0]) == ARTIFACT_STATE_DONE

    outer_filename = outer_artifact_filename(
        build_path,
        load_build_definition(build_path),
        0,
    )
    assert outer_filename.is_file()
    with h5py.File(outer_filename, "r") as handle:
        assert OUTER_DATASET_INNER in handle
        assert OUTER_DATASET_ASTERISMS in handle


def test_skipped_asterisms_still_write_outer_file_with_inner_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml", min_galactic_latitude=90.0)
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        legacy_config_path=legacy,
    )

    inner = Table(
        [
            np.array([0, 1, 2, 3], dtype=np.int64),
            np.array([0, 0, 0, 0], dtype=np.int64),
            np.array([0, 0, 0, 0], dtype=np.int64),
        ],
        names=("pix", "ngs_count", "asterism_count"),
    )

    def _unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("asterism search should not run for skipped pixels")

    monkeypatch.setattr("ao_sky.build.runner.build_outer_pixel_asterisms", _unexpected)
    monkeypatch.setattr(
        "ao_sky.build.runner.build_inner_table",
        lambda store, runtime, outer_pix, asterisms=None: inner,
    )

    run_build(build_path)

    outer_filename = outer_artifact_filename(
        build_path,
        load_build_definition(build_path),
        0,
    )
    with h5py.File(outer_filename, "r") as handle:
        assert OUTER_DATASET_INNER in handle
        assert OUTER_DATASET_ASTERISMS not in handle


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


def test_update_state_row_rejects_overlong_error_message(tmp_path: Path) -> None:
    definition = _write_build_definition(tmp_path / "build.yaml")
    legacy = _write_legacy_config(tmp_path / "legacy.yaml")
    build_path = init_build(
        definition_filename=definition,
        gaia_root=tmp_path / "gaia",
        build_root=tmp_path / "builds",
        legacy_config_path=legacy,
    )

    with pytest.raises(BuildError, match="last_error_message exceeds persisted limit"):
        update_state_row(build_path, 0, last_error_message="x" * 1025)
