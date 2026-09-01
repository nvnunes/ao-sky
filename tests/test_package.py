"""Package baseline tests."""

from __future__ import annotations

import subprocess
import sys
from gzip import open as gzip_open
from importlib.metadata import version
from pathlib import Path

import numpy as np
from astropy.table import Table

import ao_sky.predict as predict
from ao_sky import __version__, describe_package
from ao_sky.asterisms import (
    AsterismExportSummary,
    AsterismSearchOptions,
    export_asterisms,
    find_asterisms,
    load_asterism_stars,
)
from ao_sky.build import (
    BuildDefinition,
    BuildInspection,
    BuildPaths,
    check_runtime_roots,
    fetch_gaia_data,
    init_build,
    inspect_build,
    load_build_definition,
    show_build,
)
from ao_sky.gaia import (
    GaiaHealpixStore,
    GaiaStoreConfig,
    GaiaSummaryStore,
    apply_proper_motion,
    fetch_gaia_store,
)


def test_package_root_exports_version() -> None:
    assert __version__ == "0.1.0"
    assert "proper-motion transforms" in describe_package()


def test_installed_metadata_matches_package_version() -> None:
    assert version("ao-sky") == __version__


def test_gaia_surface_is_importable() -> None:
    config = GaiaStoreConfig(root="data", release="dr3", healpix_level=6)
    store = GaiaHealpixStore(config)

    assert config.release == "dr3"
    assert store.config.healpix_level == 6
    assert apply_proper_motion is not None
    assert load_asterism_stars is not None
    assert find_asterisms is not None
    assert export_asterisms is not None
    assert AsterismExportSummary is not None
    assert AsterismSearchOptions().max_stars == 1
    assert check_runtime_roots is not None
    assert fetch_gaia_data is not None
    assert init_build is not None
    assert inspect_build is not None
    assert BuildDefinition is not None
    assert BuildInspection is not None
    assert BuildPaths is not None
    assert fetch_gaia_store is not None
    assert load_build_definition is not None
    assert show_build is not None


def test_prediction_public_api_exposes_only_the_production_model_family() -> None:
    assert callable(predict.get_model)
    assert callable(predict.predict_arrays)
    assert predict.PredictionArrayTelemetry is not None
    assert not hasattr(predict, "get_point_model")
    assert not hasattr(predict, "predict_point_arrays")
    assert not hasattr(predict, "_get_legacy_field_averaged_model")
    assert not hasattr(predict, "_predict_legacy_field_averaged_arrays")


def test_module_cli_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"ao-sky {__version__}"


def test_module_cli_help_lists_build_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "{status,init,run,restart,fetch-gaia,check,show}"
        in result.stdout
    )


def test_module_cli_help_lists_fetch_gaia_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "fetch-gaia", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--gaia-release" in result.stdout
    assert "--outer-level" in result.stdout
    assert "--force" in result.stdout


def test_module_cli_help_lists_check_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "check", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--gaia-root" in result.stdout
    assert "--model-root" in result.stdout


def test_module_cli_help_lists_run_worker_options() -> None:
    run_result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "run", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    restart_result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "restart", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--workers" in run_result.stdout
    assert "--workers" in restart_result.stdout
    assert "--gaia-cache-entries" in run_result.stdout
    assert "--gaia-cache-entries" in restart_result.stdout
    assert "--gaia-cache-mb" in run_result.stdout
    assert "--gaia-cache-mb" in restart_result.stdout
    assert "--worker-memory-limit-mb" not in run_result.stdout
    assert "--worker-memory-limit-mb" not in restart_result.stdout
    assert "--parent-memory-limit-mb" in run_result.stdout
    assert "--parent-memory-limit-mb" in restart_result.stdout
    assert "--telemetry" in run_result.stdout
    assert "--telemetry" in restart_result.stdout
    assert "--ao-sky-yaml" in run_result.stdout


def test_module_cli_run_rejects_invalid_worker_count(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "run", str(tmp_path), "--workers", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "workers must be at least 1" in result.stderr


def test_module_cli_can_init_and_show_build(tmp_path: Path) -> None:
    config = tmp_path / "build.yaml"
    config.write_text(
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
prediction:
  wavelength_micron: 1.654
  models:
    1star: point_one
    2star: point_two
    3star: point_three
  legacy_field_averaged_models:
    1star: mean_one
    2star: mean_two
    3star: mean_three
traversal:
  outer_level: 0
  inner_level: 1
gaia:
  release: dr3
  epoch: 2028.0
  max_bright_star_mag: 8.0
  max_bright_star_exclusion_arcsec: 240.0
maps:
  max_level: 1
asterism:
  winner_ee_epsilon: 0.01
best:
  seeing_baseline:
    wavelength_micron: 0.5
    sr: 0.0
    ee: 0.02
    fwhm_mas: 650.0
coverage:
  on_axis_ee_threshold: 0.4
  field_averaged_ee_threshold: 0.3
""".lstrip(),
        encoding="utf-8",
    )
    gaia_root = tmp_path / "gaia"
    build_root = tmp_path / "builds"
    model_root = tmp_path / "models"
    for model_name in (
        "point_one",
        "point_two",
        "point_three",
        "mean_one",
        "mean_two",
        "mean_three",
    ):
        model_root.mkdir(parents=True, exist_ok=True)
        (model_root / f"{model_name}.pt").write_bytes(model_name.encode("utf-8") + b":pt")
        (model_root / f"{model_name}_metadata.pkl").write_bytes(
            model_name.encode("utf-8") + b":metadata"
        )
    summary = np.zeros(
        12,
        dtype=[("outer_pix", "<i8"), ("star_count", "<i8"), ("loaded", "?")],
    )
    summary["outer_pix"] = np.arange(12, dtype=np.int64)
    summary["loaded"] = True
    GaiaSummaryStore(
        GaiaStoreConfig(root=gaia_root, release="dr3", healpix_level=0)
    ).write_summary(Table(summary))
    dust_file = gaia_root / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz"
    dust_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip_open(dust_file, "wt", encoding="utf-8") as handle:
        handle.write(
            "solution_id,healpix_id,healpix_level,a0,a0_uncertainty,a0_min,a0_max,"
            "num_tracers_used,optimum_hpx_flag,status\n"
        )
        for healpix_id in range(48):
            handle.write(
                f"1,{healpix_id},1,{healpix_id + 0.5},0.1,0.0,1.0,10,\"True\",0\n"
            )

    init_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ao_sky",
            "init",
            str(config),
            "--gaia-root",
            str(gaia_root),
            "--build-root",
            str(build_root),
            "--model-root",
            str(model_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    build_path = Path(init_result.stdout.strip().splitlines()[-1])
    assert build_path.name == "v1"

    show_result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "show", str(build_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"build: {build_path}" in show_result.stdout
    assert "status: initialized" in show_result.stdout
    assert "stage: traversal" in show_result.stdout


def test_module_cli_check_reports_valid_runtime_roots(tmp_path: Path) -> None:
    gaia_root = tmp_path / "gaia"
    build_root = tmp_path / "builds"
    model_root = tmp_path / "models"
    gaia_root.mkdir()
    build_root.mkdir()
    model_root.mkdir()
    dust_file = gaia_root / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz"
    dust_file.parent.mkdir(parents=True, exist_ok=True)
    dust_file.write_text("ok", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ao_sky",
            "check",
            "--gaia-root",
            str(gaia_root),
            "--build-root",
            str(build_root),
            "--model-root",
            str(model_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"gaia_root: OK {gaia_root.resolve()}" in result.stdout
    assert f"dust_root: OK {gaia_root.resolve()}" in result.stdout
