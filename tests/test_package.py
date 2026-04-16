"""Package baseline tests."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from ao_sky import __version__, describe_package
from ao_sky.asterisms import AsterismSearchOptions, find_asterisms, load_asterism_stars
from ao_sky.build import check_runtime_roots, fetch_dust_data, init_build, load_build_definition, show_build
from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig, apply_proper_motion


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
    assert AsterismSearchOptions().max_stars == 1
    assert check_runtime_roots is not None
    assert fetch_dust_data is not None
    assert init_build is not None
    assert load_build_definition is not None
    assert show_build is not None


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
    assert "{status,init,run,restart,fetch-dust,check,show}" in result.stdout


def test_module_cli_help_lists_fetch_dust_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "fetch-dust", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--dust-root" in result.stdout
    assert "--aosky-conf" in result.stdout


def test_module_cli_help_lists_check_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "check", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--gaia-root" in result.stdout
    assert "--model-root" in result.stdout


def test_module_cli_can_init_and_show_build(tmp_path: Path) -> None:
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
                "epoch: 2028.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "\n".join(
            (
                "ao_systems:",
                "  - name: GNAO",
                "    band: R",
                "    fov: 120.0",
                "    fov_1ngs: 60.0",
                "    min_wfs: 2",
                "    max_wfs: 3",
                "    min_mag: 8.0",
                "    nom_mag: 16.0",
                "    max_mag: 18.5",
                "    min_sep: 5.0",
                "    max_sep: 120.0",
                "asterisms_max_star_density: 6.0",
                "asterisms_max_bright_star_mag: 8.0",
                "asterisms_max_overlap: 0.66",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    gaia_root = tmp_path / "gaia"
    build_root = tmp_path / "builds"
    dust_root = tmp_path / "dust"

    init_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ao_sky",
            "init",
            str(definition),
            "--gaia-root",
            str(gaia_root),
            "--build-root",
            str(build_root),
            "--dust-root",
            str(dust_root),
            "--legacy-config",
            str(legacy),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    build_path = Path(init_result.stdout.strip().splitlines()[-1])
    assert build_path.name == "GNAO-baseline-v1"

    show_result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "show", str(build_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"build: {build_path}" in show_result.stdout
    assert "status: initialized" in show_result.stdout


def test_module_cli_check_reports_valid_runtime_roots(tmp_path: Path) -> None:
    gaia_root = tmp_path / "gaia"
    build_root = tmp_path / "builds"
    dust_root = tmp_path / "dust"
    model_root = tmp_path / "models"
    gaia_root.mkdir()
    build_root.mkdir()
    model_root.mkdir()
    dust_file = dust_root / "gaia_tge" / "TotalGalacticExtinctionMap_001.csv.gz"
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
            "--dust-root",
            str(dust_root),
            "--model-root",
            str(model_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert f"gaia_root: OK {gaia_root.resolve()}" in result.stdout
    assert f"dust_root: OK {dust_root.resolve()}" in result.stdout
