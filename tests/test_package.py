"""Package baseline tests."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

from ao_sky import __version__, describe_package
from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig


def test_package_root_exports_version() -> None:
    assert __version__ == "0.1.0"
    assert "raw Gaia store" in describe_package()


def test_installed_metadata_matches_package_version() -> None:
    assert version("ao-sky") == __version__


def test_gaia_surface_is_importable() -> None:
    config = GaiaStoreConfig(root="data", release="dr3", healpix_level=6)
    store = GaiaHealpixStore(config)

    assert config.release == "dr3"
    assert store.config.healpix_level == 6


def test_module_cli_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"ao-sky {__version__}"
