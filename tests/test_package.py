"""Package baseline tests."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version

from ao_sky import __version__, describe_package


def test_package_root_exports_version() -> None:
    assert __version__ == "0.1.0"
    assert "bootstrap stage" in describe_package()


def test_installed_metadata_matches_package_version() -> None:
    assert version("ao-sky") == __version__


def test_module_cli_reports_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ao_sky", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"ao-sky {__version__}"
