from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ao_sky._paths import get_outer_pixel_bucket_path
from ao_sky.build._constants import ASTERISMS_DTYPE, STATE_DTYPE

LEGACY_INNER_DTYPE = np.dtype(
    [
        ("pix", "<i8"),
        ("gaia_A0", "<f8"),
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("best_ee", "<f8"),
        ("best_sr", "<f8"),
        ("best_fwhm", "<f8"),
        ("winner_asterism_id", "<i8"),
        ("winner_ee_resolved", "<f8"),
        ("winner_ee_averaged", "<f8"),
        ("coverage_resolved", "?"),
        ("coverage_averaged", "?"),
    ]
)
LEGACY_MAPS_DTYPE = np.dtype(
    [
        ("pix", "<i8"),
        ("gaia_A0", "<f8"),
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("winner_asterism_count", "<i8"),
        ("best_sr", "<f8"),
        ("best_ee", "<f8"),
        ("best_fwhm", "<f8"),
        ("winner_ee_resolved", "<f8"),
        ("winner_ee_averaged", "<f8"),
        ("coverage_resolved", "<f8"),
        ("coverage_averaged", "<f8"),
    ]
)


@pytest.fixture
def schema_v2_build(tmp_path: Path) -> Path:
    """Create a compact, completed schema-version-2 build fixture."""

    build_path = tmp_path / "legacy" / "v2"
    build_path.mkdir(parents=True)
    (build_path / "dust").mkdir()
    (build_path / "models").mkdir()
    gaia_root = tmp_path / "gaia"
    gaia_root.mkdir()
    source_config = tmp_path / "ao-sky.yaml"
    source_config.write_text("schema_version: 2\n", encoding="utf-8")
    runtime_config = build_path / "build.yaml"
    runtime_config.write_text(_legacy_runtime_config(), encoding="utf-8")
    (build_path / "build.log").write_text("completed\n", encoding="utf-8")

    state = np.zeros(12, dtype=STATE_DTYPE)
    state["outer_pix"] = np.arange(12, dtype=np.int64)
    state["gaia_loading_status"] = 2
    state["traversal_status"] = 2
    with h5py.File(build_path / "build.h5", "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.create_dataset(
            "config_yaml",
            data=source_config.read_text(encoding="utf-8"),
            dtype=h5py.string_dtype("utf-8"),
        )
        config = metadata.create_group("config")
        values = {
            "build_name": "v2",
            "lineage_name": "legacy",
            "lineage_version": 2,
            "gaia_release": "dr3",
            "gaia_root": str(gaia_root),
            "build_root": str(tmp_path),
            "dust_root": "dust",
            "model_root": "models",
            "runtime_config_path": str(runtime_config),
            "runtime_config_source_path": str(source_config),
            "outer_level": 0,
            "inner_level": 1,
            "max_data_level": 0,
            "survey_extent_overlays_yaml": "[]\n",
            "layout_version": 2,
            "build_status": "completed",
            "current_phase": "augmentation",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        for name, value in values.items():
            _write_scalar(config, name, value)
        handle.create_group("state").create_dataset("outer_pixels", data=state)

    manifest = {
        "schema_version": 1,
        "models": [
            {"name": "point-one", "roles": ["resolved:1star"], "files": []},
            {"name": "mean-one", "roles": ["averaged:1star"], "files": []},
        ],
    }
    (build_path / "models" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    inner = np.zeros(4, dtype=LEGACY_INNER_DTYPE)
    inner["pix"] = np.arange(4, dtype=np.int64)
    inner["star_count"] = [1, 2, 3, 4]
    inner["ngs_count"] = [1, 1, 1, 1]
    inner["winner_asterism_id"] = 7
    inner["winner_ee_resolved"] = [0.41, 0.42, 0.43, 0.44]
    inner["winner_ee_averaged"] = [0.31, 0.32, 0.33, 0.34]
    inner["coverage_resolved"] = [True, True, True, True]
    inner["coverage_averaged"] = [True, True, True, True]
    asterisms = np.zeros(1, dtype=ASTERISMS_DTYPE)
    asterisms["asterism_id"] = 7
    asterisms["num_stars"] = 1
    asterisms["pix"] = 0
    asterisms["star1_source_id"] = 1001
    outer_path = (
        build_path
        / "hpx0-1"
        / get_outer_pixel_bucket_path(0, 0)
        / "outer.h5"
    )
    outer_path.parent.mkdir(parents=True)
    with h5py.File(outer_path, "w") as handle:
        handle.create_dataset("inner", data=inner)
        handle.create_dataset("asterisms", data=asterisms)

    maps = np.zeros(12, dtype=LEGACY_MAPS_DTYPE)
    maps["pix"] = np.arange(12, dtype=np.int64)
    maps["winner_ee_resolved"] = np.linspace(0.4, 0.51, 12)
    maps["winner_ee_averaged"] = np.linspace(0.3, 0.41, 12)
    maps["coverage_resolved"] = np.linspace(0.5, 1.0, 12)
    maps["coverage_averaged"] = np.linspace(0.4, 0.9, 12)
    with h5py.File(build_path / "maps-hpx0.h5", "w") as handle:
        handle.create_dataset("maps", data=maps)
        handle.create_dataset(
            "survey_extent",
            data=np.zeros(12, dtype=[("pix", "<i8"), ("TEST", "?")]),
        )

    return build_path


def _write_scalar(group: h5py.Group, name: str, value: object) -> None:
    if isinstance(value, str):
        group.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(name, data=value)


def _legacy_runtime_config() -> str:
    return """\
schema_version: 2
ao_system:
  band: R
  fov_arcsec: 120.0
  lgs: []
  min_wfs: 1
  max_wfs: 1
  min_mag: 8.0
  max_mag: 18.5
  min_sep_arcsec: 5.0
prediction:
  wavelength_micron: 1.654
  resolved_models:
    1star: point-one
  averaged_models:
    1star: mean-one
traversal:
  outer_level: 0
  inner_level: 1
gaia:
  epoch: 2028.0
asterism:
  winner_ee_epsilon: 0.01
best:
  seeing_baseline:
    wavelength_micron: 0.5
    sr: 0.0
    ee: 0.02
    fwhm_mas: 650.0
coverage:
  resolved_ee_threshold: 0.4
  averaged_ee_threshold: 0.3
"""
