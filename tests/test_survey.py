"""Survey-overlay tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mocpy import MOC

from ao_sky.survey import build_survey_extent_dataset, normalize_survey_extent_overlays


def _write_moc(path: Path, *, level: int, pixs: list[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    moc = MOC.from_healpix_cells(
        np.asarray(pixs, dtype=np.uint64),
        level,
        max_depth=level,
    )
    moc.save(str(path), format="fits", overwrite=True)
    return path


def test_build_survey_extent_dataset_marks_single_file_pixels(tmp_path: Path) -> None:
    moc = _write_moc(tmp_path / "single.fits", level=1, pixs=[1, 3])
    overlays = normalize_survey_extent_overlays(
        [{"name": "EWS-Yr1", "moc_files": [str(moc)]}],
        base_dir=tmp_path,
    )

    data = build_survey_extent_dataset(1, overlays)

    assert data.dtype.names == ("pix", "ews_yr1")
    assert np.array_equal(data["pix"], np.arange(48, dtype=np.int64))
    assert bool(data["ews_yr1"][1])
    assert bool(data["ews_yr1"][3])
    assert not bool(data["ews_yr1"][0])


def test_build_survey_extent_dataset_unions_multiple_moc_files(tmp_path: Path) -> None:
    moc_a = _write_moc(tmp_path / "a.fits", level=1, pixs=[1])
    moc_b = _write_moc(tmp_path / "b.fits", level=1, pixs=[2])
    overlays = normalize_survey_extent_overlays(
        [{"name": "ews", "moc_files": [str(moc_a), str(moc_b)]}],
        base_dir=tmp_path,
    )

    data = build_survey_extent_dataset(1, overlays)

    assert bool(data["ews"][1])
    assert bool(data["ews"][2])
    assert not bool(data["ews"][3])


def test_build_survey_extent_dataset_degrades_to_requested_level(tmp_path: Path) -> None:
    moc = _write_moc(tmp_path / "single.fits", level=1, pixs=[1, 3])
    overlays = normalize_survey_extent_overlays(
        [{"name": "ews", "moc_files": [str(moc)]}],
        base_dir=tmp_path,
    )

    data = build_survey_extent_dataset(0, overlays)

    assert data.dtype.names == ("pix", "ews")
    assert np.array_equal(data["pix"], np.arange(12, dtype=np.int64))
    assert bool(data["ews"][0])
    assert not bool(data["ews"][1])
