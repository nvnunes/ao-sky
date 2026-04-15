"""Survey-extent overlay normalization and dense layer construction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from mocpy import MOC

from ._exceptions import SurveyError
from ._models import SurveyExtentOverlaySpec


def _normalize_overlay_name(name: object) -> str:
    normalized = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise SurveyError("survey overlay name must normalize to a non-empty identifier")
    return normalized


def _normalize_overlay_paths(
    moc_files: object,
    *,
    base_dir: Path,
) -> tuple[Path, ...]:
    if isinstance(moc_files, (str, Path)):
        raw_paths = [moc_files]
    elif isinstance(moc_files, Sequence):
        raw_paths = list(moc_files)
    else:
        raise SurveyError("survey overlay moc_files must be a path or a list of paths")
    if not raw_paths:
        raise SurveyError("survey overlay moc_files must not be empty")

    normalized: list[Path] = []
    for item in raw_paths:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        normalized.append(path)
    return tuple(normalized)


def normalize_survey_extent_overlays(
    raw_overlays: object,
    *,
    base_dir: Path,
) -> tuple[SurveyExtentOverlaySpec, ...]:
    """Normalize human-authored survey-extent overlay definitions."""

    if raw_overlays in (None, ""):
        return ()
    if not isinstance(raw_overlays, list):
        raise SurveyError("survey_extent_overlays must be a list")

    overlays: list[SurveyExtentOverlaySpec] = []
    seen_names: set[str] = set()
    for raw in raw_overlays:
        if not isinstance(raw, dict):
            raise SurveyError("each survey overlay entry must be a mapping")
        if "name" not in raw:
            raise SurveyError("survey overlay entries require a name")
        if "moc_files" not in raw:
            raise SurveyError("survey overlay entries require moc_files")

        name = _normalize_overlay_name(raw["name"])
        if name in seen_names:
            raise SurveyError(f"duplicate survey overlay name: {name}")
        seen_names.add(name)

        overlays.append(
            SurveyExtentOverlaySpec(
                name=name,
                moc_files=_normalize_overlay_paths(raw["moc_files"], base_dir=base_dir),
            )
        )

    return tuple(overlays)


def survey_extent_dtype(
    overlays: Sequence[SurveyExtentOverlaySpec],
) -> np.dtype:
    """Return the dense survey-extent dtype for one level."""

    return np.dtype([("pix", "<i8"), *[(overlay.name, "?") for overlay in overlays]])


def _load_overlay_moc(overlay: SurveyExtentOverlaySpec) -> MOC:
    mocs = [MOC.from_fits(str(filename)) for filename in overlay.moc_files]
    if not mocs:
        raise SurveyError(f"overlay {overlay.name!r} has no MOC files")
    if len(mocs) == 1:
        return mocs[0]
    return mocs[0].union(*mocs[1:])


def build_survey_extent_dataset(
    level: int,
    overlays: Sequence[SurveyExtentOverlaySpec],
) -> np.ndarray:
    """Build one dense survey-extent overlay dataset for one HEALPix level."""

    npix = 12 * (4**int(level))
    data = np.zeros(npix, dtype=survey_extent_dtype(overlays))
    data["pix"] = np.arange(npix, dtype=np.int64)

    for overlay in overlays:
        moc = _load_overlay_moc(overlay)
        if int(level) < int(moc.max_order):
            level_moc = moc.degrade_to_order(int(level))
        else:
            level_moc = moc
        level_pixs = np.asarray(level_moc.flatten(), dtype=np.int64)
        data[overlay.name][level_pixs] = True

    return data
