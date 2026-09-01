"""Constants for persisted build contracts."""

from __future__ import annotations

from typing import Final

import numpy as np

BUILD_LAYOUT_VERSION: Final[int] = 3
RUNTIME_CONFIG_SCHEMA_VERSION: Final[int] = 3
ARTIFACT_LAYOUT_VERSION_ATTRIBUTE: Final[str] = "layout_version"
BUILD_FILENAME: Final[str] = "build.h5"
BUILD_LOG_FILENAME: Final[str] = "build.log"
RUNTIME_CONFIG_FILENAME: Final[str] = "build.yaml"
MODEL_SNAPSHOT_DIRNAME: Final[str] = "models"
MODEL_SNAPSHOT_MANIFEST_FILENAME: Final[str] = "manifest.json"
SURVEY_SNAPSHOT_DIRNAME: Final[str] = "surveys"
SURVEY_SNAPSHOT_MANIFEST_FILENAME: Final[str] = "manifest.json"
OUTER_FILENAME: Final[str] = "outer.h5"
OUTER_DATASET_ASTERISMS: Final[str] = "asterisms"
OUTER_DATASET_INNER: Final[str] = "inner"
MAPS_FILENAME_TEMPLATE: Final[str] = "maps-hpx{level}.h5"
MAPS_DATASET: Final[str] = "maps"
SURVEY_EXTENT_DATASET: Final[str] = "survey_extent"
BUILD_PHASE_GAIA_LOADING: Final[str] = "gaia_loading"
BUILD_PHASE_TRAVERSAL: Final[str] = "traversal"
BUILD_PHASE_AGGREGATION: Final[str] = "aggregation"
BUILD_PHASE_AUGMENTATION: Final[str] = "augmentation"

WORK_STATUS_PENDING: Final[int] = 0
WORK_STATUS_RUNNING: Final[int] = 1
WORK_STATUS_DONE: Final[int] = 2
WORK_STATUS_FAILED: Final[int] = 3

BUILD_STATUS_INITIALIZED: Final[str] = "initialized"
BUILD_STATUS_RUNNING: Final[str] = "running"
BUILD_STATUS_COMPLETED: Final[str] = "completed"
BUILD_STATUS_FAILED: Final[str] = "failed"

STATE_ERROR_MAX_BYTES: Final[int] = 1024

STATE_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("outer_pix", "<i8"),
        ("gaia_loading_status", "<i2"),
        ("gaia_loading_attempt_count", "<i4"),
        ("gaia_loading_last_error_message", f"S{STATE_ERROR_MAX_BYTES}"),
        ("traversal_status", "<i2"),
        ("traversal_attempt_count", "<i4"),
        ("traversal_last_error_message", f"S{STATE_ERROR_MAX_BYTES}"),
    ]
)

INNER_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("pix", "<i8"),
        ("gaia_A0", "<f8"),
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("best_ee", "<f8"),
        ("best_sr", "<f8"),
        ("best_fwhm", "<f8"),
        ("winner_asterism_id", "<i8"),
        ("on_axis_winner_ee", "<f8"),
        ("field_averaged_winner_ee", "<f8"),
        ("on_axis_coverage", "?"),
        ("field_averaged_coverage", "?"),
    ]
)

ASTERISMS_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("asterism_id", "<i8"),
        ("ra", "<f8"),
        ("dec", "<f8"),
        ("num_stars", "<i8"),
        ("pix", "<i8"),
        ("star1_source_id", "<i8"),
        ("star1_ra", "<f8"),
        ("star1_dec", "<f8"),
        ("star1_mag", "<f8"),
        ("star2_source_id", "<i8"),
        ("star2_ra", "<f8"),
        ("star2_dec", "<f8"),
        ("star2_mag", "<f8"),
        ("star3_source_id", "<i8"),
        ("star3_ra", "<f8"),
        ("star3_dec", "<f8"),
        ("star3_mag", "<f8"),
    ]
)

MAPS_DTYPE: Final[np.dtype] = np.dtype(
    [
        ("pix", "<i8"),
        ("gaia_A0", "<f8"),
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("winner_asterism_count", "<i8"),
        ("best_sr", "<f8"),
        ("best_ee", "<f8"),
        ("best_fwhm", "<f8"),
        ("on_axis_winner_ee", "<f8"),
        ("field_averaged_winner_ee", "<f8"),
        ("on_axis_coverage", "<f8"),
        ("field_averaged_coverage", "<f8"),
    ]
)

MAPS_SUM_FIELDS: Final[tuple[str, ...]] = (
    "star_count",
    "ngs_count",
)

MAPS_MEAN_FIELDS: Final[tuple[str, ...]] = (
    "gaia_A0",
    "best_sr",
    "best_ee",
    "best_fwhm",
    "on_axis_winner_ee",
    "field_averaged_winner_ee",
    "on_axis_coverage",
    "field_averaged_coverage",
)
