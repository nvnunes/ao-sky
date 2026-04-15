"""Constants for persisted build contracts."""

from __future__ import annotations

from typing import Final

import numpy as np

BUILD_LAYOUT_VERSION: Final[int] = 1
BUILD_FILENAME: Final[str] = "build.h5"
BUILD_LOG_FILENAME: Final[str] = "build.log"
OUTER_FILENAME: Final[str] = "outer.h5"
OUTER_DATASET_ASTERISMS: Final[str] = "asterisms"
OUTER_DATASET_INNER: Final[str] = "inner"
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
        ("star_count", "<i8"),
        ("ngs_count", "<i8"),
        ("asterism_count", "<i8"),
        ("best_ee", "<f8"),
        ("best_sr", "<f8"),
        ("best_fwhm", "<f8"),
        ("winner_asterism_id", "<i8"),
        ("winner_distance_arcsec", "<f8"),
        ("winner_ee_resolved", "<f8"),
        ("winner_ee_averaged", "<f8"),
        ("coverage_resolved", "?"),
        ("coverage_averaged", "?"),
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
