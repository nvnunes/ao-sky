"""Raw Gaia store interfaces for ao-sky."""

from ._constants import (
    GAIA_SUMMARY_DATASET_NAME,
    GAIA_SUMMARY_DTYPE,
    GAIA_SUMMARY_FILENAME,
    GAIA_SCHEMA_COLUMNS,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_OPTS,
    HDF5_DATASET_NAME,
    HDF5_SHUFFLE,
)
from ._exceptions import GaiaError
from .cache import CachedGaiaHealpixStore, GaiaTableCacheStats
from .store import GaiaHealpixStore, GaiaStoreConfig, GaiaSummaryStore, fetch_gaia_store
from .transform import apply_proper_motion, compute_legacy_r_magnitude

__all__ = [
    "CachedGaiaHealpixStore",
    "GAIA_SUMMARY_DATASET_NAME",
    "GAIA_SUMMARY_DTYPE",
    "GAIA_SUMMARY_FILENAME",
    "GAIA_SCHEMA_COLUMNS",
    "GaiaError",
    "GaiaHealpixStore",
    "GaiaTableCacheStats",
    "GaiaStoreConfig",
    "GaiaSummaryStore",
    "HDF5_COMPRESSION",
    "HDF5_COMPRESSION_OPTS",
    "HDF5_DATASET_NAME",
    "HDF5_SHUFFLE",
    "apply_proper_motion",
    "compute_legacy_r_magnitude",
    "fetch_gaia_store",
]
