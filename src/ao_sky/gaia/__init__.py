"""Raw Gaia store interfaces for ao-sky."""

from ._constants import (
    GAIA_SCHEMA_COLUMNS,
    HDF5_COMPRESSION,
    HDF5_COMPRESSION_OPTS,
    HDF5_DATASET_NAME,
    HDF5_SHUFFLE,
)
from ._exceptions import GaiaError
from .store import GaiaHealpixStore, GaiaStoreConfig
from .transform import apply_proper_motion, compute_legacy_r_magnitude

__all__ = [
    "GAIA_SCHEMA_COLUMNS",
    "GaiaError",
    "GaiaHealpixStore",
    "GaiaStoreConfig",
    "HDF5_COMPRESSION",
    "HDF5_COMPRESSION_OPTS",
    "HDF5_DATASET_NAME",
    "HDF5_SHUFFLE",
    "apply_proper_motion",
    "compute_legacy_r_magnitude",
]
