"""Shared Gaia table cache support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table


@dataclass(frozen=True, slots=True)
class GaiaTableCacheStats:
    """Runtime counters for one in-memory Gaia table cache."""

    hits: int
    misses: int
    evictions: int
    current_bytes: int
    peak_bytes: int
    entries: int
    oversized_skips: int = 0
    load_seconds: float = 0.0
    raw_load_seconds: float = 0.0
    prepare_seconds: float = 0.0


def estimate_table_bytes(table: Table) -> int:
    """Return the approximate array bytes owned by one Astropy table."""

    total = 0
    for name in table.colnames:
        total += int(np.asarray(table[name]).nbytes)
    return total
