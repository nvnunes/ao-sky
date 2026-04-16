"""Worker-local in-memory Gaia table cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from .store import GaiaHealpixStore


@dataclass(frozen=True, slots=True)
class GaiaTableCacheStats:
    """Runtime counters for one in-memory Gaia table cache."""

    hits: int
    misses: int
    evictions: int
    current_bytes: int
    peak_bytes: int
    entries: int


class CachedGaiaHealpixStore:
    """LRU cache wrapper for canonical Gaia outer-pixel tables."""

    def __init__(
        self,
        store: GaiaHealpixStore,
        *,
        max_entries: int,
        max_bytes: int,
    ) -> None:
        self.store = store
        self.config = store.config
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._cache: OrderedDict[tuple[str, str, int, int], tuple[Table, int]] = OrderedDict()
        self._current_bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def healpix_filename(self, outer_pix: int) -> Path:
        return self.store.healpix_filename(outer_pix)

    def load_healpix(
        self,
        outer_pix: int,
        *,
        force_reload: bool = False,
        read_only: bool = True,
    ) -> Table:
        if force_reload:
            table = self.store.load_healpix(
                outer_pix,
                force_reload=True,
                read_only=True if self.enabled else read_only,
            )
            if self.enabled:
                self._replace(int(outer_pix), table)
                return table if read_only else table.copy(copy_data=True)
            return table

        if not self.enabled:
            return self.store.load_healpix(
                outer_pix,
                force_reload=False,
                read_only=read_only,
            )

        key = self._key(outer_pix)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            table, _ = cached
            self._cache.move_to_end(key)
            return table if read_only else table.copy(copy_data=True)

        self._misses += 1
        table = self.store.load_healpix(
            outer_pix,
            force_reload=force_reload,
            read_only=True,
        )
        size = estimate_table_bytes(table)
        self._cache[key] = (table, size)
        self._current_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        self._evict()
        return table if read_only else table.copy(copy_data=True)

    def stats(self) -> GaiaTableCacheStats:
        return GaiaTableCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            current_bytes=self._current_bytes,
            peak_bytes=self._peak_bytes,
            entries=len(self._cache),
        )

    def _key(self, outer_pix: int) -> tuple[str, str, int, int]:
        return (
            str(Path(self.config.root).expanduser().resolve()),
            str(self.config.release),
            int(self.config.healpix_level),
            int(outer_pix),
        )

    def _replace(self, outer_pix: int, table: Table) -> None:
        key = self._key(outer_pix)
        previous = self._cache.pop(key, None)
        if previous is not None:
            self._current_bytes -= previous[1]
        size = estimate_table_bytes(table)
        self._cache[key] = (table, size)
        self._current_bytes += size
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        self._evict()

    def _evict(self) -> None:
        while self._cache and (
            len(self._cache) > self.max_entries or self._current_bytes > self.max_bytes
        ):
            _, (_, size) = self._cache.popitem(last=False)
            self._current_bytes -= size
            self._evictions += 1


def estimate_table_bytes(table: Table) -> int:
    """Return the approximate array bytes owned by one Astropy table."""

    total = 0
    for name in table.colnames:
        total += int(np.asarray(table[name]).nbytes)
    return total
