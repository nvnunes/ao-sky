"""Traversal-local runtime Gaia table cache."""

from __future__ import annotations

from collections import OrderedDict
import threading
import time

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np

from ..gaia import (
    GAIA_SCHEMA_COLUMNS,
    GaiaHealpixStore,
    GaiaTableCacheStats,
    apply_proper_motion,
    compute_r_magnitude,
)
from ..gaia.cache import estimate_table_bytes
from ..predict._models import PredictRuntime
from ..spatial import get_pixel_from_skycoord

RUNTIME_HPX_LEVEL = 14
RUNTIME_HPX_COLUMN = "hpx14"


class RuntimeGaiaHealpixStore:
    """Worker-local cache of epoch-shifted Gaia rows used by Traversal."""

    def __init__(
        self,
        store: GaiaHealpixStore,
        runtime: PredictRuntime,
        *,
        max_entries: int,
        max_bytes: int,
        hpx_level: int = RUNTIME_HPX_LEVEL,
    ) -> None:
        self.store = store
        self.config = store.config
        self.runtime = runtime
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self.hpx_level = int(hpx_level)
        self._cache: OrderedDict[int, tuple[Table, int]] = OrderedDict()
        self._current_bytes = 0
        self._peak_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversized_skips = 0
        self._load_seconds = 0.0
        self._raw_load_seconds = 0.0
        self._prepare_seconds = 0.0
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def healpix_filename(self, outer_pix: int):
        return self.store.healpix_filename(outer_pix)

    def load_healpix(
        self,
        outer_pix: int,
        *,
        force_reload: bool = False,
        read_only: bool = True,
    ) -> Table:
        key = int(outer_pix)
        if force_reload:
            table = self._load_runtime_table(outer_pix, force_reload=force_reload)
            if self.enabled:
                self._replace(key, table)
                return table if read_only else table.copy(copy_data=True)
            return table if read_only else table.copy(copy_data=True)

        if not self.enabled:
            table = self._load_runtime_table(outer_pix, force_reload=False)
            return table if read_only else table.copy(copy_data=True)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                table, _ = cached
                self._cache.move_to_end(key)
                return table if read_only else table.copy(copy_data=True)
            self._misses += 1

        table = self._load_runtime_table(outer_pix, force_reload=force_reload)
        self._replace(key, table)
        return table if read_only else table.copy(copy_data=True)

    def stats(self) -> GaiaTableCacheStats:
        with self._lock:
            return GaiaTableCacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                current_bytes=self._current_bytes,
                peak_bytes=self._peak_bytes,
                entries=len(self._cache),
                oversized_skips=self._oversized_skips,
                load_seconds=self._load_seconds,
                raw_load_seconds=self._raw_load_seconds,
                prepare_seconds=self._prepare_seconds,
            )

    def _load_runtime_table(self, outer_pix: int, *, force_reload: bool) -> Table:
        started = time.perf_counter()
        raw_started = time.perf_counter()
        raw = self.store.load_healpix(
            outer_pix,
            force_reload=force_reload,
            read_only=True,
        )
        raw_load_seconds = time.perf_counter() - raw_started
        prepare_started = time.perf_counter()
        table = apply_proper_motion(
            raw[list(GAIA_SCHEMA_COLUMNS)],
            epoch=self.runtime.epoch,
        )
        table["R"] = compute_r_magnitude(table[list(GAIA_SCHEMA_COLUMNS)])
        hpx = np.full((len(table),), -1, dtype=np.int64)
        valid_coords = np.isfinite(table["ra"]) & np.isfinite(table["dec"])
        if np.any(valid_coords):
            hpx[valid_coords] = np.asarray(
                get_pixel_from_skycoord(
                    self.hpx_level,
                    SkyCoord(
                        ra=table["ra"][valid_coords],
                        dec=table["dec"][valid_coords],
                        unit=(u.degree, u.degree),
                    ),
                ),
                dtype=np.int64,
            )
        table[RUNTIME_HPX_COLUMN] = hpx
        _mark_table_read_only(table)
        prepare_seconds = time.perf_counter() - prepare_started
        load_seconds = time.perf_counter() - started
        with self._lock:
            self._raw_load_seconds += raw_load_seconds
            self._prepare_seconds += prepare_seconds
            self._load_seconds += load_seconds
        return table

    def _replace(self, outer_pix: int, table: Table) -> None:
        with self._lock:
            previous = self._cache.pop(int(outer_pix), None)
            if previous is not None:
                self._current_bytes -= previous[1]
            size = estimate_table_bytes(table)
            if size > self.max_bytes:
                self._oversized_skips += 1
                return
            self._cache[int(outer_pix)] = (table, size)
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


def _mark_table_read_only(table: Table) -> None:
    for name in table.colnames:
        table[name].flags.writeable = False
