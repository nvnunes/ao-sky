"""Internal single-process scheduler for build execution."""

from __future__ import annotations

import numpy as np

from ..spatial import get_pixel_neighbours
from ._constants import WORK_STATUS_FAILED, WORK_STATUS_PENDING


class OuterPixelScheduler:
    """Select the next outer pixel for one single-process build run.

    The scheduler is intentionally internal to `ao_sky.build`. It keeps
    ephemeral in-memory frontier state for one run, but it does not mutate the
    persisted build state. Selection rules are:

    - eligible work is `pending` or `failed`
    - local continuation uses immediate neighbours at the build outer level
    - neighbour order follows `get_pixel_neighbours(...)`
    - local traversal is depth-first
    - when the local frontier is exhausted, the next seed is the lowest
      unfinished outer-pixel id
    - each pixel is selected at most once per run
    """

    def __init__(self, *, outer_level: int) -> None:
        self.outer_level = outer_level
        self._frontier: list[int] = []
        self._selected: set[int] = set()
        self._last_completed_seen: int | None = None

    def select_next_outer_pixel(
        self,
        state: np.ndarray,
        *,
        last_completed_outer_pix: int | None,
    ) -> int | None:
        """Return the next eligible outer pixel, or `None` if none remain."""

        self._extend_frontier_from_completion(
            state,
            last_completed_outer_pix=last_completed_outer_pix,
        )

        while self._frontier:
            outer_pix = self._frontier.pop()
            if self._is_selectable(state, outer_pix):
                self._selected.add(outer_pix)
                return outer_pix

        outer_pix = self._lowest_unfinished_outer_pix(state)
        if outer_pix is None:
            return None
        self._selected.add(outer_pix)
        return outer_pix

    def _extend_frontier_from_completion(
        self,
        state: np.ndarray,
        *,
        last_completed_outer_pix: int | None,
    ) -> None:
        if (
            last_completed_outer_pix is None
            or last_completed_outer_pix == self._last_completed_seen
        ):
            return

        self._last_completed_seen = last_completed_outer_pix
        neighbours = get_pixel_neighbours(self.outer_level, last_completed_outer_pix)
        selectable = [
            int(outer_pix)
            for outer_pix in neighbours
            if self._is_selectable(state, int(outer_pix))
        ]
        for outer_pix in reversed(selectable):
            self._frontier.append(outer_pix)

    def _lowest_unfinished_outer_pix(self, state: np.ndarray) -> int | None:
        for outer_pix in np.asarray(state["outer_pix"], dtype=np.int64):
            if self._is_selectable(state, int(outer_pix)):
                return int(outer_pix)
        return None

    def _is_selectable(self, state: np.ndarray, outer_pix: int) -> bool:
        if outer_pix in self._selected:
            return False
        status = int(state["work_status"][int(outer_pix)])
        return status in (WORK_STATUS_PENDING, WORK_STATUS_FAILED)
