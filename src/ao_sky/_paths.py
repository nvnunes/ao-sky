"""Shared filesystem path helpers for outer-pixel bucketed storage."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from .spatial import get_pixel_skycoord


HOUR_FOLDER_OVERRIDE_PIXELS: Final[frozenset[int]] = frozenset(
    {
        8960,
        8972,
        9023,
        9152,
        9200,
        9203,
        9215,
        11264,
        11312,
        11327,
        11468,
    }
)


def get_outer_pixel_bucket_path(level: int, outer_pix: int) -> Path:
    """Return the shared relative bucket path for one outer pixel.

    The returned path has the form:

    ``<hour>h/<sign><deg>/<outer_pix>``
    """

    coord = get_pixel_skycoord(level, outer_pix)
    hour = int(np.floor(coord.ra.degree / 15.0))
    if outer_pix in HOUR_FOLDER_OVERRIDE_PIXELS:
        hour = 14

    dec_bucket = int(np.floor(np.abs(coord.dec.degree / 10.0)) * 10)
    dec_sign = "+" if coord.dec.degree >= 0 else "-"
    return Path(f"{hour}h") / f"{dec_sign}{dec_bucket:02d}" / str(int(outer_pix))
