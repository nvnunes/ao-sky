"""Process-level plotting environment helpers."""

from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_cache(path: Path | str | None = None) -> Path:
    """Set a writable Matplotlib config directory before Matplotlib import.

    The helper respects an existing ``MPLCONFIGDIR`` environment variable. It
    intentionally imports only the standard library so script entrypoints can
    call it before importing Matplotlib or other `ao_sky.plotting` modules.
    """

    configured = Path(os.environ.get("MPLCONFIGDIR", path or "/tmp/ao-sky-mplconfig"))
    configured.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(configured))
    return configured
