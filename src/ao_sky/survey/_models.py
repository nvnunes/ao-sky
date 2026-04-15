"""Survey-overlay typed records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SurveyExtentOverlaySpec:
    """One named survey-extent overlay backed by one or more MOC files."""

    name: str
    moc_files: tuple[Path, ...]
