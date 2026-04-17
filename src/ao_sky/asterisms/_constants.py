"""Canonical in-memory asterism output constants."""

from __future__ import annotations

from typing import Final

ASTERISM_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "asterism_id",
    "ra",
    "dec",
    "num_stars",
    "star1_idx",
    "star1_source_id",
    "star1_ra",
    "star1_dec",
    "star1_pmra",
    "star1_pmdec",
    "star1_ref_epoch",
    "star1_mag",
    "star2_idx",
    "star2_source_id",
    "star2_ra",
    "star2_dec",
    "star2_pmra",
    "star2_pmdec",
    "star2_ref_epoch",
    "star2_mag",
    "star3_idx",
    "star3_source_id",
    "star3_ra",
    "star3_dec",
    "star3_pmra",
    "star3_pmdec",
    "star3_ref_epoch",
    "star3_mag",
    "radius_arcsec",
    "area_arcsec2",
    "separation_arcsec",
)
