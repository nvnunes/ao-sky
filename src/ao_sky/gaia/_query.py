"""Gaia archive query seam for canonical per-pixel materialization.

This module owns the translation from a canonical Gaia outer-pixel request
into the archive query used to materialize one raw per-pixel table. It does
not persist results or apply any loader-time star processing.
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout

from astroquery.gaia import Gaia
import requests
from astropy.table import Table

from ._exceptions import GaiaError
from ..spatial import get_pixel_resolution, get_pixel_skycoord


# Query construction

def _get_archive_table_name(release: str) -> str:
    release = release.strip().lower()
    if not release or not release.isalnum():
        raise GaiaError(f"Unsupported Gaia release value: {release!r}")
    return f"gaia{release}.gaia_source"


def build_healpix_query(release: str, healpix_level: int, outer_pix: int) -> str:
    """Build the canonical Gaia archive query for one outer pixel.

    The query targets the Gaia source table for the requested release and
    aliases the selected fields into the exact canonical raw-store schema:
    ``source_id``, ``ra``, ``dec``, ``G``, ``BP``, ``RP``, ``ref_epoch``,
    ``pmra``, ``pmdec``, ``non_single_star``, and ``ruwe``.

    Rows are restricted by both ``gaia_healpix_index`` and a retained circular
    region prefilter centred on the target outer pixel. The query also excludes
    Gaia QSO and galaxy candidates and orders results by ``SOURCE_ID`` so the
    materialized canonical file has stable row ordering.

    The returned query is intentionally raw: it does not derive additional
    photometric columns, rename the canonical fields again, or apply any
    loader-time transforms.

    Args:
        release: Gaia release identifier such as ``"dr3"``.
        healpix_level: Outer nested HEALPix level used for both the archive
            filter and the canonical storage layout.
        outer_pix: Outer nested HEALPix pixel index at ``healpix_level``.

    Returns:
        The ADQL query string used to materialize the canonical raw Gaia table
        for the requested outer pixel.

    Raises:
        GaiaError: If the release value is invalid or the HEALPix inputs are
            outside the supported range.
    """

    coord = get_pixel_skycoord(healpix_level, outer_pix)
    radius = 2.0 * get_pixel_resolution(healpix_level).to_value("degree")
    table_name = _get_archive_table_name(release)

    return f"""
SELECT SOURCE_ID AS source_id
     , ra AS ra
     , dec AS dec
     , phot_g_mean_mag AS G
     , phot_bp_mean_mag AS BP
     , phot_rp_mean_mag AS RP
     , ref_epoch AS ref_epoch
     , pmra AS pmra
     , pmdec AS pmdec
     , non_single_star AS non_single_star
     , ruwe AS ruwe
  FROM {table_name}
 WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS',{coord.ra.degree},{coord.dec.degree},{radius}))
   AND gaia_healpix_index({healpix_level}, SOURCE_ID) = {outer_pix}
   AND in_qso_candidates = '0'
   AND in_galaxy_candidates = '0'
 ORDER BY SOURCE_ID
    """.strip()


def query_healpix_table(release: str, healpix_level: int, outer_pix: int) -> Table:
    """Query Gaia for one outer pixel and return the raw canonical table.

    This function is the runtime archive seam used by ``GaiaHealpixStore`` when
    a canonical file must be materialized or refreshed. The returned table is
    expected to match the canonical raw-store schema exactly:
    ``source_id``, ``ra``, ``dec``, ``G``, ``BP``, ``RP``, ``ref_epoch``,
    ``pmra``, ``pmdec``, ``non_single_star``, and ``ruwe``.

    The function does not persist results, apply proper motion, stitch
    neighbouring pixels, or derive band- or instrument-specific photometric
    products.

    Args:
        release: Gaia release identifier such as ``"dr3"``.
        healpix_level: Outer nested HEALPix level used by the canonical store.
        outer_pix: Outer nested HEALPix pixel index at ``healpix_level``.

    Returns:
        An `astropy.table.Table` exposing the canonical Gaia columns in
        canonical order.

    Raises:
        GaiaError: If the query cannot be built, the archive request fails, or
        the archive is unavailable.
    """

    query = build_healpix_query(release, healpix_level, outer_pix)

    try:
        with open(os.devnull, "w", encoding="utf-8") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                job = Gaia.launch_job_async(query)
    except requests.exceptions.HTTPError as exc:
        if str(exc) == "OK":
            raise GaiaError("Gaia archive is likely down for maintenance") from exc
        raise GaiaError("Gaia archive query failed") from exc
    except Exception as exc:
        raise GaiaError("Gaia archive query failed") from exc

    results = job.get_results()
    if not isinstance(results, Table):
        return Table(results)
    return results
