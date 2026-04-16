"""Live legacy comparison adapter for Traversal-output parity checks."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap

from astropy.table import Table
import numpy as np

from ..predict import PredictRuntime
from ._exceptions import BuildError
from .traversal import should_skip_asterisms


def _get_legacy_runtime_root(config_path: Path) -> Path:
    return Path(config_path).resolve().parents[1]


def load_live_legacy_traversal_outputs(
    runtime: PredictRuntime,
    outer_pix: int,
) -> tuple[Table | None, Table]:
    """Return live legacy retained asterisms and rich inner outputs."""

    legacy_python = _get_legacy_runtime_root(runtime.legacy_config_path) / ".conda" / "bin" / "python"
    if not legacy_python.exists():
        raise BuildError(f"Legacy Python runtime not found: {legacy_python}")

    skip_asterisms, _ = should_skip_asterisms(runtime, outer_pix)
    with tempfile.TemporaryDirectory(prefix="ao-sky-phase5-legacy-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        asterism_filename = tmpdir_path / "asterisms.fits"
        inner_filename = tmpdir_path / "inner.fits"
        code = textwrap.dedent(
            """
            import sys
            from pathlib import Path
            import numpy as np
            from astropy.table import Table

            survey_root = Path(sys.argv[1]).resolve()
            config_filename = Path(sys.argv[2]).resolve()
            asterism_filename = Path(sys.argv[3]).resolve()
            inner_filename = Path(sys.argv[4]).resolve()
            outer_pix = int(sys.argv[5])
            ao_system_name = sys.argv[6]
            skip_asterisms = sys.argv[7] == "1"

            sys.path.insert(0, str(survey_root))
            import aomap.aomap as aomap

            config = aomap.read_config(str(config_filename))
            config.asterisms_max_dust_extinction = None
            original_folder = Path(config.folder).resolve()
            temp_folder = Path(inner_filename).parent / "legacy-data"
            temp_folder.mkdir(parents=True, exist_ok=True)
            (temp_folder / "gaia").symlink_to(original_folder / "gaia", target_is_directory=True)
            config.folder = str(temp_folder)
            ao_system = aomap.get_ao_system(config, ao_system_name)

            inner_hdul = aomap._create_inner(config, outer_pix)
            inner = Table(inner_hdul[1].data)

            local_asterisms = None
            if not skip_asterisms:
                build_pixs = [outer_pix]
                for pix in aomap.healpix.get_neighbours(config.outer_level, outer_pix):
                    if pix >= 0:
                        build_pixs.append(int(pix))

                for pix in build_pixs:
                    asterisms = aomap.find_outer_asterisms(config, pix, ao_system_name)
                    if asterisms is None:
                        continue
                    if pix == outer_pix:
                        local_asterisms = asterisms.copy(copy_data=True)
                    if len(asterisms) > 0:
                        aomap._save_asterisms(config, pix, ao_system_name, asterisms)

            if local_asterisms is not None and len(local_asterisms) > 0:
                local_asterisms.write(asterism_filename, format="fits", overwrite=True)

            asterism_count = aomap._get_inner_pixel_asterism_count(config, outer_pix, ao_system)
            stats = aomap._get_inner_pixel_asterism_coverage_and_performance(config, outer_pix, ao_system)

            winner_ids = np.full(len(stats.ee_max_id), -1, dtype=np.int64)
            finite_winners = np.isfinite(stats.ee_max_id)
            finite_winners &= np.asarray(stats.ee_max_id) >= 0
            winner_ids[finite_winners] = np.asarray(stats.ee_max_id[finite_winners], dtype=np.int64)

            inner_phase5 = Table()
            inner_phase5["pix"] = np.asarray(inner[aomap.FITS_COLUMN_PIX], dtype=np.int64)
            inner_phase5["star_count"] = np.asarray(inner[aomap.FITS_COLUMN_STAR_COUNT], dtype=np.int64)
            inner_phase5["ngs_count"] = np.asarray(inner[aomap._get_ngs_count_field(ao_system)], dtype=np.int64)
            inner_phase5["asterism_count"] = np.asarray(asterism_count, dtype=np.int64)
            inner_phase5["best_ee"] = np.asarray(stats.ee_max, dtype=np.float64)
            inner_phase5["best_sr"] = np.asarray(stats.sr_max, dtype=np.float64)
            inner_phase5["best_fwhm"] = np.asarray(stats.fwhm_min, dtype=np.float64)
            inner_phase5["winner_asterism_id"] = winner_ids
            inner_phase5["winner_distance_arcsec"] = np.asarray(stats.ee_max_distance, dtype=np.float64)
            inner_phase5["winner_ee_resolved"] = np.asarray(stats.ee_max, dtype=np.float64)
            inner_phase5["winner_ee_averaged"] = np.asarray(stats.ee_max_field_mean, dtype=np.float64)
            inner_phase5["coverage_resolved"] = np.asarray(stats.resolved_coverage > 0.5, dtype=np.bool_)
            inner_phase5["coverage_averaged"] = np.asarray(stats.mean_coverage > 0.5, dtype=np.bool_)
            inner_phase5.write(inner_filename, format="fits", overwrite=True)
            """
        )
        result = subprocess.run(
            [
                str(legacy_python),
                "-c",
                code,
                str(_get_legacy_runtime_root(runtime.legacy_config_path)),
                str(runtime.legacy_config_path),
                str(asterism_filename),
                str(inner_filename),
                str(outer_pix),
                runtime.ao_system.name,
                "1" if skip_asterisms else "0",
            ],
            cwd=runtime.legacy_config_path.parent,
            env={**os.environ, "MPLCONFIGDIR": str(tmpdir_path / "mpl")},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not inner_filename.exists():
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            details = stderr or stdout or "no output"
            raise BuildError(f"Legacy traversal adapter failed for outer pixel {outer_pix}: {details}")

        asterisms = (
            Table.read(asterism_filename, format="fits")
            if asterism_filename.exists()
            else None
        )
        inner = Table.read(inner_filename, format="fits")
        return asterisms, inner
