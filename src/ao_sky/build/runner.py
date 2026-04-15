"""Build creation, execution, and summary helpers."""

from __future__ import annotations

from pathlib import Path

from astropy.table import Table

from ..gaia import GaiaHealpixStore, GaiaStoreConfig
from ._constants import (
    ARTIFACT_STATE_DONE,
    ARTIFACT_STATE_FAILED,
    ARTIFACT_STATE_PENDING,
    ARTIFACT_STATE_SKIPPED,
    BUILD_STATUS_FAILED,
    BUILD_STATUS_RUNNING,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_RUNNING,
)
from .artifacts import write_outer_artifact
from .config import load_build_definition, resolve_build_root_only, resolve_build_roots
from .control import (
    build_artifact_root,
    create_build_root,
    latest_build_path,
    load_build_definition as load_persisted_build_definition,
    load_build_roots,
    load_legacy_config_path,
    load_state,
    outer_artifact_filename,
    refresh_build_status,
    set_build_status,
    summarize_build,
    update_state_row,
)
from ._exceptions import BuildError
from .legacy_runtime import build_inner_table, build_outer_pixel_asterisms, load_legacy_runtime


def init_build(
    *,
    definition_filename: Path,
    gaia_root: Path | None,
    build_root: Path | None,
    aosky_conf: Path | None = None,
    legacy_config_path: Path,
) -> Path:
    """Create a new build root and seed its initial metadata/state."""

    definition, definition_yaml = load_build_definition(definition_filename)
    roots = resolve_build_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        aosky_conf=aosky_conf,
    )
    legacy_runtime = load_legacy_runtime(definition, legacy_config_path)
    return create_build_root(
        definition=definition,
        definition_yaml=definition_yaml,
        roots=roots,
        legacy_runtime=legacy_runtime,
    )


def _to_persisted_asterisms(asterisms: Table) -> Table:
    result = Table()
    result["asterism_id"] = asterisms["asterism_id"]
    result["ra"] = asterisms["ra"]
    result["dec"] = asterisms["dec"]
    result["num_stars"] = asterisms["num_stars"]
    result["pix"] = asterisms["pix"]
    for index in (1, 2, 3):
        result[f"star{index}_source_id"] = asterisms[f"star{index}_source_id"]
        result[f"star{index}_ra"] = asterisms[f"star{index}_ra"]
        result[f"star{index}_dec"] = asterisms[f"star{index}_dec"]
        result[f"star{index}_mag"] = asterisms[f"star{index}_mag"]
    return result


def build_outer_pixel_products(build_path: Path, outer_pix: int) -> None:
    """Build and persist one outer-pixel artifact container."""

    definition = load_persisted_build_definition(build_path)
    roots = load_build_roots(build_path)
    legacy_runtime = load_legacy_runtime(definition, load_legacy_config_path(build_path))
    store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=roots.gaia_root,
            release=definition.gaia_release,
            healpix_level=definition.outer_level,
        )
    )

    current_state = load_state(build_path)[int(outer_pix)]
    asterisms_skipped = int(current_state["asterisms_state"]) == ARTIFACT_STATE_SKIPPED

    update_state_row(
        build_path,
        outer_pix,
        work_status=WORK_STATUS_RUNNING,
        attempt_count=int(current_state["attempt_count"]) + 1,
        last_error_message="",
    )

    try:
        asterisms = None
        if not asterisms_skipped:
            _, _, final_asterisms = build_outer_pixel_asterisms(store, legacy_runtime, outer_pix)
            asterisms = _to_persisted_asterisms(final_asterisms)
            asterisms_state = ARTIFACT_STATE_DONE
            skip_reason = ""
        else:
            asterisms_state = ARTIFACT_STATE_SKIPPED
            skip_reason = current_state["asterisms_skip_reason"].decode("utf-8").strip()

        inner = build_inner_table(store, legacy_runtime, outer_pix, asterisms=asterisms)
        filename = outer_artifact_filename(build_path, definition, outer_pix)
        write_outer_artifact(filename, inner=inner, asterisms=asterisms)
        update_state_row(
            build_path,
            outer_pix,
            work_status=WORK_STATUS_DONE,
            outer_file_state=ARTIFACT_STATE_DONE,
            inner_state=ARTIFACT_STATE_DONE,
            asterisms_state=asterisms_state,
            asterisms_skip_reason=skip_reason,
            last_error_message="",
        )
    except Exception as exc:
        update_state_row(
            build_path,
            outer_pix,
            work_status=WORK_STATUS_FAILED,
            outer_file_state=ARTIFACT_STATE_FAILED,
            inner_state=ARTIFACT_STATE_FAILED,
            asterisms_state=(
                ARTIFACT_STATE_SKIPPED
                if asterisms_skipped
                else ARTIFACT_STATE_FAILED
            ),
            last_error_message=str(exc),
        )
        raise


def run_build(build_path: Path) -> Path:
    """Run all unfinished work in one build."""

    if not build_path.is_dir():
        raise BuildError(f"Build path does not exist: {build_path}")
    build_artifact_root(build_path, load_persisted_build_definition(build_path)).mkdir(
        parents=True,
        exist_ok=True,
    )
    set_build_status(build_path, BUILD_STATUS_RUNNING)
    state = load_state(build_path)
    failed = False
    for outer_pix in range(len(state)):
        if int(state["work_status"][outer_pix]) == WORK_STATUS_DONE:
            continue
        try:
            build_outer_pixel_products(build_path, outer_pix)
        except Exception:
            failed = True
    final_status = refresh_build_status(build_path)
    if failed and final_status != BUILD_STATUS_FAILED:
        set_build_status(build_path, BUILD_STATUS_FAILED)
    return build_path


def restart_build(
    *,
    ao_system_short_name: str,
    config_short_name: str,
    build_root: Path | None,
    aosky_conf: Path | None = None,
) -> Path:
    """Restart the latest build in one lineage."""

    resolved_build_root = resolve_build_root_only(
        build_root=build_root,
        aosky_conf=aosky_conf,
    )
    build_path = latest_build_path(
        resolved_build_root,
        ao_system_short_name=ao_system_short_name,
        config_short_name=config_short_name,
    )
    return run_build(build_path)


def show_build(build_path: Path) -> str:
    """Return a summary string for one build."""

    summary = summarize_build(build_path)
    lines = [
        f"build: {summary['build_path']}",
        f"status: {summary['build_status']}",
        "work:",
        f"  pending={summary['work_counts']['pending']}",
        f"  running={summary['work_counts']['running']}",
        f"  done={summary['work_counts']['done']}",
        f"  failed={summary['work_counts']['failed']}",
    ]
    skip_reasons: dict[str, int] = summary["skip_reasons"]  # type: ignore[assignment]
    if skip_reasons:
        lines.append("asterism skips:")
        for reason, count in sorted(skip_reasons.items()):
            lines.append(f"  {reason}={count}")
    return "\n".join(lines)
