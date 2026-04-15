"""Build creation, execution, and summary helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..gaia import GaiaHealpixStore, GaiaStoreConfig
from ._constants import (
    BUILD_PHASE_TRAVERSAL,
    BUILD_STATUS_FAILED,
    BUILD_STATUS_RUNNING,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ._exceptions import BuildError
from .artifacts import write_outer_artifact
from .config import load_build_definition, resolve_build_root_only, resolve_build_roots
from .control import (
    append_build_log,
    build_artifact_root,
    create_build_root,
    latest_build_path,
    load_build_definition as load_persisted_build_definition,
    load_build_roots,
    load_current_phase,
    load_legacy_config_path,
    load_state,
    outer_artifact_filename,
    phase_state_fields,
    refresh_build_status,
    set_build_status,
    summarize_build,
    update_state_row,
)
from .legacy_runtime import build_traversal_products, load_legacy_runtime
from .scheduler import OuterPixelScheduler


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


def build_outer_pixel_products(build_path: Path, outer_pix: int) -> None:
    """Run the Phase 5 Traversal pipeline for one outer pixel."""

    definition = load_persisted_build_definition(build_path)
    roots = load_build_roots(build_path)
    runtime = load_legacy_runtime(definition, load_legacy_config_path(build_path))
    store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=roots.gaia_root,
            release=definition.gaia_release,
            healpix_level=definition.outer_level,
        )
    )

    current_state = load_state(build_path)[int(outer_pix)]
    update_state_row(
        build_path,
        outer_pix,
        traversal_status=WORK_STATUS_RUNNING,
        traversal_attempt_count=int(current_state["traversal_attempt_count"]) + 1,
        traversal_last_error_message="",
    )

    try:
        asterisms, inner = build_traversal_products(store, runtime, outer_pix)
        filename = outer_artifact_filename(build_path, definition, outer_pix)
        write_outer_artifact(filename, inner=inner, asterisms=asterisms)
        update_state_row(
            build_path,
            outer_pix,
            traversal_status=WORK_STATUS_DONE,
            traversal_last_error_message="",
        )
    except Exception as exc:
        update_state_row(
            build_path,
            outer_pix,
            traversal_status=WORK_STATUS_FAILED,
            traversal_last_error_message=str(exc),
        )
        raise


def _repair_stale_running_rows(build_path: Path, *, status_field: str) -> int:
    state = load_state(build_path)
    stale = np.flatnonzero(state[status_field] == WORK_STATUS_RUNNING)
    for outer_pix in stale:
        update_state_row(
            build_path,
            int(outer_pix),
            **{status_field: WORK_STATUS_PENDING},
        )
    return int(len(stale))


def run_build(build_path: Path) -> Path:
    """Run all unfinished Phase 5 Traversal work in one build."""

    if not build_path.is_dir():
        raise BuildError(f"Build path does not exist: {build_path}")

    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_TRAVERSAL:
        raise BuildError(
            f"Phase 5 runner only implements {BUILD_PHASE_TRAVERSAL!r}, "
            f"but build phase is {current_phase!r}"
        )

    fields = phase_state_fields(current_phase)
    if fields is None:
        raise BuildError(f"Build phase {current_phase!r} is not outer-pixel-local")
    status_field, _, _ = fields

    definition = load_persisted_build_definition(build_path)
    build_artifact_root(build_path, definition).mkdir(parents=True, exist_ok=True)

    repaired = _repair_stale_running_rows(build_path, status_field=status_field)
    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(build_path, f"run start phase={current_phase} repaired_stale_running={repaired}")

    scheduler = OuterPixelScheduler(
        outer_level=definition.outer_level,
        status_field=status_field,
    )
    last_completed_outer_pix: int | None = None
    failed = False

    while True:
        state = load_state(build_path)
        outer_pix = scheduler.select_next_outer_pixel(
            state,
            last_completed_outer_pix=last_completed_outer_pix,
        )
        if outer_pix is None:
            break
        try:
            build_outer_pixel_products(build_path, outer_pix)
            last_completed_outer_pix = outer_pix
        except Exception as exc:
            failed = True
            append_build_log(build_path, f"phase={current_phase} outer_pix={outer_pix} failed: {exc}")

    final_status = refresh_build_status(build_path)
    if failed and final_status != BUILD_STATUS_FAILED:
        set_build_status(build_path, BUILD_STATUS_FAILED)
        final_status = BUILD_STATUS_FAILED

    summary = summarize_build(build_path)
    phase_counts = summary["phase_counts"]
    if phase_counts is None:
        raise BuildError(f"Build summary missing counts for active phase {current_phase!r}")
    append_build_log(
        build_path,
        "run complete "
        f"phase={current_phase} "
        f"status={final_status} "
        f"pending={phase_counts['pending']} "
        f"running={phase_counts['running']} "
        f"done={phase_counts['done']} "
        f"failed={phase_counts['failed']}",
    )
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
        f"phase: {summary['current_phase']}",
    ]
    phase_counts: dict[str, int] | None = summary["phase_counts"]  # type: ignore[assignment]
    if phase_counts is not None:
        lines.extend(
            [
                "phase work:",
                f"  pending={phase_counts['pending']}",
                f"  running={phase_counts['running']}",
                f"  done={phase_counts['done']}",
                f"  failed={phase_counts['failed']}",
            ]
        )
    return "\n".join(lines)
