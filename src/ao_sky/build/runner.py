"""Build creation, execution, and summary helpers."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path

from joblib.externals.loky import ProcessPoolExecutor
import numpy as np

from ..gaia import GaiaHealpixStore, GaiaStoreConfig, GaiaSummaryStore
from ..predict import configure_inference_threads, warm_model_cache
from .augmentation import build_survey_extent_layers
from .aggregation import build_maps
from ._constants import (
    BUILD_PHASE_AGGREGATION,
    BUILD_PHASE_AUGMENTATION,
    BUILD_PHASE_TRAVERSAL,
    BUILD_STATUS_COMPLETED,
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
    set_current_phase,
    set_build_status,
    summarize_build,
    update_state_row,
)
from .legacy_config import load_native_runtime
from ._models import TraversalTaskContext, TraversalTaskResult
from .scheduler import OuterPixelScheduler
from .traversal import build_traversal_products


def init_build(
    *,
    definition_filename: Path,
    gaia_root: Path | None,
    build_root: Path | None,
    dust_root: Path | None,
    legacy_config_path: Path,
    model_root: Path | None = None,
    aosky_conf: Path | None = None,
) -> Path:
    """Create a new build root and seed its initial metadata/state."""

    definition, definition_yaml = load_build_definition(definition_filename)
    roots = resolve_build_roots(
        gaia_root=gaia_root,
        build_root=build_root,
        dust_root=dust_root,
        model_root=model_root,
        default_model_root=Path(legacy_config_path).resolve().parents[1] / "data" / "models",
        aosky_conf=aosky_conf,
    )
    _require_gaia_summary(
        gaia_root=roots.gaia_root,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
    )
    load_native_runtime(
        definition,
        legacy_config_path=legacy_config_path,
        model_root=roots.model_root,
    )
    return create_build_root(
        definition=definition,
        definition_yaml=definition_yaml,
        roots=roots,
        legacy_config_path=legacy_config_path,
    )


def _load_gaia_star_counts(
    *,
    gaia_root: Path,
    gaia_release: str,
    outer_level: int,
) -> np.ndarray:
    summary = GaiaSummaryStore(
        GaiaStoreConfig(
            root=gaia_root,
            release=gaia_release,
            healpix_level=outer_level,
        )
    ).load_summary()
    expected_rows = 12 * (4 ** outer_level)
    if len(summary) != expected_rows:
        raise BuildError(
            "Gaia summary has wrong size; rerun "
            f"`ao-sky fetch-gaia --gaia-release {gaia_release} --outer-level {outer_level}`"
        )
    loaded = np.asarray(summary["loaded"], dtype=np.bool_)
    if not np.all(loaded):
        missing = np.flatnonzero(~loaded)
        raise BuildError(
            "Gaia summary is incomplete; rerun "
            f"`ao-sky fetch-gaia --gaia-release {gaia_release} --outer-level {outer_level}` "
            f"(first missing outer_pix={int(missing[0])})"
        )
    return np.asarray(summary["star_count"], dtype=np.int64)


def _require_gaia_summary(
    *,
    gaia_root: Path,
    gaia_release: str,
    outer_level: int,
) -> None:
    try:
        _load_gaia_star_counts(
            gaia_root=gaia_root,
            gaia_release=gaia_release,
            outer_level=outer_level,
        )
    except Exception as exc:
        if isinstance(exc, BuildError):
            raise
        raise BuildError(
            "Missing Gaia summary for build initialization; run "
            f"`ao-sky fetch-gaia --gaia-release {gaia_release} --outer-level {outer_level}` "
            "first"
        ) from exc


def _load_traversal_task_context(build_path: Path) -> TraversalTaskContext:
    """Load build metadata once in the parent process for worker tasks."""

    return TraversalTaskContext(
        build_path=build_path,
        definition=load_persisted_build_definition(build_path),
        roots=load_build_roots(build_path),
        legacy_config_path=load_legacy_config_path(build_path),
    )


def _materialize_outer_pixel_products(
    context: TraversalTaskContext,
    outer_pix: int,
) -> None:
    """Run the Traversal pipeline for one outer pixel without mutating build state."""

    runtime = load_native_runtime(
        context.definition,
        legacy_config_path=context.legacy_config_path,
        model_root=context.roots.model_root,
    )
    warm_model_cache(runtime)
    store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=context.roots.gaia_root,
            release=context.definition.gaia_release,
            healpix_level=context.definition.outer_level,
        )
    )
    asterisms, inner = build_traversal_products(
        store,
        runtime,
        outer_pix,
        dust_root=context.roots.dust_root,
        max_data_level=context.definition.max_data_level,
    )
    filename = outer_artifact_filename(context.build_path, context.definition, outer_pix)
    write_outer_artifact(filename, inner=inner, asterisms=asterisms)


def _run_outer_pixel_traversal_task(
    context: TraversalTaskContext,
    outer_pix: int,
) -> TraversalTaskResult:
    """Run one worker-owned outer-pixel Traversal task and serialize the outcome."""

    try:
        configure_inference_threads(1)
        _materialize_outer_pixel_products(context, outer_pix)
    except Exception as exc:
        return TraversalTaskResult(
            outer_pix=int(outer_pix),
            success=False,
            error_message=str(exc),
        )
    return TraversalTaskResult(outer_pix=int(outer_pix), success=True)


def build_outer_pixel_products(build_path: Path, outer_pix: int) -> None:
    """Run one outer-pixel Traversal task and update persisted build state."""

    context = _load_traversal_task_context(build_path)
    state = load_state(build_path)
    _mark_outer_pixel_running(build_path, state, outer_pix)
    result = _run_outer_pixel_traversal_task(context, outer_pix)
    _record_traversal_result(build_path, state, result)


def _mark_outer_pixel_running(build_path: Path, state: np.ndarray, outer_pix: int) -> None:
    current_state = state[int(outer_pix)]
    update_state_row(
        build_path,
        outer_pix,
        traversal_status=WORK_STATUS_RUNNING,
        traversal_attempt_count=int(current_state["traversal_attempt_count"]) + 1,
        traversal_last_error_message="",
    )
    state["traversal_status"][int(outer_pix)] = WORK_STATUS_RUNNING
    state["traversal_attempt_count"][int(outer_pix)] = (
        int(current_state["traversal_attempt_count"]) + 1
    )
    state["traversal_last_error_message"][int(outer_pix)] = b""


def _record_traversal_result(
    build_path: Path,
    state: np.ndarray,
    result: TraversalTaskResult,
) -> None:
    if result.success:
        update_state_row(
            build_path,
            result.outer_pix,
            traversal_status=WORK_STATUS_DONE,
            traversal_last_error_message="",
        )
        state["traversal_status"][int(result.outer_pix)] = WORK_STATUS_DONE
        state["traversal_last_error_message"][int(result.outer_pix)] = b""
        return

    update_state_row(
        build_path,
        result.outer_pix,
        traversal_status=WORK_STATUS_FAILED,
        traversal_last_error_message=result.error_message,
    )
    state["traversal_status"][int(result.outer_pix)] = WORK_STATUS_FAILED
    state["traversal_last_error_message"][int(result.outer_pix)] = str(
        result.error_message
    ).encode("utf-8")


def _create_traversal_executor(workers: int) -> ProcessPoolExecutor:
    """Create the loky process pool used by the parallel Traversal runner."""

    return ProcessPoolExecutor(max_workers=int(workers))


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


def _reset_running_rows(
    build_path: Path,
    state: np.ndarray,
    *,
    status_field: str,
    error_field: str,
) -> int:
    running = np.flatnonzero(state[status_field] == WORK_STATUS_RUNNING)
    for outer_pix in running:
        update_state_row(
            build_path,
            int(outer_pix),
            **{
                status_field: WORK_STATUS_PENDING,
                error_field: "",
            },
        )
        state[status_field][int(outer_pix)] = WORK_STATUS_PENDING
        state[error_field][int(outer_pix)] = b""
    return int(len(running))


def _dispatch_parallel_outer_pixel(
    *,
    build_path: Path,
    context: TraversalTaskContext,
    state: np.ndarray,
    scheduler: OuterPixelScheduler,
    executor: ProcessPoolExecutor,
    workers: int,
    active: dict[Future[TraversalTaskResult], int],
    dispatch_after_outer_pix: int | None,
) -> None:
    while len(active) < workers:
        outer_pix = scheduler.select_next_outer_pixel(
            state,
            last_completed_outer_pix=dispatch_after_outer_pix,
        )
        dispatch_after_outer_pix = None
        if outer_pix is None:
            return
        _mark_outer_pixel_running(build_path, state, outer_pix)
        append_build_log(build_path, f"phase=traversal outer_pix={outer_pix} dispatched")
        future = executor.submit(_run_outer_pixel_traversal_task, context, outer_pix)
        active[future] = int(outer_pix)


def _run_traversal_phase(build_path: Path, *, workers: int) -> tuple[bool, dict[str, int]]:
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_TRAVERSAL:
        raise BuildError(f"Expected traversal phase, got {current_phase!r}")

    fields = phase_state_fields(current_phase)
    if fields is None:
        raise BuildError(f"Build phase {current_phase!r} is not outer-pixel-local")
    status_field, _, error_field = fields

    definition = load_persisted_build_definition(build_path)
    build_artifact_root(build_path, definition).mkdir(parents=True, exist_ok=True)
    roots = load_build_roots(build_path)
    context = TraversalTaskContext(
        build_path=build_path,
        definition=definition,
        roots=roots,
        legacy_config_path=load_legacy_config_path(build_path),
    )
    star_counts = _load_gaia_star_counts(
        gaia_root=roots.gaia_root,
        gaia_release=definition.gaia_release,
        outer_level=definition.outer_level,
    )

    repaired = _repair_stale_running_rows(build_path, status_field=status_field)
    state = load_state(build_path)
    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(
        build_path,
        f"run start phase={current_phase} repaired_stale_running={repaired} workers={workers}",
    )

    scheduler = OuterPixelScheduler(
        outer_level=definition.outer_level,
        status_field=status_field,
        star_counts=star_counts,
    )
    failed = False

    if workers == 1:
        last_completed_outer_pix: int | None = None
        while True:
            outer_pix = scheduler.select_next_outer_pixel(
                state,
                last_completed_outer_pix=last_completed_outer_pix,
            )
            if outer_pix is None:
                break
            append_build_log(build_path, f"phase=traversal outer_pix={outer_pix} dispatched")
            build_outer_pixel_products(build_path, outer_pix)
            state = load_state(build_path)
            row = state[int(outer_pix)]
            if int(row["traversal_status"]) == WORK_STATUS_DONE:
                append_build_log(
                    build_path,
                    f"phase=traversal outer_pix={outer_pix} completed",
                )
                last_completed_outer_pix = outer_pix
            else:
                failed = True
                error_message = row["traversal_last_error_message"].decode("utf-8")
                append_build_log(
                    build_path,
                    f"phase=traversal outer_pix={outer_pix} failed: {error_message}",
                )
                last_completed_outer_pix = None
    else:
        executor = None
        try:
            executor = _create_traversal_executor(workers)
            active: dict[Future[TraversalTaskResult], int] = {}
            _dispatch_parallel_outer_pixel(
                build_path=build_path,
                context=context,
                state=state,
                scheduler=scheduler,
                executor=executor,
                workers=workers,
                active=active,
                dispatch_after_outer_pix=None,
            )

            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    outer_pix = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = TraversalTaskResult(
                            outer_pix=outer_pix,
                            success=False,
                            error_message=str(exc),
                        )

                    _record_traversal_result(build_path, state, result)
                    if result.success:
                        append_build_log(
                            build_path,
                            f"phase=traversal outer_pix={result.outer_pix} completed",
                        )
                        dispatch_after_outer_pix: int | None = result.outer_pix
                    else:
                        failed = True
                        append_build_log(
                            build_path,
                            f"phase=traversal outer_pix={result.outer_pix} failed: {result.error_message}",
                        )
                        dispatch_after_outer_pix = None

                    _dispatch_parallel_outer_pixel(
                        build_path=build_path,
                        context=context,
                        state=state,
                        scheduler=scheduler,
                        executor=executor,
                        workers=workers,
                        active=active,
                        dispatch_after_outer_pix=dispatch_after_outer_pix,
                    )
            executor.shutdown(wait=True, kill_workers=True)
            executor = None
        except Exception as exc:
            if executor is not None:
                try:
                    executor.shutdown(wait=False, kill_workers=True)
                except Exception as shutdown_exc:
                    append_build_log(
                        build_path,
                        f"phase=traversal executor shutdown failed: {shutdown_exc}",
                    )
            reset_running = _reset_running_rows(
                build_path,
                state,
                status_field=status_field,
                error_field=error_field,
            )
            set_build_status(build_path, BUILD_STATUS_FAILED)
            append_build_log(build_path, f"phase=traversal infrastructure failed: {exc}")
            append_build_log(
                build_path,
                "run complete "
                "phase=traversal "
                "status=failed "
                f"reset_running={reset_running}",
            )
            raise

    phase_counts = {
        "pending": int(np.count_nonzero(state[status_field] == WORK_STATUS_PENDING)),
        "running": int(np.count_nonzero(state[status_field] == WORK_STATUS_RUNNING)),
        "done": int(np.count_nonzero(state[status_field] == WORK_STATUS_DONE)),
        "failed": int(np.count_nonzero(state[status_field] == WORK_STATUS_FAILED)),
    }
    final_status = BUILD_STATUS_FAILED if failed or phase_counts["failed"] else BUILD_STATUS_RUNNING
    set_build_status(build_path, final_status)
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
    return final_status != BUILD_STATUS_FAILED and phase_counts["pending"] == 0, phase_counts


def _run_aggregation_phase(build_path: Path) -> None:
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_AGGREGATION:
        raise BuildError(f"Expected aggregation phase, got {current_phase!r}")

    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(build_path, "run start phase=aggregation")
    try:
        level_maps = build_maps(build_path)
    except Exception as exc:
        set_build_status(build_path, BUILD_STATUS_FAILED)
        append_build_log(build_path, f"phase=aggregation failed: {exc}")
        append_build_log(build_path, "run complete phase=aggregation status=failed")
        raise
    definition = load_persisted_build_definition(build_path)
    if definition.survey_extent_overlays:
        set_build_status(build_path, BUILD_STATUS_RUNNING)
        status = "running"
    else:
        set_build_status(build_path, BUILD_STATUS_COMPLETED)
        status = "completed"
    append_build_log(
        build_path,
        "run complete "
        "phase=aggregation "
        f"status={status} "
        f"levels={','.join(str(level) for level in sorted(level_maps))}",
    )


def _run_augmentation_phase(build_path: Path) -> None:
    current_phase = load_current_phase(build_path)
    if current_phase != BUILD_PHASE_AUGMENTATION:
        raise BuildError(f"Expected augmentation phase, got {current_phase!r}")

    set_build_status(build_path, BUILD_STATUS_RUNNING)
    append_build_log(build_path, "run start phase=augmentation")
    try:
        level_layers = build_survey_extent_layers(build_path)
    except Exception as exc:
        set_build_status(build_path, BUILD_STATUS_FAILED)
        append_build_log(build_path, f"phase=augmentation failed: {exc}")
        append_build_log(build_path, "run complete phase=augmentation status=failed")
        raise

    set_build_status(build_path, BUILD_STATUS_COMPLETED)
    append_build_log(
        build_path,
        "run complete "
        "phase=augmentation "
        "status=completed "
        f"levels={','.join(str(level) for level in sorted(level_layers))}",
    )


def run_build(build_path: Path, *, workers: int = 1) -> Path:
    """Run all unfinished build work through the implemented phases."""

    if workers < 1:
        raise BuildError(f"workers must be at least 1, got {workers}")
    if not build_path.is_dir():
        raise BuildError(f"Build path does not exist: {build_path}")

    current_phase = load_current_phase(build_path)
    if current_phase not in (
        BUILD_PHASE_TRAVERSAL,
        BUILD_PHASE_AGGREGATION,
        BUILD_PHASE_AUGMENTATION,
    ):
        raise BuildError(
            f"Runner implements only {BUILD_PHASE_TRAVERSAL!r} and "
            f"{BUILD_PHASE_AGGREGATION!r}, and {BUILD_PHASE_AUGMENTATION!r}, "
            f"but build phase is {current_phase!r}"
        )

    if current_phase == BUILD_PHASE_TRAVERSAL:
        traversal_complete, _ = _run_traversal_phase(build_path, workers=int(workers))
        if not traversal_complete:
            return build_path
        set_current_phase(build_path, BUILD_PHASE_AGGREGATION)
        current_phase = BUILD_PHASE_AGGREGATION

    if current_phase == BUILD_PHASE_AGGREGATION:
        _run_aggregation_phase(build_path)
        definition = load_persisted_build_definition(build_path)
        if not definition.survey_extent_overlays:
            return build_path
        set_current_phase(build_path, BUILD_PHASE_AUGMENTATION)

    _run_augmentation_phase(build_path)
    return build_path


def restart_build(
    *,
    ao_system_short_name: str,
    config_short_name: str,
    build_root: Path | None,
    aosky_conf: Path | None = None,
    workers: int = 1,
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
    return run_build(build_path, workers=workers)


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
