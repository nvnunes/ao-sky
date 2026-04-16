"""Build creation, execution, and summary helpers."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from collections.abc import Iterator
import multiprocessing
from pathlib import Path
from queue import Empty

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
    STATE_ERROR_MAX_BYTES,
    WORK_STATUS_DONE,
    WORK_STATUS_FAILED,
    WORK_STATUS_PENDING,
    WORK_STATUS_RUNNING,
)
from ._exceptions import BuildError
from .artifacts import write_outer_artifact
from .config import (
    load_build_definition,
    resolve_build_root_only,
    resolve_build_roots,
    resolve_traversal_execution_config,
)
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
from ._models import (
    TraversalCacheStats,
    TraversalExecutionConfig,
    TraversalTaskContext,
    TraversalTaskResult,
    TraversalWorkerMessage,
    TraversalWorkerPlan,
)
from .regional import build_regional_worker_plans
from .runtime_gaia import RuntimeGaiaHealpixStore
from .scheduler import OuterPixelScheduler
from .traversal import TraversalGeometry, build_traversal_products

TRAVERSAL_PROGRESS_LOG_INTERVAL = 100
TRUNCATED_ERROR_SUFFIX = "... [truncated]"


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
    *,
    runtime=None,
    store=None,
    geometry: TraversalGeometry | None = None,
) -> None:
    """Run the Traversal pipeline for one outer pixel without mutating build state."""

    if runtime is None:
        runtime = load_native_runtime(
            context.definition,
            legacy_config_path=context.legacy_config_path,
            model_root=context.roots.model_root,
        )
        warm_model_cache(runtime)
    if store is None:
        base_store = GaiaHealpixStore(
            GaiaStoreConfig(
                root=context.roots.gaia_root,
                release=context.definition.gaia_release,
                healpix_level=context.definition.outer_level,
            )
        )
        store = RuntimeGaiaHealpixStore(
            base_store,
            runtime,
            max_entries=0,
            max_bytes=0,
        )
    if geometry is None:
        geometry = TraversalGeometry.from_runtime(runtime)
    asterisms, inner = build_traversal_products(
        store,
        runtime,
        outer_pix,
        dust_root=context.roots.dust_root,
        max_data_level=context.definition.max_data_level,
        geometry=geometry,
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

    error_message = _truncate_state_error_message(result.error_message)
    update_state_row(
        build_path,
        result.outer_pix,
        traversal_status=WORK_STATUS_FAILED,
        traversal_last_error_message=error_message,
    )
    state["traversal_status"][int(result.outer_pix)] = WORK_STATUS_FAILED
    state["traversal_last_error_message"][int(result.outer_pix)] = error_message.encode(
        "utf-8"
    )


def _truncate_state_error_message(message: object) -> str:
    """Return a UTF-8-safe error message that fits the persisted state field."""

    text = str(message)
    encoded = text.encode("utf-8")
    if len(encoded) <= STATE_ERROR_MAX_BYTES:
        return text

    suffix = TRUNCATED_ERROR_SUFFIX.encode("utf-8")
    limit = max(0, STATE_ERROR_MAX_BYTES - len(suffix))
    truncated = encoded[:limit].decode("utf-8", errors="ignore")
    return truncated + TRUNCATED_ERROR_SUFFIX


def _create_traversal_executor(workers: int) -> ProcessPoolExecutor:
    """Create the loky process pool used by the parallel Traversal runner."""

    return ProcessPoolExecutor(max_workers=int(workers))


def _create_regional_process_context():
    """Return the process context used by cache-aware regional Traversal."""

    return multiprocessing.get_context("spawn")


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
        future = executor.submit(_run_outer_pixel_traversal_task, context, outer_pix)
        active[future] = int(outer_pix)


def _gaia_cache_enabled(config: TraversalExecutionConfig) -> bool:
    return config.gaia_cache_entries > 0 and config.gaia_cache_mb > 0


def _run_regional_traversal_worker(
    context: TraversalTaskContext,
    plan: TraversalWorkerPlan,
    execution_config: TraversalExecutionConfig,
    result_queue,
) -> None:
    """Worker entrypoint for one region-owned Traversal plan."""

    for message in _iter_long_lived_traversal_worker_messages(
        context,
        plan,
        execution_config,
    ):
        result_queue.put(message)


def _iter_long_lived_traversal_worker_messages(
    context: TraversalTaskContext,
    plan: TraversalWorkerPlan,
    execution_config: TraversalExecutionConfig,
) -> Iterator[TraversalWorkerMessage]:
    """Yield Traversal messages from one reusable worker runtime."""

    configure_inference_threads(1)
    runtime = load_native_runtime(
        context.definition,
        legacy_config_path=context.legacy_config_path,
        model_root=context.roots.model_root,
    )
    warm_model_cache(runtime)
    base_store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=context.roots.gaia_root,
            release=context.definition.gaia_release,
            healpix_level=context.definition.outer_level,
        )
    )
    store = RuntimeGaiaHealpixStore(
        base_store,
        runtime,
        max_entries=execution_config.gaia_cache_entries,
        max_bytes=execution_config.gaia_cache_mb * 1024 * 1024,
    )
    geometry = TraversalGeometry.from_runtime(runtime)

    for outer_pix in plan.outer_pixs:
        yield TraversalWorkerMessage(
            worker_id=plan.worker_id,
            kind="started",
            outer_pix=int(outer_pix),
        )
        try:
            _materialize_outer_pixel_products(
                context,
                int(outer_pix),
                runtime=runtime,
                store=store,
                geometry=geometry,
            )
        except Exception as exc:
            yield TraversalWorkerMessage(
                worker_id=plan.worker_id,
                kind="failed",
                outer_pix=int(outer_pix),
                result=TraversalTaskResult(
                    outer_pix=int(outer_pix),
                    success=False,
                    error_message=str(exc),
                ),
            )
            continue
        yield TraversalWorkerMessage(
            worker_id=plan.worker_id,
            kind="completed",
            outer_pix=int(outer_pix),
            result=TraversalTaskResult(outer_pix=int(outer_pix), success=True),
        )

    if isinstance(store, RuntimeGaiaHealpixStore) and store.enabled:
        stats = store.stats()
        yield TraversalWorkerMessage(
            worker_id=plan.worker_id,
            kind="cache_stats",
            cache_stats=TraversalCacheStats(
                hits=stats.hits,
                misses=stats.misses,
                evictions=stats.evictions,
                current_bytes=stats.current_bytes,
                peak_bytes=stats.peak_bytes,
                entries=stats.entries,
            ),
        )
    yield TraversalWorkerMessage(worker_id=plan.worker_id, kind="done")


def _handle_regional_worker_message(
    *,
    build_path: Path,
    state: np.ndarray,
    message: TraversalWorkerMessage,
    progress: dict[int, dict[str, int]] | None = None,
    progress_interval: int = TRAVERSAL_PROGRESS_LOG_INTERVAL,
) -> bool:
    """Apply one regional worker message. Return whether it marks a failure."""

    worker_progress = None
    if progress is not None:
        worker_progress = progress.setdefault(
            int(message.worker_id),
            {"completed": 0, "failed": 0},
        )

    if message.kind == "started":
        if message.outer_pix is None:
            raise BuildError("Regional worker start message is missing outer_pix")
        _mark_outer_pixel_running(build_path, state, int(message.outer_pix))
        return False

    if message.kind in ("completed", "failed"):
        if message.result is None:
            raise BuildError(f"Regional worker {message.kind} message is missing result")
        _record_traversal_result(build_path, state, message.result)
        if message.result.success:
            if worker_progress is not None:
                worker_progress["completed"] += 1
                if (
                    progress_interval > 0
                    and worker_progress["completed"] % progress_interval == 0
                ):
                    append_build_log(
                        build_path,
                        "phase=traversal "
                        f"worker={message.worker_id} "
                        f"progress completed={worker_progress['completed']} "
                        f"failed={worker_progress['failed']} "
                        f"last_outer_pix={message.result.outer_pix}",
                    )
            return False
        if worker_progress is not None:
            worker_progress["failed"] += 1
        append_build_log(
            build_path,
            f"phase=traversal worker={message.worker_id} outer_pix={message.result.outer_pix} failed: {message.result.error_message}",
        )
        return True

    if message.kind == "cache_stats":
        stats = message.cache_stats
        if stats is None:
            raise BuildError("Regional worker cache_stats message is missing stats")
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={message.worker_id} "
            "cache_stats "
            f"hits={stats.hits} "
            f"misses={stats.misses} "
            f"evictions={stats.evictions} "
            f"entries={stats.entries} "
            f"current_bytes={stats.current_bytes} "
            f"peak_bytes={stats.peak_bytes}",
        )
        return False

    if message.kind == "done":
        return False

    raise BuildError(f"Unknown regional worker message kind {message.kind!r}")


def _run_regional_traversal_workers(
    *,
    build_path: Path,
    context: TraversalTaskContext,
    state: np.ndarray,
    plans: tuple[TraversalWorkerPlan, ...],
    execution_config: TraversalExecutionConfig,
) -> bool:
    """Run long-lived region-owned Traversal workers."""

    if not plans:
        return False

    process_context = _create_regional_process_context()
    result_queue = process_context.Queue()
    processes = []
    active_worker_ids = {plan.worker_id for plan in plans}
    progress: dict[int, dict[str, int]] = {
        plan.worker_id: {"completed": 0, "failed": 0} for plan in plans
    }
    failed = False

    for plan in plans:
        append_build_log(
            build_path,
            "phase=traversal "
            f"worker={plan.worker_id} "
            f"regions={','.join(str(pix) for pix in plan.region_pixs)} "
            f"outer_pixels={len(plan.outer_pixs)} "
            f"estimated_star_count={plan.estimated_star_count} started",
        )
        process = process_context.Process(
            target=_run_regional_traversal_worker,
            args=(context, plan, execution_config, result_queue),
        )
        process.start()
        processes.append(process)

    try:
        while active_worker_ids:
            try:
                message = result_queue.get(timeout=0.1)
            except Empty:
                for process, plan in zip(processes, plans, strict=True):
                    if plan.worker_id not in active_worker_ids:
                        continue
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"regional worker {plan.worker_id} exited with code {process.exitcode}"
                        )
                continue

            failed = _handle_regional_worker_message(
                build_path=build_path,
                state=state,
                message=message,
                progress=progress,
            ) or failed
            if message.kind == "done":
                active_worker_ids.discard(message.worker_id)
                worker_progress = progress[message.worker_id]
                append_build_log(
                    build_path,
                    "phase=traversal "
                    f"worker={message.worker_id} "
                    f"done completed={worker_progress['completed']} "
                    f"failed={worker_progress['failed']}",
                )
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()
    return failed


def _run_in_process_traversal_worker(
    *,
    build_path: Path,
    context: TraversalTaskContext,
    state: np.ndarray,
    plan: TraversalWorkerPlan,
    execution_config: TraversalExecutionConfig,
) -> bool:
    """Run one long-lived Traversal worker in the parent process."""

    failed = False
    progress: dict[int, dict[str, int]] = {
        plan.worker_id: {"completed": 0, "failed": 0}
    }
    append_build_log(
        build_path,
        "phase=traversal "
        f"worker={plan.worker_id} "
        f"regions={','.join(str(pix) for pix in plan.region_pixs)} "
        f"outer_pixels={len(plan.outer_pixs)} "
        f"estimated_star_count={plan.estimated_star_count} started",
    )
    for message in _iter_long_lived_traversal_worker_messages(
        context,
        plan,
        execution_config,
    ):
        failed = _handle_regional_worker_message(
            build_path=build_path,
            state=state,
            message=message,
            progress=progress,
        ) or failed
        if message.kind == "done":
            worker_progress = progress[message.worker_id]
            append_build_log(
                build_path,
                "phase=traversal "
                f"worker={message.worker_id} "
                f"done completed={worker_progress['completed']} "
                f"failed={worker_progress['failed']}",
            )
    return failed


def _run_traversal_phase(
    build_path: Path,
    *,
    execution_config: TraversalExecutionConfig,
) -> tuple[bool, dict[str, int]]:
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
        "run start "
        f"phase={current_phase} "
        f"repaired_stale_running={repaired} "
        f"workers={execution_config.workers} "
        f"region_level={execution_config.region_level} "
        f"gaia_cache_entries={execution_config.gaia_cache_entries} "
        f"gaia_cache_mb={execution_config.gaia_cache_mb}",
    )

    scheduler = OuterPixelScheduler(
        outer_level=definition.outer_level,
        status_field=status_field,
        star_counts=star_counts,
    )
    failed = False

    if execution_config.workers == 1:
        try:
            plans = build_regional_worker_plans(
                state=state,
                outer_level=definition.outer_level,
                region_level=int(execution_config.region_level),
                workers=1,
                status_field=status_field,
                star_counts=star_counts,
            )
            if plans:
                failed = _run_in_process_traversal_worker(
                    build_path=build_path,
                    context=context,
                    state=state,
                    plan=plans[0],
                    execution_config=execution_config,
                ) or failed
        except Exception as exc:
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
    elif _gaia_cache_enabled(execution_config):
        try:
            plans = build_regional_worker_plans(
                state=state,
                outer_level=definition.outer_level,
                region_level=int(execution_config.region_level),
                workers=execution_config.workers,
                status_field=status_field,
                star_counts=star_counts,
            )
            failed = _run_regional_traversal_workers(
                build_path=build_path,
                context=context,
                state=state,
                plans=plans,
                execution_config=execution_config,
            ) or failed
        except Exception as exc:
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
    else:
        executor = None
        try:
            executor = _create_traversal_executor(execution_config.workers)
            active: dict[Future[TraversalTaskResult], int] = {}
            completed_successes = 0
            _dispatch_parallel_outer_pixel(
                build_path=build_path,
                context=context,
                state=state,
                scheduler=scheduler,
                executor=executor,
                workers=execution_config.workers,
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
                        completed_successes += 1
                        if completed_successes % TRAVERSAL_PROGRESS_LOG_INTERVAL == 0:
                            append_build_log(
                                build_path,
                                "phase=traversal "
                                f"progress completed={completed_successes} "
                                f"last_outer_pix={result.outer_pix}",
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
                        workers=execution_config.workers,
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


def run_build(
    build_path: Path,
    *,
    workers: int | None = 1,
    gaia_cache_entries: int | None = None,
    gaia_cache_mb: int | None = None,
    region_level: int | None = None,
    aosky_conf: Path | None = None,
) -> Path:
    """Run all unfinished build work through the implemented phases."""

    if workers is not None and int(workers) < 1:
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
        definition = load_persisted_build_definition(build_path)
        execution_config = resolve_traversal_execution_config(
            outer_level=definition.outer_level,
            workers=workers,
            gaia_cache_entries=gaia_cache_entries,
            gaia_cache_mb=gaia_cache_mb,
            region_level=region_level,
            aosky_conf=aosky_conf,
        )
        traversal_complete, _ = _run_traversal_phase(
            build_path,
            execution_config=execution_config,
        )
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
    workers: int | None = 1,
    gaia_cache_entries: int | None = None,
    gaia_cache_mb: int | None = None,
    region_level: int | None = None,
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
    return run_build(
        build_path,
        workers=workers,
        gaia_cache_entries=gaia_cache_entries,
        gaia_cache_mb=gaia_cache_mb,
        region_level=region_level,
        aosky_conf=aosky_conf,
    )


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
