"""Build-definition and root-resolution helpers."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..survey import SurveyError, normalize_survey_extent_overlays
from ._exceptions import BuildError
from ._models import BuildDefinition, BuildPaths, TraversalExecutionConfig


DEFAULT_GAIA_CACHE_ENTRIES = 16
DEFAULT_GAIA_CACHE_MB = 256
AO_SKY_CONFIG_FILENAME = "ao-sky.yaml"


def _load_aosky_yaml(
    *,
    aosky_yaml: Path | None,
    cwd: Path | None = None,
) -> dict[str, object]:
    conf_path = aosky_yaml
    if conf_path is None:
        search_root = Path.cwd() if cwd is None else Path(cwd)
        candidate = search_root / AO_SKY_CONFIG_FILENAME
        if candidate.is_file():
            conf_path = candidate
        else:
            project_root = _find_python_project_root(search_root)
            if project_root is not None:
                candidate = project_root / AO_SKY_CONFIG_FILENAME
                if candidate.is_file():
                    conf_path = candidate

    if conf_path is None:
        return {}

    with Path(conf_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _find_python_project_root(start: Path) -> Path | None:
    """Return the nearest ancestor that looks like a Python project root."""

    root = Path(start).expanduser().resolve()
    if root.is_file():
        root = root.parent
    for candidate in (root, *root.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def resolve_runtime_root_candidates(
    *,
    gaia_root: Path | None,
    build_root: Path | None,
    model_root: Path | None,
    aosky_yaml: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, Path | None]:
    """Resolve optional runtime roots from CLI arguments or `ao-sky.yaml`."""

    conf_data = _load_aosky_yaml(aosky_yaml=aosky_yaml, cwd=cwd)
    root_data = _build_root_config(conf_data)

    resolved_gaia_root = Path(gaia_root) if gaia_root is not None else None
    resolved_build_root = Path(build_root) if build_root is not None else None
    resolved_model_root = Path(model_root) if model_root is not None else None

    if resolved_gaia_root is None and root_data.get("gaia") is not None:
        resolved_gaia_root = Path(str(root_data["gaia"]))
    if resolved_build_root is None:
        resolved_build_root = Path.cwd() if cwd is None else Path(cwd)
    if resolved_model_root is None and root_data.get("model") is not None:
        resolved_model_root = Path(str(root_data["model"]))

    return {
        "gaia_root": (
            None if resolved_gaia_root is None else resolved_gaia_root.expanduser().resolve()
        ),
        "build_root": (
            None if resolved_build_root is None else resolved_build_root.expanduser().resolve()
        ),
        "dust_root": (
            None if resolved_gaia_root is None else resolved_gaia_root.expanduser().resolve()
        ),
        "model_root": (
            None if resolved_model_root is None else resolved_model_root.expanduser().resolve()
        ),
    }


def _build_root_config(conf_data: dict[str, object]) -> dict[str, object]:
    build_raw = conf_data.get("build")
    if not isinstance(build_raw, dict):
        return {}
    roots_raw = build_raw.get("roots")
    if not isinstance(roots_raw, dict):
        return {}
    return roots_raw


def load_build_definition(filename: Path) -> tuple[BuildDefinition, str]:
    """Load and validate one merged build-config YAML file."""

    with Path(filename).open("r", encoding="utf-8") as handle:
        raw_text = handle.read()
    raw = yaml.safe_load(raw_text) or {}

    traversal_raw = _required_mapping(raw, "traversal", "Build config")
    gaia_raw = _required_mapping(raw, "gaia", "Build config")
    maps_raw = _required_mapping(raw, "maps", "Build config")

    try:
        overlays = normalize_survey_extent_overlays(
            raw.get("survey_overlays"),
            base_dir=Path(filename).resolve().parent,
            resolve_paths=False,
        )
    except SurveyError as exc:
        raise BuildError(str(exc)) from exc
    definition = BuildDefinition(
        lineage_name=_derive_lineage_name(filename),
        gaia_release=_required_str(gaia_raw, "release", "gaia").lower(),
        outer_level=_required_int(traversal_raw, "outer_level", "traversal"),
        inner_level=_required_int(traversal_raw, "inner_level", "traversal"),
        max_data_level=_required_int(maps_raw, "max_level", "maps"),
        survey_extent_overlays=overlays,
    )
    if not definition.gaia_release:
        raise BuildError("gaia.release must be a non-empty string")
    if definition.outer_level < 0:
        raise BuildError("traversal.outer_level must be non-negative")
    if definition.inner_level <= definition.outer_level:
        raise BuildError("traversal.inner_level must be larger than traversal.outer_level")
    if definition.max_data_level < definition.outer_level:
        raise BuildError("maps.max_level must be greater than or equal to traversal.outer_level")
    if definition.max_data_level > definition.inner_level:
        raise BuildError("maps.max_level must be less than or equal to traversal.inner_level")
    return definition, raw_text


def _derive_lineage_name(filename: Path) -> str:
    """Derive the build lineage from the config location."""

    resolved = Path(filename).expanduser().resolve()
    if resolved.name == AO_SKY_CONFIG_FILENAME:
        return resolved.parent.name
    return resolved.stem


def _required_mapping(
    payload: dict[str, object],
    key: str,
    context: str,
) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BuildError(f"{context} is missing required section: {key}")
    return value


def _required_str(payload: dict[str, object], key: str, context: str) -> str:
    if key not in payload:
        raise BuildError(f"{context} is missing required field: {key}")
    value = str(payload[key]).strip()
    if not value:
        raise BuildError(f"{context}.{key} must be a non-empty string")
    return value


def _required_int(payload: dict[str, object], key: str, context: str) -> int:
    if key not in payload:
        raise BuildError(f"{context} is missing required field: {key}")
    try:
        return int(payload[key])
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{context}.{key} must be an integer") from exc


def resolve_build_roots(
    *,
    gaia_root: Path | None,
    build_root: Path | None,
    model_root: Path | None,
    aosky_yaml: Path | None = None,
    cwd: Path | None = None,
) -> BuildPaths:
    """Resolve build, Gaia-derived dust, and model roots from CLI or `ao-sky.yaml`."""
    candidates = resolve_runtime_root_candidates(
        gaia_root=gaia_root,
        build_root=build_root,
        model_root=model_root,
        aosky_yaml=aosky_yaml,
        cwd=cwd,
    )
    resolved_gaia_root = candidates["gaia_root"]
    resolved_build_root = candidates["build_root"]
    resolved_dust_root = candidates["dust_root"]
    resolved_model_root = candidates["model_root"]

    if resolved_gaia_root is None:
        raise BuildError(
            "gaia_root must be provided either via CLI or ao-sky.yaml"
        )
    if resolved_build_root is None:
        resolved_build_root = Path.cwd() if cwd is None else Path(cwd)
    if resolved_model_root is None:
        raise BuildError(
            "model_root must be provided either via CLI or ao-sky.yaml"
        )

    return BuildPaths(
        gaia_root=resolved_gaia_root.expanduser().resolve(),
        build_root=resolved_build_root.expanduser().resolve(),
        dust_root=resolved_dust_root.expanduser().resolve(),
        model_root=resolved_model_root.expanduser().resolve(),
    )


def resolve_build_root_only(
    *,
    build_root: Path | None,
    aosky_yaml: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve only the build root from CLI arguments or `ao-sky.yaml`."""
    candidates = resolve_runtime_root_candidates(
        gaia_root=None,
        build_root=build_root,
        model_root=None,
        aosky_yaml=aosky_yaml,
        cwd=cwd,
    )
    resolved_build_root = candidates["build_root"]
    if resolved_build_root is None:
        return (Path.cwd() if cwd is None else Path(cwd)).expanduser().resolve()
    return resolved_build_root.expanduser().resolve()


def resolve_gaia_root_only(
    *,
    gaia_root: Path | None,
    aosky_yaml: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve only the Gaia root from CLI arguments or `ao-sky.yaml`."""

    candidates = resolve_runtime_root_candidates(
        gaia_root=gaia_root,
        build_root=None,
        model_root=None,
        aosky_yaml=aosky_yaml,
        cwd=cwd,
    )
    resolved_gaia_root = candidates["gaia_root"]
    if resolved_gaia_root is None:
        raise BuildError(
            "gaia_root must be provided either via CLI or ao-sky.yaml"
        )
    return resolved_gaia_root.expanduser().resolve()


def resolve_survey_root_only(
    *,
    survey_root: Path | None,
    aosky_yaml: Path | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """Resolve the optional survey source root from CLI arguments or `ao-sky.yaml`."""

    conf_data = _load_aosky_yaml(aosky_yaml=aosky_yaml, cwd=cwd)
    root_data = _build_root_config(conf_data)
    resolved_survey_root = Path(survey_root) if survey_root is not None else None
    if resolved_survey_root is None and root_data.get("survey") is not None:
        resolved_survey_root = Path(str(root_data["survey"]))
    if resolved_survey_root is None:
        return None
    return resolved_survey_root.expanduser().resolve()


def resolve_traversal_execution_config(
    *,
    outer_level: int,
    workers: int | None = None,
    scheduler: str | None = None,
    low_latitude_workers: int | None = None,
    gaia_cache_entries: int | None = None,
    gaia_cache_mb: int | None = None,
    parent_memory_limit_mb: int | None = None,
    telemetry: str | None = None,
    aosky_yaml: Path | None = None,
    cwd: Path | None = None,
) -> TraversalExecutionConfig:
    """Resolve runtime-only Traversal execution settings."""

    conf_data = _load_aosky_yaml(aosky_yaml=aosky_yaml, cwd=cwd)
    build_data = conf_data.get("build") if isinstance(conf_data.get("build"), dict) else {}
    prediction_devices = _resolve_prediction_device_defaults(conf_data)
    resolved_workers = _resolve_int_setting(
        workers,
        build_data.get("workers"),
        default=1,
        name="workers",
    )
    resolved_low_latitude_workers = _resolve_optional_int_setting(
        low_latitude_workers,
        build_data.get("low_latitude_workers"),
        name="low_latitude_workers",
    )
    resolved_scheduler = str(
        scheduler if scheduler is not None else build_data.get("scheduler", "static")
    ).strip().lower()
    resolved_cache_entries = _resolve_int_setting(
        gaia_cache_entries,
        conf_data.get("gaia_cache_entries"),
        default=DEFAULT_GAIA_CACHE_ENTRIES,
        name="gaia_cache_entries",
    )
    resolved_cache_mb = _resolve_int_setting(
        gaia_cache_mb,
        conf_data.get("gaia_cache_mb"),
        default=DEFAULT_GAIA_CACHE_MB,
        name="gaia_cache_mb",
    )
    if "parent_memory_limit_mb" in build_data:
        raise BuildError(
            "build.parent_memory_limit_mb was renamed; use build.memory_limit_mb"
        )
    resolved_parent_memory_limit_mb = _resolve_int_setting(
        parent_memory_limit_mb,
        build_data.get("memory_limit_mb"),
        default=0,
        name="memory_limit_mb",
    )
    resolved_telemetry = (telemetry or "basic").strip().lower()

    if resolved_workers < 1:
        raise BuildError(f"workers must be at least 1, got {resolved_workers}")
    if resolved_scheduler not in {"static", "dynamic"}:
        raise BuildError(
            "scheduler must be either 'static' or 'dynamic', "
            f"got {resolved_scheduler!r}"
        )
    if resolved_scheduler == "dynamic" and resolved_low_latitude_workers is not None:
        raise BuildError("low_latitude_workers is only supported by the static scheduler")
    if resolved_low_latitude_workers is not None and (
        resolved_low_latitude_workers < 1
        or resolved_low_latitude_workers > resolved_workers
    ):
        raise BuildError(
            "low_latitude_workers must be between 1 and workers "
            f"({resolved_workers}), got {resolved_low_latitude_workers}"
        )
    if resolved_cache_entries < 0:
        raise BuildError(
            f"gaia_cache_entries must be non-negative, got {resolved_cache_entries}"
        )
    if resolved_cache_mb < 0:
        raise BuildError(f"gaia_cache_mb must be non-negative, got {resolved_cache_mb}")
    if resolved_parent_memory_limit_mb < 0:
        raise BuildError(
            "memory_limit_mb must be non-negative, "
            f"got {resolved_parent_memory_limit_mb}"
        )
    if resolved_telemetry not in {"basic", "detailed"}:
        raise BuildError(
            "telemetry must be either 'basic' or 'detailed', "
            f"got {resolved_telemetry!r}"
        )

    return TraversalExecutionConfig(
        workers=resolved_workers,
        scheduler=resolved_scheduler,
        low_latitude_workers=resolved_low_latitude_workers,
        gaia_cache_entries=resolved_cache_entries,
        gaia_cache_mb=resolved_cache_mb,
        region_level=derive_traversal_region_level(
            outer_level=int(outer_level),
            workers=resolved_workers,
        ),
        parent_memory_limit_mb=resolved_parent_memory_limit_mb,
        telemetry=resolved_telemetry,
        prediction_device=prediction_devices[0],
        averaged_prediction_device=prediction_devices[1],
    )


def _resolve_prediction_device_defaults(
    conf_data: dict[str, object],
) -> tuple[str, str]:
    prediction_raw = conf_data.get("prediction")
    if not isinstance(prediction_raw, dict):
        return "cpu", "cpu"

    resolved_device = prediction_raw.get("resolved_device")
    averaged_device = prediction_raw.get("averaged_device")
    shared_device = prediction_raw.get("device", "cpu")
    if shared_device is None:
        shared_device = "cpu"
    if resolved_device is None:
        resolved_device = shared_device
    if averaged_device is None:
        averaged_device = shared_device

    return (
        _validate_prediction_device_default(
            resolved_device,
            field_name="prediction.resolved_device",
        ),
        _validate_prediction_device_default(
            averaged_device,
            field_name="prediction.averaged_device",
        ),
    )


def _validate_prediction_device_default(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise BuildError(f"{field_name} must be either 'cpu' or 'gpu'")
    return normalized


def derive_traversal_region_level(*, outer_level: int, workers: int) -> int:
    """Derive the coarse HEALPix region level used for Traversal locality."""

    target_outer_pixels_per_region = max(4, int(workers) * 4)
    region_depth = math.ceil(math.log(target_outer_pixels_per_region, 4))
    return max(0, int(outer_level) - int(region_depth))


def _resolve_int_setting(
    explicit: int | None,
    configured: object,
    *,
    default: int,
    name: str,
) -> int:
    value = explicit if explicit is not None else configured
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{name} must be an integer") from exc


def _resolve_optional_int_setting(
    explicit: int | None,
    configured: object,
    *,
    name: str,
) -> int | None:
    value = explicit if explicit is not None else configured
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BuildError(f"{name} must be an integer") from exc
