"""Build-definition and root-resolution helpers."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from ..survey import SurveyError, normalize_survey_extent_overlays
from ._exceptions import BuildError
from ._models import BuildDefinition, BuildPaths


DEFAULT_LEGACY_CONFIG = (
    Path(__file__).resolve().parents[3] / "survey_tools" / "aomap" / "config.yaml"
)


def load_build_definition(filename: Path) -> tuple[BuildDefinition, str]:
    """Load and validate a minimal build-definition YAML file."""

    with Path(filename).open("r", encoding="utf-8") as handle:
        raw_text = handle.read()
    raw = yaml.safe_load(raw_text) or {}

    required = (
        "ao_system_short_name",
        "config_short_name",
        "gaia_release",
        "outer_level",
        "inner_level",
        "max_data_level",
        "epoch",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise BuildError(
            "Build definition is missing required fields: " + ", ".join(missing)
        )

    try:
        overlays = normalize_survey_extent_overlays(
            raw.get("survey_extent_overlays"),
            base_dir=Path(filename).resolve().parent,
        )
    except SurveyError as exc:
        raise BuildError(str(exc)) from exc
    definition = BuildDefinition(
        ao_system_short_name=str(raw["ao_system_short_name"]).strip(),
        config_short_name=str(raw["config_short_name"]).strip(),
        gaia_release=str(raw["gaia_release"]).strip().lower(),
        outer_level=int(raw["outer_level"]),
        inner_level=int(raw["inner_level"]),
        max_data_level=int(raw["max_data_level"]),
        epoch=float(raw["epoch"]),
        min_galactic_latitude=(
            None
            if raw.get("min_galactic_latitude") is None
            else float(raw["min_galactic_latitude"])
        ),
        survey_extent_overlays=overlays,
    )
    if not definition.ao_system_short_name:
        raise BuildError("ao_system_short_name must be a non-empty string")
    if not definition.config_short_name:
        raise BuildError("config_short_name must be a non-empty string")
    if not definition.gaia_release:
        raise BuildError("gaia_release must be a non-empty string")
    if definition.outer_level < 0:
        raise BuildError("outer_level must be non-negative")
    if definition.inner_level <= definition.outer_level:
        raise BuildError("inner_level must be larger than outer_level")
    if definition.max_data_level < definition.outer_level:
        raise BuildError("max_data_level must be greater than or equal to outer_level")
    if definition.max_data_level > definition.inner_level:
        raise BuildError("max_data_level must be less than or equal to inner_level")
    if not math.isfinite(definition.epoch):
        raise BuildError("epoch must be a finite number")
    if definition.min_galactic_latitude is not None and not math.isfinite(
        definition.min_galactic_latitude
    ):
        raise BuildError("min_galactic_latitude must be finite when provided")

    return definition, raw_text


def resolve_build_roots(
    *,
    gaia_root: Path | None,
    build_root: Path | None,
    dust_root: Path | None,
    model_root: Path | None,
    default_model_root: Path | None = None,
    aosky_conf: Path | None = None,
    cwd: Path | None = None,
) -> BuildPaths:
    """Resolve build, Gaia, dust, and model roots from CLI or `aosky.conf`."""

    conf_path = aosky_conf
    if conf_path is None:
        search_root = Path.cwd() if cwd is None else Path(cwd)
        candidate = search_root / "aosky.conf"
        if candidate.is_file():
            conf_path = candidate

    conf_data: dict[str, object] = {}
    if conf_path is not None:
        with Path(conf_path).open("r", encoding="utf-8") as handle:
            conf_data = yaml.safe_load(handle) or {}

    resolved_gaia_root = Path(gaia_root) if gaia_root is not None else None
    resolved_build_root = Path(build_root) if build_root is not None else None
    resolved_dust_root = Path(dust_root) if dust_root is not None else None
    resolved_model_root = Path(model_root) if model_root is not None else None

    if resolved_gaia_root is None and conf_data.get("gaia_root") is not None:
        resolved_gaia_root = Path(str(conf_data["gaia_root"]))
    if resolved_build_root is None and conf_data.get("build_root") is not None:
        resolved_build_root = Path(str(conf_data["build_root"]))
    if resolved_dust_root is None and conf_data.get("dust_root") is not None:
        resolved_dust_root = Path(str(conf_data["dust_root"]))
    if resolved_model_root is None and conf_data.get("model_root") is not None:
        resolved_model_root = Path(str(conf_data["model_root"]))
    if resolved_model_root is None and default_model_root is not None:
        resolved_model_root = Path(default_model_root)

    if resolved_gaia_root is None:
        raise BuildError(
            "gaia_root must be provided either via CLI or aosky.conf"
        )
    if resolved_build_root is None:
        raise BuildError(
            "build_root must be provided either via CLI or aosky.conf"
        )
    if resolved_dust_root is None:
        raise BuildError(
            "dust_root must be provided either via CLI or aosky.conf"
        )
    if resolved_model_root is None:
        raise BuildError(
            "model_root must be provided either via CLI or aosky.conf"
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
    aosky_conf: Path | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve only the build root from CLI arguments or `aosky.conf`."""

    conf_path = aosky_conf
    if conf_path is None:
        search_root = Path.cwd() if cwd is None else Path(cwd)
        candidate = search_root / "aosky.conf"
        if candidate.is_file():
            conf_path = candidate

    conf_data: dict[str, object] = {}
    if conf_path is not None:
        with Path(conf_path).open("r", encoding="utf-8") as handle:
            conf_data = yaml.safe_load(handle) or {}

    resolved_build_root = Path(build_root) if build_root is not None else None
    if resolved_build_root is None and conf_data.get("build_root") is not None:
        resolved_build_root = Path(str(conf_data["build_root"]))
    if resolved_build_root is None:
        raise BuildError(
            "build_root must be provided either via CLI or aosky.conf"
        )
    return resolved_build_root.expanduser().resolve()
