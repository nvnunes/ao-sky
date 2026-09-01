"""All-sky augmentation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..survey import build_survey_extent_dataset
from ._constants import SURVEY_EXTENT_DATASET
from ._exceptions import BuildError
from ._schema_compat import require_current_build_layout
from .artifacts import write_maps_family_dataset
from .control import load_build_definition, maps_artifact_filename


def build_survey_extent_layers(
    build_path: Path,
) -> dict[int, np.ndarray]:
    """Build and persist dense survey-extent layers for all aggregated levels."""

    require_current_build_layout(build_path, operation="augment")
    definition = load_build_definition(build_path)
    if not definition.survey_extent_overlays:
        return {}

    level_layers: dict[int, np.ndarray] = {}
    for level in range(definition.outer_level, definition.max_data_level + 1):
        maps_filename = maps_artifact_filename(build_path, level)
        if not maps_filename.is_file():
            raise BuildError(f"Maps artifact not found for augmentation: {maps_filename}")
        layer = build_survey_extent_dataset(level, definition.survey_extent_overlays)
        write_maps_family_dataset(
            maps_filename,
            dataset_name=SURVEY_EXTENT_DATASET,
            data=layer,
        )
        level_layers[level] = layer

    return level_layers
