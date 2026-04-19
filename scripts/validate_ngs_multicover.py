"""Validate FOR-optimized NGS selection for one outer pixel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from ao_sky.build.runtime_config import load_runtime_config
from ao_sky.build.runtime_gaia import RuntimeGaiaHealpixStore
from ao_sky.build.traversal import (
    TraversalGeometry,
    _bitset_to_mask,
    _build_bright_star_allowed_bitset,
    _build_regional_candidate_graph,
    _build_star_for_coverage,
    _count_candidate_identities,
    _select_ngs_for_multicover,
    prepare_search_inputs,
)
from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig


def _peak_rss_mib() -> float:
    # macOS reports ru_maxrss in bytes.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 * 1024)


def _fraction_ge(depth: np.ndarray, threshold: int, mask: np.ndarray) -> float:
    total = int(np.count_nonzero(mask))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(np.asarray(depth)[mask] >= int(threshold)) / total)


def _depth_for_star_indices(
    coverage,
    star_indices: np.ndarray,
    *,
    cap: int,
) -> np.ndarray:
    depth = np.zeros(coverage.target_depth.shape, dtype=np.uint16)
    for star_idx in np.asarray(star_indices, dtype=np.int64):
        start = int(coverage.starts[int(star_idx)])
        end = int(coverage.starts[int(star_idx) + 1])
        if start == end:
            continue
        pixels = coverage.pixel_indices[start:end]
        depth[pixels] = np.minimum(depth[pixels] + 1, int(cap)).astype(
            np.uint16,
            copy=False,
        )
    return depth


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate FOR-optimized NGS selection for one outer pixel without "
            "running model inference."
        )
    )
    parser.add_argument("outer_pix", type=int)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="Schema-v2 runtime config, usually a build.yaml from a benchmark build.",
    )
    parser.add_argument(
        "--gaia-root",
        type=Path,
        default=Path("/Users/nelsonnunes/ao-sky-cache/gaia"),
    )
    parser.add_argument("--gaia-release", default="dr3")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Model root used only to satisfy runtime config loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    model_root = args.model_root or args.runtime_config.parent / "models"
    runtime = load_runtime_config(args.runtime_config, model_root=model_root)

    store = GaiaHealpixStore(
        GaiaStoreConfig(
            root=args.gaia_root,
            release=args.gaia_release,
            healpix_level=runtime.outer_level,
        )
    )
    runtime_store = RuntimeGaiaHealpixStore(store, runtime, max_entries=0, max_bytes=0)
    geometry = TraversalGeometry.from_runtime(runtime)
    outer_pix = int(args.outer_pix)

    star_started = time.perf_counter()
    stars, ngs = prepare_search_inputs(
        runtime_store,
        runtime,
        outer_pix,
        geometry=geometry,
    )
    star_selection_s = time.perf_counter() - star_started

    inner_centres = geometry.inner_centres(outer_pix)
    inner_count = len(inner_centres)
    required_depth = int(runtime.ao_system.max_wfs)

    bright_started = time.perf_counter()
    allowed_bits = _build_bright_star_allowed_bitset(stars, inner_centres, runtime)
    allowed_pixels = _bitset_to_mask(allowed_bits, inner_count)
    bright_mask_s = time.perf_counter() - bright_started

    coverage_started = time.perf_counter()
    coverage = _build_star_for_coverage(
        ngs,
        inner_centres,
        runtime,
        required_depth=required_depth,
        allowed_pixels=allowed_pixels,
    )
    coverage_s = time.perf_counter() - coverage_started

    selection_started = time.perf_counter()
    selection = _select_ngs_for_multicover(ngs, coverage)
    selected_ngs = ngs[selection.star_indices]
    selection_s = time.perf_counter() - selection_started

    count_started = time.perf_counter()
    selected_counts = _count_candidate_identities(selected_ngs, runtime)
    candidate_count_s = time.perf_counter() - count_started

    source_to_index = {
        int(source_id): index
        for index, source_id in enumerate(np.asarray(ngs["source_id"], dtype=np.int64))
    }

    regional_started = time.perf_counter()
    regional_graph, regional_stats = _build_regional_candidate_graph(
        ngs,
        coverage,
        runtime,
    )
    regional_s = time.perf_counter() - regional_started
    regional_ngs = regional_graph.sorted_ngs[: regional_graph.final_count]
    regional_indices = np.asarray(
        [
            source_to_index[int(source_id)]
            for source_id in np.asarray(regional_ngs["source_id"], dtype=np.int64)
        ],
        dtype=np.int64,
    )
    regional_depth = _depth_for_star_indices(
        coverage,
        regional_indices,
        cap=required_depth,
    )

    target_mask = allowed_pixels
    selected_target_complete = (
        float(
            np.count_nonzero(selection.depth[target_mask] >= coverage.target_depth[target_mask])
        )
        / int(np.count_nonzero(target_mask))
        if int(np.count_nonzero(target_mask))
        else 0.0
    )
    output = {
        "outer_pix": outer_pix,
        "inner_pixels": inner_count,
        "allowed_inner_pixels": int(np.count_nonzero(allowed_pixels)),
        "search_star_rows": int(len(stars)),
        "total_ngs": int(len(ngs)),
        "required_depth": required_depth,
        "full_depth_ge_1": _fraction_ge(coverage.full_depth, 1, target_mask),
        "full_depth_ge_2": _fraction_ge(coverage.full_depth, 2, target_mask),
        "full_depth_ge_3": _fraction_ge(coverage.full_depth, 3, target_mask),
        "selected_ngs": int(len(selected_ngs)),
        "selected_max_mag": (
            float(np.max(np.asarray(selected_ngs["R"], dtype=np.float64)))
            if len(selected_ngs)
            else None
        ),
        "selected_depth_ge_1": _fraction_ge(selection.depth, 1, target_mask),
        "selected_depth_ge_2": _fraction_ge(selection.depth, 2, target_mask),
        "selected_depth_ge_3": _fraction_ge(selection.depth, 3, target_mask),
        "selected_target_complete": selected_target_complete,
        "selected_candidate_total": int(selected_counts.total),
        "selected_candidate_singles": int(selected_counts.singles),
        "selected_candidate_pairs": int(selected_counts.pairs),
        "selected_candidate_triples": int(selected_counts.triples),
        "regional_selected_ngs": int(regional_graph.final_count),
        "regional_selected_max_mag": (
            float(np.max(np.asarray(regional_ngs["R"], dtype=np.float64)))
            if len(regional_ngs)
            else None
        ),
        "regional_depth_ge_1": _fraction_ge(regional_depth, 1, target_mask),
        "regional_depth_ge_2": _fraction_ge(regional_depth, 2, target_mask),
        "regional_depth_ge_3": _fraction_ge(regional_depth, 3, target_mask),
        "regional_candidate_total": int(regional_graph.candidate_count),
        "regional_exact_regions": int(regional_stats.exact_regions),
        "regional_for_optimized_regions": int(regional_stats.for_optimized_regions),
        "regional_split_regions": int(regional_stats.split_regions),
        "regional_max_depth": int(regional_stats.max_depth),
        "regional_max_combination_work": int(
            regional_stats.max_regional_combination_work,
        ),
        "regional_exact_combination_work": int(
            regional_stats.exact_combination_work,
        ),
        "regional_final_resolved_inferences": int(
            regional_stats.final_resolved_inferences,
        ),
        "regional_incomplete_for_regions": int(regional_stats.incomplete_for_regions),
        "star_selection_s": star_selection_s,
        "bright_mask_s": bright_mask_s,
        "coverage_build_s": coverage_s,
        "selection_s": selection_s,
        "candidate_count_s": candidate_count_s,
        "regional_resolver_s": regional_s,
        "elapsed_s": time.perf_counter() - started,
        "peak_rss_mib": _peak_rss_mib(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
