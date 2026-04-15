# Python API Documentation

This document describes the current code-first API exposed by `ao_sky`.

The supported public surface currently centers on:

- `ao_sky.gaia` for canonical Gaia storage and proper-motion transforms
- `ao_sky.spatial` for reusable non-plotting HEALPix helpers
- `ao_sky.asterisms` for outer-pixel star assembly and in-memory search

## Current Public Surface

### `ao_sky.gaia`

The current package-supported Gaia API exposes:

- `GaiaStoreConfig`
- `GaiaHealpixStore`
- `GaiaError`
- `apply_proper_motion`
- `compute_legacy_r_magnitude`
- `GAIA_SCHEMA_COLUMNS`
- `HDF5_DATASET_NAME`
- `HDF5_COMPRESSION`
- `HDF5_COMPRESSION_OPTS`
- `HDF5_SHUFFLE`

The canonical raw Gaia schema is:

- `source_id`
- `ra`
- `dec`
- `G`
- `BP`
- `RP`
- `ref_epoch`
- `pmra`
- `pmdec`
- `non_single_star`
- `ruwe`

The canonical raw Gaia path contract is:

`<root>/gaia-<release>-hpx<healpix_level>/<hour>h/<sign><deg>/<outer_pix>/gaia.h5`

The canonical stored dataset is:

- HDF5 dataset name: `gaia`
- compression: `gzip`
- compression options: `9`
- shuffle: `True`

### `ao_sky.spatial`

The current package-supported spatial API exposes:

- `get_pixel_resolution`
- `get_pixel_area`
- `get_pixel_from_skycoord`
- `get_pixel_skycoord`
- `get_parent_pixel`
- `get_pixel_neighbours`
- `get_subpixels`
- `get_subpixel_indexes`

### `ao_sky.asterisms`

The current package-supported asterism API exposes:

- `AsterismSearchOptions`
- `load_asterism_stars`
- `find_asterisms`
- `ASTERISM_TABLE_COLUMNS`

## Core Read And Search Paths

### `GaiaHealpixStore.healpix_filename(outer_pix: int) -> Path`

Return the canonical HDF5 filename for one outer nested HEALPix pixel.

### `GaiaHealpixStore.load_healpix(outer_pix: int, *, force_reload: bool = False) -> Table`

Load one raw Gaia table for an outer pixel.

Behavior:

- if the canonical HDF5 file already exists and `force_reload=False`, read it
- otherwise query the Gaia archive seam, write the canonical file, then return
  the canonical table

### `apply_proper_motion(table, *, epoch=None, dt_years=None) -> Table`

Return a canonical Gaia table with in-memory proper motion applied. The schema
is preserved exactly, and the returned table updates `ra`, `dec`, and
`ref_epoch`.

### `compute_legacy_r_magnitude(table) -> ndarray`

Return the empirical Gaia-to-`R` magnitude estimate used by the legacy guide-star
workflow. This is an in-memory Gaia-domain helper and is not part of the raw
canonical HDF5 contract.

### `load_asterism_stars(store, outer_pix, *, neighbour_level=None, epoch=None, dt_years=None, include_locality=False) -> Table`

Return the Gaia rows needed to search one outer pixel for asterisms.

Behavior:

- load the local outer pixel fully
- when `neighbour_level` is set, include only the border-trimmed neighbour rows
- optionally apply Gaia proper motion in-memory
- add the legacy empirical `R` magnitude used by the current asterism search
- return Gaia-schema rows, optionally enriched with locality columns

### `find_asterisms(stars, options) -> Table`

Run the legacy-faithful in-memory asterism search over a prepared Gaia star
table.

Behavior:

- search brightness is fixed to the legacy empirical Gaia-derived `R`
- supported star-count range is 1-3
- output is a clean flat table with fixed `star1_*`/`star2_*`/`star3_*` slots
- the table carries the temporary full legacy-comparison field set

Non-goals of the current Gaia API:

- instrument-specific photometric proxies
- repo-root config discovery

## Working Example

```python
from pathlib import Path

from ao_sky.asterisms import AsterismSearchOptions, find_asterisms, load_asterism_stars
from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig

config = GaiaStoreConfig(
    root=Path("/data/gaia"),
    release="dr3",
    healpix_level=6,
)
store = GaiaHealpixStore(config)

filename = store.healpix_filename(outer_pix=0)
stars = load_asterism_stars(store, outer_pix=0, neighbour_level=7, epoch=2016.0)
asterisms = find_asterisms(stars, AsterismSearchOptions(min_stars=1, max_stars=3))

print(filename)
print(stars.colnames)
print(asterisms.colnames)
print(len(asterisms))
```

## Reference

Use the generated reference page for the complete public module surface:

- [Gaia API Reference](reference/gaia/api.md)
- [Spatial API Reference](reference/spatial/api.md)
- [Asterisms API Reference](reference/asterisms/api.md)
