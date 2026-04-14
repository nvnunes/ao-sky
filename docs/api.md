# Python API Documentation

This document describes the current code-first API exposed by `ao_sky`.

The supported raw Gaia entrypoint is `ao_sky.gaia`. Use
`GaiaStoreConfig(root, release, healpix_level)` to declare the canonical store
root and tiling, then use `GaiaHealpixStore(config)` to read or materialize one
outer-pixel Gaia HDF5 file at a time.

## Current Public Surface

### `ao_sky.gaia`

The current package-supported Gaia API exposes:

- `GaiaStoreConfig`
- `GaiaHealpixStore`
- `GaiaError`
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

## Core Read Path

### `GaiaHealpixStore.healpix_filename(outer_pix: int) -> Path`

Return the canonical HDF5 filename for one outer nested HEALPix pixel.

### `GaiaHealpixStore.load_healpix(outer_pix: int, *, force_reload: bool = False) -> Table`

Load one raw Gaia table for an outer pixel.

Behavior:

- if the canonical HDF5 file already exists and `force_reload=False`, read it
- otherwise query the Gaia archive seam, write the canonical file, then return
  the canonical table

Non-goals of the current Gaia API:

- proper-motion application
- neighbour stitching
- derived working-band magnitudes
- instrument-specific photometric proxies
- repo-root config discovery

## Working Example

```python
from pathlib import Path

from ao_sky.gaia import GaiaHealpixStore, GaiaStoreConfig

config = GaiaStoreConfig(
    root=Path("/data/gaia"),
    release="dr3",
    healpix_level=6,
)
store = GaiaHealpixStore(config)

filename = store.healpix_filename(outer_pix=0)
table = store.load_healpix(outer_pix=0)

print(filename)
print(table.colnames)
print(len(table))
```

## Reference

Use the generated reference page for the complete public module surface:

- [Gaia API Reference](reference/gaia/api.md)
