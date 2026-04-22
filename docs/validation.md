# Validation

This source-tree research log records physical validation evidence for `ao-sky`
build artifacts.

Timeline and entry routing live in `docs/validation/index.md`. Detailed
evidence lives under `docs/validation/entries/`.

## Summary

### All-sky

#### Observations

- The completed `v1` all-sky build produced HEALPix level `6` through `9` map
  artifacts for all `49,152` outer pixels, with no failed, pending, or running
  traversal states
  ([e001](validation/entries/2026-04-20-all-sky-e001-v1-all-sky-map-validation/index.md)).
- Level-6 summaries were finite and in expected ranges for density, dust, AO
  metrics, and coverage fields
  ([e001](validation/entries/2026-04-20-all-sky-e001-v1-all-sky-map-validation/index.md)).
- Density, AO metric, winner, and coverage maps showed coherent all-sky
  structure: low-latitude density appeared in the expected orientation, AO maps
  were smooth at large scale, and coverage stayed within fractional bounds
  ([e001](validation/entries/2026-04-20-all-sky-e001-v1-all-sky-map-validation/index.md)).
- Best-vs-winner checks preserved the expected ordering at every retained map
  level, with no finite `winner_ee_resolved` value greater than `best_ee`
  ([e001](validation/entries/2026-04-20-all-sky-e001-v1-all-sky-map-validation/index.md)).

#### Conclusions

- The first retained all-sky validation pass supports the completed `v1`
  aggregation and plotting path at the broad sanity-check level
  ([e001](validation/entries/2026-04-20-all-sky-e001-v1-all-sky-map-validation/index.md)).
