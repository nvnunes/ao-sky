# Validation

This source-tree research log records physical validation evidence for `ao-sky`
build artifacts.

## Contents

- `2026-04-20` [All Sky Map Validation](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)
- `2026-05-06` [Local Winner-Map Validation](validation/entries/2026-05-06-e002-local-winner-map-validation/e002.md)

## Summary

### All-sky

#### Observations

- The completed `v1` all-sky build produced HEALPix level `6` through `9` map
  artifacts for all `49,152` outer pixels, with no failed, pending, or running
  traversal states
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- Level-6 summaries were finite and in expected ranges for density, dust, AO
  metrics, and coverage fields
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- The retained `best_fwhm` plot and value summary showed finite all-sky FWHM
  structure, with level-6 values spanning `75.9` to `326.2` mas
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- Density, AO metric, winner, and coverage maps showed coherent all-sky
  structure: low-latitude density appeared in the expected orientation, AO maps
  were smooth at large scale, and coverage stayed within fractional bounds
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- Best-vs-winner checks preserved the expected ordering at every retained map
  level, with no finite `winner_ee_resolved` value greater than `best_ee`
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).

#### Conclusions

- The first retained all-sky validation pass supports the completed `v1`
  aggregation and plotting path at the broad sanity-check level
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
