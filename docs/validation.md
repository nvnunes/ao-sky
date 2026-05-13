# Validation

This source-tree research log records physical validation evidence for `ao-sky`
build artifacts.

## Contents

- [Entries](#entries)
- [Summary](#summary)

## Entries

- `2026-04-20` [All Sky Map Validation](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)
- `2026-05-06` [Small-Scale Winner-Map Validation](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)
- `2026-05-10` [v1 vs v3 All-Sky Map Comparison](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)
- `2026-05-13` [Compare Winner EE Resolved to Best EE](validation/entries/2026-05-13-e004-compare-winner-ee-resolved-to-best-ee/e004.md)
- `2026-05-13` [Regularization Trade Study](validation/entries/2026-05-13-e005-regularization-trade-study/e005.md)

## Summary

### All-Sky v1

#### Observations

- The completed `v1` all-sky build produced HEALPix level `6` through `9` map
  artifacts for all `49,152` outer pixels, with no failed, pending, or running
  traversal states
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- Level-6 summaries had numbers in expected ranges for density, dust, AO
  metrics, and coverage fields
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- The `best_fwhm` plot and value summary showed the expected all-sky FWHM
  pattern, with level-6 values spanning `75.9` to `326.2` mas
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- Density, AO metric, winner, and coverage maps looked physically sensible:
  low-latitude density appeared in the expected orientation, AO maps were
  smooth at large scale, and coverage stayed between `0` and `1`
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).
- Best-vs-winner checks preserved the expected ordering at every map level,
  with no valid `winner_ee_resolved` value greater than `best_ee`
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).

#### Conclusions

- The completed `v1` all-sky maps look physically sensible and internally
  consistent
  ([e001](validation/entries/2026-04-20-e001-all-sky-map-validation/e001.md)).

### Small-Scale Map Validation

#### Observations

- Winner-map asterism selection intentionally differs from the
  legacy-compatible retained catalog: it predicts candidate performance first,
  regularizes, or smooths, per-inner-pixel winners, and keeps asterisms from
  where they win rather than from legacy overlap pruning
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).
- The retained-asterism center definition was changed to use a
  minimum-enclosing-FOV center. In the COSMOS local rebuild, this changed
  retained centers while preserving retained asterism membership, meaning the
  stars in each asterism, and the inner-pixel winner source IDs relative to v1
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).
- Removing the expensive triangle pre-filter did not change the COSMOS outputs
  when no-winner recovery was disabled: retained asterism membership, retained
  centers, winner source IDs, and resolved/averaged EE values matched the
  previous new-center local rebuild
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).
- No-winner recovery addresses almost all of the missing COSMOS legacy-covered
  inner pixels by adding `4515` additional retained asterisms and raising inner
  pixels with winners from `71.55%` to `98.76%`
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).
- No-winner recovery coverage comes from valid FoV pointings that need not be
  centered on the science inner pixel; the added recovered area has lower
  performance than already-covered inner-pixel-centered regions but represents
  the best achievable performance for those recovered pixels
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).
- The COSMOS region in the completed all-sky `v3` build matches the local
  no-winner recovery rebuild in retained asterism membership, retained asterism
  centers, and winner coverage. Remaining winner differences are boundary
  near-tie swaps, with EE differences at numerical-noise scale
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).

#### Conclusions

- The COSMOS tests showed that the `v1` winner-map coverage was not as
  complete as it should be. Several small-scale issues were found, including
  asterism center differences and valid inner pixels that did not receive
  winners
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).
- No-winner recovery, the second pass over inner pixels with no winners, should
  be applied going forward to increase coverage. It fixes the uncovered valid
  COSMOS regions without disturbing the existing best-performance field.
- The completed all-sky `v3` map with no-winner recovery enabled matches expectations
  ([e002](validation/entries/2026-05-06-e002-small-scale-winner-map-validation/e002.md)).

### All-Sky v3

#### Observations

- The completed `v3` all-sky `stellar_density` maps match `v1` exactly at
  levels `6` and `9` as would be expected
  ([e003](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)).
- No-winner recovery produces `10%`+ EE gains on average by filling coverage gaps, particularly
  away from the Galactic plane. 
- At level `9`, `65.6%` of pixels improved in `best_ee`; mean `best_ee` increased by `11.2%`
  ([e003](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)).
- The winner maps realize the same improvement: no map pixel has lower
  `winner_ee_resolved` or `winner_ee_averaged` in `v3` than in `v1`, and the
  level-9 means increased by `11.2%` for resolved EE and `12.3%` for averaged EE
  ([e003](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)).
- The coverage increase is smaller than the direct EE increase because the
  recovered off-center winners have poorer performance. Level-9
  `coverage_resolved` increased by only `1.2%`
  ([e003](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)).
- Averaged coverage changes more, but this is expected because recovered
  off-center pointings more often clear the lower averaged-performance
  threshold. Level-9 `coverage_averaged` increased by `8.4%`
  ([e003](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)).

#### Conclusions

- The completed `v3` maps make sense: the changes are in the expected
  direction and are consistent across the EE and coverage maps
  ([e003](validation/entries/2026-05-10-e003-v1-vs-v3-all-sky-map-comparison/e003.md)).

### Winner Regularization

#### Observations

- The all-sky `winner_ee_resolved` map stays very close to `best_ee`: the
  median relative EE difference is `-0.0004`, and the p05 value is about
  `10%` of the allowed fractional EE loss
  ([e004](validation/entries/2026-05-13-e004-compare-winner-ee-resolved-to-best-ee/e004.md)).
- COSMOS has regularization preserving best EE for most inner
  pixels while producing a moderately smooth winner map
  ([e004](validation/entries/2026-05-13-e004-compare-winner-ee-resolved-to-best-ee/e004.md)).
- A high stellar density pixel, shows the map can
  still be quite fragmented at small scale, even when the EE loss remains inside the
  configured tolerance
  ([e004](validation/entries/2026-05-13-e004-compare-winner-ee-resolved-to-best-ee/e004.md)).

#### Conclusions

- Regularization does not appear to be working optimally across the sky. It
  keeps winner EE close to best EE, but some small-scale regions still choose
  inconsistent winners
  ([e004](validation/entries/2026-05-13-e004-compare-winner-ee-resolved-to-best-ee/e004.md)).

#### Follow-up

- A trade study should test whether different coupled `K-top` and
  `winner_ee_epsilon` values produce smoother maps while keeping EE loss
  acceptable. This follow-up is tracked in
  [e005](validation/entries/2026-05-13-e005-regularization-trade-study/e005.md).
