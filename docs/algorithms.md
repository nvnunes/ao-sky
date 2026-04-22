# Traversal Algorithms

This document records the current Traversal asterism-selection algorithm and
the legacy context it replaced. It is authoritative for algorithm intent and
implementation decisions around candidate generation, dense-field handling,
per-inner-pixel winner selection, and retained asterism catalog construction.

This document uses `resolved` for performance predicted at the inner-pixel
center, equivalent to the earlier "on-axis" language. It uses `averaged` for
performance predicted as a mean over the science field of view, equivalent to
the earlier "FOV-mean" language. Rotation or orientation of the guide-star
configuration is not part of the algorithm.

Use [`architecture.md`](architecture.md) for package boundaries and persisted
artifact ownership, [`plan.md`](plan.md) for migration phase context, and
`docs/benchmarking.md` for measured runtime and memory evidence.

## Legacy Compatible Algorithm

The legacy-compatible Traversal path is the known-good behavior from before
the Phase 14 winner-selection and dense-field candidate-control changes. It is
retained here as design context for comparisons against saved legacy products,
not as the active schema-version-2 algorithm.

For each outer pixel, legacy Traversal:

1. Builds a two-ring expanded Gaia star footprint around the outer pixel.
2. Prepares runtime Gaia rows with build-epoch coordinates, empirical `R`, and
   `hpx14`.
3. Filters NGS rows by configured magnitude and coarse density policy.
4. Generates candidate asterisms through `find_asterisms()`.
5. Applies bright-star exclusion to candidate centers.
6. Applies legacy-style overlap pruning by ranking candidates with predicted
   mean EE and greedily dropping lower-ranked candidates whose center circles
   overlap above `asterism.max_overlap`.
7. Assigns surviving asterism centers to inner pixels and persists only local
   center asterisms.
8. Builds a dense inner table from local Gaia rows and local retained
   asterism-center counts.
9. Builds a full materialized inner-pixel/asterism candidate-pair table from
   the expanded candidate set.
10. Runs resolved-model prediction for candidate pairs and updates per-pixel
    `best_sr`, `best_ee`, and `best_fwhm`.
11. Updates `winner_*` only from local asterisms, even though expanded
    non-local asterisms can influence `best_*`.
12. Runs averaged prediction only for selected winners and computes coverage.

The legacy path has known weaknesses that motivated replacement:

- overlap pruning happens before the per-inner-pixel winner field is known
- center-distance matching is more important than it should be
- dense fields are guarded mostly by coarse sky and density cuts
- a large inner-pixel/asterism pair table is materialized before prediction
- retained asterisms are determined by overlap pruning rather than by where
  they actually win

## Replacement Algorithm

The Phase 14 replacement combines winner-map-first asterism selection with
regional FOR-optimized NGS selection. The goal is to preserve exact all-NGS
candidate enumeration where the resulting outer-pixel resolved-inference work
is tractable, and to use FOR-optimized NGS selection only in locally dense
regions where exact enumeration is too expensive.

The replacement algorithm is:

1. Load the expanded outer-pixel Gaia footprint.
2. Build persisted `star_count` and `ngs_count` from the local Gaia rows.
   `ngs_count` always uses the configured `ao_system.max_mag`.
3. Build the bright-star allowed-pixel mask.
4. Build field-of-regard coverage for all configured NGS rows.
5. Partition the outer pixel into working regions. Use all configured NGS in
   regions whose exact candidate graph keeps cumulative outer-pixel
   resolved-inference work tractable, subdivide regions that are too dense, and
   apply FOR-optimized NGS selection only in regions that remain too dense at
   the minimum working scale.
6. Deduplicate regional candidate identities by exact Gaia source-id set.
7. Build per-star field-of-regard bitsets for the NGS that appear in retained
   candidate identities.
8. Stream exact 1-, 2-, and 3-star candidates according to
   `ao_system.min_wfs..ao_system.max_wfs`.
9. For each candidate, intersect member-star bitsets and the bright-star mask
   to determine eligible inner pixels.
10. Stream eligible candidate-pixel rows into resolved predicted-EE inference
   batches.
11. Maintain per-inner-pixel `best_*` and a bounded top-K shortlist.
12. Filter each top-K shortlist by `winner_ee_epsilon`.
13. Regularize the winner field using local inner-pixel neighbors.
14. Run averaged prediction only for final regularized winners.
15. Persist unique regularized winning asterisms and write final inner
    winner/coverage fields.

### Configuration

The new user-facing Traversal controls are:

```yaml
gaia:
  max_bright_star_exclusion_arcsec: null

asterism:
  winner_ee_epsilon: 0.01
```

`gaia.max_bright_star_exclusion_arcsec` is the angular radius around bright
stars where inner pixels are masked before model evaluation. When it is
`null`, the effective value is:

```text
2 * ao_system.fov_arcsec
```

Bright-star masking is active only when `gaia.max_bright_star_mag` is set.
If no bright-star magnitude threshold is configured, the exclusion distance is
ignored.

`winner_ee_epsilon` controls regularization tolerance. A pixel may switch from
its raw best asterism to another retained candidate only when:

```text
candidate_ee >= (1 - winner_ee_epsilon) * best_ee
```

Validate:

```text
max_bright_star_exclusion_arcsec is null or positive
0 <= winner_ee_epsilon < 1
1 <= min_wfs <= max_wfs <= 3
```

`asterism.max_overlap` is legacy-only. It is not used by the replacement
algorithm and should not be accepted as an active version 2 runtime setting.

Internal defaults should remain non-public until benchmark evidence says they
need to be exposed:

```text
max_regional_combination_work = inner pixels per outer pixel
winner_top_k = 3
winner_regularization_passes = 3
```

`max_regional_combination_work` is an internal dense-field tractability
threshold for complete regional candidate enumeration. For the default
`outer_level=6`, `inner_level=14` grid, it is `65,536`. It is not a final
outer-pixel cap on candidate identities or resolved inference rows. Prediction
batch sizing should use the model/traversal default rather than a new public
setting.

### Regional FOR-Optimized NGS Selection

Dense-field control happens before asterism combinations are generated. For
each outer pixel, Traversal first computes which inner pixels not excluded by
the bright-star mask are inside each configured NGS star's field of regard.

For each inner pixel `p` not excluded by the bright-star mask, define:

```text
full_availability[p] = number of configured NGS stars whose FOR contains p
target_availability[p] = min(full_availability[p], ao_system.max_wfs)
```

Bright-star masked pixels have `target_availability = 0`. If a pixel has only
one or two available guide stars at the configured `ao_system.max_mag`, the
target is that achievable availability rather than an impossible availability
of three.

The selector then sorts NGS stars by sensing magnitude and `source_id`. A star
is retained only if it increases selected NGS availability for at least one
inner pixel that is still below `target_availability`. Selection stops once
every inner pixel not excluded by the bright-star mask has `ao_system.max_wfs`
NGS within a FOR centered on it, or all available NGS if fewer are available.

Regional selection applies this FOR-optimized selector only after complete
regional enumeration is judged too expensive. Starting from the whole outer
pixel, Traversal collects the NGS stars whose FOR covers at least one target
inner pixel in the current region. If the complete regional candidate graph is
known to stay below `max_regional_combination_work`, all of those NGS are used
without reduction. If not, the region is subdivided along the nested HEALPix
hierarchy. The minimum working scale is the finest HEALPix level whose
characteristic resolution remains above the configured FOR size. At that scale,
any still-too-dense region uses the greedy FOR-optimized NGS selector described
above.

`max_regional_combination_work` controls local complete-enumeration decisions
only. Final outer-pixel candidate identities may exceed this value after
regional graphs are merged and deduplicated, and final resolved inference rows
may exceed it because one candidate identity can be valid for many inner
pixels. Final resolved inference rows are telemetry, not a normal-path pruning
target.

Candidate identities from all regions are deduplicated by exact Gaia
`source_id` set before model evaluation. This preserves exact all-NGS
enumeration in sparse parts of an outer pixel while preventing dense parts from
forcing a single blunt magnitude cap across the whole outer pixel.

The current implementation is still a greedy approximation in dense floor
regions, but its objective matches the physical requirement: preserve
guide-star availability across the field of regard. The faintest retained
magnitude is an outcome of the spatial coverage problem, not a public
configuration parameter.

Final candidate count is a measured consequence of the regional complete/FOR
decisions, close-pair geometry, and per-candidate pixel eligibility.

### Candidate Semantics

Candidates are exact star sets. A 3-star asterism is valid for an inner pixel
only where all three member stars are inside the field of regard. A 2-star
subset is a separate candidate with a separate identity.

When `min_wfs=1` and `max_wfs=3`, all enabled orders compete directly:

```text
1-star candidates
2-star candidates
3-star candidates
```

Runtime model mappings must include every enabled star count for both resolved
and averaged models. For example, `min_wfs=1` requires configured 1-star
resolved and averaged models.

Predicted EE is considered the same physical quantity across 1-, 2-, and
3-star models, so no per-order normalization or preference is applied.

Candidate identity has no rotation or orientation component. The candidate is
only the exact sorted set of Gaia member stars.

Candidate generation uses a close-pair graph. Edges satisfy:

```text
ao_system.min_sep_arcsec <= separation <= ao_system.fov_arcsec
```

One-star candidates are graph vertices, two-star candidates are graph edges,
and three-star candidates are graph triangles. Field-of-regard bitset
intersection remains the final truth for whether a candidate is valid at a
specific inner pixel.

Internal candidate identity should be based on sorted Gaia `source_id` values:

```text
candidate_key = (source_id_1, source_id_2, source_id_3)
```

The persisted `asterism_id` remains a compact per-outer-pixel ID assigned only
after final retained winners are known.

### Field-Of-Regard Bitsets

For each selected NGS star, precompute the inner pixels whose centers are
inside that star's field of regard:

```text
angular_distance(star, inner_pixel_center) <= ao_system.fov / 2
```

The bitset shape for one outer pixel is:

```text
P = number of inner pixels
W = ceil(P / 64)
star_pixel_bits.shape = (selected_ngs, W)
star_pixel_bits.dtype = uint64
```

Candidate eligibility becomes a bitwise intersection:

```text
1-star: bits[s1]
2-star: bits[s1] & bits[s2]
3-star: bits[s1] & bits[s2] & bits[s3]
```

Bright-star masking is applied before model evaluation by intersecting with an
allowed-pixel bitset:

```text
eligible_pixels = member_star_bits & bright_star_allowed_bits
```

The allowed-pixel bitset is built from the expanded prepared Gaia footprint.
Stars with `R < gaia.max_bright_star_mag` mask inner pixels whose centers fall
within `gaia.max_bright_star_exclusion_arcsec`.

Candidate-pixel rows with an empty eligibility bitset are skipped without
model inference.

### Resolved Best Fields

Resolved predicted EE drives selection. `best_*` is the continuous
best-performance field and is tied to the candidate, or seeing baseline, with
the highest resolved EE.

For an AO candidate:

```text
best_ee = resolved predicted EE of the EE-best candidate
best_sr = SR of that same candidate
best_fwhm = FWHM of that same candidate
```

`best_sr` is not independently maximized and `best_fwhm` is not independently
minimized.

Seeing baseline is retained for continuous maps. If no AO candidate exists,
persist the seeing baseline in `best_*`, leave `winner_asterism_id = -1`, and
persist the seeing baseline in `winner_ee_resolved` and `winner_ee_averaged`.

If an AO candidate exists but performs below seeing in EE, `best_*` may remain
the seeing baseline while `winner_*` still records the selected AO asterism.

### Top-K State

During streaming, each inner pixel keeps a bounded internal top-K shortlist:

```text
top_candidate_refs[P, K]
top_ee[P, K]
top_sr[P, K]
top_fwhm[P, K]
```

The shortlist is ranked by predicted EE. Candidate payloads are retained only
when they appear in at least one pixel shortlist; a later implementation may
drop zero-reference payloads during streaming.

After streaming, each pixel's shortlist is filtered by
`winner_ee_epsilon`. Regularization can only choose candidates that remain in
that pixel's epsilon-eligible shortlist.

### Regularization

Regularization operates on the winner assignment, not on the continuous
`best_*` performance field.

Initial winner labels are the top predicted-EE candidates:

```text
winner_ref[p] = top_candidate_refs[p, 0]
```

The regularizer uses local HEALPix neighbor agreement inside the current outer
pixel only. It does not smooth across outer-pixel boundaries in the first
implementation.

For each synchronous pass, each pixel considers labels currently assigned to
neighboring inner pixels. A pixel may switch to a neighbor-supported label only
when:

- the label is present in the pixel's epsilon-eligible top-K shortlist
- the switch strictly improves local neighbor-label agreement
- the EE loss remains within `winner_ee_epsilon`

Tie-breakers are:

1. higher local predicted EE
2. smaller stable candidate key

After regularization, `winner_ee_resolved` is filled from the chosen label's
resolved predicted EE for that pixel.

### Averaged Winners

Averaged prediction is not run for every candidate-pixel row. It is run only
after regularization, for final winner-pixel pairs.

For final regularized winners:

1. Group winner-pixel pairs by star count.
2. Rebuild NGS payloads from retained candidate member coordinates and inner
   pixel centers.
3. Run the averaged model in batches.
4. Fill `winner_ee_averaged`.
5. Compute averaged coverage.

This avoids storing Python NGS payload objects during candidate streaming.

### Retained Asterisms

Retained asterisms are the unique asterisms that win at least one inner pixel
after regularization:

```text
retained_refs = unique(winner_ref[p] for p where winner_ref[p] >= 0)
```

The retained asterism catalog keeps the existing center and member-star fields.
Catalog `pix` remains the inner pixel that contains the asterism center; it is
metadata for the asterism table, not the rule used to assign winners.

Additional retained-footprint metadata, such as winner-pixel counts or median
winner performance, is deferred until after the core replacement algorithm is
implemented and benchmarked.

## Derived Coverage And Physical Statistics

The inner HEALPix grid is the source of truth for physical statistics. Each
inner pixel represents one candidate pointing center and stores the best
resolved performance field plus the regularized winning asterism and its
resolved and averaged EE values.

The first-class coverage metric for Phase 15 validation should be averaged EE
from the final regularized winner:

```text
M(x) = winner_ee_averaged(x)
```

Resolved EE remains useful as a diagnostic, but averaged EE is the natural
field-usable quantity because it measures mean performance over the science
field of view for the selected pointing.

For any metric `M(x)` and threshold `tau`, coverage over a footprint is the
area fraction of inner pixels that satisfy the metric criterion:

```text
higher-is-better: C(tau) = count(M(x) >= tau) / count(x)
lower-is-better:  C(tau) = count(M(x) <= tau) / count(x)
```

Because inner HEALPix pixels are equal-area, this is a simple fraction within
one outer pixel or any footprint selected directly from inner-pixel centers.
Configured inner fields such as `coverage_resolved` and `coverage_averaged`
are threshold booleans; aggregating them over a footprint gives the
corresponding area coverage fraction for the configured thresholds.

These booleans are computed from the final regularized winner fields:
`coverage_resolved` from `winner_ee_resolved`, and `coverage_averaged` from
`winner_ee_averaged`. They are not computed from raw `best_ee`.

Coverage curves are cumulative coverage evaluated across a range of thresholds.
They should be a Phase 15 validation and reporting product, not a requirement
for the Phase 14 traversal artifact. For observatory-specific comparisons,
coverage curves should be computed only over the declination band satisfying a
maximum transit airmass cut. For observatory latitude `lat` and transit
airmass limit `X0`, the reporting layer should convert `X0` to a minimum
transit elevation and use the corresponding accessible declination band. The
curves should be area-weighted over the selected equal-area inner pixels or
outer-pixel aggregates.

Uniformity measures spatial stability across a footprint. For a metric such
as averaged EE where higher values are better:

```text
IQR = Q75(M) - Q25(M)
U = 1 - IQR / Q50(M)
```

Higher `U` means the footprint has less relative variation around its typical
performance. If the metric can be zero or negative, the implementation should
guard against unstable `Q50` values. For lower-is-better metrics such as FWHM,
uniformity should be computed after transforming the metric to a
higher-is-better quantity, such as `1 / FWHM`, so larger uniformity always has
the same interpretation.

Spatial coherence measures the angular scale over which the performance field
remains correlated. For a footprint metric `M(x)`, compute an angular
two-point autocorrelation by binning inner-pixel pairs by angular separation:

```text
rho(theta) = mean((M_i - mean(M)) * (M_j - mean(M))) / var(M)
```

The coherence scale is the separation where `rho(theta)` falls to `0.5`.
Coherence is expected to be more expensive and more sensitive to sampling than
coverage or uniformity, so it belongs first as a Phase 15 prototype/reporting
statistic rather than a Phase 14 persisted product.

Outer-pixel statistics should be derived from the contained inner pixels. For
science fields whose footprint is comparable to, or not aligned with, outer
HEALPix cells, statistics should be computed directly from the inner pixels
whose centers fall inside the exact science footprint rather than by averaging
precomputed outer-pixel summaries.

## Artifact Schema Changes

The replacement algorithm removes inner fields that were tied to debugging or
center-based retained-catalog semantics:

```text
asterism_count
winner_distance_arcsec
```

The inner table should keep:

```text
pix
star_count
ngs_count
best_ee
best_sr
best_fwhm
winner_asterism_id
winner_ee_resolved
winner_ee_averaged
coverage_resolved
coverage_averaged
gaia_A0
```

Aggregated map artifacts should also include `winner_asterism_count`. This
field is special: it is derived from retained `asterisms` rows, not from the
inner-pixel table. It counts retained winner asterisms whose center lies inside
the map pixel at every retained map level. It is not the number of distinct
`winner_asterism_id` values used by inner pixels, and it is not simply the row
count of each owning `outer.h5` artifact because retained asterism centers can
fall outside the outer pixel whose inner pixels used them.

The legacy-preserving Traversal artifact layout/schema is version 1. The
replacement Traversal artifact layout/schema should be version 2 so old and new
`outer.h5` products are not silently confused.

## Dense Fields

Traversal does not apply a Galactic latitude cut or `gaia.max_star_density`.
Regional FOR-optimized NGS selection controls dense-field combinatorics directly
before full asterism enumeration. Future dust or stellar-density cuts should be
treated as field-selection policy, not as Traversal tractability guards.

## Telemetry

Basic telemetry may include many scalar counters as long as it does not retain
full candidate histories or per-pixel detail. Expensive diagnostics should be
behind detailed telemetry.

Useful basic telemetry includes:

- configured `max_mag`
- configured and effective `max_bright_star_exclusion_arcsec`
- configured NGS rows and selected NGS rows
- selected NGS faintest magnitude
- full and selected FOR-availability fractions for thresholds 1 through
  `max_wfs`
- close-pair edge count and graph-triangle count
- exact candidate identity count
- streamed candidate counts by star order
- candidate-pixel evaluations by star order
- empty candidate-footprint count
- field-of-regard bitset bytes
- bright-masked inner-pixel count
- top-K saturated pixels
- pixels with no AO candidate
- pixels with epsilon-eligible alternatives
- raw and regularized unique winners by star order
- winner pixels changed by regularization
- mean, p95, and max EE loss from regularization
- averaged winner rows by star order

Detailed telemetry may add per-pixel or per-candidate diagnostics when needed
for algorithm debugging and benchmark work.

## Verification Expectations

The replacement algorithm intentionally diverges from the legacy overlap and
winner-selection path. Legacy comparison tooling may remain useful as a
diagnostic, but it is no longer the acceptance authority for Phase 14. Legacy
comparisons after the replacement should use saved legacy data rather than
running the live `survey_tools` implementation.

Core repo-native tests should cover:

- FOR-optimized NGS selection, including pixels with only partially achievable
  FOR availability
- validation for `winner_ee_epsilon`, `max_bright_star_exclusion_arcsec`, and
  legacy-only `max_overlap`
- exact 1-, 2-, and 3-star candidate generation
- close-pair graph and graph-triangle candidate generation
- field-of-regard bitset construction and set-bit extraction
- bright-star masking before prediction
- required resolved and averaged model mappings for every enabled star count
- top-K insertion, eviction, and epsilon filtering
- local neighbor-oriented regularization
- `best_*` tied to predicted-EE selection
- coverage booleans derived from final regularized winner fields
- seeing baseline fallback with no AO winner
- averaged prediction only for final regularized winners
- retained catalog rows derived from final winners
- removal of `asterism_count` and `winner_distance_arcsec`

Benchmark acceptance should measure dense-field behavior, memory footprint,
candidate-pixel evaluations, regularization effects, and whether the Galactic
latitude cut can be revisited.
