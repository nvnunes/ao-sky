# Brief: Estimating AO WFS Sensing-Band Magnitudes from Gaia Photometry

## Goal

We want practical estimates of guide-star brightness in the relevant AO wavefront-sensing (WFS) bands using Gaia data, with approximate accuracy at the ~10% level being sufficient for the intended application.

The purpose is not precision photometric calibration. The purpose is to obtain a reasonable proxy for the photon flux seen by each instrument's WFS.

---

## Core conceptual point

Published Gaia photometric transforms are best understood as **empirical color-based regressions**, not physical blackbody reconstructions.

They work because stellar SEDs are not arbitrary. For normal stars, broad-band optical/NIR fluxes lie on a relatively low-dimensional family, so Gaia broad-band measurements can predict other band-integrated fluxes approximately.

For our purposes, the relevant quantity is not strictly a catalog magnitude in some standard band, but rather the **band-integrated photon flux through the actual WFS response curve**.

---

## Accuracy assumptions

Working assumptions from the discussion:

- Optical transforms near Gaia's native coverage are generally quite good.
- Near-IR transforms are less direct but still usable for ordinary stars.
- Approximate uncertainty levels:
  - `I`: roughly a few hundredths of a mag
  - `H`: roughly `0.05–0.10 mag` in favorable cases; `~0.1 mag` is a reasonable working assumption
- A `0.1 mag` uncertainty corresponds to about `10%` in flux, which is acceptable for this use case.

This is good enough for:
- guide-star characterization
- target preselection
- approximate WFS flux estimates
- probabilistic modeling

This is **not** intended to replace measured photometry in the true sensing band.

---

## Instrument-specific decisions

### 1. HARMONI
Use **Gaia XP synthetic photometry** through the actual WFS passband.

Reason:
- The sensing band is in the red optical, within Gaia XP spectral coverage.
- This is better than mapping to a standard `I` magnitude and treating that as exact.

Implementation direction:
- Define the instrument throughput curve for the WFS band.
- Use Gaia XP / GaiaXPy to compute synthetic photometry directly in that band.

### 2. GNAO
Treat the same way as HARMONI.

Implementation direction:
- Use **Gaia XP synthetic photometry** through the actual WFS passband, assuming the sensing band lies within XP coverage.

### 3. MORFEO
Use a **Gaia-based transform to `H`** as the WFS brightness proxy.

Reason:
- The WFS sensing is effectively in `H`.
- This is outside Gaia XP coverage, so synthetic photometry from Gaia XP alone is not available.
- A direct `H` estimate is good enough at the required accuracy level.

Implementation direction:
- Use published Gaia-to-`H` empirical transforms.
- Carry an uncertainty floor consistent with approximate transform accuracy.

### 4. MAVIS
Use Gaia-based transforms to **`J` and `H`**, then combine them into a synthetic **`J+H`** estimate.

Reason:
- The sensing band is broad and non-standard.
- A single standard `J` or `H` magnitude is not an ideal proxy.
- The desired quantity is the integrated flux over the full `J+H` sensing range.

---

## Decision on the "magic" for MAVIS

### Chosen approach
Do **not** begin by fitting a line or power law through the `J` and `H` points unless needed later.

Instead:

1. Estimate `J` and `H` magnitudes from Gaia.
2. Convert those magnitudes to fluxes.
3. Form a weighted combination of the `J` and `H` fluxes.
4. Convert the combined flux back into a synthetic `J+H` magnitude, if a magnitude-like quantity is needed.

Conceptually:

`F_WFS ≈ a * F_J + b * F_H`

where:
- `F_J` and `F_H` are fluxes corresponding to the estimated `J` and `H` magnitudes
- `a` and `b` are weights determined from the WFS throughput and the effective contribution of the `J`-like and `H`-like regions

### Why this is preferred
- Fluxes add naturally; magnitudes do not.
- A fitted continuous function is underconstrained with only two NIR anchor bands.
- Our goal is photon-rate proxying, not detailed SED reconstruction.
- A weighted flux combination is simpler, more stable, and likely sufficient at the required accuracy level.

---

## When a fitted function might still be useful

A fitted function in wavelength or log-flux space may be worth considering later if:

- the custom sensing band is not well approximated by a weighted `J` + `H` sum
- the throughput curve has significant structure
- the band spans regions not naturally represented by standard `J` and `H`
- a more faithful approximation to the true integrated photon rate is needed

In that case, a possible refinement path is:

1. estimate nearby standard bands from Gaia
2. convert to fluxes
3. fit a smooth surrogate SED
4. integrate that surrogate through the actual WFS response curve

This is a refinement, not the baseline plan.

---

## Practical implementation summary

### Optical / XP-covered case
Use Gaia XP synthetic photometry directly in the actual sensing band.

Applies to:
- HARMONI
- GNAO

### NIR standard-band proxy case
Use empirical Gaia transforms to the nearest standard band.

Applies to:
- MORFEO -> `H`

### Broad custom NIR case
Use empirical Gaia transforms to neighboring standard bands, combine in flux space.

Applies to:
- MAVIS -> estimate `J`, estimate `H`, combine into synthetic `J+H`

---

## Recommended outputs per star

For each star, compute and store:

- Gaia inputs:
  - `G`
  - `BP`
  - `RP`
- Derived colors:
  - `BP - RP`
- Instrument-specific proxy quantities:
  - `m_HARMONI_WFS` or equivalent synthetic XP-based magnitude/flux
  - `m_GNAO_WFS` or equivalent synthetic XP-based magnitude/flux
  - `H_est` for MORFEO
  - `J_est`, `H_est`, and synthetic `J+H` proxy for MAVIS
- Uncertainty estimate or uncertainty floor for each derived quantity

Where possible, store both:
- a magnitude-like proxy
- a flux-like proxy

The flux-like proxy is closer to the physically relevant WFS quantity.

---

## Bottom line

We are not trying to infer the exact stellar spectrum. We are trying to estimate the **band-integrated WFS photon flux** from Gaia data well enough for AO guide-star selection and performance modeling.

Final decisions:

- **HARMONI:** GaiaXPy / Gaia XP synthetic photometry through the actual passband
- **GNAO:** same as HARMONI
- **MORFEO:** Gaia transform to `H`
- **MAVIS:** Gaia transforms to `J` and `H`, then combine in **flux space** to form a synthetic `J+H` proxy

For MAVIS, the current preferred implementation is **weighted flux combination**, not line/power-law fitting.