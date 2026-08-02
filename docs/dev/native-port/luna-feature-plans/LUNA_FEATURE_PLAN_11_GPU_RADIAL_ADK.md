# Luna feature plan 11 — CUDA radial RealGrid thresholded ADK

Status: depends on plan 10.

## Outcome

RealGrid radial Kerr + one thresholded ADK `PlasmaCumtrapz` uses plan 10's
segmented plasma scans on CUDA and matches CPU native.

## Implementation

1. Reuse the existing pointwise `adk_ionization_kernel` constants and exact
   threshold/non-finite behavior; add the radial series launch shape only.
2. Feed plan 10's segmented fraction/current/polarization scans unchanged.
3. Broaden eligibility only for `IonRateADK(threshold=true)` in the already
   supported radial RealGrid plasma shape. Keep `threshold=false` CPU-only.
4. Add direct rate-boundary tests for zero, below threshold, threshold, above
   threshold, and non-finite fields across multiple radial columns.
5. Prove failed/invalid ADK setup does not destroy an existing radial Kerr or
   PPT setup.
6. Keep `:auto` false pending a radial ADK benchmark.

## Acceptance

The Julia ADK control must exceed comparison tolerance by at least 100×.
Require direct stage agreement, fixed-step `<1e-6`, rejected-state bit parity,
retry/adaptive agreement, and no cross-column contamination. Run strict CUDA,
focused radial ADK, existing mode-averaged ADK and CPU radial plasma tests,
Rust group, and `git diff --check`.

Update docs and append exact rate/trajectory results to `PORT_LOG.md`.

## Non-goals

Unthresholded ADK, EnvGrid plasma, radial automatic dispatch, or new ionization
physics.
