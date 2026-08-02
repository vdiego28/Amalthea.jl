# Luna feature plan 20 — CUDA free-space RealGrid thresholded ADK

Status: depends on plan 19.

## Outcome

RealGrid free-space Kerr + thresholded ADK plasma reuses plan 19's segmented
volume scans on CUDA and matches CPU native.

## Implementation

1. Launch the existing pointwise ADK rate over all `(t,y,x)` cells with exact
   threshold/non-finite semantics.
2. Feed plan 19's per-spatial-series fraction/current/polarization scans.
3. Broaden eligibility only for `IonRateADK(threshold=true)` in the supported
   free-space RealGrid shape. Keep `threshold=false` CPU-only.
4. Add multi-series rate-boundary and setup-rollback tests.
5. Keep `:auto` false pending geometry-specific performance evidence.

## Acceptance

Assert a Julia ADK effect at least 100× tolerance, direct stage agreement,
fixed-step `<1e-6`, rejected-state bit parity/retry, adaptive trajectory, and
no cross-series contamination. Run strict CUDA, mode-averaged ADK regressions,
CPU free-space plasma tests, Rust group, and `git diff --check`.

Update docs and append exact rate/trajectory evidence to `PORT_LOG.md`.

## Non-goals

Unthresholded ADK, EnvGrid plasma, z-dependent combinations, or auto dispatch.
