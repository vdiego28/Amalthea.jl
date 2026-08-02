# Luna feature plan 19 — CUDA free-space RealGrid PPT plasma

Status: depends on plan 17.

## Outcome

RealGrid free-space Kerr + PPT plasma runs resident on CUDA, with an
independent deterministic plasma scan for every `(y,x)` time series.

## Implementation

1. Generalize the completed prefix-scan pipeline to
   `n_series=n_y*n_x`, preserving contiguous time-series boundaries.
2. Allocate rate/fraction/current/polarization/scan scratch for the full
   oversampled volume with checked arithmetic.
3. Reuse the exact PPT spline upload and current formulas; accumulate plasma
   before the free-space time window and forward 3-D FFT.
4. Make free setup precede plasma setup and make failed plasma replacement
   transactional.
5. Broaden eligibility only for RealGrid free-space, one Kerr + one PPT,
   scalar density, constant norm, no Raman/noise/mixture.
6. Keep `:auto` false.

## Acceptance

Primitive scan tests must cover multiple spatial series, multiple blocks, a
partial last block, and sentinels proving no cross-series carry. Require a
non-vacuous Julia plasma effect, direct stage comparison, fixed-step `<1e-6`,
rejected-state parity/retry, and adaptive agreement. Run strict CUDA, CPU
free-space plasma, existing scan/PPT tests, Rust group, and diff check.

Update docs and append scan shape/tolerances to `PORT_LOG.md`.

## Non-goals

ADK, EnvGrid plasma, Raman, z-dependent norm combinations, or auto dispatch.
