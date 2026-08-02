# Luna feature plan 18 — CUDA free-space EnvGrid Kerr

Status: depends on plan 17.

## Outcome

`TransFree` + EnvGrid + scalar Kerr runs resident on CUDA with joint complex
three-dimensional transforms and matches CPU native.

## Implementation

1. Add transactional joint c2c 3-D plans and complex buffers to plan 17's
   free-space setup.
2. Mirror `CpuNativeSim::rhs_free_env`, including low/high spectral expansion,
   explicit c2c scaling, envelope Kerr, time window, crop, and transferred
   normalization.
3. Preserve reversed cuFFT dimensions and test a non-square transverse grid
   with asymmetric complex values.
4. Broaden eligibility only for EnvGrid free-space Kerr under existing scalar
   density/constant norm/no-noise restrictions.
5. Keep EnvGrid plasma/Raman ineligible and `:auto` false.

## Acceptance

Literal transform reference, direct stage/non-vacuity, fixed-step `<1e-6`,
reject/retry, adaptive trajectory, and transactional c2c setup on real CUDA.
Run strict CUDA, CPU free-space EnvGrid tests, plan-17 regressions, Rust group,
and `git diff --check`.

Update docs and append scaling/layout results to `PORT_LOG.md`.

## Non-goals

Plasma, Raman, noise, z-dependence, mixtures, or auto dispatch.
