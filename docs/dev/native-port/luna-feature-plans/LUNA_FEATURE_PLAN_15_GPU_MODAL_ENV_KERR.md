# Luna feature plan 15 — CUDA modal EnvGrid Kerr

Status: depends on plan 14.

## Outcome

Eligible EnvGrid modal Kerr configurations run through plan 14's CUDA modal
point-evaluation pipeline for `full=false|true` and `npol=1|2`.

## Implementation

1. Add transactional c2c modal plans and complex time/spectral scratch while
   preserving plan 14's cubature batch protocol.
2. Mirror the CPU modal EnvGrid spectrum expansion/crop and explicit c2c
   scaling; use `0.75*|E|^2E` scalar/vector envelope Kerr as appropriate.
3. Extend device field synthesis and projection to complex envelope data and
   retain mode/polarization ordering exactly.
4. Broaden eligibility only for supported EnvGrid modal Kerr. Keep Raman,
   plasma, noise, mixtures, and `StepIndexMode` excluded.
5. Keep `:auto` false.

## Acceptance

Test supplied-node point evaluation, one/two modes, both cubature modes, and a
non-vacuous two-polarization case with asymmetric complex data. Require direct
stage agreement, nonzero modal transfer, fixed-step `<1e-6`, reject/retry, and
adaptive agreement. Run strict CUDA, CPU modal EnvGrid tests, plan-14
regressions, Rust group, and `git diff --check`.

Update support docs and append exact c2c scaling/tolerance evidence to
`PORT_LOG.md`.

## Non-goals

Raman, plasma, shot noise, mixtures, z-dependence expansion, or auto dispatch.
