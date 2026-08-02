# Luna feature plan 13 — CUDA radial EnvGrid SDO Raman

Status: depends on plans 09 and 12.

## Outcome

EnvGrid radial Kerr + `RamanPolarEnv` runs resident on CUDA with one ADE series
per radial column.

## Implementation

1. Reuse plan 12's segmented ADE storage/launch and plan 09's complex radial
   buffers.
2. Form real intensity `0.5*abs2(E)` for every `(t,r)` cell; no Hilbert branch.
3. Accumulate complex `pto += E*(density*P)` before window/QDHT/forward c2c.
4. Broaden eligibility only for matching EnvGrid radial SDO Raman. Keep
   intermediate broadening, plasma, noise, mixtures, and z-dependence out.
5. Keep `:auto` false.

## Acceptance

Use distinct complex signals in at least two radial columns and assert series
isolation. Prove a non-vacuous Julia Raman effect, compare direct stages, run a
fixed-step trajectory `<1e-6`, and exercise rejection/retry/adaptive behavior.
Run strict CUDA, focused EnvGrid radial Raman, CPU radial EnvGrid Raman,
mode-averaged Raman regressions, Rust group, and `git diff --check`.

Update support docs and append `PORT_LOG.md` with achieved errors.

## Non-goals

`:SiO2`, EnvGrid plasma, plasma composition, auto dispatch, or new Raman math.
