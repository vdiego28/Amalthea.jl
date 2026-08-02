# Luna feature plan 12 — CUDA radial RealGrid SDO Raman

Status: depends on plans 02 and 08.

## Outcome

RealGrid radial Kerr + one supported `RamanPolarField` runs resident on CUDA
for both `thg=true` and `thg=false`, with one independent ADE series per radial
column.

## Implementation

1. Generalize the CUDA ADE launch from one mode-averaged series to `n_r`
   independent contiguous time series. Use plan 02's shared capacity contract.
2. Allocate intensity and Raman polarization for `n_time_over*n_r`.
3. For `thg=true`, form `E^2` per cell. For `thg=false`, use batched c2c
   Hilbert transforms per radial column and the exact parity mask/scaling.
4. Accumulate `pto += density*eto*P` before the radial time window/QDHT.
5. Ensure oscillator state and Hilbert scratch cannot leak between columns or
   RK stages; no host arrays transfer per RHS.
6. Broaden eligibility only for RealGrid radial matching SDO Raman, scalar
   density, constant linop/norm, no plasma/noise/mixture. Keep `:SiO2` out.
7. Keep `:auto` false.

## Acceptance

Test at least two radial columns with distinct signals, vibration-only and N2
rotational Raman, both THG values, and a column-isolation sentinel. Require a
non-vacuous Julia Raman control, direct stage comparison, fixed-step `<1e-6`,
reject/retry, and adaptive agreement. Run strict CUDA, CPU radial Raman,
mode-averaged Raman, Rust group, and diff check.

Update docs and append oscillator/series counts and errors to `PORT_LOG.md`.

## Non-goals

EnvGrid, `:SiO2`, plasma composition, mixtures, z-dependence, or auto dispatch.
