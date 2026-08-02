# Luna feature plan 21 — CUDA free-space RealGrid SDO Raman

Status: depends on plans 02 and 17.

## Outcome

RealGrid free-space Kerr + supported SDO `RamanPolarField` runs resident on
CUDA, with one independent ADE series per transverse point.

## Implementation

1. Generalize the ADE launch to `n_series=n_y*n_x` contiguous time series and
   use plan 02's oscillator-capacity contract.
2. Allocate intensity/polarization/Hilbert scratch for the full oversampled
   volume using checked sizes.
3. Support `thg=true` through `E^2` and `thg=false` through batched Hilbert
   transforms along time only; spatial axes must not enter the Hilbert FFT.
4. Accumulate `pto += density*E*P` before the free-space time window and
   forward 3-D FFT.
5. Broaden eligibility only for RealGrid free-space SDO Raman, scalar density,
   constant norm, no plasma/noise/mixture. Keep `:SiO2` and EnvGrid excluded.
6. Keep `:auto` false.

## Acceptance

Use non-square transverse dimensions and distinct per-point signals to detect
axis or state leakage. Cover both THG values and N2 vibration plus rotation.
Require Julia Raman non-vacuity, direct stage agreement, fixed-step `<1e-6`,
reject/retry, and adaptive trajectory. Run strict CUDA, CPU free-space Raman,
mode-averaged Raman, 3-D FFT regressions, Rust group, and diff check.

Update support docs and append oscillator/series/layout evidence to
`PORT_LOG.md`.

## Non-goals

EnvGrid Raman, intermediate broadening, plasma composition, z-dependent norm,
or auto dispatch.
