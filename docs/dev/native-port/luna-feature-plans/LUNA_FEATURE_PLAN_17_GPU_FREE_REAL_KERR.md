# Luna feature plan 17 — CUDA free-space RealGrid Kerr foundation

Status: ready after plan 03; standing CUDA CI is strongly preferred.

## Outcome

`TransFree` + RealGrid + scalar Kerr runs entirely resident on CUDA using a
joint three-dimensional transform and agrees with CPU native.

## Geometry contract

Mirror `CpuNativeSim::rhs_free`: spectrum expansion per spatial column, one
joint `(t,y,x)` c2r transform, pointwise Kerr, time window, one joint r2c
transform, spectral crop, and the transferred complex normalization. Julia's
column-major `(n_t,n_y,n_x)` maps to cuFFT dimensions in reversed order; the
halved dimension must be time.

## Implementation

1. Implement transactional CUDA `set_free_params` for dimensions, plans,
   buffers, window, Kerr coefficient, and normalization.
2. Add joint 3-D cuFFT plans with explicit normalization
   `1/(n_t*n_y*n_x)` and overflow checks.
3. Add pad/crop, Kerr/window, and final normalization kernels respecting
   column-major layout and non-square `n_y != n_x` grids.
4. Add a literal CUDA-vs-Julia transform reference using nonsymmetric data;
   a CUDA round trip alone cannot catch swapped axes.
5. Broaden eligibility only for RealGrid free-space scalar Kerr with constant
   norm/linop, no plasma/Raman/noise/mixture.
6. Keep `:auto` false.

## Acceptance

Test non-square transverse dimensions, direct stage/non-vacuous Kerr control,
fixed-step `<1e-6`, reject/retry, adaptive trajectory, invalid dimensions, and
transactional second setup. Run strict CUDA, existing CPU free-space/3-D FFT
tests, focused Julia item, Rust group, and `git diff --check`.

Update docs and append exact dimension/scaling evidence to `PORT_LOG.md`.

## Non-goals

EnvGrid, plasma, Raman, shot noise, z-dependent norm, or auto dispatch.
