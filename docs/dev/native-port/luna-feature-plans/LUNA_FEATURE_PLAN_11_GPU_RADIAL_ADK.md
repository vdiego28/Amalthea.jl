# Luna feature plan 11 — CUDA radial RealGrid thresholded ADK

Status: complete 2026-08-02 (depends on plan 10).

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

## Verification — complete 2026-08-02

The radial series uses the existing flat column-major layout
`i = column*n_time_over + t`.  The ADK kernel reads the post-QDHT radial field
and applies the Julia-precomputed constants to each element; its exact
boundary is `abs(E) >= thr`, while non-finite fields produce zero.  The
subsequent fraction, phase/current, and polarization recurrences are the
Plan 10 segmented cumtrapz contract, with each 256-thread block total and
every finalizer offset restricted to the same radial column.  Therefore the
only new numerical operation is the pointwise rate launch; no radial scan
normalization or QDHT convention changes were introduced.

`set_plasma_params_adk` validates all constants and radial RealGrid shape,
allocates the flattened scratch and per-column block totals in locals, and
commits those buffers and parameters only after all allocations succeed.  An
invalid/null ADK replacement consequently leaves the active radial state
unchanged.  The Julia capability predicate admits only `threshold=true`; the
resident path remains explicit-on and radial `:auto` remains disabled.

On the RTX 5060 Ti (CUDA 13.3, driver 610.43.02), strict
`test_native_cuda_radial_adk.jl` passed **43/43**.  The direct CPU-vs-CUDA
stage relative error was `1.4991322388752626e-15`; the fixed five-step solve
error was `1.712696193041123e-16`; the strong-field Julia ADK-on/off effect
was `2.786765208889846e-8` (at least 278× the `1e-10` comparison floor); and
the strong native-vs-Julia error was `3.253050910467547e-16`.  The test also
covers below/above threshold columns, the existing Rust exact-threshold and
non-finite pointwise kernel contract, failed setup rollback, rejected-state
preservation, and adaptive retry.
