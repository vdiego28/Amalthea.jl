# Luna feature plan 14 — CUDA modal RealGrid Kerr

Status: ready after plan 03; standing CUDA CI is strongly preferred.

## Outcome

A bounded modal Kerr surface—RealGrid, constant-radius eligible Marcatili/
Zeisberger/Vincetti mode collections, constant linop, `full=false|true`, and
`npol=1|2`—runs through a resident CUDA backend while retaining libcubature's
adaptive node placement.

## Architecture decision

Do not invent a different quadrature rule. Keep the same host
`libcubature` binary and adaptive batches so node placement and stopping
criteria remain comparable. For each callback batch, copy only node
coordinates to CUDA, evaluate field synthesis/FFT/Kerr/projection on device,
and copy the small `fval` batch back; the resident spectral state and large
scratch arrays must not return to the host. Document this bounded control-data
exception to the general traffic budget and benchmark it.

## Implementation

1. Implement transactional CUDA `set_modal_params`, including mode metadata,
   polarization selectors, normalization factors, nonlinear prefactors,
   dimensions, and c2r plans/scratch.
2. Preserve Julia column-major mode/time/polarization layout. Add literal
   layout tests using nonsymmetric mode and polarization data.
3. Implement device kernels for mode-field synthesis at `(r,theta)`, inverse
   FFT, scalar/vector Kerr, window/normalization, forward FFT, and modal
   projection including the polar Jacobian.
4. Use stable Bessel formulas matching the CPU native path; do not recompute
   normalization or dispersion on CUDA.
5. Support both cubature callbacks (`full=false` radial and `full=true`
   two-dimensional) and both polarization counts. Tests must energize both
   polarizations so vector cross-coupling is non-vacuous.
6. Broaden eligibility only for the implemented constant-radius,
   constant-linop modal Kerr surface. Keep tapered/z-dependent modes,
   Raman/plasma/noise/mixtures, and `StepIndexMode` excluded.
7. Keep `:auto` false.

## Acceptance

- Device point-evaluator vs CPU native at fixed supplied nodes before testing
  adaptive cubature.
- One-mode and two-mode cases, `full=false` and `full=true`, `npol=1` and a
  genuinely two-polarization `npol=2` case.
- Nonzero HE11→HE12 transfer control, direct stage comparison at the modal
  method tier, fixed-step `<1e-6`, reject/retry, and adaptive trajectory.
- Transactional setup/lifecycle failures on real CUDA.
- Record host/device callback bytes and wall time to ensure the design is not
  dominated by transfers; no auto threshold follows from this benchmark.

Run strict CUDA, CPU modal/cubature/threading tests, focused Julia item, Rust
group, and `git diff --check`. Update docs and append `PORT_LOG.md`.

## Non-goals

Tapered/z-dependent modes, plasma, Raman, shot noise, mixtures,
`StepIndexMode`, or replacing cubature.
