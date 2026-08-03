# Luna feature plan 09 — CUDA radial EnvGrid Kerr

Status: complete (2026-08-02).

## Outcome

`TransRadial` + EnvGrid + scalar Kerr runs on the resident CUDA radial backend,
using complex envelope FFT/QDHT conventions and matching CPU native.

## Implementation

1. Extend plan 08's radial state transactionally with c2c oversampled batched
   plans and complex QDHT buffers; preserve `(n_time, n_r)` column-major layout.
2. Mirror `CpuNativeSim::rhs_radial_env`, including low/high spectrum
   expansion/cropping, c2c inverse/forward scaling, `0.75*|E|^2E` envelope
   Kerr, time window, QDHT directions, and transferred complex normalization.
3. Reuse the plan-08 device QDHT primitive for complex data with explicit real
   and imaginary handling; add an independent nonsymmetric reference test.
4. Broaden Julia eligibility only for EnvGrid radial Kerr with the same
   constant-linop/scalar-density/no-noise restrictions.
5. Keep EnvGrid radial plasma ineligible and `:auto` false.

## Tests and acceptance

Use an asymmetric complex field so low/high crop and sign/order errors cannot
hide. Require a non-vacuous Julia Kerr control, direct stage agreement, a
fixed-step trajectory `<1e-6`, rejection preservation/retry, and adaptive
agreement. Test transactional c2c plan replacement and invalid shapes on real
CUDA. Run strict CUDA, focused radial EnvGrid, existing CPU radial EnvGrid and
QDHT tests, Rust group, and `git diff --check`.

Update support docs and append `PORT_LOG.md` with scaling and layout evidence.

Verification: the strict CUDA item passed **24/24** on the RTX 5060 Ti. The
asymmetric direct stage relative error was `4.262893614543232e-16`, and the
fixed full-solve relative error was `2.871085295458848e-15`; invalid setup
rollback and rejected adaptive retry also passed. CPU EnvGrid radial coverage
passed 3/3, and the Rust group completed with 42,717 passing assertions and
one pre-existing CUDA-driver broken item in the non-hardware sandbox.

## Non-goals

Plasma, Raman, shot noise, mixtures, z-dependence, and automatic dispatch.
