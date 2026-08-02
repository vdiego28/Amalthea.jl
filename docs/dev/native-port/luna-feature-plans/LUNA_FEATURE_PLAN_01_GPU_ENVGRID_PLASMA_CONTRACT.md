# Luna feature plan 01 — Make the EnvGrid plasma contract safe

Status: complete (2026-08-02). Source: 2026-08-02 review finding, correctness.

## Outcome

An EnvGrid transform containing `PlasmaCumtrapz` must never select
`CudaNativeSim` until an EnvGrid plasma implementation exists. Low-level users
must receive the correct CPU-native/Julia result rather than a CUDA trajectory
that silently omits plasma.

## Evidence and cause

- `src/RK45.jl::_gpu_kernel_supports` accepts `EnvGrid` and plasma
  independently and currently returns true for the combination.
- `CudaNativeSim::compute_rhs_mode_avg_env` applies Kerr and Raman only; it
  contains no `has_plasma` branch.
- High-level `Interface.jl` rejects envelope plasma, but a low-level
  `TransModeAvg` can construct it, so the eligibility predicate is still a
  correctness boundary.

This is a support-predicate over-acceptance error, not a numerical-tolerance
problem.

## Implementation

1. Add an explicit grid/response compatibility guard to
   `_gpu_kernel_supports`: any nonempty plasma response requires `RealGrid`.
2. Keep RealGrid PPT and thresholded ADK eligibility unchanged.
3. Add a pure-Julia, no-CUDA regression to `test/test_native_gpu_dispatch.jl`
   that constructs an EnvGrid `TransModeAvg` with thresholded ADK plasma and
   proves both `_gpu_kernel_supports` and `_gpu_native_eligible(..., :on)` are
   false.
4. Construct a `RustNativeStepper` for that low-level transform and prove it
   uses CPU fallback. If plan 03 is not yet landed, assert the support decision
   and compare one fixed step against an explicitly CPU-native stepper; do not
   use `isa RustNativeStepper` as backend evidence.
5. Move any existing unsupported-response dispatch assertions out of the live
   hardware branch so they execute on CPU-only CI.
6. Correct every statement that implies EnvGrid plasma is supported by CUDA,
   especially `amalthea/README.md`, `GPU.md`, and
   `NATIVE_SUPPORT_MATRIX.md`. Do not alter the CPU-native support table.

## Tests and acceptance

- The new low-level construction must reproduce the pre-fix
  `gpu_kernel_supports=true` result before the guard is added.
- Post-fix: support and forced eligibility are false; the CPU fallback result
  is finite and agrees with explicit CPU native at the normal single-step
  reassociation tier.
- Existing RealGrid Kerr/PPT/ADK dispatch tests remain unchanged and green.
- Run the focused dispatch item, affected CPU-native mode-averaged tests,
  `LUNA_TEST_GROUP=rust julia --project test/runtests.jl`, and
  `git diff --check`.
- No CUDA hardware is required for the new regression; if CUDA is available,
  also run the focused strict CUDA suite to prove no RealGrid regression.

## Non-goals

Do not implement envelope plasma here. Do not broaden high-level APIs, change
plasma formulas, or alter `:auto` thresholds.

## Handoff

Append `PORT_LOG.md` with the exact low-level reproducer, guard location, test
counts, and the explicit statement that EnvGrid plasma remains CPU-only.

## Completion evidence

- `_gpu_kernel_supports` now rejects every EnvGrid transform containing
  `PlasmaCumtrapz`; RealGrid PPT and thresholded ADK remain eligible.
- The no-hardware dispatch regression constructs the previously accepted
  low-level EnvGrid+ADK shape. Forced GPU dispatch falls back to CPU native and
  agrees with an explicitly CPU-native fixed step exactly (`rel = 0.0`, required
  `< 1e-13`).
- The focused dispatch item passed 35/35. The strict real-hardware CUDA suite
  passed 189/189, Rust passed 79/79 unit plus 3/3 build-policy tests, and the
  full Julia Rust group passed 42,645 assertions with one expected broken CUDA
  item in the sandbox (the strict out-of-sandbox CUDA run was green).
- General unsupported-response test placement and backend observability remain
  the separate work item in plan 03; this plan's EnvGrid-plasma assertion is
  hardware-independent as required.
