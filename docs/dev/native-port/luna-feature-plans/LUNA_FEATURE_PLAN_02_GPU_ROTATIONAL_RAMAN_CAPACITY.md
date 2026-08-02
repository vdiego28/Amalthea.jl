# Luna feature plan 02 — Deliver CUDA rotational Raman capacity

Status: complete 2026-08-02. Source: 2026-08-02 review finding.

## Outcome

The documented flattened rotational Raman scope must be real: ordinary N2
rotation and rotation+vibration configurations must construct a CUDA backend,
produce nonzero Raman physics, and agree with CPU native on real hardware.

## Existing mismatch

`Raman.flatten_sdo_oscillators` returns 49 oscillators for N2 rotation and 50
for the usual rotation+vibration combination. Julia checks only non-emptiness,
while `CudaNativeSim::set_raman_params` and `MAX_OSCILLATORS` reject/cap above
32. The vibration-only test has one oscillator and cannot expose this.

## Capacity contract and resource model

The resident CUDA ADE kernel assigns one time series to one CUDA thread and
keeps the two first-order state values `(q_i, dq_i)` for every oscillator in
thread-local storage while it scans the time samples. For `M` oscillators this
is `2M` `Float64` values, or `16M` bytes per active thread; a capacity of 64
therefore requires 1024 bytes of state per active thread. The recurrence is
still the same exact-coefficient update as `PrecomputedStepCoeffs` on the CPU;
only the bounded oscillator loop changes. No oscillator may be dropped or
silently capped, because the Raman polarization is the sum of every
`q_i` contribution.

The single source of truth for the CUDA/PTX and Rust-side limit is generated
by `amalthea/build.rs` as both `cuda_raman_limits.h` (included by
`kernels.cu`) and `cuda_raman_limits.rs` (included by `cuda_native.rs` and
`raman.rs`). The chosen `CUDA_RAMAN_MAX_OSCILLATORS = 64` covers the measured
N2 rotation count (49) and rotation+vibration count (50) with 14 free slots,
while remaining a finite resource contract. Julia mirrors this public
contract as `_GPU_RAMAN_MAX_OSCILLATORS = 64` and rejects larger flattened
responses before constructing CUDA state; the boundary test covers both 64
and 65. This duplication is deliberate at the language boundary, while the
Rust/PTX value itself is generated from one build-time literal.

All Raman device allocation byte counts, including the coefficient array,
must use checked multiplication and return setup failure on overflow. The
kernel receives only `1 ≤ M ≤ 64`; its loop therefore has no truncating
fallback. Callers of the older standalone Rust Raman solver use the same
validation and fall back to its scalar CPU implementation when a larger list
is presented.

## Implementation contract

1. Define one named CUDA oscillator capacity shared by Rust validation and PTX
   compilation; choose at least 64 so the 50-line N2 case is covered with
   margin. Avoid two unrelated magic numbers.
2. Increase the ADE kernel state safely. Verify compiler resource usage from
   the real PTX/cubin build; reject the change if spills make the retained
   mode-averaged configuration unusable.
3. Keep a finite upper bound and make Julia eligibility enforce the same
   bound. Configurations above it must remain CPU fallback, never reach a
   setter error after being declared supported, and never be silently capped.
4. Replace `checked_bytes(...).unwrap_or(0)` in the touched Raman allocation
   path with explicit checked failure; an overflow must return an error rather
   than allocate a zero-length buffer.
5. Extend `test/test_native_cuda_raman.jl` with N2 `rotation=true,
   vibration=false` and, if runtime remains bounded, rotation+vibration.
6. Add no-hardware dispatch tests for counts at capacity and capacity+1.
7. Update `PLANS.md` §12, `GPU.md`, the support matrix, README, and backlog to
   state the exact supported maximum rather than saying “rotational” without a
   capacity contract.

## Non-vacuous verification

- Assert the Julia rotational-Raman-on trajectory differs from Raman-off by
  at least 100 times the GPU/CPU trajectory tolerance.
- Compare direct GPU and CPU-native stage derivatives; their Raman-bearing
  scale must be comparable and nonzero.
- Compare a fixed-step full trajectory at `<1e-7` relative error unless the
  measured same-method floor justifies a tighter bound.
- Deliberately reject one adaptive trial, prove the resident field is bit-exact
  before retry, and complete an adaptive trajectory.

## Commands and gate

Build with real PTX and run strict Rust CUDA tests, the focused Raman item, the
full focused CUDA suite, CPU Raman regressions, the Rust group, and
`git diff --check`. Record oscillator counts, kernel resource evidence, test
counts, and achieved errors in `PORT_LOG.md`.

## Non-goals

Do not add `:SiO2`, density-dependent damping, radial/modal/free-space Raman,
or automatic Raman dispatch in this run.

## Completion evidence

The capacity is implemented in `amalthea/build.rs:6-39`, which emits the same
64-oscillator value to `cuda_raman_limits.rs` and `cuda_raman_limits.h`.
`amalthea/src/cuda_native.rs:1774-1870` validates the bound in the resident
`set_raman_params` path (called by the existing
`native_set_raman_params` FFI symbol), checks every Raman allocation size, and
returns setup failure on overflow. `amalthea/src/raman.rs:147-218` applies the
same bound to the legacy standalone GPU solver and falls back to its scalar CPU
implementation above the limit. `amalthea/src/kernels.cu:4-48` uses the
generated capacity for `q[64]`/`dq[64]` and no longer clamps `num_oscillators`.
Julia eligibility mirrors the contract at `src/RK45.jl:1065-1074` and
`src/RK45.jl:1143-1153`, so 65 is rejected before CUDA construction.

The real CUDA 13.3 cubin build of `raman_ade_kernel` reported a 1024-byte
stack frame, 62 registers, and zero spill stores/loads. This is the retained
resource result for the 64-oscillator kernel.

`test/test_native_cuda_raman.jl:142-226` verifies N₂ rotation (49) and
rotation+vibration (50), direct stage agreement below `1e-9`, fixed-step
GPU/CPU relative errors of `4.946766533430483e-16` and
`5.068506594278426e-16`, Raman-on/off effects of
`3.5716896665064484e-3` and `4.108995868691615e-3`, and rejected-step
bit-exactness/retry. `test/test_native_gpu_dispatch.jl:118-163` verifies the
hardware-independent 64/65 eligibility boundary.

Verification completed:

- strict CUDA Julia suite: 209/209;
- focused no-hardware dispatch item: 41/41;
- strict Rust CUDA tests: 79 unit tests plus 3 build-policy tests;
- full Julia Rust group: 42,651 passed, 1 expected broken CUDA-driver item;
- real PTX/cubin compilation and `git diff --check`: passed.
