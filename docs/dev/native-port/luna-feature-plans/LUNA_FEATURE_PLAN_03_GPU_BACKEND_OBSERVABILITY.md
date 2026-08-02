# Luna feature plan 03 — Make CPU-vs-CUDA selection observable and testable

Status: complete 2026-08-02. Source: review test-topology and
fallback-observability findings.

## Outcome

Tests and diagnostics can determine whether a `RustNativeStepper` owns a CPU or
CUDA resident backend without inferring it from type, environment, timing, or
successful construction.

## Backend-kind contract

`RustNativeSimHandle` stores a Julia-only `backend::Symbol` with exactly two
values: `:cpu` for `init_native_sim` and `:cuda` for
`init_cuda_native_sim`. The value is assigned from the constructor request,
not inferred from timing, environment after construction, or an FFI query.
The z-dependent constructors always call `init_native_sim` and therefore must
record `:cpu`. A null pointer is still a hard construction failure; it must
not produce a handle whose observable kind is `:cpu` merely because CUDA was
requested and failed. The existing `native_set_*` FFI surface and opaque Rust
layout remain unchanged.

`RK45._native_backend(s::RustNativeStepper)` returns that stored symbol for
tests and diagnostics. It is not a public dispatch control and does not alter
eligibility. Hardware-independent tests may construct CPU-selected steppers
for supported or unsupported configurations, while a supported `:cuda`
construction is attempted only inside the successful-device branch. CUDA
numerical tests assert `:cuda` before reading stages; fallback tests assert
`:cpu` directly.

The pure dispatch/fallback testsets execute before the CUDA initialization gate.
They therefore count on CPU-only runs: `gpu_dispatch=:off` and small `:auto`
configurations construct `:cpu`, `:on` proves only the pure eligibility
decision for a supported config, and an unsupported config constructs `:cpu`
even when `:on` is requested. Only stage, trajectory, and adaptive retry
comparisons require a successful CUDA device.

## Implementation

1. Store the selected backend kind on `RustNativeSimHandle` at construction.
   A Julia-side `use_gpu::Bool` (or a small internal enum) is sufficient; do
   not add an FFI round-trip merely to recover a choice Julia just made.
2. Add an internal accessor such as `RK45._native_backend(s)` returning
   `:cpu` or `:cuda`. Keep it test/diagnostic-facing unless there is a clear
   public API reason to expose it.
3. Ensure z-dependent constructors report `:cpu` and a failed CUDA
   construction cannot masquerade as a CPU handle.
4. Replace `s isa RustNativeStepper` fallback assertions with explicit backend
   assertions.
5. Restructure `test/test_native_cuda_raman.jl`: pure eligibility, unsupported
   response, capacity, and CPU-fallback tests run before the CUDA hardware
   gate; only kernel comparisons remain inside the successful-device branch.
6. Add focused constructor tests for `gpu_dispatch=off/on/auto`, including a
   supported small config and an unsupported config. On no-GPU hosts, do not
   force construction of a supported CUDA backend; the pure decision and CPU
   fallback paths must still execute.
7. Document the accessor in testing documentation and use it in future plan
   tests.

## Acceptance

- The old unsupported `:SiO2` test proves `:cpu`, not merely
  `RustNativeStepper`.
- CPU-only CI executes and counts all pure dispatch/fallback assertions.
- Existing CUDA tests prove `:cuda` before numerical comparisons on hardware.
- No public FFI symbol or backend layout changes are required.
- Run focused dispatch/Raman items, CPU-native phase tests affected by handle
  construction, the Rust group, and `git diff --check`.

## Non-goals

Do not change dispatch eligibility, thresholds, or physics kernels.

## Handoff

Append `PORT_LOG.md` with the accessor contract and before/after assertion
counts on a no-CUDA run and, if available, strict hardware.

## Completion evidence

`src/RK45.jl:927-973` now stores `backend::Symbol` on
`RustNativeSimHandle`: the array constructor records `:cpu` or `:cuda` from
the selected `init_native_sim`/`init_cuda_native_sim` call, while the
z-dependent constructor records `:cpu`. `src/RK45.jl:1018-1030` adds the
internal `_native_backend` accessor, and `src/RK45.jl:1221-1233` makes a null
native pointer a hard construction failure with the requested backend named in
the error. No FFI symbol or Rust opaque layout changed.

`test/test_native_cuda_raman.jl:72-145` moves pure Raman eligibility,
49/50-oscillator capacity, EnvGrid support, and `:SiO2` CPU-fallback checks
before the CUDA gate. `test/test_native_gpu_dispatch.jl:147-189` covers
constructor observability for `gpu_dispatch=:off`, below-threshold `:auto`,
pure `:on`, and an unsupported forced-on configuration. Existing CUDA tests
assert `:cuda` before stage or trajectory comparisons, and z-dependent tests
assert `:cpu` explicitly.

The no-CUDA dispatch baseline was 41 passing assertions; the Raman item had no
pure assertions before its hardware gate. After the topology change, dispatch
passes 49/49 and the Raman item executes 17 pure assertions before its one
expected no-driver broken CUDA branch. The strict CUDA suite passes 248/248.
The affected z-dependent tests pass 16/16, 4/4, and 10/10; backend-report
tests pass 15/15. The full Rust group passes 42,682 assertions with one
expected CUDA-driver broken item in the sandbox. `cargo build --release` and
`git diff --check` pass.
