# Luna feature plan 08 — CUDA radial RealGrid Kerr foundation

Status: complete 2026-08-02; standing CUDA CI remains strongly preferred.

## Outcome

`TransRadial` + RealGrid + one scalar Kerr response runs entirely through a
resident `CudaNativeSim`, matching the CPU-native radial oracle without
per-stage host array transfers.

## Geometry contract

Mirror `CpuNativeSim::rhs_radial`: oversampled c2r transforms per radial
column, inverse QDHT using Julia's transferred `T`, pointwise Kerr, time
window, forward QDHT, r2c transforms, and the transferred complex
normalization array. Preserve Julia column-major `(n_time, n_r)` layout.

## Implementation

1. Extend CUDA state with radial dimensions, device QDHT matrix, real
   time-domain buffers, normalization, and transactional plans.
2. Implement `set_radial_params` validation/allocation. Reuse its existing FFI
   signature and return errors without partial state replacement.
3. Use batched cuFFT plans or a measured equivalent. Implement QDHT device
   matrix multiplication with cuBLAS if available through a bounded loader, or
   a deterministic CUDA kernel; it must use Julia's `T`, never recompute it.
4. Implement spectrum pad/crop, QDHT orientation, Kerr/window, and final
   normalization kernels with explicit shape/overflow guards.
5. Route radial stages through a dedicated CUDA RHS and seed stage zero in
   `set_field` exactly as mode-averaged does.
6. Broaden Julia GPU eligibility only for RealGrid radial, scalar density,
   constant linop/norm, one scalar Kerr, no noise/plasma/Raman/mixture.
7. Keep `:auto` false for radial until a separate benchmark establishes a
   threshold.

## Tests and acceptance

- Primitive test: CUDA QDHT direction/normalization against CPU resident using
  a nonsymmetric matrix/field so transposition errors cannot round-trip away.
- Direct stage derivative agreement at the radial reassociation floor; assert
  nonzero Kerr scale and a Julia Kerr-on/off control at least 100× tolerance.
- Fixed-step full trajectory `<1e-6`, with tighter measured value recorded.
- Deliberate reject/retry and adaptive trajectory against CPU native.
- Invalid dimensions/nulls and transactional second-setup rollback on real
  CUDA.
- Run strict Rust CUDA, new focused Julia item, existing CPU radial/QDHT tests,
  the Rust group, and `git diff --check`.

## Documentation and handoff

Update `GPU.md`, support matrix, backlog, README, and plan status. Append
`PORT_LOG.md` with layouts, kernels, FFI symbol reuse, errors, and commands.

## Completion record (2026-08-02)

The narrow slice is implemented. `CudaNativeSim::stage_radial_setup` and
`commit_radial_setup` in `amalthea/src/cuda_native.rs` stage checked,
device-resident buffers, the Julia-transferred QDHT matrix, normalization,
time window, and separate D2Z/Z2D cuFFT plans before one transactional commit.
The existing `native_set_radial_params` FFI symbol is reused. The CUDA kernels
`expand_radial_spectrum_kernel`, `qdht_radial_real_kernel`,
`apply_radial_time_window_kernel`, and `finalize_radial_spectrum_kernel` keep
the complete RHS on device; cuFFT is launched independently for each radial
column. Julia eligibility in `src/RK45.jl` admits only RealGrid, scalar density,
constant linop/norm, one plain Kerr, and no noise/plasma/Raman/mixture. Explicit
`AMALTHEA_NATIVE_GPU=on` is required; radial `:auto` remains false.

One important convention is recorded here because it is easy to regress: the
temporal zero-padding scale `(n_spec_over-1)/(n_spec-1)` is separate from
Julia's QDHT `scaleRK`. Reusing `scaleRK` for temporal padding passed symmetric
physical tests but failed the nonsymmetric primitive and suppressed the GPU
stage; the corrected distinction is in `compute_rhs_radial`.

Acceptance is complete: the focused CUDA item passed 25/25 on the RTX 5060 Ti,
including non-vacuity, nonsymmetric QDHT direction/normalization, invalid/null
setup rejection with rollback, fixed solve, and adaptive reject/retry. The
measured fixed-solve CPU-vs-CUDA relative error was `4.772174254620178e-16`.
The CPU radial oracle passed 3/3 with `1.142189692971526e-17` single-step and
`1.2869428033620095e-16` full-solve relative errors. Strict Rust CUDA tests
passed 80/80 plus 3/3 build-policy tests; dispatch coverage passed 63/63.
The writable-depot Rust-group rerun passed all applicable assertions; one
expected CUDA-driver-broken item remained, and the new timing-manifest entry
is included in this worktree.

## Non-goals

EnvGrid, plasma, Raman, noise, z-dependent radial normalization, mixtures, and
automatic dispatch.
