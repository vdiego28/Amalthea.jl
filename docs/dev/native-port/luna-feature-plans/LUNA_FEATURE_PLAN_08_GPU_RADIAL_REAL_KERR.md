# Luna feature plan 08 — CUDA radial RealGrid Kerr foundation

Status: ready after plan 03; standing CUDA CI is strongly preferred.

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

## Non-goals

EnvGrid, plasma, Raman, noise, z-dependent radial normalization, mixtures, and
automatic dispatch.
