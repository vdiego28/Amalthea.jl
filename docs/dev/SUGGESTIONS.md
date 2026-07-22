# Suggestions — beyond the current backlog

Ideas out of scope of the fix/parity plan in `BACKLOG.md`. Roughly ordered
by expected payoff per unit effort for the typical workload (mode-averaged
HCF, 8k–16k time samples, ~10³–10⁴ RK stages per solve).

## Performance

### 1. GPU-resident propagation (the real GPU win)
The current CUDA/Vulkan paths accelerate individual kernels but the field
round-trips host↔device per call, so they rarely beat AVX2 at typical grid
sizes. The payoff structure changes completely if the *whole* `NativeSim`
state lives on the GPU for the duration of a `solve`: field, RK stage
buffers, cuFFT plans, ionisation LUT, Raman ADE state — with only per-step
scalars (z, dt, err) crossing PCIe. The Phase 0–8 architecture is already
shaped for this (one opaque handle owning all state); a `CudaNativeSim`
implementing the same FFI surface is the natural next tier. Free-space
(3-D FFT dominated) and multi-mode modal (many independent per-node
evaluations) benefit first; mode-averaged 1-D cases may stay CPU-bound —
keep the runtime dispatcher deciding per problem size.

### 2. Threading the native RHS
The resident RHS is currently single-threaded per stage. Three easy seams,
all embarrassingly parallel: (a) rayon over radial nodes in `rhs_radial`
(each r is an independent 1-D FFT + Kerr); (b) rayon over cubature
evaluation points in the modal path (the *reason* the Julia
`Threads.@threads` attempt raced was shared scratch — give each rayon task
its own scratch slab and the race disappears by construction); (c) split
the ω-loop of the exp-linop application + windowing broadcasts. FFTW plans
can also be created with `fftw_plan_with_nthreads` for the 3-D free-space
case, where FFT time dominates everything else.

### 3. Instruction-level parallelism / SIMD layout
- The Kerr kernel (`E → E³` or `|E|²E`) and the window/norm broadcasts are
  pure streaming FMA loops — ensure they compile to packed AVX2/AVX-512
  via explicit `f64x4`/`f64x8` chunks (the crate already has a dispatch
  layer; today only Raman uses it).
- Complex arrays are stored interleaved (re,im,re,im). For the exp-linop
  multiply and error-norm reductions, a split (SoA) layout doubles
  effective SIMD width and removes shuffle traffic. Worth benchmarking on
  the exp(linop)·y hot loop specifically before committing — the FFT
  requires interleaved, so SoA needs a transpose or FFTW's split-array API
  (`fftw_plan_guru_split_dft`, same library, already dlopened).
- The RK45 stage-accumulation loops (`y + h·Σaᵢⱼkⱼ`) can be fused into a
  single pass per stage (read each kⱼ once, FMA into an accumulator)
  instead of one pass per coefficient — memory-bound win at 16k points.
- Precompute `exp(L·dt_stage)` once per accepted dt rather than per stage
  where the stage offsets allow (c-coefficients share values: c2=1/5,
  c3=3/10, … — cache keyed on dt since dt changes only on step
  accept/reject).

### 4. Persistent FFTW wisdom for the native plans
Native plans are created with `FFTW_ESTIMATE` (fast setup, slower
transforms). Julia's side already persists wisdom via
`Utils.loadFFTwisdom`; export/import that same wisdom through
`fftw_export_wisdom_to_filename` in the Rust binding so native `solve`s get
`FFTW_MEASURE`-quality plans with zero per-run planning cost.

### 5. BLAS-3 QDHT
The QDHT is a dense (n_r×n_r)·(n_r×n_time) product currently done as
Rayon-parallel dot products. That is a GEMM; binding a real BLAS
(OpenBLAS is already in-process via Julia — `libblastrampoline` symbols can
be dlopened like FFTW) turns it into an L3-blocked kernel with 2–4× the
throughput at n_r ≥ 128. Alternatively implement a register-blocked
micro-kernel in Rust; but reusing the in-process BLAS is nearly free.

## Architecture

### 6. Replace the env-var toggle zoo with a config struct
Seven `AMALTHEA_USE_RUST_*` env vars now interact (REVIEW §3.2 is a direct
casualty: two toggles with a hidden dependency). Suggest a single
`Luna.config` (or kwarg `backend=:native/:julia/:auto`) resolved once at
setup, with env vars kept only as overrides for CI. This also makes
eligibility *inspectable*: `Luna.backend_report(output)` telling the user
which path actually ran — today a one-time `@warn` is the only signal, and
it's easy to benchmark the wrong backend without noticing.

### 7. Unify the FFI error model
FFI calls return ad-hoc `Cint` codes mapped to `error(...)`/
`NativeIneligible` case-by-case on the Julia side. A single error enum in
`ffi.rs` (negative = bug/crash-worthy, positive = ineligibility/fallback)
plus one Julia-side `check_ffi(rc)` would make the "fall back vs crash
loudly" decision systematic instead of per-call-site.

### 8. Rust-side setup, not just the hot loop
`RustNativeStepper`'s Julia-side setup pokes into implementation details
(`getfield(norm_func, :pre)`, `occursin("γ3", string(fld))` to find the
Kerr coefficient). These reflective probes are brittle against upstream
refactors — a `NonlinearRHS`-side explicit accessor API (`kerr_γ3(resp)`,
`norm_pre(norm!)`) would break loudly at method-resolution time instead of
silently extracting the wrong field.

### 9. In-process HDF5 streaming for scans
Scans currently serialize through the Julia HDF5 path per scan point. The
Rust `io.rs` dlopen binding could own a single shared scan file with the
flock queue, writing each point's dataset directly from the native buffers
(zero Julia-side copies) — pairs naturally with the Windows `LockFileEx`
fix.

## Numerics

### 10. Mixed-precision trial steps
The embedded error estimate only needs ~3 significant digits to steer the
PI controller. Evaluating *rejected-step probing* in f32 (or doing the
first RHS stage in f32 and promoting on acceptance) can roughly halve
memory bandwidth in the hot loop, which is the actual bottleneck at 16k
points. Needs care with the known near-cancellation in `b5-b4`
(TESTING.md §3) — the error *estimate* is the one place f32 is risky, so
compute the estimate's reduction in f64 from f32 stage values.

### 11. Deterministic-mode flag
The adaptive-path divergence documented in TESTING.md §3 (FP-noise in the
error estimate → different step sequences) also affects users comparing
runs across machines/ISAs. A `deterministic=true` mode that (a) pins
reduction order (no rayon in the error norm) and (b) disables
target-cpu-dependent dispatch would give bit-reproducible trajectories —
valuable for papers and regression archaeology, cheap to implement since
the dispatcher already exists.

### 12. Higher-order dense output everywhere
Phase 8 fixed dense output to Julia's quartic `interpC`. Consider exposing
the full DP5(4) continuous extension (5th-order Shampine dense output) on
both paths — the coefficients are standard, the cost is one extra k-vector
combination per saved point, and it removes a documented Julia-vs-native
tolerance tier (saved-grid interpolation) entirely.

## Ecosystem

### 13. Prebuilt binaries via a JLL / cargo-dist
Requiring a Rust toolchain at `Pkg.build` time is the biggest adoption
barrier vs upstream Luna. Publishing `libluna_rust` as a JLL artifact
(Yggdrasil build recipe, or GitHub-release binaries fetched by
`deps/build.jl` with a source-build fallback) makes `]add Amalthea`
just work — and pins the exact rustc for reproducibility.

### 14. Standalone Rust CLI / WASM demo
The crate is already a self-contained UPPE engine (grid, dispersion,
stepper, FFT). A thin `luna-rust-cli` (TOML config in → HDF5 out) would
serve HPC users who don't want a Julia runtime on compute nodes; the same
core compiled to WASM + a small web UI makes a compelling teaching demo
(mode-averaged Kerr-only fits comfortably in a browser).

## Numerics — post-port audit additions (2026-07-08)

Full derivations and verification requirements in
`docs/dev/native-port/MATH.md` §8; tracked in BACKLOG.md Phase J. Like the
analytic-β1 precedent, each of these deliberately breaks bit-parity with
the Julia oracle to be *more* correct or cheaper at equal accuracy — so
each needs a controlled-divergence verification against a ground truth
(BigFloat / closed form), which is most of its cost.

### 15. Direct embedded-error coefficients
Compute the DP5(4) error estimate as `Σᵢ eᵢ·kᵢ` with precomputed
`eᵢ = b5ᵢ − b4ᵢ` (exact rationals) instead of subtracting the two
solutions — removes the near-total cancellation documented in
TESTING.md §3 that makes the PI controller FP-noise-sensitive and
forces the fixed-step test discipline. Must land on `PreconStepper` and
the native stepper in the same commit (both step sequences change).
Potentially the highest-leverage numerics item: it attacks the root
cause behind the adaptive-path divergence, the deterministic-mode
motivation (#11), and part of the mixed-precision risk (#10).

### 16. Direct PPT evaluation (replace the spline LUT on both sides)
`IonRatePPTAccel` is a spline LUT in Julia too; evaluating the PPT
series directly with good special-function code is more accurate than
both current paths and structurally eliminates the out-of-range
segfault (BACKLOG Phase J item 2) — no fitted range to fall off of.
Verify the direct sum against BigFloat, then existing triangulating
sim tests unchanged.

### 17. Short-kernel (overlap-save) Raman convolution
For strongly damped responses (SiO2: h ≈ 0 beyond ~100 fs on a
multi-ps grid), replace the double-length-grid FFT convolution with
overlap-save using a kernel truncated where |h| < f64 noise (checked
at setup; full double grid kept as fallback for slowly-decaying
responses). Pairs with the r2c/c2r halving (BACKLOG Phase J item 3).
Hard boundary: no recursive/IIR fits to the Gaussian-damped response —
that's the multi-SDO approximation trap Phase I item 2 rules out.

---

## Status and execution tracking

The 17 ideas above were grouped into tracks **S1-S6** and scheduled for
execution. That plan — and, more importantly, the *live status* of every
track and item — lives in [`BACKLOG.md`](BACKLOG.md), which declares itself
the single owner of status ("synced so status lives in one place"). The
duplicate copy of the track plan that used to sit here was removed on
2026-07-22 because the two had begun to drift.

| Track | Covers ideas | Where |
|---|---|---|
| S1 — Hot-loop CPU performance | 3, 4, 5 | ✅ closed — [`ARCHIVE.md`](ARCHIVE.md) |
| S2 — Threading the native RHS | 2 | live — [`BACKLOG.md`](BACKLOG.md) |
| S3 — GPU-resident propagation | 1 | live — [`BACKLOG.md`](BACKLOG.md) |
| S4 — Architecture cleanups | 6, 7, 8 | ✅ closed — [`ARCHIVE.md`](ARCHIVE.md) |
| S5 — Numerics options | 10, 11, 12 | live — [`BACKLOG.md`](BACKLOG.md) |
| S6 — Distribution & ecosystem | 9, 13, 14 | live — [`BACKLOG.md`](BACKLOG.md) |

Ideas 15-17 (post-port audit additions) are tracked as ARCHIVE.md's
Phase J item 6, surfaced in `BACKLOG.md`'s open remainders.

**Read this file for the *why* — the rationale, equations and code
sketches. Read `BACKLOG.md` for the *whether* and *when*.**

