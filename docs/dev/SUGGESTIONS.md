# Suggestions — rationale and outcomes

This file preserves the original *why* behind the 17 suggestions. It is not a
queue and some proposal text intentionally describes the pre-implementation
state. Current execution status belongs to `BACKLOG.md`.

Status snapshot (2026-07-25):

| Idea | Outcome |
|---|---|
| 1 GPU-resident propagation | **Correctness-blocked.** Narrow `CudaNativeSim` exists, but currently omits nonlinear scaling/normalization and behaves linearly; BACKLOG S3 item 0 is the first priority |
| 2 Threading | **Complete.** Radial, modal, and free-space seams landed; reductions remain sequential |
| 3 SIMD/layout | Fused/de-branched loops landed; full SoA conversion **parked** after an end-to-end ceiling of ~1% |
| 4 FFTW wisdom | **Complete, opt-in.** `AMALTHEA_NATIVE_FFTW_WISDOM=1`, default off for determinism |
| 5 BLAS-3 QDHT | Correctness wiring fixed; remains opt-in pending the default-flip performance bar |
| 6 Config struct | **Complete.** `BackendConfig` plus backend reporting |
| 7 FFI error model | **Complete.** Shared `RK45.check_ffi`/`NativeIneligible` policy |
| 8 Explicit setup accessors | **Complete.** Reflective probes were replaced |
| 9 Native scan HDF5 writer | **Complete**, opt-in |
| 10 Mixed precision | **Rejected after measurement** (~1.0–1.06×, below bar; numerically risky) |
| 11 Deterministic mode | **Complete, re-scoped** to process-global BLAS eligibility |
| 12 Order-5 dense output | **Complete 2026-07-23**, including the upstream FSAL/k1 bug fix |
| 13 Prebuilt binaries | Workflow/install fallback **implemented**, but v1.0.0 assets use legacy `libluna_rust-*` names and do not match the current installer; fix/validate remains |
| 14 Standalone CLI/WASM | **Parked.** Cold-start CLI has negative ROI; dump-and-replay is the only recommended variant |
| 15 Direct error coefficients | **Do not pursue.** Both steppers already precompute the coefficient differences |
| 16 Direct PPT | **Do not pursue.** BigFloat-quadrature tail is unsuitable for the hot loop; LUT hardening is already fixed |
| 17 Short-kernel Raman | **Open and recommended**, benchmark first |

The original ordering below was by expected payoff per unit effort for the
typical workload (mode-averaged HCF, 8k–16k time samples, ~10³–10⁴ RK
stages per solve). The table above supersedes present-tense claims in those
historical proposal paragraphs.

## Performance

### 1. GPU-resident propagation (the real GPU win)

This proposal produced the resident `CudaNativeSim` and the
`AMALTHEA_NATIVE_GPU=off/on/auto` policy for a narrow mode-averaged
RealGrid Kerr/PPT scope. It eliminated per-kernel PCIe round-trips, but the
current RHS is missing required scaling, oversampling, normalization, and
windowing and therefore behaves linearly. Correctness restoration and
non-vacuous GPU tests precede any scope expansion; Vulkan remains
unimplemented.

### 2. Threading the native RHS

This track is complete. Rayon now covers independent radial nodes and modal
cubature points using per-worker scratch, and the free-space 3-D FFT uses
threaded FFTW plans. Reductions and Julia's shared-scratch `TransModal`
integration remain sequential by design.

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
**Implemented 2026-07-23.** Both steppers now use the same
Calvo–Montijano–Rández order-5 extra-stage extension. This work also found
and fixed the eager FSAL carry that had silently reduced every dense-output
path to first order.

## Ecosystem

### 13. Prebuilt binaries via release assets / JLL
Requiring a Rust toolchain at `Pkg.build` time is the biggest adoption
barrier vs upstream Luna. Publishing `libamalthea` as a JLL artifact
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
**Studied; do not pursue.** The premise was wrong: both steppers already
compute `Σᵢ eᵢ·kᵢ` with `eᵢ=b5ᵢ-b4ᵢ` precomputed. The sensitivity comes from
the mathematically cancelling stage sum itself, so rewriting the same
coefficients cannot remove it.

### 16. Direct PPT evaluation (replace the spline LUT on both sides)
**Studied; do not pursue.** `IonRatePPTAccel` is a spline LUT in Julia too,
but the true PPT series contains a BigFloat-quadrature tail that cannot
replace a LUT in the hot loop at acceptable cost. LUT error is already below
physical significance, and fitted-range/non-finite safety was fixed directly.

`IonRatePPTAccel` is a spline LUT in Julia too; evaluating the PPT
series directly remains useful only as an offline ground-truth tool.

### 17. Short-kernel (overlap-save) Raman convolution
**Open; recommended, benchmark first.**

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
| S2 — Threading the native RHS | 2 | ✅ closed — [`BACKLOG.md`](BACKLOG.md) |
| S3 — GPU-resident propagation | 1 | live — [`BACKLOG.md`](BACKLOG.md) |
| S4 — Architecture cleanups | 6, 7, 8 | ✅ closed — [`ARCHIVE.md`](ARCHIVE.md) |
| S5 — Numerics options | 10, 11, 12 | ✅ closed — [`BACKLOG.md`](BACKLOG.md) |
| S6 — Distribution & ecosystem | 9, 13, 14 | implementation resolved/parked; v1.0.0 asset-name repair and validation remain — [`BACKLOG.md`](BACKLOG.md) |

Ideas 15-17 (post-port audit additions) are tracked as ARCHIVE.md's
Phase J item 6. Only idea 17 remains open.

**Read this file for the *why* — the rationale, equations and code
sketches. Read `BACKLOG.md` for the *whether* and *when*.**
