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
Seven `LUNA_USE_RUST_*` env vars now interact (REVIEW §3.2 is a direct
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
`deps/build.jl` with a source-build fallback) makes `]add Luna-Rust`
just work — and pins the exact rustc for reproducibility.

### 14. Standalone Rust CLI / WASM demo
The crate is already a self-contained UPPE engine (grid, dispersion,
stepper, FFT). A thin `luna-rust-cli` (TOML config in → HDF5 out) would
serve HPC users who don't want a Julia runtime on compute nodes; the same
core compiled to WASM + a small web UI makes a compelling teaching demo
(mode-averaged Kerr-only fits comfortably in a browser).

## Numerics — post-port audit additions (2026-07-08)

Full derivations and verification requirements in
`docs/native-port/MATH.md` §8; tracked in BACKLOG.md Phase J. Like the
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

# Implementation plan

Executable plan for the 14 suggestions above, grouped into tracks (S1–S6)
that can each be run as an independent work session on a later date. Track
order is the recommended order, but only the dependencies stated inside
each track are hard requirements.

**To resume on a later date, paste this to a fresh session:**
> Read `SUGGESTIONS.md` (Implementation plan section), `AGENTS.md`, and the
> latest `docs/native-port/PORT_LOG.md` entry in Luna-Rust.jl. Then
> implement Track S<N>, following its steps and gate. Log the postmortem in
> PORT_LOG.md when done.

**Universal gate (every track):** `cargo test` green in `luna-rust/`, then
full `LUNA_TEST_GROUP=All` Julia suite green (46590+/0/0, 12 pre-existing
broken) — not a subset. Performance tracks additionally require a
before/after benchmark recorded in `docs/native-port/PORT_LOG.md`; a track
whose benchmark shows no win gets reverted, not merged ("keep it because we
wrote it" is not a merge criterion).

**Prerequisite for all performance tracks (S1–S3):** BACKLOG Phase C.3's
benchmark harness must exist first (a fixed-seed default-HCF wall-time +
per-step-cost script, e.g. `luna-rust/benches/e2e_hcf.jl` + Criterion
micro-benches). Without it, none of these tracks can prove their win.
Build it as step 0 of whichever track runs first.

## Track S1 — Hot-loop CPU performance (suggestions 3, 4, 5)
*Effort: ~2–4 sessions. No new dependencies. Biggest certain win per effort.*

1. **FFTW wisdom for native plans (item 4; smallest, do first).**
   - Rust: in `fftw.rs`, bind `fftw_import_wisdom_from_filename` /
     `fftw_export_wisdom_to_filename` (same dlopen pattern as existing
     symbols). Add an optional `wisdom_path: Option<CString>` to plan
     creation; on hit, plan with `FFTW_MEASURE` instead of `FFTW_ESTIMATE`.
   - Julia: pass `Utils.cachedir()`'s existing wisdom file path through
     `native_set_fftw_plans` (new trailing `Cstring` arg; keep the old
     signature working by accepting `C_NULL`). Tests keep `:estimate`
     (pass NULL) so CI timing is unchanged.
   - Gate: universal + benchmark shows setup cost unchanged (wisdom hit)
     and per-step FFT time ≤ before.
2. **Fused RK45 stage accumulation (item 3c).**
   - Rust: in `native.rs`'s stage loops, replace per-coefficient passes
     (`y + h·a_i1·k1`, then `+ h·a_i2·k2`, …) with one fused pass per stage
     reading each `k_j` once and FMA-ing into the accumulator. Pure
     reordering of the same FP ops — but summation order changes, so
     expect the documented adaptive-path sensitivity: validate with
     fixed-step equivalence tests (max_dt=min_dt pattern, TESTING.md §3).
   - Gate: universal + single-step vs Julia still ≤1e-13 in the phase
     tests + measurable per-step win at n_t ≥ 8192.
3. **exp(L·c_i·dt) caching (item 3d).**
   - Rust: `native.rs` — cache the per-stage propagator arrays keyed on
     `dt`; invalidate on dt change (accept/reject) and on
     `ensure_linop_at` (z-dep case). Note c2..c7 include repeated values
     only via the FSAL structure — verify which stage offsets actually
     coincide before assuming shared factors; the win even without sharing
     is skipping recomputation on *rejected* steps (same dt retried is
     rare; measure first, skip this sub-item if rejects <5% in the
     benchmark).
4. **SIMD Kerr + window/norm broadcasts (item 3a).**
   - Rust: route the Kerr cube and the ωwin/twin/norm loops through
     `dispatch.rs` (the AVX2/AVX-512 lanes exist; only Raman uses them
     today). Portable-scalar fallback stays the reference implementation;
     add a Rust unit test asserting SIMD lane == scalar lane bitwise (same
     FP ops, just packed) or within 1 ulp where FMA contraction differs.
5. **BLAS-3 QDHT (item 5).**
   - Rust: dlopen the in-process BLAS Julia already ships
     (`libblastrampoline`; resolve `dgemm_64_`/`zgemm_64_` — note the
     ILP64 suffix convention) with the `io.rs`/`fftw.rs` runtime-loading
     pattern. Julia passes the library path at init (like the FFTW path),
     `LUNA_QDHT_BLAS=0` opt-out. Fall back to the existing Rayon kernel if
     symbols are missing.
   - Gate: universal + `test_qdht_rust.jl` tolerance unchanged (~1e-13;
     BLAS reorders sums — if it exceeds this, widen ONLY with a written
     justification in the test) + ≥1.5× QDHT micro-bench at n_r=256 or
     revert.
6. **Split-complex exp-linop experiment (item 3b) — timeboxed spike.**
   - Benchmark a split-layout (SoA) version of only the exp(linop)·y
     multiply on synthetic data in Criterion first. Only if >1.3× do the
     invasive part (guru split-DFT FFTW plans). Otherwise record the
     negative result in PORT_LOG.md and close the item.

## Track S2 — Threading the native RHS (suggestion 2)
*Effort: ~2 sessions. Depends on: S1.4 (dispatch plumbing touched there).*

1. Add a thread-count knob: `native_set_threads(handle, n)` FFI, wired
   from `Threads.nthreads()` on the Julia side, default 1 (today's
   behaviour) so single-thread CI results are bit-identical.
2. `rhs_radial`: rayon `par_iter` over radial nodes — each node's 1-D FFT
   plan must be per-thread (FFTW plan execution is thread-safe only via
   `fftw_execute_dft` new-array variants; use those against one shared
   plan). Scratch: one slab per rayon worker (`thread_local` or
   `map_init`), never shared — this is precisely the bug the Julia
   `Threads.@threads` attempt had.
3. Modal path: rayon over cubature evaluation points with per-task scratch.
   Caveat: `libcubature`'s vectorized interface (`hcubature_v`) hands us
   batches — parallelize *inside* the batch callback, keeping cubature's
   adaptive node placement sequential/deterministic (bit-identical
   integral regions, only the batch loop parallel).
4. 3-D free-space FFT: bind `fftw_plan_with_nthreads` +
   `fftw_init_threads` (libfftw3_threads, dlopened; fall back silently if
   absent).
5. Reductions (error norm) stay sequential — determinism (see S5.2).
   Document that in `native.rs` where the norm is computed.
- Gate: universal + fixed-step equivalence tests pass with
  `JULIA_NUM_THREADS=4` + radial/free benchmark ≥2× at 4 threads, and
  n=1 results bit-identical to pre-track.

## Track S3 — GPU-resident propagation (suggestion 1)
*Effort: large (5+ sessions); needs a CUDA machine. Depends on: BACKLOG
Phase G.2 (GPU CI) or it will rot immediately; do NOT start before that
exists.*

1. Design doc first (`docs/native-port/GPU.md`): which `NativeSim` fields
   move to device; per-step host↔device traffic budget (target: scalars
   only); cuFFT plan lifecycle; where Julia-side `get_field`/`set_field`/
   `native_resync_field` force a transfer (they do — that's the resync
   seam, once per accepted step, n_t floats; acceptable).
2. `CudaNativeSim` in `cuda.rs` implementing the same internal trait as
   the CPU `NativeSim` (extract a `NativeBackend` trait from `native.rs`
   first — pure refactor, its own commit, universal gate).
3. Kernels in `kernels.cu`: exp-linop multiply, Kerr, window/norm, RK
   stage fusion (reuse S1.2's fused formulation), cumtrapz (plasma scan —
   use a work-efficient prefix scan), Raman ADE (sequential in t —
   parallelize across oscillators/modes only, or keep Raman configs on
   CPU in v1: document the eligibility split like `NativeIneligible`).
4. Dispatch: extend `dispatch.rs`'s existing cascade with a *problem-size
   threshold* (measured crossover, not guessed) so small grids stay on
   CPU. Env override `LUNA_NATIVE_GPU=0/1/auto`.
5. Scope v1 to mode-averaged RealGrid Kerr(+plasma) — the Phase 1/2
   equivalent — with the same phase-test structure
   (`test/test_native_gpu.jl`, self-skip without CUDA, mirroring
   `test_gpu_cuda.jl`).
- Gate: universal (CPU paths untouched) + GPU equivalence vs CPU native
  ≤1e-10 fixed-step + benchmark ≥3× on free-space or large-grid
  mode-averaged; record crossover sizes in PORT_LOG.md.

## Track S4 — Architecture cleanups (suggestions 6, 7, 8)
*Effort: ~2 sessions. Do BEFORE S3 if possible (S3 multiplies the toggle
zoo otherwise). Coordinate with BACKLOG Phase C (it touches the same
toggle logic — ideally land C first, then this).*

1. **Config struct (item 6).**
   - Julia: `src/Config.jl` — `struct BackendConfig` (native::Symbol
     ∈ (:auto,:on,:off), per-kernel overrides, gpu, threads, qdht_blas).
     One resolver `Luna.backend_config()` reads env vars ONCE at first
     use (keeps CI/env compatibility; env semantics unchanged), then
     everything (`solve_precon`, `IonRatePPTAccel`,
     `_make_rust_raman_handle`, …) consults the struct — grep for every
     `get(ENV, "LUNA_USE_RUST` and replace.
   - Introspection: `Luna.backend_report()` returning which stepper/kernels
     ran last (populate a `Ref` from `solve_precon` and the fallback catch;
     this also implements BACKLOG Phase C.2's test hook).
2. **FFI error enum (item 7).**
   - Rust: `#[repr(i32)] enum FfiStatus { Ok=0, Ineligible>0…, Bug<0… }`
     in `ffi.rs`; migrate `native_set_*` returns to it.
   - Julia: one `check_ffi(rc, what)` that throws `NativeIneligible` for
     positive codes and `error()` for negative — replace the per-call-site
     `rc == 0 ||` patterns in `RK45.jl`.
3. **Explicit accessor seams (item 8).**
   - Julia: in `NonlinearRHS.jl`/`Nonlinear.jl`, add `kerr_γ3(resp)::Float64`
     (dispatch per response type; `NativeIneligible` for unknown) and
     `norm_pre(norm!)`, `norm_βfun(norm!)`; delete the
     `occursin("γ3", string(fld))` and `getfield(norm_func, :pre)` probes
     in `RK45.jl`.
- Gate: universal + a new `@testitem` asserting `backend_report()` says
  `RustNativeStepper` for a default `prop_capillary` (post-Phase-C) and
  Julia stepper when `native=:off`.

## Track S5 — Numerics options (suggestions 10, 11, 12)
*Effort: ~2–3 sessions, independent of each other; S5.1 is a spike.*

1. **Mixed-precision spike (item 10) — timeboxed, expect to reject.**
   Criterion bench of an f32 stage evaluation + f64 error reduction on
   synthetic 16k grids. Only proceed to a real implementation if >1.4×
   end-to-end projected AND the b5−b4 cancellation analysis (TESTING.md §3)
   shows the f64-reduced estimate keeps the controller on the same step
   sequence for the benchmark case. Record either way in PORT_LOG.md.
2. **Deterministic mode (item 11).**
   - Rust: `native_set_deterministic(handle, bool)` — forces sequential
     reductions (no-op until S2 lands parallel ones) and pins
     `dispatch.rs` to the portable lane.
   - Julia: `deterministic=true` kwarg on `prop_capillary`/`prop_gnlse`
     threaded into `BackendConfig` (S4.1 — hard dependency).
   - Test: two runs bit-identical (`===` on output arrays), and a
     documented statement of what it does NOT guarantee (cross-machine
     libm differences).
3. **5th-order dense output (item 12).**
   - Rust: Shampine's DP5 continuous-extension coefficients in
     `stepper.rs`/`native.rs::interpolate` path (`get_ks_stage` already
     exposes the stages — Phase 8 built the plumbing).
   - Julia: same coefficients in `RK45.jl`'s `interpC` **in the same
     commit** — the point is removing the Julia-vs-native saved-grid
     tolerance tier, which only works if both sides change together.
   - Gate: universal + saved-output equivalence tests tighten (record the
     new tier in TESTING.md; if any general-suite test needed the old
     looser tier, that's a finding, not a tolerance to widen).

## Track S6 — Distribution & ecosystem (suggestions 9, 13, 14)
*Effort: 13 is ~1–2 sessions; 9 and 14 are larger and lowest priority.*

1. **Prebuilt binaries (item 13, do first — biggest adoption win).**
   - CI: a `release.yml` workflow building `libluna_rust` for
     linux-x86_64/macos-{x86_64,aarch64}/windows-x86_64 with `RUSTFLAGS=""`
     (portable; runtime dispatcher handles ISA) and uploading to GitHub
     Releases with a sha256 manifest.
   - `deps/build.jl`: try download-by-version+checksum first, fall back to
     the existing `cargo build` (keep it as the from-source path and for
     dev checkouts — detect via `.git` presence or `LUNA_BUILD_FROM_SOURCE=1`).
   - Later (optional): proper Yggdrasil JLL.
2. **Rust-side scan HDF5 writer (item 9).** Extend `io.rs` with a
   `scan_write_point(queue, idx, name, dims, data_ptr)` writing directly
   from native buffers under the existing flock (Unix) / `LockFileEx`
   (after BACKLOG Phase G.1 — hard dependency) queue lock; wire as an
   opt-in `Output.jl` backend. Gate: `test_scans.jl` + a concurrent
   two-process write test.
3. **CLI (item 14).** New `luna-rust/src/bin/luna-cli.rs` (or separate
   crate in a workspace — note the root-workspace removal history in
   BACKLOG "Done"; prefer `[[bin]]` in the existing crate to avoid
   re-breaking test paths). TOML schema mirroring `prop_capillary`'s
   kwargs; scope v1 to exactly the native-eligible configs (mode-averaged
   RealGrid Kerr/plasma/Raman), HDF5 out via `io.rs`. Compare against a
   Julia run of the same config as the acceptance test. WASM demo only
   after the CLI exists (it reuses the same config-in/arrays-out core with
   FFT via `rustfft` behind a feature flag, since FFTW/dlopen is
   unavailable in WASM).

## Suggested execution order

S1 (with the Phase C.3 benchmark harness as step 0) → S4 → S2 → S5.2/S5.3
→ S6.1 → S3 → remaining S5/S6 items. Rationale: certain CPU wins first,
then the config/introspection groundwork that everything later reports
through, then parallelism, then the big GPU bet once CI can hold it.
