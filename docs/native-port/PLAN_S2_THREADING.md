# Plan: BACKLOG.md S2 — Threading the native RHS

Status: **Phase 1 done (2026-07-10). Phase 2 (measurement) done — result
justifies proceeding. Phase 3 (thread the safe seams) next.** See
BACKLOG.md's S2 entry for the live status line; this file is the durable
plan (survives a context reset), not the status tracker.

## Context

S1.6 (the SoA/split-complex conversion) was measured and parked: the
exp-linop multiply it targeted is only ~2% of `step()` time, so its
ceiling was too small to justify the rewrite. That investigation also
confirmed FFT/RHS work dominates `step()` time — which is exactly what S2
targets: the resident RHS (`rhs_radial`, `rhs_modal`, `rhs_free`) runs
single-threaded per stage today.

An Explore survey (2026-07-10) found the risk profile is better than the
backlog's own framing suggested. The backlog warns "one scratch slab per
rayon worker (never shared) — this is precisely the bug the Julia
`Threads.@threads` `pointcalc!` race had." In `rhs_radial`
(`luna-rust/src/native.rs:1343+`), the two candidate FFT loops (Step 1
`fft.inverse` per r-column, Step 6 `fft.forward` per r-column) already
operate on **disjoint column slices of column-major, matrix-shaped
buffers** (`radial_eoo`/`radial_eto`/`radial_pto`/`radial_poo`, each sized
`n_time_over*n_r` or `n_spec_over*n_r`) — not a single shared per-call
scratch reused sequentially. FFTW's own documented contract (`fftw.rs:32-34`)
is that `fftw_execute*` (unlike `fftw_plan_*`) is thread-safe when called
concurrently against one shared plan with distinct buffer arguments. Same
finding for `apply_plasma_radial` (native.rs:982-1032): `plas_rate`/
`plas_fraction`/`plas_phase`/`plas_j`/`plas_p` are already `n_time_over*n_r`
matrices indexed by the same `start..end` column slices — also
safe-by-construction. The one genuine shared-mutable-state hazard is
`apply_raman_radial`'s single resident `raman_solver` and (when
`raman_thg == false`) the shared Hilbert-transform scratch — both reused
sequentially across r-columns today, which *would* race under naive
parallelization. QDHT itself is a single batched dense operation across
all columns, not a per-column loop — it already has its own internal
rayon fallback path when no BLAS is found.

**Scope for this pass:** radial geometry only (BACKLOG.md S2 items 1-2).
Modal (item 3) and free-space (item 4) are structurally independent and
left as separate, not-started follow-ups. Item 5 (error-norm reduction
stays sequential) needed no code change — confirmed already sequential.

## Phase 1 — plumbing: `native_set_threads` + per-sim thread pool — DONE 2026-07-10

Added `n_threads: usize` (default 1) and `thread_pool: Option<rayon::ThreadPool>`
to `CpuNativeSim`, a `set_threads` `NativeBackend` trait method (no-op
stub for `CudaNativeSim`), and a `native_set_threads` FFI entry point,
wired from Julia's `Threads.nthreads()` at `RustNativeStepper`
construction (`src/RK45.jl`, right after `native_set_fftw_plans`).
Deliberately builds a **per-sim** `rayon::ThreadPool` (not
`rayon::ThreadPoolBuilder::build_global()`) — the global pool can only be
configured once per process and would conflict with `QdhtFfiHandle`'s
existing rayon fallback usage (currently ungated) and with multiple
sims/tests coexisting in one Julia process. Purely additive — `n_threads`
is stored but not yet read by any RHS code. Verified behavior-neutral:
full 7-group gate matches all baselines exactly.

## Phase 2 — measurement (temporary instrumentation, reverted after reading) — DONE 2026-07-10

Added temporary `Instant`-based profiling to `rhs_radial` (same pattern as
the S1.6 ceiling measurement: add, measure, revert, never commit), timing:
(a) the two FFT-per-column loops, (b) `qdht.apply_real` calls, (c)
`apply_plasma_radial`, (d) everything else (Kerr, window, noise). Measured
on `N=32` and `N=128` r-points, with and without plasma (500/200 fixed-dt
steps respectively):

| N | plasma | FFT | QDHT | plasma | other |
|---|---|---|---|---|---|
| 32 | off | 35.9% | 46.7% | — | 17.4% |
| 32 | on | 20.9% | 28.3% | 40.5% | 10.2% |
| 128 | off | 18.6% | 72.6% | — | 8.8% |
| 128 | on | 14.1% | 54.8% | 24.5% | 6.6% |

**Decision: proceed to Phase 3.** Unlike S1.6's ~2% ceiling, FFT+plasma
combined is 38-61% of `rhs_radial` time for plasma-enabled configs —
comfortably clears the bar for the parallelization work below. QDHT
dominates the rest but is already internally parallel/BLAS-backed (S1
item 5) — out of scope for this item. Instrumentation was fully reverted
(`git diff` empty on `native.rs`) before Phase 3 started.

## Phase 3 — thread the safe seams: FFT-per-column + plasma-per-column — NEXT

Gated behind `self.n_threads > 1` (via
`self.thread_pool.as_ref().unwrap().install(|| { ... })`, building the
pool lazily on first use):
- Step 1 (`fft.inverse` per r-column) and Step 6 (`fft.forward` per
  r-column) in `rhs_radial`: replace the sequential loop with
  `par_chunks_mut`/zipped `par_iter` over the disjoint column slices.
  Same for `rhs_radial_env` (EnvGrid, `ComplexFft1d`/complex chunks).
- `apply_plasma_radial`'s r-loop: same `par_chunks_mut` treatment — a
  direct swap of the loop construct, not a data-layout change (buffers
  are already matrix-shaped).
- **Raman stays sequential in this phase** (`apply_raman_radial` keeps
  its single shared `raman_solver` — parallelizing it safely needs one
  `TimeDomainRamanSolver` instance per rayon worker, a separate follow-up,
  not bundled here).
- QDHT, Kerr, windowing, noise injection: left as-is (QDHT already has
  its own internal parallelism; the others are cheap flat elementwise ops
  — the "other" column above is 6.6-17.4%, not worth the same treatment
  unless a future measurement says otherwise).

**Verify — bit-identical, not tolerance-based:** every parallelized loop
writes to disjoint memory with no cross-column reduction, so
`n_threads=1` vs. `n_threads=4` must produce **exactly** the same output.
Add this as an explicit assertion in `test_native_radial.jl` and
`test_native_radial_plasma.jl`: run the same solve at both thread counts,
assert exact equality (not `norm(...) < tol`). Any mismatch is a real bug
(an assumed-disjoint slice aliased, or a chunk-size miscalculation at a
specific `n_r % n_threads`), never a tolerance to loosen. Full 7-group
gate at `n_threads=1` (must match all baselines exactly) plus a second
`rust`-group-only run with 4 threads.

## Phase 4 (not started) — Raman per-worker solver, modal, free-space

- Raman per-worker `TimeDomainRamanSolver` instances (indexed by
  `rayon::current_thread_index()`, sized to `n_threads`).
- Modal (`rhs_modal`/`modal_integrand_v`/`modal_integrand_v_full`,
  native.rs:2313-2387): rayon over the batch's `npt` points inside the
  callback (cubature's own adaptive node placement stays sequential).
- Free-space 3-D FFT threading via `fftw_init_threads`/
  `fftw_plan_with_nthreads` (new bindings in `fftw.rs`, a planning-time
  setting under `PLANNER_LOCK`).

## Verification (every phase)

- `RUSTFLAGS="-D warnings" cargo test --release` — must stay green.
- Full 7-group Julia gate, sandbox disabled (see gotcha in
  `PLAN_S1_6_SOA_CONVERSION.md`): `rust` via
  `python3 test/parallel_rust_tests.py` (baseline 42087/42087), other 6
  via `LUNA_TEST_GROUP=<group> julia --project test/runtests.jl`.
- Phase 3's bit-identical n_threads=1-vs-4 assertion is the key new
  correctness gate — treat any mismatch as a bug, not noise.
