# Inbox note — S2 item 4: thread the 3-D free-space FFT

Agent: Claude (sonnet-5), worktree `agent-a9b023a57275d7a78`, branch
`worktree-agent-a9b023a57275d7a78`. Do not edit `PORT_LOG.md`/`BACKLOG.md`/
`NATIVE_SUPPORT_MATRIX.md` directly (8 agents editing concurrently) — this
file is the merge-ready content for whoever integrates it.

---

## 1. PORT_LOG.md entry (append verbatim, in the standard template)

```
## 2026-07-22 — S2 item 4 — thread the 3-D free-space FFT — Claude (sonnet-5)
**Status:** complete
**Did:** Threaded `TransFree`'s joint 3-D FFTW plan (`fftw.rs::RealFft3d`/
`ComplexFft3d`) using FFTW's own internal threaded planner
(`fftw_init_threads`/`fftw_plan_with_nthreads`, both resolved from the
*same* combined `libfftw3.so` FFTW_jll already ships — no separate
`libfftw3_threads` to dlopen on this build, confirmed via `nm -D`). This
closes the last open item in BACKLOG S2 — the track is now fully done.
**How:**
- `fftw.rs`: added `InitThreads` binding (`fftw_init_threads`, optional/
  best-effort like the existing `plan_with_nthreads`/`planner_nthreads`),
  `FftwApi::ensure_threads_initialized` (process-wide `OnceLock<bool>`
  guarding a single `fftw_init_threads()` call), and
  `FftwApi::with_nthreads_plan(n, f)` — the symmetric counterpart to the
  existing `with_single_threaded_plan`, forcing FFTW's global thread count
  to `n` (falling back to `with_single_threaded_plan` if `n<=1` or
  threading isn't available) around a plan-creation closure, under the
  existing `PLANNER_LOCK`.
- `RealFft3d::new`/`ComplexFft3d::new` gained an `nthreads: usize` argument,
  wrapping their `fftw_plan_r2c_3d`/`fftw_plan_c2r_3d`/`fftw_plan_dft_3d`
  calls in `with_nthreads_plan` (previously these two constructors were the
  only plan types in the file with NO thread-count control at all — they
  silently inherited whatever the ambient process-global FFTW thread count
  happened to be at call time, always 1 in practice since nothing else in
  the process ever left it otherwise).
- `native.rs::set_free_params` (`CpuNativeSim`, ~line 4225, backing the
  `native_set_free_params` FFI export at ~5693): passes `s.n_threads`
  (already populated by `native_set_threads`, called unconditionally
  before any geometry-specific `native_set_*_params` block — see
  `RK45.jl:1150-1165`) as the new `nthreads` argument. No FFI signature
  change was needed — `s.n_threads` was already plumbed through from S2
  items 1-3, just never read by the free-space plan constructors.
- `RK45.jl`: added a doc comment at the `native_set_free_params` call site
  (~line 1984) explaining the no-signature-change threading wiring; no
  functional change (the free-space guard block, `native_threads` kwarg,
  and its `Threads.nthreads()` default were all already in place from S2
  item 1).
- New `test/test_native_free_threading.jl`: RealGrid + EnvGrid Kerr,
  `native_threads=1` vs `=4` bit-identical (`s.yn ==`, not tolerance),
  native-vs-Julia parity unchanged (~1e-16/1e-17), 4-cycle forced-`GC.gc()`
  stress loop (documented as a standing regression guard, not evidence of a
  new hazard — see Decisions below).
**Decisions:**
- **The threading model here is not rayon-over-columns** (radial/modal's
  pattern) — free-space's `rhs_free`/`rhs_free_env` have no per-column loop
  over independent transforms; Steps 2/5 are each ONE joint 3-D transform
  per RK stage, called from exactly one thread, never concurrently. So the
  parallelism had to move *inside* FFTW itself: bake FFTW's own internal
  thread count into the plan, so a single `fftw_execute_dft_r2c`/`_c2r`
  call fans out to FFTW's private worker pool and blocks the one calling
  Rust thread until done. This is why `RealFft3d`/`ComplexFft3d`
  deliberately do **not** implement `Sync` (contrast `ComplexFft1d`/
  `RealFft1d`, which are `Sync` and forced single-threaded at plan-creation
  time specifically *because* they're executed concurrently from multiple
  rayon workers against one shared plan) — see §4 below for the full
  soundness argument.
- **Verified bit-identical-at-scale BEFORE wiring anything into native.rs**
  (per-advisor guidance): the S2 gate's justification for radial/modal
  ("disjoint memory, no cross-thread reduction ⇒ bit-identical") does not
  automatically transfer here, because the parallelism is *inside* FFTW's
  own decomposition, not a property of Rust-side memory layout chosen by
  this crate. Added two `fftw.rs` unit tests
  (`r2c_3d_threaded_matches_single_threaded`,
  `c2c_3d_threaded_matches_single_threaded`) at a realistically large size
  (2048×64×64, ~8.4M/16.8M elements — the existing 4×3×2 dimension-order
  fixture is far too small to prove engagement) BEFORE touching
  `native.rs`, confirmed bit-identical forward+inverse output at
  `nthreads=1` vs `4`, and only then wired `s.n_threads` into
  `set_free_params`.
- **Reused `s.n_threads`** (the existing rayon-pool thread count from S2
  item 1) as the FFT thread count, rather than adding a second knob — same
  single-lever pattern the radial/modal items established, and it means
  `native_threads` (already a documented `RustNativeStepper` kwarg,
  default `Threads.nthreads()`) transparently gets the free-space geometry
  too with zero new Julia-side API surface.
- **No min-size guard added.** Measured both a large (Nx=Ny=32, FFT ≈74%
  of `rhs_free` wall time) and a tiny (Nx=8,Ny=6, matching
  `test_native_free.jl`'s own parity-test grid) config — both showed a
  real speedup at `native_threads=4`, none showed a regression. Matches
  the modal item's own finding ("no min-npt guard needed") rather than
  guessing a threshold nobody measured.
**Gotchas:**
- `RealFft3d`/`ComplexFft3d` previously had **no** `with_single_threaded_plan`
  wrapping at all (unlike every other plan type in the file) — they
  silently inherited the ambient process-global FFTW thread count at
  construction time. Benign in practice (nothing else in the process ever
  left that count above 1), but worth knowing: this item didn't "add"
  thread-count exposure to an otherwise-isolated plan type, it made an
  already-latent, uncontrolled behavior explicit and deliberate.
- `fftw_init_threads` lives directly in the combined `libfftw3.so` on the
  FFTW_jll build this crate targets (verified via `nm -D
  $(julia -e 'using FFTW; print(FFTW.FFTW_jll.libfftw3)')`) — no separate
  `libfftw3_threads`/`libfftw3_omp` to dlopen on this host. The binding
  still degrades gracefully (silent single-threaded fallback) if a future
  build ships it split out and the symbol lookup misses, per the task's
  own stated requirement.
- Default `native_threads` is `Threads.nthreads()` (`RK45.jl:1048`) — in
  any multi-threaded Julia session, free-space 3-D FFT plans are now
  multi-threaded **by default**, whereas before this item that knob was
  silently a no-op for this geometry. Verified safe (bit-identical either
  way) but flagging as a genuine default-behavior change for the lead to
  be aware of, not just a bugfix.
**Tests:**
- `cargo test --release` (amalthea): 71/71 passed, 0 failed, including the
  two new bit-identical-at-scale tests. `RUSTFLAGS="-D warnings" cargo
  build --release`: clean.
- `julia --project -e 'using TestItemRunner; @run_package_tests
  filter=ti->occursin("free", ti.filename)'`: **197/197 passed, 0 failed**
  — the full pre-existing `test_native_free*.jl` suite (6 files) plus the
  new `test_native_free_threading.jl`, all achieved tolerances unchanged
  from their documented values (single-step ~1e-17/1e-18, full-solve
  ~1e-16/1e-17, z-dependent ~1e-11/1e-10 tier).
- Did **not** run the full 7-group gate or any other `LUNA_TEST_GROUP`
  (out of scope per the resource cap — the lead runs the gate).
**Next:** none — S2 as a track is fully closed (items 1-5 all done). The
lead should fold this file into `PORT_LOG.md`/`BACKLOG.md` and can delete
this inbox file once merged.
```

---

## 2. BACKLOG.md status line

Replace S2 item 4's current line:

> 4. 3-D free-space FFT: `fftw_plan_with_nthreads`/`fftw_init_threads`
>    (dlopened `libfftw3_threads`, silent fallback if absent). *Not started —
>    the last open S2 item.* Blocked on `fftw.rs`'s `unsafe impl Sync` holding
>    only for single-threaded plans: an `nthreads>1` plan would deadlock under
>    the existing concurrent `fftw_execute_dft` path, so it needs an isolated
>    multi-threaded plan under `PLANNER_LOCK` first.

with:

> 4. 🟢 **Done 2026-07-22.** `fftw_plan_with_nthreads`/`fftw_init_threads`
>    (resolved from the same combined `libfftw3.so` FFTW_jll already loads —
>    no separate `libfftw3_threads` needed on this build; silent
>    single-threaded fallback if the symbols are ever missing). **Not** a
>    rayon-over-columns pattern like radial/modal (free-space's 3-D
>    transform has no per-column loop to parallelize) — instead, FFTW's own
>    internal thread count is baked into `RealFft3d`/`ComplexFft3d` at plan
>    creation (new `fftw.rs::with_nthreads_plan`, isolated under
>    `PLANNER_LOCK`, symmetric to the existing `with_single_threaded_plan`),
>    so ONE `fftw_execute_dft_r2c`/`_c2r` call per RK stage fans out to
>    FFTW's own worker pool. `RealFft3d`/`ComplexFft3d` deliberately do
>    **not** implement `Sync` (unlike the radial/modal 1-D plan types) —
>    they are never called concurrently from multiple Rust threads, so no
>    `Sync` boundary was ever needed for them; the earlier "blocked on
>    `unsafe impl Sync`" framing described the 1-D plan types' constraint,
>    not a constraint on the 3-D plans themselves. Verified **bit-identical**
>    `n_threads=1` vs `4` at a realistic size (`fftw.rs` unit tests, 2048×
>    64×64, BEFORE any `native.rs` wiring) and end-to-end via
>    `test/test_native_free_threading.jl` (`s.yn ==`, RealGrid + EnvGrid
>    Kerr). Measured wall-clock speedup (min-of-3, this 12-core/7-concurrent-
>    agent host — see §3 below for full numbers and the contention caveat):
>    **1.51x** at a large, FFT-dominated config (Nx=Ny=32, FFT ≈74% of
>    `rhs_free` time) and **1.43x** even at the tiny Nx=8,Ny=6 grid the
>    existing parity tests use — no min-size guard needed (measured, not
>    assumed, matching item 3's own precedent). 71/71 Rust unit tests,
>    `-D warnings` clean, 197/197 `free`-group Julia tests (6 pre-existing
>    files + the new threading file), all pre-existing tolerances unchanged.
>
> **S2 — Threading the native RHS: track CLOSED 2026-07-22.** All five
> items (plumbing, radial FFT+plasma, radial Raman, modal, free-space) are
> done; item 5 (error-norm reduction) needed no code change (confirmed
> already sequential). No further open items in this track.

---

## 3. Benchmark numbers (full detail, with the host-contention caveat)

**Standing caveat:** this host runs 7 other concurrent agents on 12 cores
per the task's own resource note. Every number below is a genuine
measurement, not fabricated, but should be read as a lower bound on the
speedup achievable on a dedicated machine — contention suppresses the
`nthreads=4` case more than the `nthreads=1` case (4 threads competing for
oversubscribed cores vs. 1 thread that only needs one).

### 3a. Isolated FFT-only microbenchmark (`fftw.rs` unit tests, min-of-5, plan creation excluded)

Size: `n_t=2048, n_y=64, n_x=64` (RealGrid: 8.4M real / 4.2M complex
elements; EnvGrid: 16.8M complex elements both sides).

| Transform | nthreads=1 | nthreads=4 | speedup |
|---|---|---|---|
| `RealFft3d` forward (r2c) | 62.5 ms | 25.4 ms | **2.46x** |
| `ComplexFft3d` forward (c2c) | 118.3 ms | 47.1 ms | **2.51x** |

Both also bit-identical (`spec1 == spec4` exactly, every element, both
directions) at this size — the load-bearing correctness check, run and
confirmed BEFORE any `native.rs` wiring existed.

### 3b. `rhs_free` FFT time-share (temporary `Instant` profiling, added/
measured/reverted — never committed, per S1.6/S2 discipline; `git diff`
confirmed empty on `native.rs`'s `rhs_free` afterward)

Config: RealGrid, Nx=Ny=32 (1024 columns), `n_time_over=4096`, Kerr-only,
`native_threads=1`, 700 accumulated `rhs_free` calls (multiple full
solves):

**FFT (Steps 2+5) = 73.4-74.1% of `rhs_free` wall time** (stabilized
~74.0% after the first ~150 calls) — well above the track's own "proceed"
bar (radial: 38-61%; modal: 82-96%; S1.6's own rejected item: ~2%).

### 3c. End-to-end wall-clock (`RustNativeStepper` full `solve()`, fixed
`max_dt=min_dt=dt`, min-of-3, warmup run excluded)

| Config | native_threads=1 | native_threads=4 | speedup |
|---|---|---|---|
| Nx=Ny=32, L=0.01 (20 steps) | 8.424 s | 7.357 s | 1.15x |
| Nx=Ny=32, L=0.03 (60 steps, same config as §3b) | 22.821 s | 15.061 s | **1.51x** |
| Nx=8,Ny=6 (matches `test_native_free.jl`'s tiny parity grid), L=0.05 (100 steps) | 1.287 s | 0.900 s | **1.43x** |

The 20-step run's lower ratio (1.15x) vs. the 60-step run at the identical
spatial config (1.51x) is consistent with fixed per-solve overhead (JIT
warmup residue, plan creation, thread-pool spin-up) amortizing better over
more steps — not a sign the threading itself is weaker; §3b's steady-state
FFT-share measurement (74%, stable after ~150 calls) was taken from this
same longer run. The Amdahl-law ceiling implied by §3a's raw FFT speedup
(~2.5x) and §3b's 74% FFT share is `1/(0.26 + 0.74/2.5) ≈ 1.80x`; the
measured 1.51x end-to-end is below that ceiling, consistent with host
contention diluting the in-process FFT speedup below the isolated 2.5x
figure once 7 other agents are also scheduling work on the same 12 cores.

**Verdict: real, measured, positive speedup at every tested size — proceed
(not revert).** No config showed `nthreads=4` slower than `nthreads=1`.

---

## 4. Soundness argument for the `Sync` boundary

**Claim:** `RealFft3d`/`ComplexFft3d` are correctly `!Sync` (the Rust
compiler's default for any type holding a raw pointer, and neither struct
opts back in with `unsafe impl Sync`), and this is not a limitation to work
around — the design *depends on* their not being `Sync`.

**Why `ComplexFft1d`/`RealFft1d` (the 1-D per-column plan types) ARE `Sync`:**
`rhs_radial`/`rhs_modal` parallelize a loop over many independent
columns/nodes with rayon, and every worker calls `forward`/`inverse` on
the SAME shared plan object concurrently, each against its own disjoint
buffer slice. FFTW documents `fftw_execute*` (unlike `fftw_plan_*`) as
reentrant/thread-safe when called concurrently against one shared plan
with distinct buffers — but **only if that plan itself was created with
FFTW's own internal thread count forced to 1** (`with_single_threaded_plan`,
already existing code). A plan baked with FFTW-internal `nthreads>1`
dispatches to FFTW's own worker pool on every `fftw_execute*` call; if two
Rust threads then call `fftw_execute*` concurrently against that same
plan, FFTW's own thread-pool synchronization state gets entered twice at
once from unrelated call sites — this repo has already hit the resulting
deadlock once (the native-radial-threading hang this module's doc
comments reference). So `ComplexFft1d`/`RealFft1d`'s `Sync` impl is sound
specifically *because* `new` always forces `nthreads=1` at plan-creation
time via `with_single_threaded_plan` — the `Sync` grant and the
single-threaded-planning guarantee are two halves of one soundness
argument, and neither can move independently.

**Why `RealFft3d`/`ComplexFft3d` need the opposite treatment:**
`rhs_free`/`rhs_free_env` have no per-column loop over independent 3-D
transforms — Steps 2 and 5 are each exactly ONE joint transform spanning
the whole `(t,y,x)` volume, called once per RK stage from the single Rust
thread executing that stage. Nothing else in the crate calls `forward`/
`inverse` on `fft_r2c_3d`/`fft_c2c_3d` concurrently, and nothing ever will
under the current `CpuNativeSim` design (`Box<dyn NativeBackend>` carries
no `Send`/`Sync` bound and is only ever driven by one FFI caller at a
time). Because there is no concurrent Rust-side access to guard against,
granting `Sync` here would buy nothing — and would be actively wrong to
grant if this plan is ALSO built with FFTW-internal `nthreads>1` (which it
now deliberately is, via the new `with_nthreads_plan`): a hypothetical
future caller that *did* start sharing this plan across rayon workers
would hit exactly the same deadlock class `ComplexFft1d`/`RealFft1d` avoid
by staying single-threaded internally. Withholding `Sync` makes that
misuse a compile error instead of a latent deadlock, which is the
type-system enforcing the actual safety boundary rather than a comment
promising it.

**The isolation the task asked for, concretely:** `with_nthreads_plan` is
a new, separate code path from `with_single_threaded_plan` — it is never
called by `ComplexFft1d`/`RealFft1d`/`SplitComplexFft1d`/`SplitRealFft1d`
(the four `Sync` 1-D plan types all still call `with_single_threaded_plan`
exactly as before, unmodified), and `with_nthreads_plan` itself falls back
to `with_single_threaded_plan`'s behavior whenever `n<=1` or FFTW's
threaded API isn't available, so the two helpers converge to identical
behavior at the boundary rather than diverging silently. The doc comment
on `with_nthreads_plan` explicitly tells future readers not to use it for
the 1-D types, and explains why in terms of the same argument above.

---

## 5. Dependencies outside my zone

None. `native_set_free_params`'s FFI signature was unchanged (the new
`nthreads` argument is internal, threaded through the already-existing
`s.n_threads` field set by `native_set_threads`), so no other agent's
region needed to change.
