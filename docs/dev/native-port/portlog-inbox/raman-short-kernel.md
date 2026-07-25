# Inbox note — BACKLOG open remainder 5 / Phase J.6(c): short-kernel Raman convolution

Measure-first spike per `AGENTS.md`/`CLAUDE.md`'s "benchmark first" discipline
(precedents: S1.6, S2 Phase 2, S5 item 1). **Outcome: measured, does not clear
the bar, recommend against implementing.** No production code changed by this
pass — everything below was added, measured, and reverted. This file is the
retained record; the deleted files' content is transcribed in full so the
measurement is reproducible without them.

Machine context: AMD Ryzen 5 5600X (6-core/12-thread, `nproc`=12), Linux. Rust
release profile (`opt-level=3`, `lto="thin"`, `codegen-units=1`), and
`.cargo/config.toml`'s `target-cpu=native` **left active** throughout (not
overridden to `RUSTFLAGS=""`) — these numbers are single-machine, native-ISA
numbers, matching how the repo's own local dev builds run, not the
CI-portable build.

---

## 1. PORT_LOG.md entry (append verbatim, in the standard template)

```
## 2026-07-25 — Phase J.6(c) — short-kernel Raman convolution (BACKLOG open remainder 5) — Claude (sonnet-5)
**Status:** complete (measure-first spike; recommend against implementing)
**Did:** Measured whether shortening the `:SiO2` intermediate-broadening
Raman FFT-convolution pad from the current `2·n_time_over` to
`n_time_over + M` (M = the real Hollenbeck & Cantrell response's support
length at an f64-noise cutoff) is worth implementing. It is not, at any grid
size this repository's own configs or examples reach. Full numbers below.
**How:** (1) Derived M analytically/numerically from the exact SiO2
parameters already in `PhysData.jl:1179-1188`/`native.rs`'s
`set_raman_fft_params` (native.rs:4409-4483) — no guessing. (2) Wrote a
temporary Criterion bench (`raman_short_kernel_bench.rs`, modeled on
`raman_fft_r2c_bench.rs` which measured J.3) using the *real* h(t), not a
synthetic kernel, across the same n_time_over=1024..65536 sweep. (3) Added
temporary `Instant`-based profiling directly to `rhs_mode_avg_env`
(native.rs:1568, Step 3c at 1647-1688) to measure Step 3c's real share of
RHS wall time at the actual `test/test_native_raman_sio2.jl` config (via a
temporary `:tmpprofile` testitem tag, reverted after), at both its native
trange=4e-12 (n_time_over=4096) and a widened trange=16e-12
(n_time_over=16384, same λlims ⇒ same dt ⇒ same M). (4) Quantified
truncation error against a realistic sech² pulse intensity via a pure-Python
r2c convolution (no numpy in this environment; hand-rolled radix-2 FFT),
not just a kernel-norm proxy.
**Decisions:**
- Truncation cutoff eps=1e-13 (relative to h's peak) — chosen to match the
  existing native-vs-Julia SiO2 full-solve tolerance floor (1.8e-13-3.6e-13,
  `test_native_raman_sio2.jl`), so a truncation error introduced at this
  cutoff cannot itself blow that budget (confirmed empirically, see §4 below).
- Held dt fixed at the real test config's value across the bench's
  n_time_over sweep, since dt is set by λlims/λ0 (bandwidth), not by trange —
  physically, M (in samples) is roughly fixed while n_time_over grows with
  trange, so the achievable ratio is a property of *how much trange margin
  the user chose beyond the material's Raman decay time*, not of grid size
  alone.
**Gotchas (the load-bearing finding):**
- `native-port/PLANS.md` §6.3 assumed "kernel maybe 5-10% of the padded
  grid" and `MATH.md` §8.5 asserted "h ≈ 0 beyond ~100fs" for SiO2. Both were
  unmeasured guesses and both are wrong by roughly 40x: the real support is
  M≈3104 samples ≈ **4.15 ps**, not ~100fs. At the one real production-shaped
  grid in this repo (`test_native_raman_sio2.jl`, n_time_over=4096), that's
  **76% of the grid**, not 5-10%. This single wrong assumption is the entire
  reason the prior recommendation ("recommend" in BACKLOG) was wrong — it's
  independently useful to the repo, and it retroactively vindicates
  native.rs's existing zero-fill comment at Step 3c ("don't rely on h's tail
  happening to be zero at the wrap distance") — the tail genuinely reaches
  the wrap boundary at real grid sizes.
- Two independent reasons the shortened pad doesn't help even where the
  kernel *is* meaningfully shorter than the grid: (a) the natural
  `n_time_over+M` length is not a power of two, and FFTW's mixed-radix path
  measurably underperforms a pure-radix-2 transform of similar or even
  larger size — enough to erase the entire length-reduction gain at
  n_time_over=4096 (7200 vs 8192: 43.66µs vs 42.89µs, i.e. *slower*); (b)
  even where the isolated transform *is* faster (n_time_over=16384: 1.32x),
  Step 3c's non-FFT overhead (`raman_intensity_half_env`, the mandatory
  zero-fill, `raman_accumulate_env`) is untouched by pad-shortening and
  dilutes the RHS-level gain to ~1.05x — short of the >1.4x bar S5.1 was
  rejected against.
**Tests:** `cargo test` (amalthea, release): 71/71 pass, post-revert.
`test_native_raman_sio2.jl` (via `LUNA_TEST_GROUP=rust`, post-revert):
unaffected — no production code changed. During measurement (pre-revert,
same physics, only added timers), native-vs-Julia agreement was 2.95e-13
(n_time_over=4096, the file's own config) and 1.04e-12
(n_time_over=16384, widened trange) — both within the expected FFT-method
summation-order tier, confirming the instrumentation didn't perturb the
math.
**Next:** None — this item is closed as "do not implement" pending a future
config that actually uses a trange many times longer than SiO2's ~4ps decay
time (none exist in this repo today; chasing that would be optimizing for a
hypothetical workload). If BACKLOG open remainder 5 needs a live entry, the
lead should mark Phase J.6(c) "recommend against" (reversing the prior
"recommend") and cite this file.
```

---

## 2. Full measurement detail (this file is the only permanent record — the bench was deleted)

### 2.1 The support length M (correctness-adjacent, but really a performance-premise check)

`h(t) = scale · Σᵢ Aᵢ·exp(-γᵢt)·exp(-Γᵢ²t²/4)·sin(ωᵢt)`, i=0..12, reproduced
bit-for-bit from `PhysData.jl:1179-1188`'s Hollenbeck & Cantrell SiO2 table
(`ωᵢ=200π·c·[wavenumbers]`, `Γᵢ=γᵢ=100π·c·[wavenumbers]`, both rad/s) and
`native.rs`'s `set_raman_fft_params` (native.rs:4409-4483, particularly
:4444-4463) which builds this exact array at setup time. `scale` is a single
uniform multiplicative constant (Julia's `hquadrature`-normalized integral) —
irrelevant to where a *relative* cutoff falls, so it was omitted (set to 1)
throughout without loss of generality.

**Criterion:** M = smallest index such that `max(|h[M:]|) < eps · max(|h|)`,
swept `eps ∈ {1e-12, 1e-13, 1e-14, 1e-15}`. Chose **eps=1e-13** to line up
with the existing native-vs-Julia SiO2 tolerance floor (see §4 for the
confirmation this choice doesn't blow that budget).

**Real test config's dt** (`test/test_native_raman_sio2.jl`: λlims=(450nm,
8000nm), λ0=835nm, `prop_gnlse` fixes thg=false ⇒ ffac=2,
f_lims=(fmin,1.1·fmax)): `δt_f = 1/(2·max|f_lims - c/λ0|)` =
**1.337638300647872e-15 s** (1.3376 fs). This is independent of `trange`
(Grid.jl:115-141): trange only sets how many of these fixed-width samples
(`n_time_over = 2^ceil(log2(trange/δt_f))`) are taken.

At this dt: **M=3104 samples at eps=1e-13** (t_M ≈ 4.15 ps physical decay
time — an intrinsic material-response property, not a grid property). Full
eps sweep at this dt (n_time_over=4096, hmax=165.387):

| eps | M | M/n_time_over | t_M |
|---|---|---|---|
| 1e-12 | 2953 | 72.1% | 3.95 ps |
| 1e-13 | 3104 | 75.8% | 4.15 ps |
| 1e-14 | 3253 | 79.4% | 4.35 ps |
| 1e-15 | 3380 | 82.5% | 4.52 ps |

**As a fraction of n_time_over at real repo configs:**
- `test/test_native_raman_sio2.jl` (trange=4e-12 ⇒ n_time_over=4096, this
  dt): **M/n_time_over = 75.8%**.
- `examples/low_level_interface/gnlse/simplescg_modeAvg_env.jl` (different
  λlims=(400nm,1400nm) ⇒ dt=1.0744e-15s; trange=10e-12 ⇒ n_time_over=16384):
  M at *that* dt = 3864 ⇒ **M/n_time_over = 23.6%**.
- Holding the test config's own dt fixed and hypothetically widening its
  trange to 16e-12 (n_time_over=16384, used for the share measurement in §3
  since it lets the bench's 16384 row and the share measurement share the
  same M=3104): **M/n_time_over = 18.9%**.

This ~40x-longer-than-assumed decay time (~4.15ps measured vs "~100fs"
asserted in MATH.md §8.5 / "5-10%" assumed in PLANS.md §6.3) is the root
cause of the whole recommendation reversal.

### 2.2 Criterion bench (deleted — full numbers transcribed here)

File was `amalthea/benches/raman_short_kernel_bench.rs`
(`[[bench]] name = "raman_short_kernel_bench"` in `Cargo.toml`), modeled
directly on `raman_fft_r2c_bench.rs` (which measured J.3). Both `full_pad`
and `short_pad` used `RealFft1d` (r2c/c2r, post-J.3) — this bench isolates
**only** the length-shortening effect, not the r2c gain (already banked).
`FFTW_ESTIMATE` (`1 << 6`) flag, matching production's actual
`set_raman_fft_params` plan flag (native.rs:4465) exactly — the mixed-radix
penalty measured below is the one production would actually pay, not a
bench artifact from a mismatched planning mode.

`REAL_DT = 1.337638300647872e-15` (fixed across the whole sweep, per §2.1).
`next_fast_len(n)`: smallest integer ≥ n whose only prime factors are
{2,3,5,7} (standard FFTW/scipy "fast length" heuristic). Fallback rule
applied: `pad_short = min(next_fast_len(n_time_over + min(M, n_time_over)),
pad_full)` — i.e. never pick a pad longer than today's, and if M exceeds
n_time_over (kernel doesn't even decay within this grid) there is nothing to
shorten.

100-sample Criterion means, release+LTO+target-cpu=native:

| n_time_over | pad_full | pad_short (raw → rounded) | length ratio | full_pad | short_pad | isolated speedup |
|---|---|---|---|---|---|---|
| 1024 | 2048 | 2048 (M=3104>1024, fallback fires) | 1.000 | 9.21 µs | 9.71 µs | 0.949x (fallback; no real change) |
| 2048 | 4096 | 4096 (M=3104>2048, fallback fires) | 1.000 | 20.47 µs | 20.30 µs | 1.008x (fallback; no real change) |
| 4096 | 8192 | 7200 → 7200 | 0.879 | 42.89 µs | 43.66 µs | **0.982x (slower)** |
| 8192 | 16384 | 11296 → 11340 | 0.692 | 90.23 µs | 82.83 µs | 1.089x |
| 16384 | 32768 | 19488 → 19600 | 0.598 | 205.21 µs | 155.88 µs | 1.317x |
| 32768 | 65536 | 35872 → 36000 | 0.549 | 477.95 µs | 310.70 µs | 1.538x |
| 65536 | 131072 | 68640 → 69120 | 0.527 | 1093.3 µs | 518.33 µs | 2.109x |

**Critical caveat (only the 4096/8192 rows are reachable by any config in
this repo today — see §2.1):** 7200 and 11340 are **not** powers of two
(7200=2⁵·3²·5², 11340=2²·3⁴·5·7). FFTW's mixed-radix path is measurably
slower per-sample than pure radix-2 at similar or even somewhat larger N —
enough to make the "shortened" 7200-length transform slower in absolute
terms than the "full" 8192-length one. A real implementation honoring
`MATH.md` §8.5's own stated fallback rule ("falling back to the full double
grid otherwise") would see no power-of-two length below 8192 that fits
4096+3104=7200, and would rationally just keep 8192 — i.e. the honest
production answer at the repo's actual test config is **1.00x (no change)**,
with the measured 0.98x serving as direct evidence of *why* a naive
mixed-radix implementation would actually regress instead.

### 2.3 End-to-end share (temporary Instant counters, add/measure/revert)

Added three `static AtomicU64` counters (total ns, Step-3c ns, call count) at
module scope in `native.rs`, an `Instant::now()` at `rhs_mode_avg_env`'s
entry and one bracketing only the `if self.has_raman_fft { ... }` block
(Step 3c, native.rs:1647-1688), accumulated with `Ordering::Relaxed`, and an
`eprintln!` every 500 calls. Isolated to a single config by giving
`test/test_native_raman_sio2.jl`'s `@testitem` a temporary extra tag
(`:tmpprofile`) and running `LUNA_TEST_GROUP=tmpprofile`. All of this was
reverted (`git diff` empty on both files) before finishing; `cargo build
--release` + `cargo test` reconfirmed green after reverting (§3 of the
PORT_LOG entry above).

`solve(s_ru_r, flength)` in the test's "Full-solve triangulation" testset
runs 2000 fixed RK45 steps × 6 stages = 12000 `rhs_mode_avg_env` calls; final
(calls=12000) readings:

| config | n_time_over | total ns | step3c ns | share (of total RHS) | isolated full_pad (bench) | step3c ns/call | fraction of Step3c that's the FFT itself | improvable share of total RHS |
|---|---|---|---|---|---|---|---|---|
| test's own trange=4e-12 | 4096 | 1,331,620,525 | 602,948,613 | 45.28% | 42,887 ns | 50,246 ns/call | 85.35% | **38.6%** |
| trange widened to 16e-12 | 16384 | 11,848,191,995 | 5,220,078,952 | 44.06% | 205,210 ns | 435,007 ns/call | 47.18% | **20.8%** |

The "fraction of Step3c that's the FFT itself" column matters: Step 3c also
runs `raman_intensity_half_env`, an explicit zero-fill of the upper half of
the padded buffer (native.rs:1667-1669, required every call — the previous
step's convolution tail must not survive), and `raman_accumulate_env`, none
of which pad-shortening touches. At n_time_over=4096 these are a small
addition (14.65% of Step3c); at n_time_over=16384 they are much larger in
*absolute* terms than a linear scale-up would predict (229.8 µs vs an
expected ~29 µs at 4x scale-up) — plausibly a cache effect, since the
relevant buffers (`raman_fft_e2`/`raman_fft_ew`) are 4x larger at this size
and may spill L2. Reported as measured, not force-fit to a clean model.

**Projected end-to-end effect (Amdahl, using the *measured* improvable share
`p` and the *measured* isolated speedup `s` from §2.2, not the raw share
alone):** `overall = 1 / [(1-p) + p/s]`.

- n_time_over=4096 (the repo's actual SiO2 test/production config): p=0.386,
  s=0.982 (7200-vs-8192 case) ⇒ **overall ≈ 0.993x (0.7% slower)**; using the
  "sane fallback keeps 8192" reading instead (s=1.00) ⇒ **overall = 1.000x
  (no change)**. Either way: **no gain, possibly a small loss**, at the one
  config this repo's own tests actually exercise.
- n_time_over=16384 (only reachable by manually widening trange 4x beyond
  any example in this repo): p=0.208, s=1.317 ⇒ **overall ≈ 1.053x (5.3%
  faster)** — the best case measured anywhere in this repo's reachable
  config space, and still well short of the >1.4x bar.

### 2.4 Correctness bound

**Kernel-norm proxy (diagnostic only — see below for why this alone isn't
sufficient):** `||h_tail||₂ / ||h_full||₂ = 1.078e-13` at the M=3104 cutoff
(n_time_over=4096 grid). This is *not* a safe stand-in for the actual output
error, because `I(t)=0.5|E|²` is strictly non-negative while `h` oscillates
in sign — `P = h ⊛ I` involves genuine cancellation, so `||P||` can sit well
below the naive `||h||·||I||` bound, which would inflate the *relative*
output error above the kernel-norm ratio.

**Realistic-signal bound (the actual measurement):** computed
`P_full = irfft(rfft(h_full) · rfft(I))` and
`P_trunc = irfft(rfft(h_truncated-at-M=3104) · rfft(I))`, both at a common
pad=8192 (power of two, ≥ n_time_over+M-1=7199, so no wraparound
contamination in either), for `I(t) = sech²((t-t_c)/τ0)` with
τ0=280fs (the test's own pulse width), at two pulse placements within the
buffer:

| pulse placement | rel. L2 error `‖P_trunc-P_full‖₂/‖P_full‖₂` |
|---|---|
| centered mid-buffer | 1.74e-16 |
| at the buffer's t=0 edge (worst-case proximity to the wrap boundary) | 3.38e-14 |

Both are comfortably below the existing native-vs-Julia SiO2 full-solve
tolerance floor (1.8e-13-3.6e-13, `test_native_raman_sio2.jl`, BACKLOG Phase
J.3 record) — **truncation at eps=1e-13 does not threaten that tolerance
tier.** Computed via a hand-rolled pure-Python radix-2 FFT (no numpy
available in this environment) — noted for reproducibility, not committed
anywhere.

**Correctness is not the blocker here — performance is.**

### 2.5 Recommendation

Two independent screens, framed the way the repo's own precedents are:

- **Share screen (S2-style, bar: was 38-61% proceed / ~2% park):** **passes.**
  38.6-45.3% of `rhs_mode_avg_env` wall time is Step 3c; even the
  FFT-only "improvable" portion is 20.8-38.6%. This kernel is legitimately
  worth attacking — the premise that motivated looking at it was sound.
- **Speedup gate (S5.1-style, bar: >1.4x; rejected there at 1.0-1.06x):**
  **fails**, at every size any real config in this repo reaches. ~0.99-1.00x
  at n_time_over=4096 (the repo's actual test/production config), ~1.05x at
  n_time_over=16384 (only reachable by hypothetically widening trange 4x
  beyond the largest example in-repo). Both are short of the >1.4x bar, and
  even short of the ~1.3x bar S1 item 6's SoA spike cleared to proceed.
- **Root cause:** the two design docs that recommended this item
  (`PLANS.md` §6.3, `MATH.md` §8.5) both asserted an unmeasured ~100fs decay
  time / "5-10% of the padded grid" premise. Measurement finds the true
  support is ~4.15ps — **~40x longer** — consuming 76-86% of n_time_over at
  the one real production-shaped grid this repo has. There is no
  realistically-sized trange in this repository where the SiO2 kernel is
  actually short relative to the grid. (Asymptotically, as n_time_over≫M,
  the ratio does approach the theoretical ~2x — see the 32768/65536 rows —
  but nothing in this repo is anywhere near that regime, and chasing it
  would mean optimizing for a hypothetical workload rather than the one that
  exists.)

**Recommendation: do not implement.** This reverses BACKLOG's prior
"recommend" for Phase J.6(c); the lead should update that line to
"recommend against (measured 2026-07-25)" and cite this file. If some future
user config genuinely needs trange many times longer than SiO2's ~4ps
Raman decay time, this measurement should be redone at that config's actual
grid size before reconsidering.

### 2.6 What was reverted (nothing landed)

- `amalthea/benches/raman_short_kernel_bench.rs` — written, measured,
  **deleted** (did not clear the bar; S1.6/S5.1 precedent). Its
  `[[bench]]` entry in `Cargo.toml` reverted alongside it.
- Temporary `Instant`/`AtomicU64` profiling code in `amalthea/src/native.rs`
  (three statics near the top-of-file imports, two `Instant::now()` calls in
  `rhs_mode_avg_env`, two accumulate/report blocks) — fully reverted;
  `git diff amalthea/src/native.rs` is empty.
  `cargo build --release` + `cargo test` (71/71 pass) reconfirmed green
  after reverting.
- `test/test_native_raman_sio2.jl`'s temporary `:tmpprofile` tag and
  temporary `trange=16e-12` edit (used only to reach n_time_over=16384 for
  the second share measurement point) — both reverted; `git diff` empty.
- `Manifest.toml` — this worktree had none checked in (gitignored,
  per-worktree); copied from the sibling checkout to instantiate/build
  Julia deps for testing, left in place (gitignored, not staged).
- No other file touched.
```
