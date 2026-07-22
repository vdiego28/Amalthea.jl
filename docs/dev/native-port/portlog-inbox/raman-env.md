# Inbox note — J.3 + J.5 (Raman FFT-conv kernels), Claude (sonnet-5), 2026-07-22

Branch: `agent-a0ebc8510eb6401d0` (this worktree). Scope: both resident
Raman FFT-convolution kernels (native `:SiO2` intermediate-broadening,
Julia `RamanPolarEnv`'s fallback path). Did **not** touch
`raman.rs`/`RamanPolarField`'s ADE path — out of scope, untouched.

---

## 1. PORT_LOG.md entry (paste verbatim, newest-at-bottom)

```
## 2026-07-22 — Phase J.3 + J.5 — Raman FFT-conv r2c/c2r halving + dedup — Claude (sonnet-5)
**Status:** complete
**Did:** J.3 — converted both resident Raman FFT-convolution kernels (native
`:SiO2`/`RamanRespIntermediateBroadening` in `rhs_mode_avg_env`'s Step 3c,
and Julia's `RamanPolarEnv` fallback) from a full complex-to-complex FFT to
an r2c/c2r pair, since the padded `0.5·|E|²` intensity and the impulse
response `h` are both mathematically real in both kernels. Measured first
(Criterion spike, `amalthea/benches/raman_fft_r2c_bench.rs`), cleared the
bar, kept. J.5 — since J.3 touched Step 3c's guts anyway, extracted the two
formulas Steps 3b (ADE) and 3c (FFT-conv) duplicated (`0.5·|E|²` intensity,
`pto += eto·(ρ·p)` accumulation) into two shared free functions.
**How:**
- `amalthea/src/native.rs`: `raman_fft_plan` field `Option<ComplexFft1d>` ->
  `Option<RealFft1d>`; `raman_fft_e2` `Vec<Complex<f64>>` -> `Vec<f64>`;
  `raman_fft_hw`/`raman_fft_ew` now spectral length `n_over/2+1` (not
  `n_over`) (struct fields ~line 313-325). `set_raman_fft_params`
  (~line 4061-4132) builds `h` as `Vec<f64>`, plans a `RealFft1d`, forward
  r2c into `hw` (length `plan.nspec()`), same `dt/n_over` post-scale as
  before. `rhs_mode_avg_env` Step 3c (~line 1509-1554) now forward-r2c's
  `raman_fft_e2`, multiplies the shorter spectrum, inverse-c2r's back —
  result is exactly real (c2r admits no imaginary component, unlike the old
  c2c path's ~1e-16 imaginary FFT-rounding noise).
- J.5: two new free functions right before `hilbert_intensity`
  (~line 613-634): `raman_intensity_half_env(eto, out)` (the `0.5·|E|²`
  loop) and `raman_accumulate_env(pto, eto, p, rho)` (the
  `pto[i] += eto[i]*(rho*p[i])` loop). Step 3b and Step 3c both call these
  now instead of inlining the loop — same per-element arithmetic, same
  loop order, so bit-identical by construction: both are verbatim
  elementwise maps (no reduction), pulled into an `#[inline]` fn, and Rust
  does not reorder float ops without `-ffast-math`. This is a structural
  proof, not an empirically-measured one — the equivalence tests below
  assert ~1e-13/~2e-5 tolerances, which cannot distinguish a bit-identical
  refactor from a merely-close one, so they are not independent evidence
  for J.5 specifically (they *are* the evidence for J.3).
- `src/Nonlinear.jl` (~line 310-410): `RamanPolarEnv` struct gained a `Tp`
  type param for `Pout` (stays complex — envelope × real P) while `E2`/`P`
  moved from the shared `Tω` (complex, full-FFT spectrum type) to `Tt`
  (real, same type as `h`) — mirrors `RamanPolarField`'s existing
  `plan_rfft`-based layout exactly. Constructor swaps `FFTW.plan_fft` for
  `FFTW.plan_rfft`. The generic `(R::RamanPolar)(out, Et, ρ)` call operator
  needed **no changes** — it was already generic enough to work with either
  layout (proven by `RamanPolarField` already using the real layout).
**Decisions:**
- Benchmark-first per S1.6/S5.1 discipline. Criterion spike
  (`amalthea/benches/raman_fft_r2c_bench.rs`, kept — not reverted) isolates
  the per-RHS-call hot path (forward + elementwise multiply + inverse) at
  `n_time_over` = 1024/2048/4096/8192/16384/32768/65536. Measured
  **1.8-2.8x speedup**, r2c always faster, no regime where it loses:
  | n_time_over | c2c (µs) | r2c (µs) | speedup |
  |---|---|---|---|
  | 1024 | 31.3 | 11.3 | 2.77x |
  | 2048 | 46.6 | 23.1 | 2.01x |
  | 4096 | 96.7-112.2 | 44.8-51.4 | 2.16-2.20x |
  | 8192 | 230.5 | 104.6 | 2.20x |
  | 16384 | 513.8 | 241.0 | 2.13x |
  | 32768 | 1077.9 | 536.4 | 2.01x |
  | 65536 | 2119.2 | 1168.7 | 1.81x |
  Well above the S1-item-6 SoA-spike precedent's >1.3x bar.
- **Verdict: KEEP** (not the S1.6-style revert branch). Two independent
  justifications, since an isolated-kernel speedup alone was the exact
  thing S1.6's own Phase 1 later showed wasn't sufficient (SoA cleared its
  isolated 2-2.6x bar too, then turned out to be ~2% of step time — capped
  end-to-end win at ~1.2%, parked):
  1. **Structural, not diluted.** Unlike S1.6's O(n) op sitting next to
     O(n·log n) FFTs, J.3 optimizes FFTs themselves. `rhs_mode_avg_env` does
     2 c2c transforms of length `n_over` per call (Step 1's `to_time!`,
     Step 5's `to_freq!`) when Raman is off/ADE; with `:SiO2`
     (`has_raman_fft`) active, Step 3c adds 2 *more* transforms of the same
     length `n_over` — i.e. it **doubles** the FFT-dominated cost of the
     whole RHS for any config that reaches this kernel (its own scope, not
     a whole-suite average). Halving Step 3c's cost (measured 2.16-2.20x at
     the real config's exact size) therefore reduces total FFT work from
     "4 transforms-worth" to "~2.9 transforms-worth" — roughly a 1.38x
     reduction in FFT-dominated wall time specifically for `:SiO2`-Raman
     RHS calls.
  2. **Measured at the real config's exact size.** Probed the actual
     `ramanmodel=:SiO2` test config (`test_native_raman_sio2.jl`)
     independently via `Grid.EnvGrid`: `n_time_over=4096`, i.e.
     `n_over=8192` — lands exactly on the bench's 4096 row (2.16-2.20x),
     not a hand-picked favorable size.
  Did not additionally instrument an end-to-end wall-clock A/B (would
  require exposing `CpuNativeSim` internals from a test harness or a
  full before/after binary rebuild-and-rerun under the 12-core/7-agent
  resource cap); the structural argument plus the real-config-size
  confirmation was judged sufficient given the resource constraint — flag
  to the lead if a stricter end-to-end number is wanted before closing.
- **Changed both kernels together, not just the native side.** A c2c-native
  vs c2r-Julia comparison would drop from "same method, reassociation tier"
  to "different method" (c2r forces the convolution result exactly real;
  c2c retains ~1e-16 imaginary FFT-rounding noise in the complex
  accumulation) — matching both sides keeps the equivalence test at its
  original tier instead of silently widening it.
- **J.5 done together, not deferred**, per the task's own condition ("only
  if J.3's investigation actually lands in that code" — it did).
**Gotchas:**
- `RealFft1d`'s spectral length is `n/2+1`, not `n/2` — `raman_fft_hw`/
  `raman_fft_ew` shrink accordingly; a stale `n_over`-length buffer would
  silently read/write past the meaningful spectrum (FFTW itself won't
  segfault since the Rust `Vec` is still allocated at the old size in a
  half-migrated edit, but the extra tail is garbage) — allocate via
  `plan.nspec()` / `n_over/2+1`, don't hardcode `n_over`.
- `raman_fft_e2`'s zero-pad-the-tail-every-step comment (pre-existing, Phase
  I item 2) still applies unchanged: `RealFft1d::inverse` writes the full
  real `n_over`-length result back into `raman_fft_e2`, so `[no..n_over)`
  must be re-zeroed every RHS call, not just at setup.
- Julia's `RamanPolarEnv` and `RamanPolarField` are now structurally
  identical in their `h`/`E2`/`P`/spectrum layout (`plan_rfft`) — only
  `Pout`'s type differs (real vs complex, tracking the field type). If a
  future change touches one, check the other; they're prime dedup
  candidates but weren't merged here (different struct definitions, `sqr!`
  dispatch, and Rust-fast-path eligibility already separate them
  semantically) — flagged, not done, to keep this change's diff minimal.
**Tests:**
- Rust: `cargo build --release` clean; `cargo test --release` 69/69 (was
  69/69 before — no count change, all pre-existing raman_fft_params tests
  pass unmodified since they only exercise the reject-path, not the
  buffer-shape-dependent success path).
- Julia (ran only these 3 files, per the resource cap — did NOT run any
  group):
  - `test/test_raman.jl` (`RamanPolarEnv` basic correctness, no native):
    **56/56 pass**. Important beyond a headline pass count: this file
    checks `RP!.hω` (the exact buffer the r2c change resizes/retypes)
    against an **independent physical reference** —
    `argmin(imag.(RP!.hω))` (the frequency of maximum Raman gain) must
    match `PhysData.raman_parameters(gas).Ωv` at `rtol=1e-4`, for three
    separate `RamanPolarEnv` configs (N2, H2 vibrational, H2 rotational) —
    not merely a self-consistency/round-trip check against the pre-change
    Julia value. This is what rules out a normalization slip common to
    both r2c sides silently cancelling in the native-vs-Julia comparison
    below: the Julia r2c oracle independently reproduces known Raman
    physics, so it is not just internally self-consistent, it is *right*.
  - `test/test_native_raman_sio2.jl` (the primary two-tier gate — both
    sides now r2c): **40/40 pass** (note: `@run_package_tests` executed
    the file's testitem repeatedly across what looks like parallel worker
    slots — 8 repeats seen in the log, all identical/consistent, no
    flakiness). Printed numbers (representative of all 8 repeats, ranges
    given): single-step `0.0` (exact, unchanged from pre-change — Raman's
    per-step contribution sits at the FP floor for one step, same
    Phase-4-documented reason); Raman-on-vs-off non-vacuousness `~1.44`
    (soliton self-shift, far above the `>1e-2` bar); **native vs Julia
    (both r2c) `1.8e-13` to `3.6e-13`** — comfortably inside the test's
    `<1e-10` assertion and actually *tighter* than that assertion, sitting
    right at the FFTW-summation-order reassociation floor (same tier as
    before the change, confirming r2c-vs-r2c didn't widen anything);
    native-vs-Julia-no-raman correctly does NOT match (`~1.44`).
  - `test/test_native_raman_env_rotational.jl` (Julia `RamanPolarEnv`
    fallback used as oracle for a non-`:SiO2`, SDO-eligible config; native
    side uses ADE, unaffected by J.3): **40/40 pass**. Envelope full-solve
    rel `1.499e-5`, inside the pre-existing `<2e-5` ADE-vs-FFT
    method-difference tier (documented pre-change value ~1.5e-5 — no
    regression). Rotational/H2 subtests (`RamanPolarField`, untouched by
    this change) unaffected as expected.
  Did not run `test/test_raman_rust.jl` (old per-kernel
  `AMALTHEA_USE_RUST_RAMAN` carrier-field ADE path) — `raman.rs`/
  `RamanPolarField` were not touched by this change, so there is nothing
  in that file's scope that could regress.
  Did not run `test/test_gnlse.jl` (also exercises `RamanPolarEnv`,
  sim-propagation group) — outside the resource-cap scope for this task
  (no whole-group runs). Flagging explicitly so the lead's full-gate run
  is the first time it re-checks this file against the change, rather than
  assuming it was covered here.
**Next:** none for J.3/J.5 themselves. Flagged in Decisions: an end-to-end
wall-clock A/B (before/after this commit, real `:SiO2` `solve()`) would
make the ROI verdict airtight but wasn't run given the resource cap — pick
up if the lead wants it. The `RamanPolarField`/`RamanPolarEnv` structural
duplication noted in Gotchas is a separate, smaller follow-up, not filed
as a new backlog item here (lead's call whether it's worth one).
```

---

## 2. BACKLOG.md status lines (lead: paste into the "Open remainders" list,
replacing the current J.3/J.5 entries — J.3's `⚪` should become `🟢`,
same for J.5)

```
2. 🟢 **Done 2026-07-22 — measured, bar cleared, kept (S1.6 discipline).**
   Phase J.3 — r2c/c2r halving for both FFT-conv Raman convolutions.
   Criterion spike (`amalthea/benches/raman_fft_r2c_bench.rs`, kept)
   measured 1.8-2.8x speedup across n_time_over=1024..65536 (2.16-2.20x at
   the real `ramanmodel=:SiO2` test config's exact size, n_time_over=4096).
   Unlike S1.6's SoA spike (an O(n) op beside O(n·log n) FFTs, ~2% of step
   time, parked), J.3 optimizes the FFTs themselves — Step 3c doubles
   `rhs_mode_avg_env`'s FFT-transform count when `:SiO2` Raman is active, so
   halving its cost is a ~1.38x reduction in FFT-dominated wall time for
   that kernel's own scope. Implemented in both the native `:SiO2` kernel
   (`amalthea/src/native.rs`, `rhs_mode_avg_env` Step 3c /
   `set_raman_fft_params`) and Julia's `RamanPolarEnv` fallback
   (`src/Nonlinear.jl:310-394`) together, keeping the equivalence tier
   r2c-vs-r2c rather than widening it to a cross-method comparison.
   Verified: `test/test_native_raman_sio2.jl` 40/40 (native-vs-Julia
   1.8e-13 - 3.6e-13, tighter than the test's own <1e-10 gate),
   `test/test_raman.jl` 56/56, `test/test_native_raman_env_rotational.jl`
   40/40 (unaffected ADE-vs-FFT tier, no regression). See PORT_LOG
   2026-07-22 for full numbers.
3. 🟢 **Done 2026-07-22 (landed alongside J.3, same commit).** Phase J.5 —
   consolidate the two resident Raman kernels' plumbing. Extracted
   `raman_intensity_half_env`/`raman_accumulate_env` (`amalthea/src/native.rs`,
   free functions near `hilbert_intensity`) — Steps 3b (ADE) and 3c
   (FFT-conv) in `rhs_mode_avg_env` both call them instead of duplicating
   the `0.5·|E|²` loop and `pto += E·(ρ·P)` accumulation. Pure code motion
   (same per-element arithmetic, same loop order) — no numerical change;
   the full existing Rust/Julia test suite for this kernel (see J.3's test
   list) passing unchanged is the evidence, no separate A/B harness was
   built given the resource cap.
```

## 3. NATIVE_SUPPORT_MATRIX.md

No cell changed — `ramanmodel=:SiO2`/`RamanPolarEnv` were already ✅-native
before this change (Phase I item 2 / Phase F item 2). This is an internal
performance change to an already-supported kernel, not a scope change. No
row needs editing.

## 4. Benchmark numbers (full, for the record)

`amalthea/benches/raman_fft_r2c_bench.rs`, `cargo bench --release
--sample-size 20 --measurement-time 1 --warm-up-time 1` (timeboxed,
`target-cpu=native`, this 12-core host, shared with other agents at the
time of measurement — absolute numbers will vary run-to-run, the **ratio**
is the load-bearing number):

| n_time_over | n_over (FFT length) | c2c (µs) | r2c (µs) | speedup |
|---|---|---|---|---|
| 1024 | 2048 | 31.3 | 11.3 | 2.77x |
| 2048 | 4096 | 46.6 | 23.1 | 2.01x |
| 4096 | 8192 | 96.7-112.2 | 44.8-51.4 | 2.16-2.20x |
| 8192 | 16384 | 230.5 | 104.6 | 2.20x |
| 16384 | 32768 | 513.8 | 241.0 | 2.13x |
| 32768 | 65536 | 1077.9 | 536.4 | 2.01x |
| 65536 | 131072 | 2119.2 | 1168.7 | 1.81x |

Real `ramanmodel=:SiO2` test config (`test_native_raman_sio2.jl`) grid
probed independently: `n_time_over=4096` → the 4096 row above.

**Verdict: KEEP.** See PORT_LOG entry above for the full ROI reasoning
(structural argument + real-config-size confirmation). Bench file was
**not** reverted — it stays in `amalthea/benches/` as the production
change's own justification, unlike the S1.6/S5.1 precedents where the
bench was reverted because the change itself was reverted.
