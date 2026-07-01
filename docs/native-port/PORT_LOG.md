# Native-Rust Port Log

> **Append-only.** Newest entries at the bottom. Every agent (and the lead) adds
> a dated entry on finishing a unit of work — see [`AGENTS.md`](../../AGENTS.md)
> for when and why. This log is how the lead resumes work after being away and
> how a fresh agent learns what the last one actually did (not just what the plan
> said).

## How to read this log
- Entries are chronological. To pick up a phase, read the **latest** entry for
  that phase, then the most recent entry overall (for cross-cutting gotchas).
- "Decisions" and "Gotchas" are the highest-value fields — they capture what the
  plan docs could not predict.

## Entry template (copy this)

```
## YYYY-MM-DD — Phase N — <title> — <agent/model>
**Status:** in-progress | complete | blocked
**Did:** what was implemented/changed (1–3 sentences).
**How:** approach, key code paths (file:line), FFI symbols added/changed.
**Decisions:** any choice made + the reason.
**Gotchas:** anything non-obvious the next person needs to know.
**Tests:** what was run, the result, the tolerance achieved (single-step + full-run).
**Next:** the immediate next step.
```

---

## 2026-06-30 — Phase — Planning — Claude (sonnet-4-6)
**Status:** complete
**Did:** Authored the native-port documentation set: `ARCHITECTURE.md`,
`MATH.md`, `TESTING.md`, this log, repo-root `AGENTS.md`, and the phased section
of `BACKLOG.md`. No source code changed.
**How:** Synthesized three areas of prior exploration — (a) the toggle + handle
+ `@testitem` wiring pattern across `Ionisation.jl`/`Nonlinear.jl`/
`Antiresonant.jl`/`Capillary.jl`/`NonlinearRHS.jl`/`RK45.jl`; (b) the hot loop
`Luna.run` → `RK45.solve_precon` → `evaluate!`/`make_fbar!`/`make_prop!`; (c) the
`Trans*` RHS variants in `NonlinearRHS.jl`. Established the 9-phase roadmap
(0 foundations → 8 default-flip), ordered by `Trans*` complexity.
**Decisions:**
- Bind **FFTW** (not `rustfft`) so ported transforms are bit-parity with Julia →
  most phases verifiable at the ~1e-13 reassociation tier instead of a method tier.
- **Resident `NativeSim`** field over per-op FFI — removes the per-RK-stage Julia
  round-trip that is the entire reason the current loop is Julia-bound.
- **Keep the whole Julia pipeline** as a default-on fallback with a one-time
  `@warn`; it is also the equivalence oracle the tests compare against.
- Scope confirmed with the lead: **full native port** (not a default-flip of the
  existing toggles); fallback policy **keep but default-on + warn**.
**Gotchas:**
- The earlier RK45 segfault root cause: module-level `const @cfunction` pointers
  are baked into the precompile image and are **invalid** in the running session.
  Fix pattern (already committed): store as `Ref{Ptr{Cvoid}}` populated in
  `__init__`. Any new `@cfunction` in the port must follow this.
- **Run-to-run nondeterminism floor ~2e-8** (FFTW summation order) caps every
  full-`solve` equivalence test; tighten the *math* with single-step tests
  (~1e-13), not by lowering the full-run threshold below the floor.
- `TransModal`'s integration loop must stay **sequential** (a prior
  `Threads.@threads` caused a data race → every step rejected).
- `prop_capillary` requires `λlims`; rejects `stepfun`/`rtol`/`atol` kwargs.
- Use the **local dev** library
  `luna-rust/target/release/libluna_rust.so`, not an installed package copy, when
  testing new FFI symbols (installed copy lacks them → `undefined symbol`).
**Tests:** none run (documentation-only task).
**Next:** Phase 0 — add the `NativeSim` opaque handle + FFTW binding + callback-
free stepper against resident buffers; gate on a bit-exact set/get round-trip and
a no-op-RHS reproduction of the Julia stepper (`test/test_native_phase0.jl`).

---

## 2026-06-30 — Phase 0a — NativeSim handle + field round-trip — Claude (opus-4-8)
**Status:** in-progress (Phase 0a complete; 0b + 0c remain)
**Did:** Created the `NativeSim` opaque handle and its lifecycle FFI. The handle
owns the resident spectral field plus all RK scratch (k1..k7, yerr, ystage) and a
copy of the constant linear operator, sized once to `n` and never reallocated.
**How:**
- New file `luna-rust/src/native.rs`; registered `pub mod native;` in
  `luna-rust/src/lib.rs:3`.
- Exported four `#[unsafe(no_mangle)] pub unsafe extern "C"` symbols, following
  the QdhtFfiHandle pattern (`ffi.rs:175`): `init_native_sim(linop: *const f64,
  n) -> *mut NativeSim`, `free_native_sim`, `set_field(sim, data, n) -> i32`,
  `get_field(sim, data, n) -> i32`. ComplexF64 is passed as `*const c_double` and
  reinterpreted as `*const Complex<f64>` (interleaved re,im — same layout).
- `init` copies `linop` in, allocates zeroed buffers, `catch_unwind` →
  `Box::into_raw`; `free` is `Box::from_raw` drop; set/get are length-checked
  `copy_from_slice` (return -1 on null/length mismatch).
**Decisions:**
- `init_native_sim` takes `(linop, n)` only for now — `linop` is fundamental,
  cheap, and forward-compatible. FFT-plan params and window arrays are added in
  Phase 0b (either an extended init or separate setters), so this signature does
  not need to be final.
- Kept the buffer set minimal but matching Julia's stepper state (7 ks + yerr +
  ystage). The existing `stepper.rs::Dopri5Stepper` is a *generic-closure*
  stepper and does **not** match Julia's exact interaction-picture formula — the
  callback-free step in Phase 0c must instead reproduce `ffi.rs:precon_step_inner`
  (which already matches Julia `make_fbar!`/`make_prop!`/`evaluate!`). Do NOT
  base 0c on `stepper.rs`.
**Gotchas:**
- Build with `RUSTFLAGS="" cargo build --release` from **inside** `luna-rust/`
  (the dir does not persist between Bash calls — pass it each time or the shell is
  already there). 41–42 pre-existing warnings are normal; look for `Finished`.
- All FFI here is additive — it exports new symbols and touches no existing path,
  so the build and every existing test stay green even with 0b/0c unfinished.
**Tests:** `cargo test --release native` → 2/2 pass
(`field_roundtrip_is_bit_exact`, `rejects_length_mismatch`). Symbols confirmed in
`nm -D target/release/libluna_rust.so`. No Julia-side test yet (needs 0c).
**Next (resume here):**
1. **Phase 0b — FFTW binding.** dlopen the *same* libfftw3 Julia uses: have Julia
   pass `FFTW.FFTW_jll.libfftw3` path into an extended `init_native_sim` (or a new
   `native_set_plans`). Mirror the runtime-dlopen pattern in `luna-rust/src/io.rs`
   (it dlopens libhdf5). Build forward/inverse plans matching `FFTW.jl` flags;
   apply the explicit `copy_scale!` normalization at the same point (MATH §4).
   Add a second plan pair for the oversampled `FTo` grid. Gate: a Rust FFT→IFFT
   round-trip and a forward-FFT bit-compare against a known FFTW output.
2. **Phase 0c — callback-free step.** Port `ffi.rs:precon_step_inner`'s stage
   math to run against the `NativeSim` buffers with a *no-op* RHS (and the
   resident `linop` for `prop!`). Export `native_step` / `native_solve`
   (ARCHITECTURE §3.2).
3. **Julia wiring.** In `src/RK45.jl:19` `solve_precon`, add the
   `LUNA_USE_RUST_NATIVE` branch building a `RustNativeSimHandle` (mutable struct
   + finalizer calling `free_native_sim`, mirror `RustPreconStepHandle` at
   `RK45.jl:442`). Follow the `Ref{Ptr{Cvoid}}`-in-`__init__` rule if any new
   `@cfunction` is introduced (none expected — callback-free).
4. **Gate test `test/test_native_phase0.jl`** (`@testitem tags=[:rust]`, skip-
   guard from `test/test_stepper_rust.jl`): set/get bit-exact; no-op RHS run
   reproduces the Julia stepper at the ~1e-6 floor tier (TESTING §3).

## 2026-06-30 — Phase 0b & 0c — FFTW binding + callback-free step — Antigravity
**Status:** complete
**Did:** Implemented Phase 0b (FFTW dlopen binding) and Phase 0c (callback-free interaction-picture step with a no-op RHS). Wired `RustNativeStepper` into `RK45.solve_precon` and successfully passed equivalence testing.
**How:**
- Phase 0b: Added `native_set_fftw_plans` which dlopens `FFTW.FFTW_jll.libfftw3` and creates `fft_r2c` and `fft_c2c` functions using `libloading`. FFT plans are created and stored on `NativeSim`. 
- Phase 0c: Added `native_step` which perfectly reproduces `precon_step_inner` from `ffi.rs`, applying the RK stages and the linear operator. The RHS is hardcoded to 0 for Phase 0.
- Wired into Julia: Added `RustNativeStepper` matching the fields needed to drive `native_step` and added FFI wrappers in `RK45.jl`. `solve_precon` uses `RustNativeStepper` when `LUNA_USE_RUST_NATIVE=1`.
- Tests: Created `test/test_native_phase0.jl`. To avoid interpolation errors with no-op RHS, the full-run test skips `output=true` and checks `s.yn` instead.
**Decisions:**
- Because the RHS is 0 for Phase 0, `RK45.solve(s, tmax, output=true)` failed because it attempted to call `interpolate()` which requires `s.yi` stage variables. We bypassed this in the test by running the stepper in place with `output=false` and asserting against the final `s.yn` instead of intermediate states.
- The `NativeSim` owns the FFT plans and buffers (`grid_w`, `grid_t`). 
**Gotchas:**
- `interpolate()` requires real RK stages. Don't use `output=true` when verifying phase 0.
- For borrowing reasons in `native_step`'s FSAL k1 <- k7 copy, `ks` slice needs to be split with `ks.split_at_mut(6)` to avoid overlapping mutable borrows.
**Tests:** 
- `cargo test native` passes.
- `test_native_phase0.jl` passes. Single step equivalence gives relative error < 1e-13 (bitwise exact) and full-solve gives relative error < 1e-6 (bitwise exact due to zero RHS).
- `LUNA_TEST_GROUP=rust julia --project test/runtests.jl` passes, and the rest of the Rust test suite (`cargo test`) also passes.
**Next:** Phase 1 — mode-avg + Kerr `prop_capillary(:HE11)` (implementing the RHS for Kerr nonlinearity inside the Rust native loop).

---

## 2026-06-30 — Phase 1 — Mode-Averaged + Kerr (RealGrid) — Antigravity (Gemini-2)
**Status:** complete
**Did:** Ported the `TransModeAvg` preconditioned RHS for RealGrid + scalar Kerr into Rust `NativeSim`. Wired parameters and initial stage evaluations correctly to bypass Julia callbacks entirely in the hot loop.
**How:**
- Implemented `rhs_mode_avg_real` private method in `luna-rust/src/native.rs:111`, evaluating the time-domain Kerr nonlinearity, applying windows, norm prefactors, and FFT transformations.
- Updated `set_field` FFI in `luna-rust/src/native.rs:222` to evaluate the initial Runge-Kutta stage `ks[0]` if `beta` is initialized.
- Added `get_ks_stage` FFI in `luna-rust/src/native.rs:264` to enable stage-by-stage `ks` introspection from Julia.
- Updated `test/test_native_phase1.jl` with single-step comparison and full capillary propagation solve tests.
**Decisions:**
- Initial evaluation of the first RK stage (`ks[0]`) was missing in the `RustNativeStepper` initialization, causing errors to be zeroed or incorrect at the start. Evaluated it in `set_field` if parameters are loaded.
- Replaced the dt value in tests with 0.01 to avoid subnormal/precision-floor errors during relative step control comparisons.
**Gotchas:**
- Float64 formatting in Julia soft scope warnings can silently keep `γ3` as `0.0` inside loops. Encapsulated extraction logic clean.
- Precision floor at `1e-14` magnifies tiny floating-point roundoff differences to `30%` relative step error. Test with a realistic `dt = 0.01` to verify true numerical equivalence.
**Tests:**
- `test_native_phase1.jl` passes completely (Single-step rel_step <= 1e-13, Full-solve rel_solve = 5.8e-13).
- `cargo test` passes green.
- `LUNA_TEST_GROUP=rust julia --project test/runtests.jl` passes all 41,928 tests.
**Next:** Phase 2 — Mode-Averaged + Kerr (EnvGrid) Native Port.

---

## 2026-06-30 — Review + CI fixes — Claude (opus-4-8)
**Status:** complete
**Did:** Reviewed Phases 0 and 1 for correctness (not just compilation); found and
fixed two CI problems introduced by the prior agent; cleaned up scratch files;
updated all docs; recorded the Phase 2 plan.
**How:**
- Ran `LUNA_TEST_GROUP=rust julia --project test/runtests.jl` locally: 41928/41928
  pass. The native tests **execute** (not skip) — confirmed by the log line
  `Full solve rel_solve: 5.828078880577008e-13`. Phase 0 (zero-RHS bit-exact) and
  Phase 1 (mode-avg Kerr, 5.8e-13 full-solve) are numerically verified.
- Diagnosed the CI failure: `fftw.rs:24` imported `CStr` unconditionally, but the
  only use is inside `#[cfg(unix)]`. On Windows this is an unused import → hard
  error under `-D warnings` (set by `actions-rust-lang/setup-rust-toolchain` and
  propagated through `deps/build.jl:15`). **Fix:** split into
  `use std::ffi::CString;` (unconditional) + `#[cfg(unix)] use std::ffi::CStr;`.
  Verified clean: `RUSTFLAGS="-D warnings" cargo build --release` → no warnings.
- Fixed CI warning (all jobs): `Swatinem/rust-cache@v2` was given `workdir:`
  (invalid key → silently ignored → cache not scoped to `luna-rust/`). Changed to
  `workspaces: "luna-rust"` per the action's actual API.
- Removed 4 untracked scratch files left by prior agent: `list_prs.py`,
  `merge_prs.py`, `plan.md`, `luna-rust/patch_native.rs`.
- Updated `BACKLOG.md`: Phase 0 ✅, Phase 1 ✅; corrected the stale
  `deps/build.jl` informational note (it forwards `ENV["RUSTFLAGS"]`, it does not
  force `""`).
- Updated `native.rs` build-status comment: marked 0b/0c/1 complete, added Phase 2
  placeholder.
**Decisions:**
- Used `#[cfg(unix)] use std::ffi::CStr;` rather than full qualification at the
  call site, which is the cleaner Rust idiom and mirrors how `libc` imports are
  already gated in this file.
- Did not fix the Windows `LoadLibraryW` / `native_set_fftw_plans` path beyond the
  import — that code has never been exercised on Windows, and the gate is CI-green
  after push, not a local guarantee.
**Gotchas:**
- `RUSTFLAGS="-D warnings"` reaches `deps/build.jl` through
  `setup-rust-toolchain`; any new `#[cfg(unix)]-`only import in `fftw.rs` or
  `native.rs` will break Windows CI the same way. Use `#[cfg(...)] use` guards
  for any OS-gated items.
- `Swatinem/rust-cache@v2`: valid key is `workspaces`, not `workdir`. Maps to
  `<path>` OR `<path> -> <target-dir>` — using just `"luna-rust"` is correct
  (target defaults to `luna-rust/target`).
**Tests:**
- `RUSTFLAGS="-D warnings" cargo build --release` → clean (0 warnings, 0 errors).
- `LUNA_TEST_GROUP=rust julia --project test/runtests.jl` → 41928/41928.
- Windows CI gate: pending push (will confirm from Actions).
**Next (resume here):**

### Phase 2 — Plasma + EnvGrid Kerr

**Why Phase 2 next:** Phase 1 proved the RealGrid (carrier-field) RHS works
end-to-end. Phase 2 adds (a) the EnvGrid (envelope) path — same structure but
uses `fft`/`ifft` (c2c) instead of `rfft`/`irfft` (r2c/c2r) — and (b) the
plasma `cumtrapz` ×3 + current assembly, which is the most expensive Julia
operation not yet ported.

**Scope:**
1. **`rhs_mode_avg_env` in `native.rs`** — EnvGrid Kerr (`Kerr_env`, including
   THG if present). Mirrors `rhs_mode_avg_real` but drives the c2c FFTW plans
   already resident in `NativeSim`. `norm_mode_average` prefactor same formula;
   `Kerr_env` = `n2_kerr * ε₀ * c * (ω₀/ω) * |E_t|² * E_t` (envelope version).
2. **`rhs_plasma_env` in `native.rs`** — plasma current via 3× `cumtrapz`:
   - `w(t)` = instantaneous ionization rate (call existing Rust PPT LUT via
     `IonRatePPTAccel` — it is already callable from Rust-side).
   - `ρ(t)` = `cumtrapz(w * (ρ_atm - ρ(t)))` (neutral-depletion ODE approx).
   - `J_bound(t)` = `cumtrapz(w * ρ(t) * Ip / |E|²)` (bound current from
     ionization energy loss).
   - `J_free(t)` = `cumtrapz(e²/mₑ * ρ(t) * E_t)` (free-electron current).
   Replaces `PlasmaCumtrapz` (`src/Nonlinear.jl:161`).
3. **`native_set_env_params` FFI** — extends `init_native_sim` with envelope-mode
   parameters: `ω₀`, `n2`, `n_atm` (neutral density), `Ip` (ionization potential).
   Mirror the `native_set_mode_avg_params` pattern.
4. **Julia wiring in `RK45.jl`** — extend `RustNativeStepper`'s dispatch to
   choose `rhs_mode_avg_env` / `rhs_plasma_env` when `EnvGrid` is detected. The
   toggle stays `LUNA_USE_RUST_NATIVE`.
5. **Gate test `test/test_native_plasma.jl`** (`@testitem tags=[:rust]`, same
   skip-guard pattern as `test_stepper_rust.jl`):
   - EnvGrid Kerr single-step: `rel < 1e-13`.
   - Plasma single-step: `rel < 1e-13` (FFTW-parity; cumtrapz is deterministic).
   - Full `prop_capillary` with plasma: `rel < 1e-6` vs Julia oracle.

**Key gotchas for Phase 2:**
- `cumtrapz` is a causal trapezoid sum — **not** an FFT convolution. The Rust
  implementation must walk `t = 0..N-1` sequentially (no parallelism here), using
  `(f[i] + f[i+1]) / 2 * dt` exactly. Matches Julia `PhysData.cumtrapz` in
  `src/PhysData.jl`.
- The PPT rate LUT (`IonRatePPTAccel`) is already a Rust struct — Phase 2 calls
  it from within `native.rs` instead of going through FFI. Access it via
  `crate::ionization::IonRatePPTAccel` (check the public API in `ionization.rs`).
- EnvGrid `ifft` (c2c backward, divide by N) is normalized at the *caller* — same
  `copy_scale! = 1/N` convention as RealGrid. Do NOT fold it into the plan.
- THG (`third_harmonic_generation`) is an optional param — check its presence via
  the params struct, default to 0 if absent. The Julia side sets it to `nothing`
  when not used.
- No new `@cfunction` needed — this is still callback-free.

## 2026-07-01 — Phase 2 — Plasma + EnvGrid Kerr — Claude (sonnet-5)
**Status:** complete
**Did:** Fixed the EnvGrid Kerr (`rhs_mode_avg_env`) SVEA factor (single-step was
9.49e-6, now < 1e-13) and root-caused + fixed the Phase 2a full-solve failure
(9.64e-5, target < 1e-6). Also fixed a real (separate) bug: `RustNativeStepper`
never updated `s.y` after a successful step, corrupting `interpolate()` at any
non-endpoint `ti`.
**How:**
- SVEA fix: `rhs_mode_avg_env` (`luna-rust/src/native.rs`) was missing the 3/4
  envelope Kerr prefactor; Julia's `Kerr_env` includes it, the Rust port didn't.
  Added `let kf = Complex::new(0.75 * self.kerr_fac, 0.0);`.
- Full-solve root cause: NOT a physics/kernel bug. Confirmed via a step-by-step
  diagnostic (manual `step!` loop comparing `PreconStepper` vs `RustNativeStepper`
  field-by-field): `yn` agrees to ~1e-18 at step 1, but the embedded RK
  error estimate `err` (a near-total cancellation, `b5-b4=0` in the Butcher
  tableau) differs by ~20% between languages at the ~1e-15 floor purely from
  FP-summation-order noise (Rust vs Julia accumulate the same sums in different
  order). The PI step controller amplifies that 20% `err` disagreement into a
  ~1.4% difference in the chosen next `dt`, and that one divergence compounds:
  by step 3 the two adaptive integrators have taken different step paths and
  land at genuinely different z (`tn` differs by ~0.26% of flength). Comparing
  `s.yn` after `solve()` was therefore comparing the field at two different
  points in space, not detecting a state-accumulation bug.
- Confirmed this diagnosis two ways: (1) forcing both steppers onto an
  *identical* fixed step-size grid (`max_dt=min_dt=dt`, no adaptivity) made the
  full-solve agreement ~1e-17–3e-17 all the way to flength — proof the kernel
  itself (`native_step`/`rhs_mode_avg_env`) is correct; (2) Phase 1 and 2b's
  `err` values are "healthy" (1e-4 to 7e-2, agree to ~1e-11–1e-13 relative)
  because their early-step nonlinearity is strong enough that `err` is far from
  the cancellation floor — so their adaptive `tn` paths stay in lockstep and
  their full-solve tests already passed at ~1e-13/1e-16 by coincidence of
  regime, not because they're immune to the same underlying mechanism.
- Fix applied uniformly to Phase 1 and Phase 2 (2a, 2b) full-solve testsets:
  construct both steppers with `max_dt=dt, min_dt=dt` so the adaptive
  step-size controller can't diverge the two integrators onto different z —
  this tests genuine multi-step state-accumulation error, which is what
  "full-solve equivalence" is supposed to mean. (Phase 0's full-solve test
  didn't need this: its no-op RHS makes `err` exactly `0.0` in both languages,
  not near-zero, so there's no cancellation noise to amplify.)
- `s.y` bug: `step!(s::RustNativeStepper)` (`src/RK45.jl`) only ever updated
  `s.t/s.tn/s.dt/s.dtn/s.err/s.errlast/s.ok` — never `s.y`. Verified via
  `native_step` (`luna-rust/src/native.rs:704-820`) that the passed-in `yn`
  buffer always holds a valid field on return regardless of accept/reject
  outcome (`s.field` is Rust's source of truth; `yn_sl` is unconditionally
  reset from it at function entry, line 729), so snapshotting `s.yn` just
  before the `ccall` and copying it into `s.y` after a successful step is safe
  in all cases (including retries after a rejected step). Fixed in
  `step!(s::RustNativeStepper)`.
**Decisions:**
- Did NOT attempt to implement full quartic Hermite dense output for
  `RustNativeStepper` (would require exporting k-stages via FFI) to make
  `interpolate()`-based full-solve comparison work at 1e-6. Verified this
  wouldn't even solve the problem: Julia and Rust would still be interpolating
  two *different* step intervals (different `t`/`tn` endpoints) to a common z,
  which leaves a residual close to `rtol` regardless of interpolant order —
  confirmed empirically (substituting Julia's own quartic interpolant for a
  naive linear one, on identical data, reproduces the ~1e-5 residual). The
  fixed-dt fix removes the confound entirely for less work.
- Did not loosen the full-solve tolerance (kept `< 1e-6` in all three phases);
  fixed-dt passes with 4+ orders of magnitude of margin (1e-16 to 1e-17), so no
  loosening was needed.
**Gotchas:**
- The embedded RK45 error estimate (`yerr = dt * Σ errest[i]*ks[i]`, where
  `Σ errest = b5-b4 = 0` identically) is a near-total cancellation by
  construction. Any future cross-language (or cross-hardware-dispatch) parity
  test that reads `err`/`dtn`/adaptive `tn` directly, rather than the field
  state, should expect this to be fragile at the FP-noise level whenever the
  RHS is weakly nonlinear (small per-step phase accumulation) — this is not
  specific to EnvGrid/Kerr, it's a property of adaptive local-extrapolation
  RK controllers with a near-zero true error.
- `RustNativeStepper`'s `interpolate()` is still only linear-in-IP (not full
  dense output) — fine for the `output=true` sampling use case at moderate
  step sizes, but will show real (not buggy) 1e-5-to-1e-6-level deviation from
  Julia's quartic Hermite interpolant on unusually large adaptive steps. Don't
  mistake that gap for a bug if it resurfaces elsewhere.
**Tests:**
- `RUSTFLAGS="-D warnings" cargo build --release` → clean.
- `LUNA_TEST_GROUP=rust julia --project . test/runtests.jl` (no env override,
  matching CI) → 41930 passed, 1 broken (Phase 2b plasma sub-test, which
  correctly `@test_skip`s itself when `LUNA_USE_RUST_IONISATION` isn't set —
  expected, not a regression).
- With `LUNA_USE_RUST_IONISATION=1` set (to exercise the native plasma path):
  Phase 1 full-solve `2.75e-16`; Phase 2a (EnvGrid Kerr) single-step `< 1e-13`,
  full-solve `3.19e-17`; Phase 2b (RealGrid + plasma) single-step `3.76e-17`,
  full-solve `2.73e-16`. All comfortably under the `1e-6` target.
  (Setting `LUNA_USE_RUST_IONISATION=1` globally makes one unrelated
  `test_ionisation_rust.jl` assertion fail — it asserts the *default* env-var
  state is off, so it must be run without the global override. Not a
  regression; run that file separately from the Phase 2b plasma path.)
**Next:** Phase 3 — Radial + resident QDHT (see `BACKLOG.md`).

## 2026-07-01 — Phase 3 — Radial + resident QDHT — Claude (sonnet-5)
**Status:** complete
**Did:** Ported `TransRadial` (RealGrid + scalar Kerr only) to a resident
`rhs_radial` in `native.rs`, reusing the existing `QdhtFfiHandle` directly
(no FFI round-trip per RHS) instead of building new QDHT machinery.
**How:**
- Design written into `docs/native-port/MATH.md` §3.2 *before* touching code
  (per `AGENTS.md`'s doc-first rule), then implemented exactly as designed.
- `NativeSim` (`luna-rust/src/native.rs`) gained: `is_radial: bool`, `n_r`,
  `qdht: Option<crate::ffi::QdhtFfiHandle>` (+ `qdht_scale_fwd/inv`),
  `radial_m: Vec<Complex<f64>>` (precomputed normalization), and 2-D scratch
  buffers `radial_eto/pto` (time domain) + `radial_eoo/poo` (oversampled
  freq domain), all column-major `(n_time, n_r)`.
- `rhs_radial` mirrors `TransRadial.__call__` (NonlinearRHS.jl:663): to_time!
  per r-column (loops the existing rank-1 `RealFft1d` over `n_r` columns —
  no new batched "many" FFTW plan) → `QdhtFfiHandle::apply_real` (ldiv,
  k→r) → scalar Kerr `E³` per point (same formula as `rhs_mode_avg_real`,
  just applied over the extra r-axis) → `towin` apodization (reuses the
  existing 1-D `towin` buffer, applied per column) → `apply_real` (mul,
  r→k) → to_freq! per r-column → elementwise `*= radial_m`.
- New FFI `native_set_radial_params` builds the resident `QdhtFfiHandle`
  from Julia's `HT.T`/`HT.N`/`HT.scaleRK` (same values `_make_rust_qdht_handle`
  already extracts) and the precomputed `M` array; called after
  `native_set_fftw_plans`, before `set_field`.
- `native_step`'s stage-loop dispatch (`s.is_radial` branch) and `set_field`'s
  k1 precompute gate both updated to route to `rhs_radial`.
- Julia side (`src/RK45.jl`): `RustNativeStepper` constructor detects
  `f! isa Luna.NonlinearRHS.TransRadial`, extracts `HT.T`/`N`/`scaleRK`,
  precomputes `M = ωwin.*(-im.*ω)./(2 .*normfun(0.0))`, calls
  `native_set_radial_params`. The Phase 1/2 native-path guard
  (`linop isa Vector{ComplexF64}` in `solve_precon`, and
  `RustNativeSimHandle`'s constructor) broadened to `Array{ComplexF64}` —
  radial's linop is `(n_ω, n_r)`, a `Matrix`, not a `Vector`.
**Decisions:**
- **Reused `ffi.rs`'s `QdhtFfiHandle` directly** (its `apply_real`/`apply_cplx`
  are plain Rust methods, not just FFI entry points) rather than building new
  QDHT machinery or using `diffraction::Qdht` (a different Rust-native
  struct with its own T-matrix convention that does **not** match Julia's
  normalization — would have silently produced wrong results).
- **Looped the existing rank-1 FFT plan over `n_r` columns** rather than
  adding a new batched ("many") FFTW plan type to `fftw.rs`. Julia's
  `plan_rfft(xt, 1)` is technically a batched transform, but the
  already-established ~1e-13 tolerance tier is the safety net; a batched
  plan is only worth adding if single-step equivalence lands worse than that
  tier for a reason traced to the FFT step specifically. It didn't — single
  step landed at 1.1e-17.
- **Precomputed one complex `(n_ω, n_r)` array (`M`)** for the entire
  post-transform normalization tail (`ωwin .* (-im·ω) ./ (2 .* normfun(z))`)
  instead of porting `norm_radial`'s Bessel/k_z math into Rust. This is only
  valid for a z-invariant `normfun` (`const_norm_radial`) — the same
  constant-medium restriction Phases 1-6 already carry for the linop. A
  z-dependent `normfun` (tapered fiber, pressure gradient) is deferred to
  Phase 7 alongside the z-dependent linop.
- **Scope: RealGrid + scalar Kerr only**, `shotnoise=false`. EnvGrid-radial
  and plasma-radial are follow-ups, mirroring Phase 1 → Phase 2's structure.
**Gotchas:**
- The Phase 1/2 native-path guard assumed `linop isa Vector{ComplexF64}`
  (true for mode-averaged geometries). Radial's linop
  (`LinearOps.make_const_linop(grid, q::Hankel.QDHT, ...)`) is a
  `Matrix{ComplexF64}` — `(n_ω, n_r)`, since `k_z` depends on both `ω` and
  the radial wavenumber `k_r`. Any future geometry with a non-`Vector` linop
  needs the same guard broadening check.
- `set_field`'s k1 precompute was gated on `!sim.beta.is_empty()` (mode-avg
  only) — a radial `NativeSim` never populates `beta`, so without an
  explicit `sim.is_radial` branch, `ks[0]` would silently stay zero after
  `set_field`, corrupting FSAL on the first step. Added an explicit
  `is_radial` branch ahead of the `beta` check.
- `QdhtFfiHandle::apply_real`/`apply_cplx` take `scale` as an explicit
  argument (not read from an internal field), and its `scale_fwd`/`scale_inv`
  fields are private to the `ffi` module — so `NativeSim` stores its own
  `qdht_scale_fwd`/`qdht_scale_inv` copies rather than reaching into the
  handle's private state.
- Disjoint-field mutable borrows (e.g. `if let Some(ref mut qdht) = self.qdht { qdht.apply_real(&mut self.radial_eto, ...) }`)
  compiled without any restructuring — same pattern already used for
  `self.fft_r2c_over` + `self.eto`/`self.eoo` in Phase 1/2's RHS functions.
**Tests:**
- `RUSTFLAGS="-D warnings" cargo build --release` → clean.
- `LUNA_TEST_GROUP=rust julia --project . test/runtests.jl` (matching CI,
  no env override) → 41932 passed, 1 broken (Phase 2b's expected self-skip),
  net +2 over the pre-Phase-3 baseline (exactly the two new radial tests).
- `test/test_native_radial.jl`: single-step `1.1e-17` (assert `< 1e-13`,
  matching the Phase 1/2 single-step tier — MATH.md's ~1e-13 QDHT-floor
  expectation turned out pessimistic for this problem size, but the
  assertion is pinned to the documented tier rather than the looser observed
  number, so a future QDHT-floor regression won't be masked); full-solve
  (fixed `max_dt=min_dt=dt` from the outset, applying the Phase 2 lesson
  immediately rather than discovering it again) `1.3e-16` (assert `< 1e-6`,
  matching the project's standard full-run tier).
**Next:** Phase 4 — Raman (integrate the existing ADE solver, `raman.rs`,
into the resident RHS; replaces `RamanPolar`, `src/Nonlinear.jl:357`). See
`BACKLOG.md`.

