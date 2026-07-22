# Vanilla-Luna issues found during the native-Rust port

This document catalogs problems, deficiencies, and non-obvious behaviors in
**vanilla Julia Luna** — the original upstream `LupoLab/Luna.jl` code, or this
fork's own pre-existing Julia-side code, independent of anything the Rust port
itself might get wrong — that were uncovered over the course of building and
verifying the native-Rust port (Phases A-J, `BACKLOG.md`) and the Suggestions
backlog (S1-S6). It excludes bugs that were purely artifacts of the Rust
implementation with no bearing on vanilla Luna's own correctness (e.g. a typo
in a new Rust FFI function) — those live in `PORT_LOG.md`/`BACKLOG.md` as
ordinary port work, not here.

For each finding: what the problem is, where it was found (file:line or
phase), and its status — **Fixed**, **Not fixed (open)**, or **Accepted
limitation** (a real, permanent characteristic that isn't going to be "fixed"
so much as documented).

## Summary table

| Issue | Category | Status |
|---|---|---|
| `Polarisation.ellipse` always returns θ=0 | 1. Correctness bug | Fixed |
| `SSHExec` missing `files=String[]` kwarg (scp loop dropped) | 1. Correctness bug | Fixed |
| `PhysData._safe_n` clamped below-resonance Sellmeier to n=1 instead of complex sqrt | 1. Correctness bug | Fixed |
| Native plasma polarization missing density factor (mode-avg/radial) | 1. Correctness bug (port-introduced, not vanilla) | Fixed — see note |
| `Modes.dispersion`'s adaptive-FD β1(z) has ~1e-12 relative error vs. the true derivative | 2. Numerical-accuracy shortcut | Accepted limitation (Rust deliberately diverges for accuracy) |
| PPT ionisation rate is a spline LUT, not a direct evaluation | 2. Numerical-accuracy shortcut | Not fixed (open, MATH.md §8.2 / SUGGESTIONS #16) |
| Embedded RK45 error estimate is a near-total cancellation (`b5-b4=0`) | 2. Numerical-accuracy shortcut (structural) | Accepted limitation; alternative recorded, not implemented |
| `RamanPolarEnv` uses full c2c FFT where the signal is real (no r2c halving) | 2. Numerical-accuracy shortcut (perf, not accuracy) | Not fixed (open, BACKLOG Phase J item 3) |
| `stepfun`'s per-step windowing silently dropped on the native path | 3. Port-of-vanilla-behavior bug | Fixed (Phase 8) |
| Native dense-output interpolation was linear, not Julia's quartic Hermite | 3. Port-of-vanilla-behavior bug | Fixed (Phase 8) |
| Unrecognized bare `f!` closures silently got zero nonlinearity | 3. Port-of-vanilla-behavior bug | Fixed (Phase 8) |
| Gas mixtures crashed with a raw `MethodError` instead of a graceful fallback | 3. Port-of-vanilla-behavior bug | Fixed (Phase 8) |
| `RamanPolarEnv` silently vanished (matched no `isa` branch, no error) | 3. Port-of-vanilla-behavior bug | Fixed (Phase 8) |
| Default `prop_capillary` (plasma on by default) silently never used the native path | 3. Port-of-vanilla-behavior bug | Fixed (Phase C) |
| Mode-averaged/radial shot noise (`Et_noise`) silently dropped by native path | 3. Port-of-vanilla-behavior bug | Fixed (Phase I item 1) |
| `:SiO2` intermediate-broadening Raman model not representable by native's SDO solver | 3. Port-of-vanilla-behavior gap | Fixed (Phase I item 2, second resident kernel) |
| FFTW `PATIENT` mode + `loadFFTwisdom`/`saveFFTwisdom` cause run-to-run step-path variability | 4. Non-determinism characteristic | Accepted limitation (pre-existing upstream behavior); newly *also* exposed on the native path (open design question) |
| Adaptive-path divergence amplifies tiny FP differences into different `dt` sequences | 4. Non-determinism characteristic | Accepted limitation; worked around in tests via fixed-step comparison |
| `PptIonizationRate::rate` can segfault far outside its fitted range | 5. Open/tracked problem | Not fixed (open, BACKLOG Phase J item 2) |
| BLAS-3 QDHT wiring is broken, disabled by default | 5. Open/tracked problem | Not fixed (open, BACKLOG S1 item 5) |
| Windows scan-queue `flock`/native FFTW-wisdom race under concurrent processes | 5. Open/tracked problem | Not fixed (open, newly found 2026-07-09) |

---

## 1. Correctness bugs in vanilla Julia Luna's own code

These are bugs in vanilla upstream Luna.jl (or this fork's pre-port Julia
code) that have nothing to do with the Rust port — found either during the
upstream-sync review or during math verification for the port, and fixed
directly in Julia source (so both the Julia and native-Rust paths benefit,
since the Rust path calls the same fixed Julia code for setup).

### `Polarisation.ellipse` always returned θ=0
**Found:** `BACKLOG.md` Phase A.1, upstream commit `0a52ffb` (PR #428).
**Problem:** the fork's `Polarisation.ellipse` computed the polarization
ellipse orientation as `angle(aL)/2`, which is identically zero regardless
of input — the correct formula is `θ = angle(Q + 1im*U)/2` (Stokes
parameters). This silently produced a wrong ellipse orientation for any
elliptically/circularly polarized field analysis.
**Status: Fixed.** Ported upstream's fix and its QWP-at-π/8 regression test
into `test/test_polarisation.jl`.

### `SSHExec` dropped auxiliary scp files
**Found:** `BACKLOG.md` Phase A.2, upstream commit `0fd7ac2` (PR #427).
**Problem:** the fork's version of `SSHExec` was missing the `files=String[]`
kwarg and the auxiliary-file `scp` loop present upstream — meaning any scan
script relying on auxiliary file transfer silently didn't transfer them.
**Status: Fixed.** Re-added the kwarg and loop, merged into the fork's
shell-escaped `Cmd`-array `runscan` (kept the existing shell-escaping rather
than reverting to upstream's raw string interpolation).

### `PhysData._safe_n` clamped below-resonance dispersion to n=1
**Found:** `BACKLOG.md` Phase B.1 (fork-vs-upstream review §3.1).
**Problem:** for roughly 20 glass Sellmeier models, `_safe_n` clamped any
negative `n²` (which occurs below an absorption resonance) to `n=1` instead
of taking the complex square root upstream does
(`sqrt(complex(n2))`/`sqrt(complex(n2)+1e-10im)` for the exactly-zero case).
This silently discarded the (physically meaningful, if evanescent/absorptive)
below-resonance refractive index behavior for any simulation probing deep-UV
or other below-resonance wavelengths for these glasses.
**Status: Fixed.** Restored complex-sqrt semantics; `test/test_physdata.jl`
pins BK7 at 70nm (n²≈-3.70, below resonance) against upstream's value.

### Native plasma polarization missing its density factor
**Found:** `BACKLOG.md` Phase I preamble, 2026-07-08, flagged 🔴🔴 ("critical").
**Problem:** technically a bug *introduced by this port* (in `native.rs`'s
`apply_plasma_real`/`apply_plasma_radial`), not a vanilla-Luna bug — included
here because it was a case of the Rust port failing to faithfully reproduce
vanilla Julia's own formula: Julia's `(Plas::PlasmaCumtrapz)(out, Et, ρ)` does
`out .+= ρ .* Plas.P` (`Nonlinear.jl:237-249`), but the native path added
`Plas.P` with no density multiplication at all (density implicitly 1 instead
of the real gas density, ~1e25 m⁻³) for every mode-averaged/radial plasma
simulation (PPT or ADK) since Phase C. Every existing equivalence test
happened to run at a field weak enough that ionization stayed near zero, so
"native matches Julia" was vacuously true.
**Status: Fixed** (2026-07-08). Added a `plasma_density` field, threaded
through both plasma FFI setters, multiplied into the polarization before
accumulation. Verified: full-solve error at a previously-untested strong-field
point dropped from ~3.9e-4 (the bug's signature) to ~3.8e-15 (bit-parity).
**Practical implication documented in BACKLOG.md:** any published result from
a native `prop_capillary`/`prop_gnlse` run with plasma enabled and a field
strong enough for ionization to matter, from Phase C until this fix, was
under-computing plasma self-defocusing/spectral broadening.

---

## 2. Numerical-accuracy shortcuts in vanilla Luna

Places where vanilla Luna's own math takes an approximation, LUT, or
finite-difference shortcut where a more exact form exists — some fixed with
better math in the Rust port (deliberately diverging from the Julia oracle),
some faithfully reproduced (matching vanilla's own approximation), and some
just recorded as future options.

### `Modes.dispersion`'s adaptive finite-difference β1(z)
**Found:** Phase 7, `docs/dev/native-port/BETA1_ANALYTIC.md`.
**Problem:** `β1(z) = Modes.dispersion(mode, 1, ω0; z=z)` calls
`Maths.derivative`, which uses `FiniteDifferences.central_fdm(7, 1, adapt=1)`
— a 7-point adaptive-step central difference. This has a small, real, and
repeatable (not random) discrepancy against the true derivative, measured at
~1e-12 relative (confirmed by re-running the same call with `BigFloat`
arguments, which removes the floating-point-rounding side of the
step-size/rounding-error tradeoff that limits `Float64` FD accuracy).
**Status: Accepted limitation — and deliberately NOT reproduced.** For the
z-dependent-linop phases (7, D.5), Rust computes β1(z) as an **exact closed
form** (4 constants, derived via a chain rule over `εco(ω;z) = γ(λ(ω))·dens(z)`,
differentiated once at setup using a `BigFloat` argument to `Maths.derivative`
— not hand-derived symbolics). This makes Rust's β1(z) *more accurate* than
Julia's own `dispersion` — the first (and, as documented, still only
deliberate) point in the port where Rust diverges from the Julia oracle on
purpose rather than reproducing it bit-for-bit. Two earlier LUT-based designs
(z-domain spline, density-domain spline) were tried and abandoned first
because a spline fit *validated against Julia's own noisy FD samples* cannot
converge past the samples' own inaccuracy — see `PORT_LOG.md`'s Phase 7
entries and `feedback_prefer_analytic_over_lut`. This forces full-solve
equivalence tests for z-dependent-linop configs onto a wider tolerance tier
(~1e-4 to ~1e-3) than every other phase (~1e-10 to ~1e-13), since the tiny
systematic β1 offset accumulates coherently over propagation length and
bandwidth — a documented, understood widening, not slack from a bug.

### PPT ionisation rate is a spline LUT, not a direct evaluation
**Found:** `docs/dev/native-port/MATH.md` §8.2 (recorded 2026-07-08), `ionization.rs`.
**Problem:** vanilla Luna's PPT ionisation rate (`IonRatePPTAccel`, and this
port's `PptIonizationRate`) is evaluated via a 1-D cubic B-spline LUT fitted
over `|E|`, not the closed-form PPT rate expression directly. This trades a
small interpolation error (and the fitted-range boundary problem below) for
speed.
**Status: Not fixed (open).** Recorded as an available "beyond-Luna" option
in `MATH.md` §8.2 / `SUGGESTIONS.md` item 16 — direct PPT evaluation would
also structurally fix the segfault-outside-fitted-range problem (see §5
below), but has not been implemented; would require its own
controlled-divergence verification against a ground truth, same discipline
as the β1 case above.

### Embedded RK45 error estimate is a near-total cancellation
**Found:** Phase 2, `PORT_LOG.md` 2026-07-01 entry; also `MATH.md` §8.1 /
`SUGGESTIONS.md` item 15.
**Problem:** the Dormand-Prince embedded error estimate
`yerr = dt·Σ errest[i]·ks[i]` has `Σ errest = b5-b4 = 0` identically — a
near-total cancellation by mathematical construction, not specific to this
codebase's implementation. When the RHS is weakly nonlinear (small per-step
phase accumulation), the true `err` sits near the floating-point noise floor,
so a ~1e-15 summation-order difference between two implementations (e.g.
Rust vs Julia, or two different hardware-dispatch code paths) can show up as
a ~20% *relative* disagreement in `err` — which the PI step controller then
amplifies into measurably different `dt` choices, sending two otherwise
mathematically-identical integrators down different step paths to different
z. This is a structural property of local-extrapolation RK controllers with
a near-zero true error, not a bug introduced by the port.
**Status: Accepted limitation.** Full-solve equivalence tests work around it
by constructing both steppers with `max_dt=min_dt=dt` (pinning the step-size
sequence so both integrators can't diverge onto different z, isolating
genuine multi-step accumulation error from adaptive-path divergence — see
`TESTING.md` §3). `MATH.md` §8.1 records an alternative (direct DP5(4)
embedded-error coefficients that avoid the cancellation at its root,
potentially dissolving the need for the fixed-step test discipline) as a
future option; not implemented.

### `RamanPolarEnv` uses a full complex FFT where the signal is real
**Found:** `BACKLOG.md` Phase J item 3, `Nonlinear.jl:327`.
**Problem:** the padded E² and the impulse response h are both mathematically
real in both the carrier-field and envelope Raman convolution paths. Julia's
`RamanPolarField` already exploits this via `plan_rfft` (r2c/c2r halving), but
`RamanPolarEnv` uses a full c2c `plan_fft` — and this port's native `:SiO2`
kernel faithfully mirrors that (matching parity exactly, not fixing it), so
both do double the necessary transform work.
**Status: Not fixed (open).** `BACKLOG.md` Phase J item 3, pairs with
`SUGGESTIONS.md` item 17 (short-kernel overlap-save convolution). Flagged as
a summation-order change requiring the fixed-step validation discipline, and
as something worth benchmarking first to confirm the convolution is actually
a measurable fraction of a step's cost before investing the work — not yet
done on either the Julia or the native side.

---

## 3. Bugs specific to porting vanilla Luna's real behavior into the resident native stepper

These are not bugs in vanilla Luna's *output* — Julia's own behavior was
always correct — but real, subtle things vanilla Luna does (in `Luna.run`'s
`stepfun`, in `Interface.jl`'s default configuration, in dense-output
interpolation) that the native port initially failed to faithfully replicate,
found only because making native the *default* (Phase 8) forced the full
general-purpose test suite — not just phase-specific native tests — through
the native path for the first time.

### `stepfun`'s per-step windowing silently dropped
**Found:** Phase 8, `PORT_LOG.md`/`CLAUDE.md` "non-obvious gotcha."
**Problem:** `Luna.run`'s `stepfun` callback applies the grid's frequency
window (`Eω .*= grid.ωwin`) and a time-domain window on every accepted step,
mutating the live state array in place. For `PreconStepper` this is free
(the mutated array *is* the live state). For `RustNativeStepper`,
`native_step` unconditionally overwrote the passed buffer from Rust's own
resident field at the top of every call — it never read back whatever Julia
had just written into it. Every `Luna.run`-driven simulation silently
dropped windowing on the native path, always, since Phase 1 — invisible
because every native-specific phase test called `solve()`/`step!()` directly,
bypassing `stepfun` entirely.
**Status: Fixed** (Phase 8). New `_native_field_resync!(s)` hook in
`RK45.jl`'s generic `solve` loop, called immediately after `stepfun`, pushes
the windowed field back into Rust via a new `native_resync_field` FFI —
deliberately *not* recomputing the FSAL stage-0 RHS, matching what
`PreconStepper` itself does (re-propagate the last stage linearly into the
new frame, don't re-evaluate the nonlinear RHS). Isolated via
`test_multimode.jl`'s "Radial" test: pure-Julia gave 0.043% mode-average-vs-
modal agreement, both-native gave 2.0%, before the fix.

### Native dense-output interpolation was linear, not quartic
**Found:** Phase 8, same entry.
**Problem:** `RustNativeStepper`'s dense output between accepted steps
(`interpolate`, used by any `saveN`/`MemoryOutput` config — i.e. nearly
every general-purpose test) was linear, a documented stopgap since Phase 0.
`PreconStepper`'s is a full quartic Hermite fit over all 7 RK stages
(`interpC`). This single gap explained nearly every remaining general-suite
failure (multimode, gradient, tapers, interface, output, linearprop,
full-freespace) at once, not eight separate bugs — isolated by comparing a
201-point interpolated output (matched Julia to only 1.77e-2) against the raw
final field (matched to 7.1e-15) for the same fixed-dt run.
**Status: Fixed** (Phase 8). Exported all 7 resident RK stages via
`get_ks_stage`; new `native_apply_prop` FFI re-expresses the polynomial
correction at query time; `interpolate` now ports the same `interpC` formula.
Verified: the same 201-point comparison went from 1.77e-2 to 4.9e-15.

### Unrecognized bare `f!` closures silently got zero nonlinearity
**Found:** Phase 8, item 1 of the four bugs found running the full suite.
**Problem:** `RustNativeStepper` gated its Kerr/plasma/Raman wiring on
`f! isa TransModeAvg` etc., but nothing rejected an `f!` matching none of the
known `Trans*` types (e.g. `test_rk45.jl`'s own raw RHS closures) — such a
config silently ran with **no** parameter wiring at all, i.e. pure linear
propagation with no nonlinearity and no error.
**Status: Fixed** (Phase 8). Any non-`nothing`, non-`Trans*` `f!` is now
rejected with `NativeIneligible` (falling back to Julia); `f! === nothing`
stays legal (the deliberate bare-stepper case Phase 0's own tests use).

### Gas mixtures crashed with a raw `MethodError`
**Found:** Phase 8, item 2 of the four.
**Problem:** `MarcatiliMode(a, (gas1,gas2), (p1,p2))` gives `densityfun(z)` a
per-species `Vector` return; native's mode-averaged setup assumed a scalar
density (`kerr_fac = density*ε₀*γ3`) and threw a raw `MethodError` at the FFI
boundary trying to coerce a `Vector{Float64}` into a `Float64` ccall argument
— a crash, not a graceful fallback.
**Status: Fixed** (Phase 8). `f!.densityfun(0.0) isa Real` is checked up
front; non-scalar density is rejected as `NativeIneligible`. (Gas mixtures
were later *also* natively supported for mode-averaged and radial geometries
— Phase F.4 / Phase I.4 — once the scalar-collapse math was verified valid;
this entry is specifically about the earlier crash-instead-of-fallback bug.)

### `RamanPolarEnv` silently vanished
**Found:** Phase 8, item 3 of the four.
**Problem:** the mode-averaged Raman-wiring loop only checked
`r isa RamanPolarField` (carrier-field); `RamanPolarEnv` (the response
`Interface.makeresponse` attaches for `EnvGrid`/`prop_gnlse`) matched none of
the loop's `isa` branches, so it fell through with no wiring and no error —
native ran Kerr-only, silently dropping Raman completely for every GNLSE
config. Found via `test_gnlse.jl`'s "Soliton shift" test, where the expected
self-frequency-shift was simply absent (off by ~1e15 rad/s), not subtly wrong.
**Status: Fixed** (Phase 8). A catch-all loop now rejects *any* response
object that matched none of the three known types (Kerr/plasma/Raman) as
`NativeIneligible` — closing this gap generally, not just for
`RamanPolarEnv`, and applied to the equivalent single-response checks in
radial/modal/free-space too.

### Default `prop_capillary` silently never used the native path
**Found:** Phase C, "fork-vs-upstream review §3.2."
**Problem:** `prop_capillary` defaults to `plasma = !envelope` — so every
default field-resolved run includes plasma. But native's plasma wiring
required `IonRatePPTAccel.rust_handle`, only built when
`AMALTHEA_USE_RUST_IONISATION=1` was explicitly set (default `"0"`). So the
out-of-the-box configuration (`AMALTHEA_USE_RUST_NATIVE=1` by default since
Phase 8, `AMALTHEA_USE_RUST_IONISATION=0` by default) threw `NativeIneligible`
and silently fell back to the Julia stepper for the single most common
use case — the native port's headline speedup never applied unless a user
knew to flip a second, unrelated-looking environment variable.
**Status: Fixed** (Phase C). `_make_rust_ionization_handle` now builds the
handle whenever the Rust library is present and *either* toggle is on.
Benchmarked: ~3.5x wall-time speedup (0.305s → 0.087s for 10 accepted steps)
on the exact out-of-the-box config, previously 0x.

### Mode-averaged/radial shot noise silently dropped
**Found:** `BACKLOG.md` Phase I item 1.
**Problem:** neither the mode-averaged nor radial setup blocks in
`RustNativeStepper` ever referenced `f!.Et_noise`, and no noise buffer was
ever passed to the native parameter setters — unlike the modal and
free-space blocks, which correctly guarded and fell back. Since
`shotnoise=true` (`:modified` model) is `Interface.jl`'s default, this
affected the single most common configuration in the package, silently, for
both `prop_capillary`'s and `prop_gnlse`'s default workloads (the GNLSE case
matters because `prop_gnlse` defaults to envelope + mode-averaged +
`shotnoise=true`).
**Status: Fixed** (2026-07-08). Fully ported to native (not just
guarded-and-fallback) for mode-averaged and radial, both RealGrid and
EnvGrid. Verified via dedicated equivalence + noisy-vs-clean regression
tests for each of the four combinations.

### `:SiO2` intermediate-broadening Raman model unrepresentable
**Found:** `BACKLOG.md` Phase I item 2.
**Problem:** `ramanmodel=:SiO2` (`Interface.jl:1077-1079`) returns a bare
`RamanRespIntermediateBroadening` (Gaussian-damped impulse response), not
wrapped in `CombinedRamanResponse`, so `flatten_sdo_oscillators` correctly
returned `nothing` — no finite sum of single-damped-oscillators can
reproduce a Gaussian envelope. This is a real representational gap, not a
missing wrapper: the native ADE Raman solver (built for SDO responses) has
no way to represent this response type at all.
**Status: Fixed** (2026-07-08), as a **second, independent** resident
kernel (not an SDO approximation, per the project's stated preference for
analytic methods over LUTs/approximations chasing a different method's
noise) — precomputes the padded impulse response's spectrum once at setup,
then does the same FFT-convolution math as Julia's own generic (non-SDO)
Raman path, moved resident. Verified with a triangulating test at a soliton
self-shift configuration with a large, unambiguous Raman effect.

---

## 4. Non-determinism / reproducibility characteristics inherent to vanilla Luna

Not bugs — real, permanent characteristics of how vanilla Luna is built that
users (and this port) need to know about, because they affect exact
run-to-run reproducibility even though final accuracy stays within the
requested tolerance.

### FFTW `PATIENT` planning mode + automatic wisdom persistence
**Found:** 2026-07-09, while investigating a 17-test discrepancy between
running the Julia test suite as one process vs. many.
**Problem:** `src/Luna.jl:13` sets Luna's **default** FFTW planning mode to
`FFTW.PATIENT` (not `ESTIMATE`), and `Utils.loadFFTwisdom()`/
`saveFFTwisdom()` (`src/Utils.jl:84,99`) are called around essentially every
high-level entry point (`prop_capillary`, `prop_gnlse`, etc. — a dozen+ call
sites in `src/Luna.jl`). `PATIENT` mode does empirical timing measurements to
pick the fastest FFT algorithm for a given size, and persists that choice to
a wisdom file that's imported on the next call — meaning plan selection (and
therefore the exact floating-point summation order of every FFT in the
simulation) can vary with machine load, thermal state, prior wisdom-file
content, or anything else that perturbs the timing measurements. Combined
with the RK45 near-cancellation issue (§2 above), this means vanilla Luna has
never guaranteed exact step-by-step reproducibility across runs or machines,
by design of its default settings — this has been true since long before
this port existed.
**Status: Accepted limitation** (pre-existing upstream characteristic; test
suites side-step it by forcing `set_fftw_mode(:estimate)`, which is
deterministic-by-size). **Newly relevant to this port:** S1 item 1
(2026-07-09) added FFTW wisdom import/export to the native-Rust path for
performance — before that, native plans were always fresh `ESTIMATE` plans
with no wisdom involved, so the native path was actually *more*
deterministic than vanilla Julia already was. Adding wisdom import brought the
native path toward parity with Julia's pre-existing behavior, at the cost of
losing that accidental extra determinism — flagged as an open design
question in `BACKLOG.md` (S1 item 1's "second risk" note). **Resolved
2026-07-09:** native wisdom persistence is now opt-in, default OFF
(`AMALTHEA_NATIVE_FFTW_WISDOM=1`; see `docs/dev/native-port/PLANS.md §1`),
restoring the native path's accidental extra determinism as the default —
opting in trades it back away for whatever (likely near-zero, since native
only ever plans with `ESTIMATE`) speed benefit accumulated wisdom provides.

### Adaptive-path divergence from FP summation-order noise
**Found:** Phase 2, `PORT_LOG.md` 2026-07-01; also documented in
`TESTING.md` §3.
**Problem:** covered in more detail under §2 above (the error-estimate
cancellation) — restated here because its *consequence* is a reproducibility
characteristic, not an accuracy one. Two mathematically-identical
integrators (even Julia-vs-Julia across two machines, or two compiler
versions) can take measurably different adaptive step paths from
sub-ULP-level summation-order differences, given a weakly-nonlinear-enough
RHS. Final accuracy stays within the requested `rtol`/`atol` (that is what
the adaptive controller guarantees), but exact step count and intermediate
`z` positions are not reproducible guarantees of vanilla Luna's design.
**Status: Accepted limitation.** Documented, not "fixed" (there is no fix
that doesn't change the fundamental adaptive-RK design); worked around in
this port's own tests via fixed-step (`max_dt=min_dt`) comparison.

---

## 5. Open / tracked problems, not yet fixed

Real, live gaps still sitting in `BACKLOG.md`, listed here for completeness
since they are relevant "vanilla Luna" (or vanilla-Luna-adjacent) issues even
though they haven't been resolved.

### `PptIonizationRate::rate` can segfault far outside its fitted range
**Found:** `BACKLOG.md` Phase J item 2 (promoted from a buried note in Phase
G.3).
**Problem:** repeatedly `step!`-ing far beyond `flength` with a too-large
fixed `dt` can grow the field until a PPT LUT query lands outside
`[e_min, e_max]`, and something in that path segfaults instead of hitting
the documented Phase B.2 clamp-to-`rate(e_max)` behavior. A rate query
should never be able to crash the process regardless of input.
**Status: Not fixed (open).** Needs reproduction via the
`benchmark_native_default.jl` overshoot scenario, a fix, and a Rust unit test
hammering `rate()` across ±1e10× the fitted range.

### BLAS-3 QDHT wiring is broken, disabled by default
**Found:** `BACKLOG.md` S1 item 5, wired 2026-07-07.
**Problem:** the Julia-side caller (`NonlinearRHS._init_rust_qdht_blas`) was
missing and has since been added, surfacing a real, previously-untested bug
in `blas.rs` involving `libblastrampoline`'s dynamic dispatch. Left disabled
by default pending a real fix (see `BACKLOG.md` for the specific mechanism).
**Status: Not fixed (open).**

### Concurrent-process races on shared native FFTW-wisdom / scan-lock files
**Found:** 2026-07-09, during test-gate redistribution.
**Problem:** related to the FFTW-wisdom characteristic in §4 — the native
wisdom file (`native_fftw_wisdom_$(threads)threads`) is a single shared path
across every process on a machine; concurrent processes importing/exporting
it simultaneously can race (best-effort, silently falls back to no-wisdom on
failure, never crashes, but can cause different processes in a concurrent
batch to get different plans purely from the race, not from anything
physically different about their workloads).
**Status: Fixed 2026-07-09.** On-disk wisdom persistence for the native path
is now opt-in, default OFF (`AMALTHEA_NATIVE_FFTW_WISDOM=1`; see
`docs/dev/native-port/PLANS.md §1` and `BACKLOG.md` S1 item 1). By
default no process reads or writes the shared wisdom file, so this race is
eliminated for the default configuration; it can still occur for anyone who
opts in and runs concurrent processes, which is documented as an accepted
tradeoff of the opt-in. The **determinism boundary** this leaves in place,
regardless of the toggle: native shares one process-global FFTW wisdom pool
with Julia's own `FFTW.jl` (same `dlopen`ed `libfftw3`), so exact step-path
reproducibility is only guaranteed with one fresh process per simulation —
which is how `scans.rs`'s `ScanQueue` already runs scans.
**Empirically confirmed safe to parallelize (2026-07-09):** with the
default-off fix in place, the `rust` group's 44 files were run twice as
independent concurrent processes (max 6 at once). Both runs gave identical
per-file pass/total counts (42087/42087, zero failures, zero mismatches
file-by-file) and wrote no wisdom file to disk. The total differs from the
same suite run as one shared process (42104) by the residual **pool-channel**
amount predicted above — a single process's constructions all draw from one
shared in-process wisdom pool that a fresh-per-file process doesn't have,
not a race and not a regression (both topologies are internally
self-consistent and reproducible; final accuracy still respects
`rtol`/`atol`). Splitting the `rust` group into concurrent per-file jobs is
therefore safe post-fix.
