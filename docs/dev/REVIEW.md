# Luna-Rust.jl vs upstream Luna.jl — review report (2026-07-02)

Scope: full comparison of this fork against the freshly-downloaded upstream
`Luna.jl` (`../Luna.jl`, master @ `0a52ffb`, 2026-06-24), plus a correctness
review of the fork's own changes (Julia and Rust sides). Method: upstream was
added as a local git remote (`upstream`) and diffed against both the fork's
initial import commit (`e3b8d19`) and HEAD; the Rust crate's `cargo test`
suite (31/31 green) and the full `LUNA_TEST_GROUP=All` Julia suite were run
as evidence.

## 1. How stale is the fork base? (Answer: barely)

The fork's squashed initial import `e3b8d19` differs from **current upstream
master by only 2 functional commits** — the base is upstream master as of
~June 2026, not an old release. The complete missed-upstream list:

1. **`0a52ffb` — "correct calculation of ellipse angle" (#428). This is a
   real bug the fork still has.** `Polarisation.ellipse` computes
   `θ = angle(aL)/2` where `aL = sqrt(Q^2 + U^2) ≥ 0`, so `θ` is **always
   0**. Upstream's fix is `θ = angle(Q + 1im*U)/2`. Upstream also added a
   regression test (QWP at π/8 → ellipse angle π/8) which the fork's
   `test/test_polarisation.jl` lacks. Port both.
2. **`0fd7ac2` — "Ssh scan files" (#427).** Adds a `files=String[]` kwarg to
   `SSHExec` to transfer auxiliary files with the scan script. The fork
   lacks the feature — but note the fork *rewrote* `runscan(::SSHExec)` with
   proper shell escaping (`Base.shell_escape_posixly` + `Cmd` arrays), which
   is a genuine security improvement over upstream. Porting #427 must be a
   **merge into the escaped version**, not a copy of upstream's unescaped
   loop.

Remaining upstream commits since the base are CI-only (dependabot).

## 2. Fork changes that are *better* than upstream (candidates to send back)

- `DataField(fpath; ...)` now forwards `λ0` (upstream silently drops it when
  loading from a file — an upstream bug).
- `Scans.parse_item(::Type{UnitRange{Int}}, ...)` no longer calls
  `eval(Meta.parse(x))` on a CLI string (arbitrary-code-execution hazard);
  SSH/Slurm/Condor submission uses `Cmd` arrays instead of interpolated
  backtick commands (shell-injection hazard).
- `IonRatePPTAccel` clamps instead of throwing above `Emax` (the adaptive
  stepper legitimately probes rejected trial points; a crash there was a
  usability bug). Deliberate, documented divergence.
- The β1(z) analytic closed form (Phase 7) is *more* accurate than
  upstream's adaptive-FD `Modes.dispersion` — see
  `docs/native-port/BETA1_ANALYTIC.md`.
- Assorted allocation fixes (`@views`, generator instead of array
  comprehensions) and the `pointcalc!` data-race fix (sequential loop).

## 3. Mistakes / divergences found in the fork

### 3.1 `PhysData._safe_n` silently changes glass physics (🔴 fix)
Upstream's glass Sellmeier closures return `sqrt(complex(n²))`, so regions
where `n² < 0` yield a complex index (absorption) and `0 < n² < 1` yields a
real `n < 1` (physically valid, phase velocity > c). The fork replaced ~20
sites with `_safe_n(n2) = n2 < 1 ? one(n2) : sqrt(n2)`, which **clamps both
cases to exactly 1.0** — silently wrong dispersion/loss near and beyond the
UV/IR resonances of every glass except `:SiO2` (which kept complex
handling). Recommendation: restore `sqrt(complex(...))` semantics (or apply
the SiO2-style `+1e-10im` regularisation everywhere) and keep `_safe_n` only
if some downstream Rust path genuinely requires a real return — in which
case guard it, don't clamp globally.

### 3.2 The native default path skips the most common workload (🔴🔥 biggest perf gap)
`prop_capillary` defaults to `plasma = !envelope`, i.e. **every default
field-resolved run includes plasma**. The native stepper's plasma wiring
requires `IonRatePPTAccel.rust_handle`, which is only constructed when
`LUNA_USE_RUST_IONISATION=1` — and that toggle **defaults to 0**. So in the
out-of-the-box configuration (`LUNA_USE_RUST_NATIVE=1`,
`LUNA_USE_RUST_IONISATION=0`), the flagship resident-Rust stepper throws
`NativeIneligible` and silently (one-time `@warn`) falls back to the Julia
stepper for the fork's bread-and-butter use case. The port's headline
speedup does not apply to default HCF simulations. Fix: build the ionisation
LUT handle whenever the native path is active (decouple it from the opt-in
kernel toggle), or flip `LUNA_USE_RUST_IONISATION` default to 1 after fixing
3.3.

### 3.3 Rust/Julia disagree above `Emax` on the opt-in ionisation path (🟡)
Julia's `IonRatePPTAccel` clamps above `Emax`; the Rust
`PptIonizationRate::rate` still **errors** (`ionization.rs:222`). With
`LUNA_USE_RUST_IONISATION=1`, one above-`Emax` sample makes
`ppt_ionization_rate_vector` fail the whole vector, emitting a `@warn` and
falling back to Julia for that call — i.e. the acceleration disappears (and
warns repeatedly) in exactly the strong-field scenarios that need it. The
**native** path already clamps correctly (`native.rs:536` falls back to
`rate(e_max)`), so the fix is to make the standalone LUT clamp the same way.

### 3.4 Native path freezes density at z=0 without a guard (🟡, low-level API only)
`RustNativeStepper` bakes `density = f!.densityfun(0.0)` (and
`raman_density = densityfun(0.0)`) into constants for all non-z-dep phases.
The high-level interface can't hit this (a z-varying fill produces a
function linop or `ZDepLinopMarcatili`, both handled), but the low-level API
allows a constant `Array{ComplexF64}` linop with a z-varying `densityfun` —
which Julia's `TransModeAvg` re-evaluates every stage and native silently
freezes. Add a cheap guard (`densityfun(0) ≈ densityfun(L/2) ≈
densityfun(L)` → else `NativeIneligible`).

### 3.5 Minor
- `README.md` claims "fully supported on Linux, macOS, and Windows" while
  parallel parameter scans are not process-safe on Windows (`scans.rs`
  flock no-op — already in BACKLOG). Soften the claim or fix the lock.
- Test suite: no weakened tolerances found; the widened native-tier
  tolerances (~1e-4 β1 tier etc.) are documented and justified in-file. The
  only missing upstream test is the #428 ellipse regression (comes with the
  port).

## 4. Verification evidence

- `cargo test` (release): **31/31 pass**, including GPU-equivalence
  self-skips.
- Dormand-Prince 5(4) tableau constants and 4-point Gauss-Legendre
  nodes/weights in `stepper.rs`/`integrator.rs` verified against reference
  values — correct.
- Full `LUNA_TEST_GROUP=All` Julia suite on `ac66186`: see the run log in
  the session scratchpad (`suite_all_ac66186.log`); result reported
  separately (the previous session's run was killed mid-flight, leaving no
  record of a post-β1-fix green run — this rerun closes that gap).
- Commits `50d8ca2` + `ac66186` are already on `origin/main`
  (`git ls-remote` confirmed).

## 5. What is still lacking (native-path scope gaps)

Inventory from the `NativeIneligible` guards in `src/RK45.jl` (each falls
back to the Julia stepper today): EnvGrid radial / modal / free-space;
plasma and Raman in radial/modal/free geometries (Kerr-only there); modal
`full=true`, TE/TM/`n>1` modes, tapered radius, npol=2, modified shot-noise;
Raman `thg=false` and `RamanPolarEnv` (envelope); gas mixtures (non-scalar
`densityfun`); multi-point pressure gradients and any function-typed linop
(tapers, multimode z-dependence); z-dependent free-space `normfun`. Plus
platform gaps: Windows scan locking, GPU paths never exercised in CI.

The phased plan to close all of the above is in `BACKLOG.md` ("Improvement
plan 2026-07-02"). Out-of-scope architectural ideas are in `SUGGESTIONS.md`.
