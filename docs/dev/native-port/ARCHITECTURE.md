# Native-Rust Backend Port — Architecture

> Status: design doc for the phased port. Phases 0-8 are implemented and
> passing (see `docs/native-port/PORT_LOG.md` for the latest entry) — the
> native-Rust backend port is complete.
> Companion docs: [MATH.md](MATH.md), [TESTING.md](TESTING.md),
> [PORT_LOG.md](PORT_LOG.md), [BETA1_ANALYTIC.md](BETA1_ANALYTIC.md). Agent
> workflow: `AGENTS.md`.
> Phase checklist: [`BACKLOG.md`](../BACKLOG.md).

## 1. Why this document exists

The goal is to make the propagation backend run **exclusively in Rust** — the
per-step hot loop should execute with no return into Julia. Today it does not,
even with every `AMALTHEA_USE_RUST_*` toggle enabled. This document explains the
current architecture, why it cannot reach that goal, the target architecture
that can, and the rationale behind each load-bearing decision.

## 2. Current architecture (per-kernel offload + callback stepper)

Luna offloads five kernels to Rust, each behind an opt-in env toggle that
defaults **off** (`get(ENV, "AMALTHEA_USE_RUST_*", "0") == "1"`):

| Kernel | Toggle | Julia site | Rust handle |
|--------|--------|-----------|-------------|
| PPT ionization LUT | `AMALTHEA_USE_RUST_IONISATION` | `src/Ionisation.jl:516` | `RustIonizationHandle` |
| Raman SDO ADE | `AMALTHEA_USE_RUST_RAMAN` | `src/Nonlinear.jl:379` | `RustRamanHandle` |
| Zeisberger neff/β | `AMALTHEA_USE_RUST_DISPERSION` | `src/Antiresonant.jl:251` | `RustZeisbergerHandle` |
| Marcatili neff/β | `AMALTHEA_USE_RUST_DISPERSION` | `src/Capillary.jl:445` | `RustMarcatiliHandle` |
| QDHT k↔r transform | `AMALTHEA_USE_RUST_QDHT` | `src/NonlinearRHS.jl:665,675` | `RustQdhtHandle` |
| RK45 tableau/PI control | `AMALTHEA_USE_RUST_STEPPER` | `src/RK45.jl:24` | `RustPreconStepHandle` |

Each follows the same lifecycle pattern: a `mutable struct *Handle` wrapping a
`Ptr{Cvoid}` with a GC finalizer that calls `free_*`; an env toggle gating
construction; an `@testitem tags=[:rust]` equivalence test guarding the boundary.

### 2.1 The fatal limitation

The RK45 stepper is **interaction-picture Dormand-Prince**: each accepted step
performs ~7 nonlinear-RHS evaluations and ~13 linear-propagator applications
(`src/RK45.jl:173-229`). When `AMALTHEA_USE_RUST_STEPPER=1`, the Rust side
(`amalthea/src/ffi.rs:1002` `precon_step_inner`) owns only the Butcher tableau,
FSAL bookkeeping, error norm, and the Lund PI controller. For the actual math it
calls **back into Julia** through two C function pointers on every stage:

```
ffi.rs: prop_fn(t1, t2, y, …)   →  RK45.jl:_prop_cfun  →  Julia prop!  (exp-linop)
ffi.rs: fbar_fn(t1, t2, y, k, …) →  RK45.jl:_fbar_cfun  →  Julia fbar!  (full RHS)
```

So with **all five toggles on**, one propagation step still executes in Julia
6–7×: every FFT (there is no Rust FFT at all), Kerr, the plasma `cumtrapz`
integrals and current assembly, all `norm!`/window broadcasts, the
`exp(linop·dt)` application, and — for modal — the cubature-wrapped overlap
integrals. The toggles offload only the leaf kernels listed above; the
*structure* of the loop, and most of its cost, is Julia.

**Conclusion:** "exclusively Rust" cannot be reached by flipping defaults. The
field must become resident in Rust and the RHS + propagator must execute there.

## 3. Target architecture (resident field + native RHS)

### 3.1 `NativeSim` — a Rust-resident simulation state

Introduce one opaque Rust handle that owns, for the lifetime of a single
`solve`, everything the loop touches:

- the spectral field `Eω` and all scratch buffers (`Et`, `Pt`, `Pω`, the 7 `ks`,
  `yerr`, interpolation buffer);
- the FFTW plans (forward/inverse, normal and oversampled grids);
- the static arrays copied once at setup: ω grid, time/frequency windows
  (`towin`, `ωwin`), the linear-operator array (or its z-dependent builder), the
  response coefficients, and the QDHT T-matrix where applicable.

Julia builds this once via `init_native_sim(...)`, hands over the initial field
with `set_field`, calls `native_solve` (or repeated `native_step` for dense
output), and retrieves results with `get_field`. **The FFI boundary is crossed a
handful of times per `solve`, not ~14× per step.**

### 3.2 FFI surface

```
init_native_sim(grid_spec, plan_spec, linop, windows, resp_spec, …) -> *mut NativeSim
free_native_sim(*mut NativeSim)
set_field(*mut NativeSim, *const Complex<f64>, n)
get_field(*const NativeSim, *mut Complex<f64>, n)
native_step(*mut NativeSim, …) -> NativeStepResult     # one adaptive step, for dense output
native_solve(*mut NativeSim, tmax, …) -> NativeSolveResult   # full run when no per-step callback needed
```

The existing `precon_step_ffi` is retained during the transition: phases are
validated against it by swapping a Rust-native RHS in for the Julia callback,
one `Trans*` type at a time.

### 3.3 Julia integration point

`RK45.solve_precon` (`src/RK45.jl:19`) gains a top-level branch: native path vs.
the existing `PreconStepper`/`RustPreconStepper`. Selection is once per `solve`,
not per op. The high-level `prop_capillary` / `prop_gnlse` interface is unchanged.

## 4. Key decisions and rationale

### 4.1 FFT: bind FFTW, not `rustfft`
Julia uses FFTW via `FFTW.jl`. Binding the **same FFTW C library** from Rust
makes the transforms bit-identical, so every ported phase can be validated to
~1e-13 against the Julia oracle instead of needing a method-difference tolerance.
This collapses the hardest validation risk in the whole port.
- *Trade-off:* a C link-time/runtime dependency on FFTW. Mitigated — FFTW is
  already a Julia dependency present on every Luna install.
- *Recorded alternative:* `rustfft` (pure Rust, no C dep) remains a future
  option; it would move FFT equivalence into the ~1e-13 (summation-order) tier
  and is acceptable if the FFTW dependency ever becomes a portability problem.

### 4.2 Resident field, not per-op FFI
The per-stage Julia round-trip is *the* reason the current loop is Julia-bound.
A resident `NativeSim` is the single change that removes it. Per-op FFI (e.g.
"call Rust for just the FFT") would add FFI overhead without removing the
round-trip, and is explicitly rejected.

### 4.3 Keep the Julia pipeline as a whole-pipeline fallback (default-on + warn)
Once the field is resident, there is no natural per-op fallback — it is
native-loop vs. the old Julia loop, chosen once at `solve` time. We keep the
entire Julia pipeline reachable and emit a **one-time `@warn`** on fallback.
Three reasons:
1. Portability — CPU-only / unbuilt-lib installs keep working (`.so` missing).
2. **The Julia path is the equivalence oracle.** Every test compares native
   output against it; deleting it would delete the test baseline.
3. **Scope restrictions.** Phases 1-7 each cover a narrow slice of Luna's
   configuration space (specific grid types, response combinations, mode
   kinds, ...). As implemented (Phase 8), `RustNativeStepper`'s constructor
   throws a `NativeIneligible` exception for anything outside that slice;
   `solve_precon` catches *only* that type and falls back to `PreconStepper`
   — any other exception (a genuine FFI error, an invariant violation) still
   propagates and crashes, unchanged. This is why native-eligibility is a
   runtime, per-`solve` decision (`RK45.jl`'s `native_ok`/`NativeIneligible`
   catch), not a static, config-time one: the same call site must route
   both the common case (fast, native) and the long tail of narrow-scope
   configs (correct, Julia) through one function.

### 4.4 Phase ordering by `Trans*` complexity
Mode-averaged is the simplest geometry (1 inverse + 1 forward FFT, no spatial
transform). It proves the resident-field architecture end-to-end before the hard
geometries (modal cubature, free-space 3D FFT) are attempted. Each phase is
independently shippable and independently testable.

## 5. Geometry → phase map

| RHS path | Julia source | Phase | Status | Notes |
|----------|--------------|-------|--------|-------|
| `TransModeAvg` + Kerr | `src/NonlinearRHS.jl:531` | 1 | ✅ done | RealGrid first |
| Plasma (`PlasmaCumtrapz`) + EnvGrid Kerr | `src/Nonlinear.jl:161`, `:81` | 2 | ✅ done | rate LUT already Rust |
| `TransRadial` + QDHT | `src/NonlinearRHS.jl:663` | 3 | ✅ done | QDHT already Rust, made resident |
| Raman (`RamanPolar`) | `src/Nonlinear.jl:357` | 4 | ✅ done | ADE already Rust, made resident |
| `TransModal` + overlap cubature | `src/NonlinearRHS.jl:421` | 5 | ✅ done | narrow scope: `libcubature` reused (dlopen, not reimplemented), `HE,n=1`/`full=false`/Kerr-only |
| `TransFree` (3D FFT) | `src/NonlinearRHS.jl:826` | 6 | ✅ done | joint 3-D FFTW plan (same libfftw3, new plan rank); RealGrid + const_norm_free + Kerr-only |
| z-dependent linop assembly | `src/LinearOps.jl:77,185,337` | 7 | ✅ done | narrow scope: mode-averaged, graded-core constant-radius `MarcatiliMode` (`Capillary.gradient`, two-point pressure gradient) only; radial/modal/free-space z-dependent `nfun` deferred — see MATH.md §3.5 |
| Default-flip + cleanup | — | 8 | ✅ done | `AMALTHEA_USE_RUST_NATIVE` defaults to `"1"`; every scope restriction now a catchable `NativeIneligible` → Julia fallback instead of a crash |

## 6. Out of scope (stays Julia) — the three categories, and which is a real barrier

With the port complete (Phases 0-8 + D-I), the Julia code that remains
un-ported falls into exactly three categories. Only the third is a
genuine technical barrier; the first two are choices.

**(a) One-time setup and orchestration — stays Julia by design.**
The high-level `Interface.jl` / `Output.jl` / `Grid.jl` / `Fields.jl`
setup code (grid construction, mode solvers, Sellmeier/gas data, pulse
synthesis), HDF5 output (`Output.jl`), parameter scans (`Scans.jl`),
plotting/processing/stats. This runs once per simulation, costs
milliseconds, and porting it buys nothing. The port's goal was "the
*per-step hot loop* is 100% Rust", and that is achieved: for every
configuration the documented high-level API (`prop_capillary` /
`prop_gnlse`) can construct, no Julia code runs between steps.

**(b) Ineligible-but-portable configs — "won't", not "can't".**
BACKLOG.md Phase I items 5-6: `StepIndexMode`/`ZeisbergerMode`/
`VincettiMode` in modal, plasma/Raman in free-space, mixtures in
modal/free-space, and similar low-level-API-only combinations. These
are ordinary numerics that *could* be ported exactly like everything
else was — they fall back via `NativeIneligible` because they are only
reachable through the low-level API in combinations almost nobody
uses, so effort/value rules them out, not feasibility.

**(c) Arbitrary user closures — the one genuine barrier.**
The low-level API accepts any Julia function as a nonlinear response,
density profile, or mode field. Rust cannot execute arbitrary Julia
code without a per-sample FFI callback — which is exactly the
round-trip this port exists to eliminate. This is why the guards for
"unrecognized bare `f!` closures" exist: *recognizable* structures
(splines, two/multi-point gradients, the standard response types) are
transferred as **data** and evaluated natively; anything else falls
back to the Julia stepper. Closing this category fully would mean a
declarative config format instead of Julia closures — that is
SUGGESTIONS.md item 14's CLI, which sidesteps Julia entirely rather
than porting it.

**Native-path FFTW planner wisdom (opt-in, default off).** Unlike the
per-kernel toggles in §2 (default off because the *feature* is optional),
`RustNativeStepper`'s own FFTW wisdom cache is default off for a
determinism reason: importing wisdom before planning lets accumulated
on-disk/in-process wisdom perturb plan selection, which — via the RK45
controller's near-cancellation sensitivity (CLAUDE.md's Phase 2 gotcha) —
can shift the adaptive step-size path run to run. Set
`AMALTHEA_NATIVE_FFTW_WISDOM=1` to opt back in. See
`docs/native-port/PLAN_FFTW_WISDOM_FIX.md` for the full analysis and
`BACKLOG.md` S1 item 1 for the resolution. The determinism boundary this
leaves in place: exact bit-level step-path reproducibility across runs is
only guaranteed with one fresh process per simulation (native shares
FFTW's single process-global wisdom pool with Julia's own `FFTW.jl`,
regardless of the on-disk cache toggle).
