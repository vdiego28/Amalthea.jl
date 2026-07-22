# Plan: BACKLOG.md S6 item 3 — standalone CLI (`luna-cli`)

Status: **Measured, then parked — recommend against building the item as
currently specified. A much smaller alternative ("dump-and-replay") is
sketched below as the only version of this idea that looks worth building
today.** This is a feasibility/design pass, not an implementation — no
Rust source was written, no crate dependencies were added, no existing
code was changed. Research was cut short by a mid-session scope change
("wrap up promptly, free the machine"); several sub-questions are marked
**OPEN** below rather than filled in from inference. Treat this doc as a
first pass to be extended, not a closed investigation.

## The item as currently written

> Standalone CLI (item 14) — `amalthea/src/bin/luna-cli.rs` (`[[bin]]` in
> the existing crate, not a new workspace member). TOML config in, scoped
> to native-eligible configs, HDF5 out; compare against a Julia run of the
> same config as the acceptance test. WASM demo only after the CLI exists.

Note: BACKLOG.md's own text still says `luna-rust/src/bin/luna-cli.rs`
(the crate directory is `amalthea/` — confirmed via `amalthea/Cargo.toml`'s
`[package] name = "amalthea"`). Stale path, same pattern CLAUDE.md already
flags elsewhere in this file's prose.

## Why this is not just "add a `[[bin]]`"

`docs/dev/native-port/ARCHITECTURE.md:161-195` (§6, written when the
native-port phases closed out) already answers the question this plan set
out to ask, in general terms:

> **(a) One-time setup and orchestration — stays Julia by design.** The
> high-level `Interface.jl` / `Output.jl` / `Grid.jl` / `Fields.jl` setup
> code (grid construction, mode solvers, Sellmeier/gas data, pulse
> synthesis), HDF5 output ... This runs once per simulation, costs
> milliseconds, and porting it buys nothing. The port's goal was "the
> *per-step hot loop* is 100% Rust", and that is achieved ...
>
> **(c) Arbitrary user closures — the one genuine barrier.** ... Closing
> this category fully would mean a declarative config format instead of
> Julia closures — that is SUGGESTIONS.md item 14's CLI, which sidesteps
> Julia entirely rather than porting it.

In other words: the project already looked at "should we port setup to
Rust" and answered no, because the entire point of the native port was
the *per-step* loop, and setup is a one-time, millisecond-scale cost. A
standalone CLI needs setup ported anyway — for a different reason (no
Julia present at all, not perf) — but it is the same porting work
ARCHITECTURE.md already classified as high-effort/low-value for the
per-step goal. This plan's job was to find out whether that porting work
is actually small (making the CLI cheap despite ARCHITECTURE.md's framing)
or actually large (confirming it). Evidence below points to large.

## Setup-path inventory: mode-averaged RealGrid, Kerr-only, single gas, constant pressure

Target config (the narrowest genuinely useful case, matching
`test/benchmark_native_default.jl`'s flagship workload minus plasma):
`prop_capillary(radius, flength, gas, pressure; λ0, λlims, trange, energy,
τfwhm, modes=:HE11, kerr=true, plasma=false, raman=false)`. Traced
`src/Interface.jl:385-450` (`_prop_capillary_args`) end to end. Classified
each piece the CLI would need to reproduce in Rust with no Julia present:

| Piece | Source | Classification | Evidence |
|---|---|---|---|
| Grid ω/t axes, windows | `Grid.jl` `RealGrid(...)` | **(b) small port** | `Grid.jl:38` builds axes via `FFTW.fftfreq`/`fftshift` (`Grid.jl:150,219,224`) — standard closed-form DFT-frequency arithmetic, no adaptive logic. Windows (`ωwin`/`twin`) are separately-defined closed-form tapers. |
| `MarcatiliMode` neff/β(ω) | `Modes.β` → `Capillary.jl` | **(a) already in Rust** | `amalthea/src/dispersion.rs` has `MarcatiliNeff`/`ChebyshevDispersion` (CLAUDE.md-confirmed, matches Julia bit-for-bit per existing kernel wiring). |
| `β1const` (linop group-velocity term) | `Modes.dispersion(mode,1,ω0)` = `Maths.derivative(...)` | **(b) small port** | `Modes.jl:188-199` — a genuine one-time scalar finite difference (not the Phase-7 analytic closed form, which is only for the two-point z-dependent case). Cheap to replicate once at setup. |
| `pre` (nonlinear prefactor) | `NonlinearRHS.jl:602-614` | **(b) small port** | Closed-form arithmetic on `grid.ω` + `PhysData.c`/`ε_0` constants. |
| `kerr_fac` (`density·ε_0·γ3`) | `RK45.jl:1244`, `PhysData.density`/`PhysData.γ3_gas` | **(c) large / hard dependency — worse than first assessed** | `γ3_gas` (`PhysData.jl:668-699`) is closed-form arithmetic, but it scales off `dens_1atm_0degC[material]` (`PhysData.jl:798`), a module-load-time constant computed via `density()`. **`PhysData.density(material,P,T)` (`PhysData.jl:784-795`) calls `CoolProp.PropsSI("DMOLAR",...)`** — a real-gas equation of state from the external C++ `CoolProp` library (`PhysData.jl:3` imports it), not the ideal-gas law. Reproducing Julia's actual number needs either dlopening CoolProp itself (same pattern as HDF5/FFTW, but a new native dependency with its own portability story) or accepting an unquantified systematic offset from an ideal-gas approximation at real operating pressures. This is a genuine hard dependency, not a "big but mechanical" table-transcription job as first assessed. |
| `sqrt_aeff` (effective-area normalization) | `Modes.Aeff(mode)` → `Capillary.jl:290-297` | **(b/c) medium port — smaller than first assessed** | `Aeff = radius²·aeff_intg`, where `aeff_intg = Aeff_Jintg(n,unm,kind)` is a 1-D `hquadrature` over `r·J_{n-1}(u_nm·r)^4` (`Capillary.jl:290-295`) and `unm` is a Bessel-zero from `FunctionZeros.besselj_zero` (`Capillary.jl:249-259`). The needed *quadrature primitive already exists*: `amalthea/src/cubature.rs` dlopens the native C `libcubature` (`pcubature_v`/`hcubature_v_2d`, lines 148-172) and is usable standalone today, with no Julia dependency, for exactly this kind of integral. What's actually missing is a Bessel-J evaluator and a Bessel-zero root-finder — both well-known, portable, closed-form-adjacent algorithms with no external-library dependency required — not a from-scratch quadrature engine. One-time/scalar (evaluated once per mode construction, not per RHS step). |
| Setup FFI hand-off | `RK45.jl:1250-1259` `ccall(:native_set_mode_avg_params,...)` → `amalthea/src/native.rs:5129` | confirms the shape of the gap | Rust's FFI entry point stores every argument (`towin`, `owin`, `sidx`, `pre` re/im, `beta`, `kerr_fac`, `nlscale`, `sqrt_aeff`) verbatim — it does **not** recompute any of them today. A standalone CLI must become the thing that computes all of the above, in Rust, with no fallback to ask Julia. Confirmed separately: `dispersion.rs`'s `MarcatiliNeff`/`ZeisbergerNeff` (already-in-Rust per the per-kernel wiring table) are **not** on this codepath at all — Julia's `beta` array here traces back through `NonlinearRHS.norm_βfun`/`Modes.dispersion`, never through a `ccall` into those Rust kernels. So "dispersion is already in Rust" (true for the older per-kernel accelerator) does not actually close this particular gap. |
| Pulse synthesis (`GaussPulse`/`SechPulse` → initial `Eω`) | `src/Fields.jl:105-195,654-664` | **(b) small port** | `PulseField` builds `Et` from a closed-form envelope (`Maths.gauss`/`sech`), transforms via `FT*Et` (RealGrid rfft — already in `fftw.rs`), and normalizes energy via `energyfuncs` (`Fields.jl:654-664`), a Simpson's-rule integral over `|Et|²`/`|Eω|²` — no root-finding, no external dependency. |
| Shot noise (`Et_noise`/`Emω_noise`) | `Interface.jl` `makenoise` | **OPEN — not independently verified this session** (no fork thread reached it before the cutoff). Needed even for a "Kerr-only" config since `shotnoise=true` is `prop_capillary`'s default; flag as a prerequisite check for whoever resumes this. |

**Reading across the table:** the grid axes, `pre`, `β1const`, and pulse
synthesis are genuinely small, closed-form ports. But two pieces are not:
**`kerr_fac` needs `PhysData.density`'s real-gas EOS, which is CoolProp —
an external C++ library, dlopenable in principle but a wholly new native
dependency with its own portability story**, and `Aeff` needs a Bessel-J
evaluator + Bessel-zero root-finder (smaller than first assessed, since
`cubature.rs` already supplies the quadrature primitive, but still new
special-function surface). This is materially more than "add a `[[bin]]`
and a TOML parser" — it is a second, smaller-but-real porting project
layered on top of the one that's already done, for code the native
port's own architecture doc already decided wasn't worth porting. Also
confirmed (fork B): `dispersion.rs`'s already-ported `MarcatiliNeff` is
not actually on the mode-averaged `beta`-array codepath at all (see table
above) — the "dispersion is already in Rust" comfort from the per-kernel
wiring table does not transfer to this use case.

**Additional module-level findings (fork B, `amalthea/src/*.rs` inventory):**
- `grid.rs` (58 lines) is a bare data struct with a plain positional
  constructor — no logic today builds `RealGrid`/`EnvGrid` from
  `(zmax,λ0,λlims,trange,δt)` in Rust; the native stepper receives
  Julia-built `towin`/`ωwin`/`sidx` as flat arrays already. Confirms the
  "small port" classification above is about *effort to add*, not
  "already partially done."
- `ionization.rs`: `PptIonizationRate` is a generic spline-LUT engine
  (`::from_samples`) — the actual PPT rate formula
  (`src/Ionisation.jl:467`, special-function-heavy) exists only in Julia.
  `AdkIonizationRate` looks more self-contained but wasn't fully traced
  back to its constant sources — flag as open if plasma is ever in scope.
- `raman.rs`: the ADE solver math is fully self-contained Rust; only the
  oscillator-parameter *derivation* (rotational energy levels etc., from
  `src/Raman.jl:217`) is Julia-only. Out of scope for the Kerr-only V1
  target anyway.
- `cubature.rs`, `io.rs`: both already general-purpose and
  Julia-independent (confirmed **(a) already-in-Rust-standalone**) —
  the real gaps are the physics that would need to *call* them (Bessel
  functions for `cubature.rs`'s integrand; `Output.jl`'s stats/meta
  schema for `io.rs`'s writer, addressed next).

**`Output.jl`'s actual `HDF5Output` schema** (fork A, `Output.jl:161-259`):
a top-level extensible `"Eω"` dataset (chunked, z as last index), a `"z"`
1-D extensible dataset, `"stats/*"`, `"meta/*"` (sourcecode, git commit,
cache hash, `meta/cache` sub-group), and a `"prop_capillary_args"`
string-dict group (`Interface.jl:462-468`). Dim ordering matches the
column-major reversal `io.rs`'s existing `scan_write_point`/
`create_dataset_nd_julia` already handles — the field+z part is
essentially already solved by existing Rust infra; only `meta`/`stats`
would need adding for a byte-for-byte-comparable file, and (per
`Stats.default`, `Stats.jl:495-532`) stats are not required for a V1
acceptance test — comparison can be done directly on `Eω`/`z`, with
Julia able to recompute any stats post-hoc from a Rust-written
field/z file if deeper comparison is later wanted.

**Not verified this session (flag for whoever resumes):**
- `src/Interface.jl`'s `makenoise` (shot noise) — no fork thread reached
  this before the cutoff.
- `AdkIonizationRate::rate`'s (`ionization.rs:410-458`) full independence
  from Julia-only constants — read the struct, didn't trace every
  constant to its `PhysData.jl`/`Ionisation.jl` origin. Only matters if
  plasma is ever brought into CLI scope.
- ~~Whether `optional = true` deps + `required-features` on a `[[bin]]`
  leaves `cargo build --release` (no `--features`) unaffected~~ —
  **resolved**: empirically verified in a scratch crate (not this repo,
  per the "no Cargo.toml changes" constraint): a `[[bin]]` with
  `required-features = ["cli"]` paired with `optional = true` deps gated
  behind `cli = ["dep:serde"]` is correctly skipped by plain `cargo
  build` (no bin built, no optional deps compiled); `--features cli`
  builds both. The mechanism works exactly as hoped — this is not a
  blocker.

## "Scoped to native-eligible configs" — what that boundary actually is

`docs/dev/native-port/NATIVE_SUPPORT_MATRIX.md` gives the current
eligibility surface: mode-averaged is the most complete geometry (Kerr,
plasma, Raman, shot noise, Kerr-only mixtures, and a narrow
two-point-gradient z-dependent case are all native-eligible for RealGrid).
If "native-eligible" is read as broadly as the matrix allows rather than
just Kerr-only, the CLI's setup-porting burden grows further: plasma needs
`PhysData.ionisation_potential` plus PPT/ADK rate construction, Raman needs
gas-specific spectroscopic constants from `PhysData.jl`. None of that was
traced this session — flagged as **OPEN**.

## TOML config and the cargo dependency question

`amalthea/Cargo.toml` today has exactly three dependencies: `libc`,
`num-complex`, `rayon` (plus `windows-sys` gated on `cfg(windows)`) — **no
`serde`, `toml`, or `clap`**. The crate is `crate-type = ["cdylib",
"rlib"]`, built by `deps/build.jl` via plain `cargo build --release` and
loaded by every Julia session at package init.

The standard Cargo mechanism for adding an opt-in binary without touching
the default library build is: mark `serde`/`toml`/`clap` `optional = true`,
define a `cli` feature that enables them, and add `required-features =
["cli"]` to the `[[bin]]` entry. Under that shape, `cargo build --release`
(what `deps/build.jl` runs) should not build `luna-cli` or pull in its
dependencies at all — only `cargo build --release --features cli` would.
**Confirmed this session** (in a scratch crate, not this repo, per the
"no Cargo.toml changes" constraint): this shape works exactly as
described — plain `cargo build` skips both the bin and the optional deps,
`--features cli` builds both. This is not a blocker; the mechanism is
sound and this repo's `Cargo.toml` starts from zero `serde`/`toml`/`clap`
deps today, so there is nothing to disentangle from an existing graph.

## HDF5 output and the acceptance test

`amalthea/src/io.rs` already dlopens `libhdf5` at runtime and (S6 item 2,
done 2026-07-19) has `scan_write_point` + `create_dataset_nd_julia` for
writing arbitrary named ND datasets matching HDF5.jl's column-major
dim-reversal. That's a real head start — the low-level "can Rust write an
HDF5 file Julia can read back" question is already answered yes. What's
open is how much of `Output.jl`'s actual `HDF5Output` schema (stats
groups, `meta`/`meta/cache`, the `chash` consistency check, compression
options) a V1 CLI needs to reproduce for "compare against a Julia run of
the same config" to mean anything more than "the field arrays are
plausible." A defensible **V1 scope-down**: skip `stats` entirely (Stats.jl
computation is itself Julia-only machinery — energy, etc. — and isn't
needed to validate the field propagation itself), write only `Eω`/`z`
under the same dataset names Julia would use, and compare those two
arrays directly rather than trying to byte-match the whole HDF5Output
file structure.

**On the acceptance test's meaningfulness:** because the CLI (as
specified) reimplements setup independently in Rust rather than consuming
Julia's already-computed arrays, "compare against a Julia run of the same
config" is **not a bit-parity test** — it inherits every divergence in the
setup path (Sellmeier table transcription errors, `β1const`'s finite
difference implemented with a possibly-different step size or scheme than
Julia's `Maths.derivative`, `Aeff`'s quadrature/root-finder using
different convergence tolerances than `HCubature.jl`). This is
structurally the same situation CLAUDE.md documents for Phase 7's β1
analytic-vs-Julia-FD divergence (`docs/dev/native-port/BETA1_ANALYTIC.md`):
a real, small, systematic difference that can accumulate coherently over
a propagation. Expect a tolerance tier in the ~1e-4 to ~1e-6 range at best
(not the ~1e-8/~1e-12 tiers the per-kernel wiring table in CLAUDE.md
reports for kernels that consume identical, Julia-computed inputs) —
and only after every setup-path piece above is actually built and its own
divergence from Julia is characterized individually, the same way Phase 7
did (a `kerr=false` control run to isolate one variable at a time). No
number can be honestly stated before that work exists.

## WASM (per the item: "only after the CLI exists")

Independent of whether the CLI is built, WASM would additionally block on
the fact that **FFTW and HDF5 are both `dlopen`ed native shared libraries**
(`amalthea/src/fftw.rs`'s `Library::sym` dlopen infra; `io.rs` same
pattern). Neither `wasm32-unknown-unknown` nor WASI has a general
`dlopen` of arbitrary host `.so`/`.dylib` files — there is no library to
find at runtime in a WASM sandbox. A WASM build would need either (a)
statically-linked/vendored WASM builds of FFTW and HDF5 (a real question
mark for HDF5 in particular — this needs its own research, not assumed
possible), or (b) dropping HDF5 for the WASM target and replacing FFTW
with a pure-Rust FFT crate at the native-port's FFT boundary — itself a
non-trivial, separate rewrite of a hot-path dependency the rest of the
crate is built around. Either path is its own multi-session project,
independent of and larger than the CLI item itself.

## Alternative shapes considered

1. **The item as specified** (TOML → pure-Rust setup → HDF5 out, no Julia
   anywhere). Evidence above: large port (Bessel root-finder + quadrature
   for `Aeff`, full gas-table transcription, plus two open items — pulse
   synthesis and noise — that could add more). **Not recommended now.**

2. **Thin CLI that shells out to / embeds Julia.** Doesn't eliminate the
   Julia dependency at all — still needs a full Julia install +
   `Amalthea.jl` loaded, just wrapped in a different entry point. Defeats
   the stated purpose (a Julia-free artifact) and doesn't unblock WASM
   either, since embedding a Julia runtime in WASM is its own much larger
   problem than the FFTW/HDF5 one above. **Not useful as a way to satisfy
   this item's goal.**

3. **Dump-and-replay: Julia exports a one-time setup dump, a genuinely
   standalone Rust CLI replays it.** Julia runs `prop_capillary_args`
   once (as it does today) and, instead of constructing a
   `RustNativeStepper` in-process, serializes every argument the existing
   `native_set_mode_avg_params`/`native_set_plasma_params`/etc. FFI calls
   already pass (the exact arrays in the inventory table above — this
   reuses S6 item 2's `create_dataset_nd_julia` writer, or a simpler flat
   binary format) to a file. `luna-cli` (still behind an opt-in `cli`
   feature) loads that file and drives `native_step` to completion using
   the *existing, unmodified* `native.rs`, writing `Eω`/`z` out via
   `io.rs`. **This is dramatically smaller** — zero new setup-porting work
   (no Sellmeier tables, no Bessel root-finder, no pulse-synthesis port),
   because Rust never computes any of the setup; it only consumes
   Julia's numbers, exactly like every existing per-kernel wiring already
   does. The acceptance test is then genuinely bit-identical (same input
   arrays, same Rust code path, run once from Julia's process and once
   from `luna-cli` — the only variable is which process is driving
   `native_step`), which is a real, checkable acceptance criterion, unlike
   option 1. **Trade-off:** it is not what item 14 literally asks for — a
   Julia session still has to run once to produce the dump, so this
   doesn't remove Julia from the workflow, only from *repeated* runs of an
   already-configured simulation. Useful for exactly the parameter-scan
   use case the rest of S6 targets (re-running one setup many times with
   different seeds/energies without paying Julia startup cost per run) —
   a real, if narrower, value proposition. **This is the only version of
   "a standalone CLI" that looked buildable at reasonable cost from this
   session's evidence**, if there's appetite for something in this space
   at all.

## Recommendation

**Do not build the CLI as specified in BACKLOG.md's current one-paragraph
item.** The setup-porting cost is materially larger than the item implies
— it duplicates, for a different reason, exactly the work
`ARCHITECTURE.md` §6a already classified as high-effort/low-value
("porting it buys nothing"), and at least one piece (`Aeff`'s Bessel-zero
root-finder + adaptive quadrature) is genuine new special-function surface
this crate doesn't have today, not a mechanical translation. Two more
pieces (pulse synthesis, shot noise) weren't even confirmed closed-form
this session. No current user of this fork has asked for a Julia-free
artifact; the WASM motivation the item cites is blocked on a separate,
larger FFTW/HDF5-in-WASM problem regardless of whether the CLI exists.

**If there is appetite for something in this space**, option 3
(dump-and-replay) above is the only shape this session's evidence
supports as a reasonably-scoped V1: it reuses the entire existing
native-port stepper unmodified, needs no new numerics, and has a genuine
bit-identical acceptance test. It is a different, narrower feature than
"standalone CLI" — recommend re-titling/re-scoping the backlog item if
this path is chosen, rather than silently redefining item 14.

**A well-argued no is treated as a fully successful outcome for this
research task** (per the task's own framing, mirroring
`PLAN_S1_6_SOA_CONVERSION.md`'s parked-with-reasoning precedent) — this is
that outcome.

## What would change this recommendation

- A concrete user/downstream request for a Julia-free CLI or WASM demo
  (nothing currently in the repo motivates this beyond the backlog item's
  own suggestion-list origin — worth checking `SUGGESTIONS.md` for the
  original ask's context if this is revisited).
- If the still-open items (pulse synthesis, shot noise, the exact
  `PhysData.jl` gas-table portability) turn out to be smaller than
  `Aeff`'s Bessel/quadrature dependency suggests the whole path is —
  possible, since they weren't checked this session, but nothing found so
  far points that way.
- If the dump-and-replay alternative (option 3) is deemed worth building
  as its own, smaller, differently-named item — that would be a
  reasonable next step; it is not this item as written.

## Session limitations (explicit, per the interrupted research)

This pass was cut short mid-investigation by a scope change to free the
machine, partway through four parallel research threads. All four did
return by the time this doc was finalized (the setup-path trace, the
`amalthea/src/*.rs` inventory, the `Fields`/`PhysData`/`Output`/`Stats`
trace, and the config/eligibility/Cargo-mechanics check), and their
findings are folded into the tables and sections above with file:line
citations. The one item no thread reached before the cutoff is
`Interface.jl`'s `makenoise` (shot noise construction) — left as an
explicit **OPEN** item above rather than inferred. Given the convergent,
mutually-reinforcing findings from three independent threads (the
CoolProp dependency, the `dispersion.rs`/`grid.rs` "already-in-Rust but
not on this codepath" pattern, and the dump-and-replay alternative being
independently proposed by two separate threads), the overall recommendation
below is reasonably well-supported despite the early cutoff — but
`makenoise` and `AdkIonizationRate`'s constant provenance (noted inline
above) should still be checked before treating this as a closed
investigation.
