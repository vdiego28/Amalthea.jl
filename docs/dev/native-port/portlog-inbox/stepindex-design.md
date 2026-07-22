## 5. Native multi-mode `StepIndexMode` (BACKLOG Phase I item 5)

Status: **Feasibility studied, not built. Recommend against building now — a
value/priority call, not a difficulty finding.** Unlike §4's CLI item, there
is no hard external dependency blocking this: no root-finder needs porting
(the crux fact below), the crate already binds the same `libcubature`/FFTW
this would reuse, and the net new surface is one special function plus a
parameter-shape widening — roughly a Phase-5-sized, 1-2 day slice, not a
multi-week project. It stays parked because (a) nothing in the repo
currently constructs this specific configuration (`prop_stepindex` is
single-mode only; multi-mode `StepIndexMode` only reaches `TransModal`
through `prop_capillary(...; modes=[m1,m2,...])`, an unusual combination —
solid-fibre modes inside a gas-capillary high-level entry point — that no
example or test exercises today), and (b)
`docs/dev/native-port/ARCHITECTURE.md` §6b already classifies exactly this
row of the support matrix as "ordinary numerics that could be ported... but
effort/value rules them out, not feasibility" — this write-up confirms that
classification rather than overturning it. **If a concrete consumer shows
up, this is a clean, boundable yes** — see "If someone picks this up" below
for the exact narrow scope and first steps.

### 1. What the native modal RHS actually needs from a mode (traced, not assumed)

`native_set_modal_params` (`amalthea/src/native.rs:5613`) takes, per mode:
`a` (one shared scalar radius), `unm`, `inv_sqrt_n` (`= 1/√N(m,z=0)`,
precomputed in Julia), `order` (`Int32`), `kind` (`u8` code 0/1/2 for
HE/TE/TM), `phi`, plus one shared `towin`/`kerr_fac`/`nlfac` array — see
`src/RK45.jl:1684-1791`. No Bessel-zero, root-find, or normalization math
crosses the boundary; only the already-solved scalars do.

The per-node hot loop (`modal_pointcalc`, `native.rs:1942`) confirms my
reading exactly: the mode field is synthesized as

```rust
let x = r * ro.modal_unm[m] / ro.modal_a;
let base = jn(ro.modal_order[m], x) * ro.modal_inv_sqrt_n[m];
let (ax, ay) = mode_angle_xy(ro.modal_kind[m], ro.modal_order[m], ro.modal_phi[m], theta);
// Ex = base*ax, Ey = base*ay
```

i.e. **exactly the factored form the task description predicted**:
`J_order(x)·(ax,ay)`, with `(ax,ay)` a closed-form trig function of `θ`
computed in Rust (`mode_angle_xy`, `native.rs:2988`) and `x` a linear
rescale of `r` by two precomputed-in-Julia scalars (`unm`, `a`). `modal_a`
does double duty here: it is *both* the cubature integration bound
(`pcubature_v(fdim, ..., 0.0, self.modal_a, ...)`, `native.rs:1909-1921`)
*and* the mode's own field-argument scale (`r * unm / a` above, and the
`r <= 0 || r >= modal_a → 0` early-out at `native.rs:1949`). For Marcatili
those two roles coincide (the mode is genuinely zero outside the capillary
core, `radius(m,z)`) — this coincidence is Marcatili-specific, not a
general property of `modal_a`, and it is exactly where §2 below breaks.

Normalization (`N(m)`) never calls a Bessel function inside Rust at all —
`MarcatiliMode` overrides the generic `Modes.N` with a closed form
(`Capillary.jl:285-288`), and Julia precomputes `1/√N` once, per mode, at
setup. This is why Phase 5's stated tolerance ceiling is set by `jn`'s
agreement with `SpecialFunctions.besselj`, not by any normalization
integral (MATH.md §3.3, TESTING.md §4).

### 2. How different `StepIndexMode`'s field actually is

`field(m::StepIndexMode, xs; z=0, ω=wlfreq(1030e-9))`
(`src/StepIndexFibre.jl:265-276`) is Snyder & Love 1983 Table 12-3, not a
Marcatili-shaped formula at all:

```julia
f1, f2, u, w, v = f1f2(radius(m,z), ω/c, m.coren(ω,z=z), m.cladn(ω,z=z), neff(m,ω;z=z), m.n)
a1 = (f2 - 1)/2 ; a2 = (f2 + 1)/2
fv = m.parity == :even ? cos(m.n*θ) : sin(m.n*theta)   # NB: `theta`, not `θ` — see below
gv = m.parity == :even ? -sin(m.n*θ) : cos(m.n*θ)
R = r/radius(m,z)
Er, Et = abs(R) <= 1.0 ? corefields(m.n,u,R,a1,a2,fv,gv) : cladfields(m.n,u,w,R,a1,a2,fv,gv)
SVector(Er*cos(θ) - Et*sin(θ), Er*sin(θ) + Et*cos(θ))
```

Contrast with Marcatili (`Capillary.jl:271-283`), which returns `(Ex,Ey)`
directly from one closed-form trig expression with no radial branch and no
polar-to-Cartesian rotation step. `StepIndexMode` differs in every
structural respect that matters for a port:

- **Two radial branches, not one.** `corefields` (Bessel **J**, orders
  `n-1`/`n+1`, evaluated at `u·R`) for `R ≤ 1`; `cladfields` (modified
  Bessel **K**, same orders, at `w·R`) for `R > 1`. Marcatili has no
  cladding branch — the mode is defined (and zero) only inside the core.
- **Wider domain.** `dimlimits(m::StepIndexMode) = (:polar, (0,0),
  (10*radius(m,z), 2π))` (`StepIndexFibre.jl:211`) vs. Marcatili's
  `(radius(m,z), 2π)` (`Capillary.jl:268`) — the field is genuinely
  evaluated (and integrated, for `N`/`Aeff`) out to 10 core radii into the
  cladding, not cut off at the core wall.
- **A real (Er,Eθ)→(Ex,Ey) rotation**, not a single closed-form Cartesian
  expression — `mode_angle_xy` has no equivalent step because Marcatili's
  formula is already Cartesian by construction.
- **A pre-existing bug in the `:odd`-parity branch**, found while reading
  this: line 271 above reads `sin(m.n*theta)` — lowercase `theta`, not the
  destructured `θ` (line 267 binds `θ`, never `theta`) — so `parity=:odd`
  throws `UndefVarError` today, on the Julia oracle, independent of any
  native work. No test in the repo exercises `parity=:odd` (confirmed:
  `test_interface.jl`'s only `StepIndexMode` test and `prop_stepindex`'s
  default both use `parity=:even`). This is a reason to scope any port to
  `:even` only, not a native-specific finding — `:odd` has no working
  oracle to validate against until this upstream bug is fixed.

### 3. The ω-dependence question — traced, and it is the single biggest simplification

`field`'s `ω=wlfreq(1030e-9)` default keyword, plus its own
`# TODO: how do we handle wavelength dependence?` comment, look like an open
design question. Tracing the actual call chain shows **Julia's own modal
pipeline never resolves it** — every caller invokes `field`/`Exy` without
an `ω` argument, so the keyword default is silently load-bearing everywhere,
for every mode type, not just `StepIndexMode`:

- `Modes.field(m::AbstractMode; z=0) = (xs) -> field(m, xs, z=z)`
  (`Modes.jl:48`) — no `ω`.
- `Modes.N` (`Modes.jl:55-66`) calls `field(m, z=z)` → the closure above.
- `Modes.Exy(m, xs; z=0.0) = field(m, xs, z=z) ./ sqrt(N(m, z=z))`
  (`Modes.jl:73`) — no `ω`.
- `Modes.ToSpace.to_space!` (`Modes.jl:491`), the function that actually
  drives `TransModal`'s per-node mode synthesis, calls
  `Exy(ts.ms[i], xs, z=z)` — no `ω`.
- `Modes.Aeff` (`Modes.jl:102-120`) uses `absE(m,z=z)` → `Exy(...)` → same
  chain, no `ω`.

So for **every** mode type reachable through `TransModal` (not a
`StepIndexMode`-specific carve-out), the *field profile itself* is always
evaluated at the fixed reference `ω = wlfreq(1030nm)`. For `MarcatiliMode`
this is invisible because its `field` has no `ω` parameter at all — the
weakly-guided capillary mode shape genuinely doesn't depend on frequency in
that model, only `neff(ω)`/`β(ω)` (evaluated separately, correctly,
per-frequency-bin, in `norm_modal`/the linop) do. `StepIndexMode`'s exact
Snyder & Love solution *does* have a real per-frequency mode shape (`u`,
`w` depend on `ω` through the dispersion relation) — but Julia's own
`TransModal` machinery never asks for it at any `ω` but the hardcoded
default, for every wavelength bin in the grid alike.

**Consequence for the port:** the native side only has to reproduce one
fixed `(u, w, a1, a2)` per mode — solved once, at one frequency, at setup —
not an `ω`-dependent field synthesized per grid bin. This collapses what
looked like "port a dispersion-relation root-finder into the RHS hot loop"
into "pass 4-6 more setup-time scalars per mode," exactly the same shape as
`unm`/`inv_sqrt_n` already are for Marcatili. **This is also a trap, not
just a simplification**: an implementer must reproduce this fixed-`ω`
behavior faithfully, including its physical inexactness, rather than
"fixing" it by making the native path genuinely `ω`-dependent. Native-only
`ω`-dependence would silently diverge from the Julia oracle with no
independent ground truth to justify the change (contrast Phase 7's β1,
which deliberately diverges from Julia but only after validating the new
value against an independent BigFloat truth — MATH.md §3.5, TESTING.md's
"deliberate divergence" tier). If the `# TODO` is ever resolved, it must be
resolved in Julia first — the oracle changes, then the port follows it,
never the other way round.

`Modes.N`/`Modes.Aeff` reach `field` the same no-`ω` way (confirmed above)
— no other call site in the low-level pipeline passes an explicit `ω`.

### 4. What new numerics Rust would actually need

**Reused, not new:** general-order Bessel **J** (`jn`, `diffraction.rs:102`,
Miller downward recurrence, already generalized past `n=1` for Phase E's
`TE`/`TM`/`HE_{n>1}` Marcatili support) covers the core-field Bessel J
evaluations (`besselj(n±1, u·R)`) directly — no new J-side code.

**Genuinely new: modified Bessel K, general integer order.** Confirmed
absent — `grep`ing the whole crate for `besselk`/`BesselK` returns nothing.
The cladding field needs `K_{n-1}`/`K_{n+1}` evaluated at `w·R` for
arbitrary `R ∈ (1, 10]` in the per-node hot loop (this cannot be
precomputed at setup — it varies with the cubature node's `r`, unlike
`u`/`w` themselves). Unlike `J_n` (which needs a *downward*-stable
recurrence because it decays), `K_n` is the numerically easier direction —
it is stably computed via the standard *upward* three-term recurrence
seeded from `K_0`/`K_1` (rational/Chebyshev approximations, e.g.
Abramowitz & Stegun 9.8.5-9.8.8 or Numerical Recipes' `bessk0`/`bessk1`),
since `K_n` grows (not decays) with order and the recurrence amplifies
error in the numerically safe direction. This is precedented work for this
crate — `j0`/`j1` (`diffraction.rs:36-84`) are already from-scratch
rational/power-series approximations of exactly this kind — but it is a
genuinely new special function, not a reuse.

**Can `u`/`w`/`v` (and more) be precomputed in Julia and passed as
constants?** Yes, and further than just `u`/`w`/`v`. Tracing exactly what
`field` computes and which pieces are `r`/`θ`-dependent (must run per
cubature node, in Rust) versus not (fixed once `ω`, `a`, `n`, `z` are fixed
— i.e. everything, given §3's finding that `ω` never varies):

| Quantity | Depends on `(r,θ)`? | Where it can live |
|---|---|---|
| `u`, `w` (dispersion-relation solve, `Roots.find_zeros` inside `findneff`) | no | Julia, precomputed once per mode |
| `v` | no — and unused after `f1f2` returns it (`field` never reads it) | dead value, doesn't need to cross FFI at all |
| `a1 = (f2-1)/2`, `a2 = (f2+1)/2` | no | Julia, precomputed once per mode |
| `j_u = besselj(n, u)`, `k_w = besselk(n, w)` (the *normalizing* denominators in `corefields`/`cladfields`) | no — `u`,`w` are fixed scalars | Julia, precomputed once per mode (Julia already has `SpecialFunctions.besselk` for this — no new Julia-side code needed either) |
| `besselj(n∓1, u·R)`, `besselk(n∓1, w·R)` | **yes** | must run in Rust, per node |
| `fv`, `gv` (parity-dependent `cos`/`sin` of `n·θ`) | yes | Rust, per node (cheap trig, same shape as `mode_angle_xy` already does) |
| the final `(Er,Eθ)→(Ex,Ey)` rotation | yes | Rust, per node |

So **no root-finder ports to Rust at all** — `find_zeros`/`findneff` stays
exactly where the accel-spline machinery already keeps it, in Julia, run
once per mode at setup (this is also why `test_interface.jl`'s
`StepIndexMode` test warns to always pass `accellims` for a real
propagation — that machinery exists for the `neff(ω)` sweep used
elsewhere, e.g. dispersion/`Stats.default`, not for this single-`ω` field
evaluation, but confirms the root-find is a known, already-mitigated setup
cost independent of this design). Per mode, Julia would hand Rust roughly
7 new scalars (`a_core`, `u`, `w`, `a1`, `a2`, `j_u`, `k_w`, a `parity` flag,
`n`) alongside the existing `inv_sqrt_n`/`order`/`kind`/`phi` — the same
shape of addition Marcatili's own `unm`/`inv_sqrt_n` already are, not a
new category of FFI payload.

**The one real structural (not just numerical) change: `modal_a` must
split into two radii.** Per §1, `modal_a` today is simultaneously the
cubature upper bound *and* the mode's field-argument radius
(`x = r·unm/a`) *and* the "return zero beyond here" cutoff — a coincidence
that only holds because Marcatili's mode is exactly zero outside its core.
`StepIndexMode` needs **two** distinct radii: `a_core` (core/clad branch
selector, `R = r/a_core`) and `a_outer = 10·a_core` (the actual cubature
bound, from `dimlimits`). `native_set_modal_params`'s signature and
`modal_pointcalc`'s cutoff/scaling logic both need widening for this — a
real (if small) change to the function's parameter shape and its hot-loop
branch structure, not a guard-only relaxation. Confirmed by grepping every
use of `self.modal_a`/`ro.modal_a` in `native.rs`: exactly the two roles
above, both currently assuming they're the same number.

### 5. Contrast with the sibling task (why this is not the same easy category)

`ZeisbergerMode`/`VincettiMode` (`src/Antiresonant.jl` `@eval` loops at
`:126` and `:381`) **delegate** `field`/`N`/`dimlimits` straight through to
a real, wrapped inner `MarcatiliMode` — their mode profile *is*
Marcatili's, bit-for-bit, by construction, not merely "similar." The
concurrent guard-relaxation for that case is exactly that: relax
`RK45.jl`'s `isa MarcatiliMode` check to also accept the wrapper types
(and unwrap `.m` before extracting `unm`/`kind`/`n`/`ϕ`) — the *existing*
`modal_pointcalc`/`mode_angle_xy`/`x = r·unm/a` Rust code runs completely
unmodified, because the underlying math genuinely is Marcatili's math.

`StepIndexMode` has no such inner `MarcatiliMode` and no delegation — §2
above shows its field formula is structurally unrelated (two radial
branches vs. one, Bessel K vs. no Bessel K at all, a genuine polar-to-
Cartesian rotation vs. a single closed-form Cartesian expression, a
different — and wider — integration domain). **The same guard line cannot
be widened to cover it**; supporting `StepIndexMode` needs a second,
parallel field-synthesis code path in Rust (new params, new per-node
branch, one new special function), not a relaxation of the existing one.
This is the single most useful fact this document settles: do not assume
the Zeisberger/Vincetti guard relaxation (landing concurrently with this
research) incidentally covers `StepIndexMode` too — it does not, and
cannot, by construction.

### 6. Verification strategy, if built

Two-tier equivalence per TESTING.md §4:

- **Single-step tier: expect ~1e-10, Phase-5-shaped.** Node placement is
  bit-identical (same `libcubature` binary, unchanged). The floor is set by
  Rust's new `besselk` agreeing with `SpecialFunctions.besselk` — verify
  this **standalone, before writing any native.rs code**, exactly like
  Phase 5 verified `j0` against `SpecialFunctions.besselj` first
  (MATH.md §3.3: "~1.5e-15 max absolute error... verified... not
  assumed"). Range to check: `w·R` for `R ∈ (1, 10]` at realistic `w`
  (order ~1-10 for typical step-index geometries) — this is the number
  that sets the honest tier to assert, not a guessed 1e-10.
- **Full-solve tier: ~1e-6 floor (fixed `max_dt=min_dt=dt`), per the
  standard discipline** (TESTING.md §3) — sidesteps adaptive-path
  divergence, isolates state accumulation.
- **Non-vacuousness, mirroring Phase 5's own lesson (MATH.md §3.3, "not
  skipped"):** use **two** co-eligible `StepIndexMode`s (different `n`/`m`,
  e.g. `HE11`+`HE12`-analogues) so the `to_space!` sum-over-modes matmul and
  the back-projection matmul are genuinely exercised — a single-mode test
  would leave both matmuls implicitly untested, exactly the Phase 5
  precedent this doc's task framing points at. Independently assert (before
  trusting the Rust-vs-Julia comparison) that inter-mode coupling actually
  moves energy between modes by a margin unmistakably above the FP floor —
  same two-part structure Raman's gate test used (MATH.md §5.3) after a
  first attempt at a Raman gate test passed for the wrong reason (an
  exact-zero effect, not a genuine null result).
- **OPEN, verify at implementation time, does not block this write-up:**
  whether `prop_capillary(...; modes=[stepindex1, stepindex2])` — a solid
  step-index-fibre mode basis inside a gas-capillary high-level entry
  point — actually constructs a well-formed, numerically sensible
  `TransModal` today (density/gas-fill semantics for a mode type that
  isn't gas-based are untested territory). If it doesn't construct
  cleanly, the equivalence test should build `TransModal` directly via the
  low-level API (`Modes.ToSpace`/`NonlinearRHS.TransModal`) instead of
  routing through `prop_capillary`, matching how several existing
  native-port tests already bypass the high-level entry points when the
  high-level combination isn't itself the thing under test.
- Scope the gate test (and any first implementation) to `parity=:even`
  only — §2's `theta`/`θ` typo means `:odd` has no working Julia oracle to
  validate against until that upstream bug is fixed independently of this
  work.

### If someone picks this up

1. **De-risk the one new special function first, in isolation.**
   Implement `besselk(order, x)` in `diffraction.rs` (or a new module) via
   upward recurrence from `K_0`/`K_1` rational approximations; write a
   standalone Rust unit test asserting agreement against
   `SpecialFunctions.besselk` (computed via a tiny throwaway Julia script,
   the same way `j0`/`jn`'s docstrings cite their verification) over
   `x ∈ (0, ~30]` (covers `w·R` for `R` up to 10 at realistic `w`). This
   single step sets the honest single-step tolerance tier for everything
   downstream and is almost all of the genuinely new work.
2. **Widen `native_set_modal_params`'s signature** to carry `a_core` (used
   for the `R = r/a_core` branch selector and the cubature's
   core/clad-transition point, not the outer bound) alongside a new
   `a_outer` (the `pcubature_v`/`hcubature_v_2d` domain bound,
   `= 10·a_core` from `StepIndexMode`'s own `dimlimits`) plus, per mode:
   `u`, `w`, `a1`, `a2`, `j_u`, `k_w`, a `parity` flag. Keep `unm`/existing
   Marcatili fields as-is (still used for co-propagating Marcatili modes in
   a mixed-type `TransModal` — check whether that combination needs
   supporting or can be rejected as out of scope for a first cut).
3. **Add a `StepIndexMode`-specific branch to `modal_pointcalc`** (or a
   parallel function selected by a per-mode type tag) implementing
   `corefields`/`cladfields`'s formula directly from §4's precomputed
   scalars plus the new `besselk`, followed by the `(Er,Eθ)→(Ex,Ey)`
   rotation — this is new Rust code, not a data-only change.
4. **Widen `src/RK45.jl`'s modal guard** (~line 1657) from `all(m -> m isa
   MarcatiliMode, modes)` to also accept `StepIndexMode`, dispatching to
   whichever native params-setup path matches each mode's concrete type;
   compute and pass the 7 new per-mode scalars from Julia (`f1f2`/`uwv`,
   already exported functions in `StepIndexFibre.jl`, reusable as-is for
   this purpose — no new Julia-side math needed, only new glue code calling
   already-existing functions and threading their outputs through the
   FFI call).
5. **Scope the first cut narrowly**, mirroring every prior phase's
   discipline: `parity=:even` only (§2), constant (non-tapered) radius
   only, Kerr-only to start (defer Raman — `modal_pointcalc`'s existing
   Raman block is written against the Marcatili scalar-field assumption
   and would need its own check for whether it generalizes unchanged),
   RealGrid only, `full=false` (radial-only cubature) only. Confirm the
   `prop_capillary(...; modes=[...])` reachability question (§6's OPEN
   item) before writing the equivalence test.
6. Follow `AGENTS.md` §3/§4 exactly: build the Rust lib, wire Julia, write
   the two-tier `@testitem tags=[:rust]` test, run the `rust` group plus a
   check for regressions in `sim-multimode`, and append a `PORT_LOG.md`
   entry recording both achieved tolerances and every gotcha found —
   in particular the `modal_a` split and the `theta`/`θ` typo, since a
   future reader will hit both again if they aren't written down here.
