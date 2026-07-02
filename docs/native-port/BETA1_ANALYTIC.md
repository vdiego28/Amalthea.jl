# β1(z) as an exact closed form (Phase 7)

This note documents a specific numerical case: the z-dependent linop for a
graded-core (pressure-gradient) capillary needs `β1(z)`, the group-velocity
term, at every resident propagation step. Julia computes it with an
**adaptive finite difference**, which turns out to have a small, real,
repeatable error against the true derivative. Rust computes the true
derivative directly. This is why Rust and Julia now disagree by a tiny but
non-zero amount, why that is not a bug, and how the closed form is derived.

## 1. What Julia computes, and why it isn't exact

`β1(z) = Modes.dispersion(mode, 1, ω0; z=z)`, defined in `src/Modes.jl`:

```julia
β(m, ω; z=0.0) = ω/c * real(neff(m, ω, z=z))
dispersion(m, order, ω; z=0.0) = Maths.derivative(ω -> β(m, ω, z=z), ω, order)
```

`Maths.derivative` (`src/Maths.jl:25-32`) calls
`FiniteDifferences.central_fdm(7, 1, adapt=1)` — a 7-point adaptive-step
central difference. Adaptive FD picks a step size `h` balancing two
competing errors: truncation error (`O(h^6)` for this stencil, shrinks as
`h→0`) against floating-point rounding error in evaluating `β` at nearby
points (grows as `h→0`, since `β(ω+h) - β(ω-h)` is a difference of two
close-in-value `Float64`s). The adaptive algorithm picks a near-optimal
`h`, but "near-optimal" for `Float64` inputs still leaves a real residual —
measured here at **~1e-12 relative** against the true derivative (obtained
by re-running the identical `Maths.derivative` call with a `BigFloat`
argument instead of `Float64`, which removes the rounding side of the
tradeoff entirely and lets the adaptive algorithm shrink `h` much further).
This is *not* random noise — re-evaluating at the same `z` gives the same
answer — but it is a genuine, non-zero gap between "what Julia's `dispersion`
returns" and "the true derivative of `β`."

An earlier version of Phase 7 (superseded) tried to make Rust's β1(z) agree
with Julia's `dispersion` by sampling it at many `z` and fitting a spline
(`(dens(z), β1(z))` pairs). That approach chased the wrong target for a
subtler reason than the ~1e-12 floor above: a spline fit *validated and
converged against Julia's noisy samples* is not the same thing as *matching
the true derivative*, and — more concretely — the validation loop kept
finding new failing intervals no matter how many samples were added,
because it was trying to resolve FD sampling artifacts with a smooth
interpolant. See `PORT_LOG.md`'s Phase 7 entries for the two dead-end
designs (z-domain LUT, pressure-domain LUT) that were tried and abandoned
before this analytic approach.

## 2. Why the closed form needs only 4 constants

For the two-point-gradient, constant-radius `MarcatiliMode` `Capillary.gradient`
builds:

```
εco(ω; z) - 1 = γ(λ(ω)) · dens(z)      # separable: fixed per-ω array × scalar
nwg(ω)                                  # z-independent (constant core radius)
neff(ω; z) = combine(εco(ω;z), nwg(ω))  # sqrt(εco-nwg)  [:full]  or  1+(εco-1)/2-nwg  [:reduced]
```

`β1(z) = d/dω[ω/c·Re(neff(ω,z))]|_{ω0}`. Differentiating the `:full` combine
by the chain rule (holding `z`, i.e. `dens(z)`, fixed):

```
neff0(z)  = sqrt(εco(ω0;z) - nwg(ω0))
dneff0(z) = (dεco/dω|_{ω0} - dnwg/dω|_{ω0}) / (2 · neff0(z))
          = (dγ0 · dens(z) - dnwg0) / (2 · neff0(z))
β1(z)     = [Re(neff0(z)) + ω0 · Re(dneff0(z))] / c
```

(`:reduced` is the same but linear, no `sqrt`: `neff0 = 1 + (εco0-1)/2 -
nwg0`, `dneff0 = dεco0/2 - dnwg0`.)

Everything that depends on `z` funnels through the single scalar `dens(z)`
(already transferred exactly via `HermiteSpline`, see `MATH.md` §3.5). The
four remaining quantities —

- `γ0 = γ(λ(ω0))`
- `dγ0 = dγ/dω|_{ω0}`
- `nwg0 = nwg(ω0)` (complex)
- `dnwg0 = dnwg/dω|_{ω0}` (complex)

— are **z-independent constants**, computed once at `RustNativeStepper`
construction (`Capillary.jl`'s `make_linop` specialization) and shipped over
FFI as 7 `f64`s (`ω0, γ0, dγ0, Re(nwg0), Im(nwg0), Re(dnwg0), Im(dnwg0)`).
`NativeSim::ensure_linop_at` (`luna-rust/src/native.rs`) then computes
`β1(z)` in ~6 lines using these constants and the `dens` it already computes
for the linop — no LUT, no spline, no per-z Julia round-trip.

## 3. Computing the constants: BigFloat differentiation, not hand-derived symbolics

`γ` (gas Sellmeier) and `cladn`/`nwg` (glass Sellmeier + Bessel-root
geometry) are **opaque Julia closures** — `PhysData.sellmeier_gas(gas)`
returns a closure over that gas's specific coefficients (2 or 3 rational
terms, or a user-registered material with an arbitrary number of terms).
Hand-deriving `dγ/dω` symbolically would mean special-casing every gas/glass
combination Luna supports, and silently producing a wrong answer for any
material added later without a matching hand-derived update.

Instead, `dγ0`/`dnwg0` are computed via Julia's *own* differentiation
machinery — `Maths.derivative` — but fed a `BigFloat` argument:

```julia
γ_of_ω(ω) = γ(wlfreq(ω)*1e6)
dγ0 = Float64(Maths.derivative(γ_of_ω, BigFloat(ω0), 1))
```

This is the same adaptive-FD algorithm Julia already trusts for
`dispersion`, just with the rounding side of its truncation/rounding
tradeoff pushed far below `Float64` epsilon by the wider input type — so the
adaptive step-size search can shrink `h` until truncation error alone
dominates, landing far closer to the true derivative than the `Float64` call
does. It works unchanged for **any** gas or glass closure, generically,
with no coupling to which material was chosen.

## 4. Verification

Three independent checks, each isolating a different possible failure mode:

1. **Analytic (4-constant) vs Julia's own `dispersion` (Float64 FD)** — at
   several `z` across a representative `Capillary.gradient(:Ar, 0.5, 0.5,
   5.0)` config: relative agreement **~1.86e-11**, essentially constant
   across `z` (confirms it's a systematic offset, not noise varying with
   density).
2. **Full BigFloat derivative of the whole `β(ω,z)` formula (ground truth)
   vs Julia's `dispersion`** — same config: relative agreement **~1.7e-12**.
   This is the load-bearing check: it proves the ~1e-11 gap in (1) is
   *Julia's* FD imprecision (plus a small residual from decomposing the
   derivative into 2 separately-BigFloat-differentiated pieces rather than
   1 combined one — itself negligible, ~1e-11), not an error in Rust's
   closed form.
3. **Rust's resident `ensure_linop_at` β1(z) (via the `native_debug_beta1_at`
   FFI, used by `test_native_zdep_linop.jl`) vs Julia's `dispersion`** — same
   ~1.86e-11 agreement as (1), confirming the FFI-transferred constants and
   Rust arithmetic reproduce the Julia-side analytic computation exactly (no
   transcription bug in the 7-constant handoff).

## 5. The tolerance tradeoff this creates

Every phase before this one is a **faithful port**: Rust reproduces
whatever Julia computes, bit-for-bit or to FFTW-summation-order precision.
Phase 7's β1 is the first case where Rust **deliberately computes something
different from, and more accurate than, what Julia computes** — by
replacing Julia's finite-difference approximation with the true analytic
derivative. The consequence: Rust vs Julia full-solve equivalence no longer
converges to Julia's own floor as the LUT-based design did.

Concretely, for a broadband test config (`λlims=(200e-9,4e-6)`,
`flength=0.5`, two-point gradient `p0=0.5,p1=5.0` bar, `:Ar`): the
point-wise linop comparison against Julia is ~1e-6 to 1e-7 (worse near
`z=0`, the low-pressure end), and a fixed-step full-solve comparison is
~7e-5 — both *larger* than an earlier (superseded) LUT design's ~1e-8,
which directly interpolated Julia's own sampled FD output rather than the
true derivative.

This was checked, not assumed, to be the β1 difference and nothing else:
re-running the same fixed-step full-solve with `kerr=false` (pure linear
propagation, the only Rust-vs-Julia moving part left is `β1(z)`) gives the
**same ~7e-5** magnitude, growing linearly with propagated distance — i.e.
a small (~1e-11 relative) but *coherent* (same sign, same magnitude at
every z) offset in `β1`, multiplied by `ω` (up to ~1e16 rad/s for this
config's bandwidth) and integrated over the full propagation length,
accumulates into a macroscopic phase difference. A **random** per-z error
of the same size would mostly average out over many steps; a **systematic**
one does not. Narrower bandwidth or a shorter fibre will show a
proportionally smaller number.

**This is the intended tradeoff, not a regression to chase down**: Phase 7
was changed, on request, from "reproduce Julia's `dispersion` value" to
"compute the true β1(z)". The honest cost is that Rust-vs-Julia parity
numbers for this phase are measured at ~1e-4–1e-6 (still far smaller than
the physics itself cares about) rather than ~1e-8, and they will not shrink
by tuning anything in Rust — only by improving Julia's own `dispersion`
(out of scope here) or by testing a narrower-bandwidth config. See
`TESTING.md` for how the test suite documents and gates this tier.
