## 6. Beyond-Luna math options (BACKLOG Phase J item 6)

Feasibility/design pass over the three items `native-port/MATH.md` §8 +
`SUGGESTIONS.md` items 15-17 grouped as "beyond-Luna math options": direct
DP5(4) embedded-error coefficients, direct PPT evaluation replacing the
spline LUT, and short-kernel overlap-save Raman convolution. Each is
individually assessed below against the β1 bar (`BETA1_ANALYTIC.md`): a
divergence from the Julia oracle is acceptable only with a derivation, a
ground truth that isn't Julia itself, a quantified error against that
ground truth, and a stated validation blast radius. This is a
feasibility/design pass, not an implementation — no Rust or Julia source
was changed.

| Item | Candidate | Verdict |
|---|---|---|
| 6.1 | Direct DP5(4) embedded-error coefficients (SUGGESTIONS #15) | **Recommend against — already implemented on both sides.** The backlog's own premise (`errest` computed as a runtime `y5−y4` cancellation) does not match the current code. Same class of finding as S5 item 3 ("backlog premise wrong"), not a genuine open design question. |
| 6.2 | Direct PPT evaluation replacing the spline LUT (SUGGESTIONS #16) | **Recommend against, as specified.** The accuracy gain is real but physically immaterial (LUT error is already far below anything the physics cares about); the cost is not — plasma is already a measured 24.5-40.5% of `rhs_radial` time on the *cheap* path, and direct evaluation is 1-3 orders of magnitude more expensive per call, with an unbounded tail (BigFloat adaptive quadrature) that cannot run in a hot loop at all. The stated secondary motivation (structurally eliminating an out-of-range segfault) is also moot — that segfault was already fixed by a cheaper, already-shipped guard. |
| 6.3 | Short-kernel overlap-save Raman convolution (SUGGESTIONS #17) | **Recommend, narrowly scoped.** Not full block-based overlap-save — just shortening the zero-pad to the truncated kernel length. Complementary to, not superseded by or exclusive with, item J.3's r2c/c2r halving (BACKLOG open remainder 2); the two multiply. Scope is the FFT-convolution Raman paths only (native `:SiO2` kernel + Julia `RamanPolarEnv`); the default carrier-field path (`raman.rs`'s ADE solver) is already O(N) and out of scope. Unlike the other two items and unlike β1, this one need not diverge from the Julia oracle at all if the cutoff is chosen below the f64 noise floor — the ground truth is the existing full-grid convolution, not a "more correct" reformulation. |

---

### 6.1 Direct DP5(4) embedded-error coefficients

**Verdict: recommend against — the code already does this.**

#### What the backlog item asserts

`MATH.md` §8.1 and `SUGGESTIONS.md` #15 both describe the problem as: "Luna
computes the DP5(4) embedded error as the difference of the 5th- and
4th-order solutions — a near-total cancellation," and propose replacing it
with "the estimate directly as `err = Σᵢ eᵢ·kᵢ` with precomputed
`eᵢ = b5ᵢ − b4ᵢ` coefficients (exact rationals, no runtime cancellation)."

#### What the code actually does

Neither side forms two full solution vectors and subtracts them.

- **Rust (`amalthea/src/native.rs`):** `DP_ERREST` (`native.rs:5889-5897`) is
  an array of *directly-typed rational literals* — `-71.0/57600.0`,
  `71.0/16695.0`, `-71.0/1920.0`, `17253.0/339200.0`, `-22.0/525.0`,
  `1.0/40.0` — computed once at compile time. The error estimate
  (`native.rs:4668-4710`, inside `native_step`) is `Σᵢ eᵢ·kᵢ` directly:
  `e0*k0 + e1*k1 + ... + e6*k6`. There is **no `DP_B4` array anywhere in
  `native.rs`** (confirmed by grep) — the accepted solution `yn` is formed
  only from `DP_B5` (the 5th-order weights, `native.rs:4622-4665`); a
  4th-order solution is never computed, let alone subtracted from the
  5th-order one.
- **Julia (`src/dopri.jl:14-18`):** `b5` and `b4` are each defined as
  `Float64`-rounded rational literals, and `const errest = b5 .- b4` is
  computed **once at module load** (a `const`, not recomputed per step).
  `RK45.jl:288-291`'s `step!` then accumulates `s.yerr += s.dt*s.ks[ii]*errest[ii]`
  — the same `Σᵢ eᵢ·kᵢ` direct-dot-product form, not a per-step `y5−y4`
  subtraction of full state vectors. `PreconStepper` (`RK45.jl:239`) reuses
  the same generic `step!`.

So both backends already implement "the standard, numerically superior
formulation" the item proposes. There is no runtime `y5−y4` cancellation to
remove from either hot loop.

#### The real (much smaller) discrepancy, and why fixing it doesn't help

The one genuine gap: Julia's `errest[i]` is computed via
`Float64(b5ᵢ_rational) − Float64(b4ᵢ_rational)` (double-rounded), while
Rust's `DP_ERREST[i]` is `Float64(eᵢ_rational)` (singly-rounded, from the
already-reduced fraction). Checked numerically (Python, reproducing both
computations bit-for-bit):

| Coefficient | Julia (`b5−b4`) | Rust (`DP_ERREST`, direct) | Match |
|---|---|---|---|
| E1 | −0.0012326388888888873 | −0.0012326388888888888 | 1 ULP off |
| E3 | 0.004252770290506136 | 0.0042527702905061394 | 1 ULP off |
| E4 | −0.036979166666666674 | −0.03697916666666667 | 1 ULP off |
| E5 | 0.05086379716981132 | 0.05086379716981132 | exact |
| E6 | −0.04190476190476192 | −0.0419047619047619 | 1 ULP off |
| E7 | 0.025 | 0.025 | exact |

4 of 6 nonzero coefficients differ by exactly 1 ULP (~1e-15 to ~1e-17
relative to the coefficient's own magnitude — the same order as the
FFTW/BLAS summation-order noise already documented as unavoidable, TESTING.md
§3). This is a genuine, tiny, **one-time** (not per-step) discrepancy
between the two backends' constants — but it is not the mechanism CLAUDE.md's
"Phase 2 gotcha" and TESTING.md §3 describe. That mechanism is: the DP5(4)
embedded pair is constructed so its error estimate is, by design,
mathematically small whenever the true local truncation error is small —
which is exactly the regime a weakly-nonlinear early propagation step is
in. When the *true* `err` sits near the ~1e-15 floor, any independent
~1e-15-relative noise in the `kᵢ` themselves (from Julia's BLAS/FFTW vs.
Rust's Rayon-sequential summation order — an unrelated, unavoidable source,
same TESTING.md §3) dominates the signal, producing the documented ~20%
relative swings and the resulting adaptive-path divergence. That mechanism
lives entirely in the `kᵢ` (the RHS evaluations), not in how the fixed
`eᵢ` constants are computed or represented — and no reformulation of the
coefficients touches it.

**Concretely: aligning Julia's `errest` to Rust's directly-typed rational
literals** (a one-line, free, zero-runtime-cost change — replace
`errest = b5 .- b4` with the reduced-fraction literals, mirroring
`DP_ERREST`) **would close the ~1e-15-relative constant gap without
addressing the actual, much larger, and irreducible source of divergence.**
It is available and harmless, but should not be sold as "dissolving the
fixed-step test discipline" — it doesn't, because the dominant noise
source it leaves untouched (cross-backend `kᵢ` summation-order noise,
amplified by an intrinsically tiny true `err` for smooth problems) is
unrelated to coefficient precision.

#### Acceptance test / ground truth, if the micro-fix were made anyway

Not a "more correct than Julia" claim in the β1 sense — both constants are
already accurate to their own last bit; this is purely a
bit-for-bit-alignment question. Ground truth: the exact rational value of
each `eᵢ`, computed once and compared to both backends' current constants
(the table above already does this). No BigFloat/analytic derivation
needed — the "ground truth" is exact rational arithmetic, trivially exact.

#### Validation blast radius

Because the micro-fix changes a controller input (`err`) used by every
adaptive (non-fixed-`dt`) test, it is technically downstream of the entire
suite (Phase 8's gate: 46590 pass / 12 broken, 46602 total,
`TESTING.md` §4). In practice: the perturbation (~1e-15 to ~1e-17 per
coefficient) is smaller than the FFTW/BLAS run-to-run noise floor
(~2e-8, TESTING.md §3) already tolerated by every full-solve test, so no
currently-passing assertion is expected to flip. The honest statement is
"technically all ~46.6k assertions are downstream of the step controller;
practically none should move, because the change is smaller than noise
already priced into every tolerance" — not "zero blast radius." Given the
fix buys nothing for the actual documented failure mode, this doesn't
clear the bar for spending a validation pass on it.

---

### 6.2 Direct PPT evaluation replacing the spline LUT

**Verdict: recommend against, as specified** (replacing the LUT with a
per-call direct evaluation in the hot RHS loop). Below is the accuracy
case, the cost case, and the one narrower variant that might still be
worth a bench — restated as an open question with an explicit measurement
gate, not a recommendation.

#### What's being proposed

`amalthea/src/ionization.rs`'s `PptIonizationRate` evaluates a natural
cubic spline fit to `ln(rate)` over `|E|` (`CubicSplineLUT`, built from
`N=2^16` knots by default — `src/Ionisation.jl:533`'s
`IonRatePPTAccel(...; N=2^16, ...)` — with a strict lower-bound cutoff
`e_min` and a clamp-not-error policy above `e_max`,
`ionization.rs:242-289`). Julia's own `IonRatePPTAccel` (`Ionisation.jl:480-518`)
is the same idea one level up — a `Maths.CSpline` fit to the same
log-rate data, cached to disk (`~/.luna/pptcache`, `Ionisation.jl:528-570`)
so the expensive fit is amortized across process launches. The item
proposes evaluating the true PPT series (`IonRatePPT`, `Ionisation.jl:367-422`)
directly on every call, on both sides, removing interpolation error
entirely.

#### Accuracy argument: real, but immaterial

Evaluating the PPT formula directly is unambiguously "more correct" than
interpolating a spline through it, in the same direction as β1 (Rust
computing something exact instead of an approximation). But unlike β1,
where Julia's FD error against the true derivative was ~1e-12 and
accumulated *coherently* over a broadband, long-propagation to a physically
meaningful ~1e-4–1e-7 full-solve effect, the LUT here is fit at
`N=2^16` = 65536 knots in log-space over the field-strength axis. A
natural cubic spline at that knot density interpolates a smooth,
monotonic-ish function (`IonRatePPT` output) to an error far below any
physical tolerance the simulation carries elsewhere (this is exactly why
`TESTING.md` §2 classifies "LUT/spline interpolation" as its own
~1e-8-tier bucket, already tighter than the ~1e-6 floor tier most
full-solve tests live at). There is no evidence, measured or structural, of
a coherent, propagation-scale error mechanism here the way there was for
β1 — the interpolation error is uncorrelated, essentially white noise
across the field-strength axis, at a magnitude below the noise floor the
suite already tolerates. **"More correct" is true; "buys anything" is not
demonstrated,** which is the opposite of β1's case.

#### Cost argument: this is where it fails

`amalthea/src/ionization.rs`'s `PptIonizationRate::rate` today is a binary
search (`~log2(65536)≈16` comparisons) plus a cubic evaluation
(`a + dx*(b + dx*(c + dx*d))`, ~6 flops) — call it O(10ns) per grid point,
per stage, and it vectorizes/parallelizes trivially (`rate_vector`,
`ionization.rs:291-306`, with an optional GPU path).

The PPT formula it replaces (`Ionisation.jl:367-422`) is materially more
expensive, and in its default configuration has an **unbounded** tail:

- `sum_integral` defaults to `false` (`Ionisation.jl:347-348`), meaning the
  rate is not the closed-form integral approximation but an explicit
  series summed via `Maths.converge_series` to `rtol=1e-6`
  (`Ionisation.jl:409-413`) — an iteration count that depends on the
  Keldysh parameter `γ` and isn't bounded a priori.
- Each series term calls `φ(m, x)` (`Ionisation.jl:427-461`), which for
  `m=0` is a Dawson-function evaluation, for `m≠0, x≤26` is a confluent
  hypergeometric function (`pFq`) times gamma functions, and **for
  `m≠0, x>26` falls back to adaptive quadrature (`hquadrature`) over a
  `BigFloat` integrand** (`Ionisation.jl:453-459`). That last branch is
  categorically impossible to run in a Rust hot loop without either
  reimplementing arbitrary-precision arithmetic and adaptive quadrature in
  Rust (a large, separate porting project with no existing primitive to
  reuse — `cubature.rs`'s `libcubature` binding is `f64`-only) or accepting
  silent divergence from Julia's answer exactly in the regime it was added
  to handle.
- Even the cheaper `sum_integral=true` closed-form variant
  (`Ionisation.jl:407`, `s = sqrt(π)*factorial(mabs)*β^mabs/...`) requires
  several `pow`/`exp`/`asinh`/`sqrt`/`gamma`-adjacent evaluations per call
  — call it conservatively 20-30 transcendental-function-class operations,
  each tens of ns, i.e. **~0.5-1.5µs per call**, a rough 50-150x per-call
  slowdown versus the spline even before counting the accuracy cost of
  using a cheaper formula than Julia's own default oracle uses (see below).

The plasma path is not a cold path: `apply_plasma_radial`/the mode-averaged
plasma step calls the ionisation rate once per time-domain grid point,
every RHS stage, every accepted-or-rejected step. Measured
(`docs/dev/BACKLOG.md`, S2 Phase 2 item 2, `Instant`-based profiling,
add/measure/revert discipline): **plasma is 24.5-40.5% of `rhs_radial`
wall time today, on the cheap spline path**
(N=32 r-points: 40.5%; N=128 r-points: 24.5%). A 50-150x per-call slowdown
on a term that is already a quarter-to-two-fifths of RHS time would not
shave a few percent off a solve — it would make plasma-enabled propagation
dominated by ionisation-rate evaluation, likely a multi-x regression on
total wall time for exactly the configs (strong-field, plasma-heavy) where
the native port's performance case matters most. And that estimate is for
the *cheap* variant; the *default* (`sum_integral=false`, unbounded series
+ occasional BigFloat quadrature) is not boundable at all — it cannot be
given a worst-case per-call cost, which alone disqualifies it from a
allocation-free, real-time-budgeted hot loop.

There is also a version-mismatch problem baked into "replace the LUT with
direct evaluation": if the *cheap* (`sum_integral=true`) formula is used
for speed, that is no longer the same ground truth the LUT was fit to
(which uses Julia's default `sum_integral=false`) — so switching would
trade a well-characterized, sub-noise-floor interpolation error for an
uncharacterized, physically real "neglects the multiphoton thresholds"
approximation error (the docstring's own words, `Ionisation.jl:285-286`).
That is not obviously a net accuracy win, and is not the β1 pattern
("more correct, unambiguously") at all.

#### The stated secondary motivation is already moot

`SUGGESTIONS.md` #16 and `MATH.md` §8.2 both cite "structurally eliminates
the out-of-range segfault (BACKLOG Phase J item 2)" as a benefit. That
segfault is **already resolved** (`ARCHIVE.md`, Phase J archived item 2,
"Resolved 2026-07-09"): the root cause was a NaN/±inf comparison gap (IEEE
754 makes every comparison with NaN false, silently falling through both
the `< e_min`/`> e_max` guards), fixed by an explicit `is_finite()` check
in both `CubicSplineLUT::evaluate` and `PptIonizationRate::rate`
(`ionization.rs:242-289`), backed by 11 new Rust unit tests
(`ionization.rs:449-591`, including `ppt_non_finite_inputs_never_panic`,
`ppt_extreme_sweep_never_panics`) sweeping ±1e10× the fitted range in both
signs without a panic. The LUT already has no fitted-range-based failure
mode left to eliminate; a direct-eval rewrite would not be closing an open
safety gap, it would be redundant with a cheaper fix already shipped.

#### Acceptance test / ground truth, if pursued anyway

The direct sum against a `BigFloat` evaluation of the same series (as
`SUGGESTIONS.md` #16 itself proposes) is the right ground truth *for the
accuracy question* — but accuracy was never the blocker here. Cost is. Any
future revisit should lead with a Criterion microbenchmark of
`IonRatePPT`-direct (both `sum_integral` settings) against
`PptIonizationRate::rate`, at the actual per-RHS-call rate (once per
`n_time × n_r` point, or `n_time` for mode-averaged), gated the same way
S5 item 1's mixed-precision spike was: **measure first, in a
timeboxed/reverted benchmark, before writing any production code**, against
an explicit threshold (e.g. "total plasma-path time must not exceed
current + X% of a representative solve's wall time" — pick X based on
how much of the native port's advertised speedup budget plasma is allowed
to consume, not an arbitrary number invented here).

#### Validation blast radius

If measurement somehow cleared the cost bar (unlikely given the analysis
above, and only conceivable for the `sum_integral=true` variant with a
hard cap disallowing the `m≠0,x>26` BigFloat branch — i.e., a genuinely
different, narrower formula than Julia's default, which would itself need
its own accuracy sign-off before being called a drop-in replacement): every
plasma-enabled test in `sim-propagation`, `sim-multimode`, and the `rust`
group's ionisation-specific tests (`test_ionisation_rust.jl`) would need
re-validation at whatever tier the new method/ground-truth pairing
justifies — likely the "method/spline" ~1e-8 tier stays, since both old
and new paths interpolate or approximate the same underlying physics, just
via a different intermediate representation.

---

### 6.3 Short-kernel overlap-save Raman convolution

**Verdict: recommend, narrowly scoped to pad-shortening — not full
block-based overlap-save.**

#### Scope: which Raman kernel this touches

There are two distinct native Raman kernels (`CLAUDE.md`'s kernel-wiring
table, `MATH.md` §5.3, §8.4):

1. The **default carrier-field path** (`raman.rs`'s
   `TimeDomainRamanSolver`, an explicit exponential-integrator ADE
   solve) — already O(N), allocation-free, no FFT at all. **Out of scope**
   for this item; there is no convolution to shorten.
2. The **intermediate-broadening `:SiO2` path** (`native.rs`'s
   `raman_fft_e2`/`raman_fft_hw`/`raman_fft_plan`, `native.rs:301-325,
   1490-1523`, wired via `native_set_raman_fft_params`) and its Julia
   counterpart `RamanPolarEnv` (`Nonlinear.jl:327` onward) — both use a
   **full c2c FFT over a zero-padded double-length grid**
   (`2·n_time_over`), every RHS call. This is the one the item, and BACKLOG
   open remainder "Phase J.3," both target.

#### Why the grid is doubled today, and why truncation is safe in principle

Luna's own comment (`Nonlinear.jl:443-447`) explains the doubling: it gives
an "accurate full convolution between the full field grid and the full
extent of the impulse response function, with no truncation of the
response function... this is safe, until we come up with [something more
efficient]." For the SiO2 intermediate-broadening (Hollenbeck & Cantrell
Gaussian-damped) response, `h ≈ 0` beyond roughly 100 fs on a
multi-picosecond simulation grid — already noted as the reason a stale-tail
bug in the padded buffer was numerically invisible in every existing test
(`ARCHIVE.md`, Phase I archived item 1). That is the concrete basis for
truncating the kernel: measure the index `M` where `|h|` drops below (say)
a few ULPs of its peak in `f64`, and use a pad of `~n_time_over + M`
instead of `2·n_time_over`.

#### Decomposing the actual speedup — the "overlap-save" name overclaims

The naive framing ("kernel is a small fraction of the grid → convolution
gets proportionally cheaper") is wrong because FFT cost is `O(N log N)`,
not `O(N)`, and there are two structurally different ways to exploit a
short kernel:

- **Pad-shortening only (recommended here):** keep a single FFT per RHS
  call, but size the zero-pad to `n_time_over + M` (rounded to an
  FFT-friendly length) instead of `2·n_time_over`. This is a **clean,
  low-risk ~2x** in the typical case where `M ≈ n_time_over` shrinks the
  transform length by roughly half — no algorithmic change, same
  convolution theorem, same single resident FFTW plan (just shorter),
  same code shape as the existing `raman_fft_plan`. It **stacks
  multiplicatively** with item J.3's r2c/c2r halving (also ~2x, from using
  a real-valued transform instead of c2c on data that's real in both
  domains) — combined, roughly **~4x** on this specific kernel.
- **Full block-based overlap-save** (many short FFTs, one per
  `M`-sized input block, standard fast-convolution machinery): adds only
  the *additional* `log(N)/log(M)` factor on top of the above — for
  realistic `M/N` ratios here (kernel maybe 5-10% of the padded grid),
  that's roughly **another 1.3-1.5x**, at materially higher implementation
  complexity (block bookkeeping, save/discard logic, edge handling at
  segment boundaries) and more surface area for a correctness bug.

**The bulk of the achievable win is pad-shortening alone.** Full
block-based overlap-save is a real but much smaller additional
multiplier for a lot more code — worth deferring until pad-shortening
plus J.3's r2c/c2r halving are landed and measured, not worth bundling into
the same change.

#### Relationship to item J.3 (r2c/c2r halving)

**Complementary, not superseded by or exclusive with it.** J.3 attacks the
*representation* of the transform (complex-valued FFT on real-valued data →
real-valued FFT, same length, ~2x from skipping the redundant
conjugate-symmetric half). Pad-shortening attacks the *length* of the
transform (shorter effective kernel → shorter pad → smaller `N` in the
`O(N log N)` cost, another ~2x). Neither subsumes the other; a combined
r2c/c2r-on-a-shortened-pad implementation captures both multipliers
(~4x) with one coordinated change, and is a natural single unit of work
alongside — or immediately following — whichever agent lands J.3, so the
two don't produce a merge conflict on the same `native.rs` FFT-setup code.

#### Why this one, uniquely among the three, need not diverge from the oracle

Items 6.1 and 6.2 either found nothing to change (6.1) or would need a
genuine, physically real accuracy/cost tradeoff against the Julia oracle
(6.2). Item 6.3 is different in kind: **if the truncation cutoff `M` is
chosen so that everything discarded is below the f64 noise floor relative
to the kept kernel's peak, the truncated convolution and the full-grid
convolution are the same computation to within noise — not a "more
correct than Julia" divergence in the β1 sense, but a "provably
equivalent to the existing native SiO2 kernel and to `RamanPolarEnv`,
just computed over a shorter buffer" claim.** The ground truth is the
existing full-double-length-grid convolution (Rust-vs-Rust, or
Rust-vs-Julia's `RamanPolarEnv`), not a BigFloat/analytic derivation — a
materially easier bar than β1's, precisely because nothing here is
claimed to be *more accurate than Luna's physics*, only *cheaper to
compute the same physics*.

Two implementation choices follow from this:

- **Truncate only where measured, per-material, at setup** — not a fixed
  cutoff. `MATH.md` §8.5's own framing ("checked at setup, falling back to
  the full double grid otherwise") is right: compute `M` from the actual
  `h` array being used (SiO2's specific Gaussian-damping constants,
  already resident), not a hardcoded 100fs guess, so the mechanism is
  correct for any future material with a longer-tailed response.
- **If the cutoff is chosen conservatively enough that truncation error is
  below f64 noise, both `RamanPolarEnv` (Julia) and the native `:SiO2`
  kernel can adopt the shorter pad in the same commit**, preserving exact
  bit-parity between them (same discipline `SUGGESTIONS.md` #15 correctly
  insists on for the DP5(4) coefficients: land both sides together). This
  keeps the change inside the *existing* SiO2 equivalence tier
  (`~1e-6`-ish, FFT-method class, `TESTING.md` §2) rather than opening a
  new "deliberate divergence" tier.

**Hard boundary (already correctly stated in `MATH.md` §8.5, repeated
here for completeness):** do not go further and fit a recursive/IIR filter
to the Gaussian-damped response to avoid the FFT altogether — that is the
multi-SDO approximation trap `ARCHIVE.md`'s Phase I item 2 explicitly
rules out ("prefer analytic over LUT/fit-the-noise").

#### Acceptance test / ground truth

Rust-vs-Rust (truncated-pad kernel vs. the existing full-`2·n_time_over`
kernel, same inputs, single RHS evaluation) at whatever tier the measured
truncation error justifies — expect ~bitwise-to-reassociation
(`< 1e-13`) if the cutoff is chosen conservatively per the discussion
above, since the discarded tail is defined to be below the f64 noise
floor. If Julia's `RamanPolarEnv` is changed in the same commit, the
existing native-vs-Julia SiO2 full-solve test keeps its current tolerance
tier unchanged (fixed-step discipline per `TESTING.md` §3, since any
transform-length change alters summation order).

#### What to measure before implementing

1. **The effective kernel length `M`** for the SiO2 response actually used
   in the test suite and in representative user configs: the smallest `M`
   such that `max(|h[M:]|) < ε · max(|h|)` for `ε` a few ULPs of `f64`
   (e.g. `1e-15`-`1e-14` relative). This single number decides both
   pad-shortening's multiplier (`M / n_time_over`, roughly) and whether
   full block-based overlap-save's extra `log(N)/log(M)` factor is worth
   its complexity — if `M` turns out close to `n_time_over` (i.e. the
   response barely decays on the grid), there is little to gain and this
   item should be shelved for that config.
2. **A wall-clock benchmark of the shortened-pad FFT vs. the current
   `2·n_time_over` FFT**, isolated (Criterion, add/measure/revert
   discipline per S1.6/S5.1), to confirm the ~2x (or combined ~4x with
   r2c/c2r) actually materializes on this codebase's FFTW binding and
   plan-reuse pattern, before committing to a production change.

#### Validation blast radius

Narrow by construction: only the `:SiO2`/`RamanPolarEnv` path is touched
(`prop_gnlse(...; ramanmodel=:SiO2)`'s reachable configs, per `native.rs`'s
own comment at `native.rs:5516-5519`). The default carrier-field SDO
path, every Kerr-only/plasma-only config, and every other geometry are
untouched. If both Julia and Rust change together (recommended), the
existing SiO2-specific tests are the entire blast radius — no
suite-wide adaptive-step-path perturbation the way 6.1's micro-fix would
cause, because the *result* is designed to be numerically unchanged, only
the FFT length differs.
