# Inbox note — S5 item 3: order-5 dense output (continuous extension)

Started by an agent in worktree `agent-ada2bad7efb980155` (branch
`worktree-agent-ada2bad7efb980155`, WIP commit `63b6003`), which hit the
account spend limit mid-investigation with a self-declared blocker. Finished
by the lead on branch `s53-dense-order5` (rebased onto `main`), which
diagnosed and fixed the blocker. This file is the merge-ready record; every
code comment added by this item that points at
`docs/dev/native-port/portlog-inbox/dense-order5.md` points here.

---

## 1. PORT_LOG.md entry (append verbatim, in the standard template)

```
## 2026-07-23 — S5 item 3 — order-5 dense output + the FSAL/k1 dense-output bug — Claude (opus-4.8, finishing sonnet-5's WIP)
**Status:** complete
**Did:** Replaced the quartic (order-4) continuous extension used for dense
output between accepted steps with the Calvo–Montijano–Rández order-5
interpolant, on *both* the resident-native and the pure-Julia steppers. In
the process, found and fixed a pre-existing correctness bug — inherited
verbatim from upstream Luna and faithfully re-ported into all three of
Amalthea's own steppers — that had been silently collapsing dense output to
**first order** everywhere.
**How:**
- **The bug.** `RK45.jl`'s `step!` performed the FSAL carry
  `s.ks[1] .= s.ks[end]` (k7→k1) at the moment a step was *accepted*. But
  `interpolate(s, ti)` is called *after* that, for output points lying inside
  the interval that just finished, and it needs that interval's genuine k1.
  It was instead handed k7, which differs from k1 by O(h) — so the
  continuous extension reproduced only `y0 + σ·h·y'(t0)` correctly and its
  local defect degraded from O(h⁵) to O(h²). Measured on a real
  `prop_capillary` config: order-4 local defect ratios of **3.996 / 3.999 /
  4.000** per halving (i.e. exactly h²) instead of the expected 32.
  Identical eager copies existed in `native.rs::step` (`ks.split_at_mut(6)`,
  accepted branch), `ffi.rs::precon_step_ffi`, and
  `cuda_native.rs`'s accepted branch.
- **The fix.** Defer the FSAL carry from accept-time to the *top of the next
  step*, immediately before the pre-existing re-framing of `ks[0]` into the
  new interaction-picture frame (`prop!(ks[1], t, tn)` in Julia,
  `apply_prop_cached(&mut s.ks[0], …, t_new - t_old)` in Rust). The
  arithmetic is untouched — copy still precedes reframe — it just moves
  across the step boundary, so accepted-step values stay bit-identical while
  `ks[0]` survives long enough to be interpolated with. Guarded so a
  rejected-step retry (where `ks[0]` is already the right k1) does not copy:
  `s.ok` in Julia, a new `CpuNativeSim::fsal_pending` flag in `native.rs`
  (also cleared by `set_field`, which recomputes `ks[0]` from scratch), and
  `t_new > t_old` in `ffi.rs`/`cuda_native.rs`, which is the same predicate
  given Julia leaves `s.tn == s.t` on rejection and on the initial state.
- **The order-5 interpolant.** `dopri.jl` gained `dp5_extra_c`,
  `dp5_extra_a7`, `dp5_extra_a8` and the 5×9 `interpC5` basis table;
  `native.rs` gained the mirrored `DP5_EXTRA_*` constants, two trailing
  stage slots (`ks` is now `[Vec; 9]`), `CpuNativeSim::eval_extra_stage`
  (per-geometry dispatch generalized to an arbitrary `(c_node, stage_idx)`,
  reusing the unmodified `rhs_*` kernels), and the
  `native_compute_extra_stages`/`get_extra_stage` FFI pair. `RK45.jl` gained
  the shared `interpC5_weights` and `_dp5_extra_stages!` helpers so the two
  steppers cannot diverge on interpolant math.
- **Laziness.** The 2 extra RHS evaluations happen only inside
  `interpolate`, never in `step!` — an accepted step nobody asks for dense
  output on pays nothing. Backends without the FFI (the CUDA-resident one)
  return -1 and fall back to the order-4 `interpC` branch, which is now
  itself correct again thanks to the FSAL fix.
**Verified:** `test/test_native_dense_order5.jl` (4 testsets, 14 assertions,
`:rust` group) — see §4 below. Full 7-group gate green.
**Gotcha for the next person:** a "physically sensible" step size is far too
fine to see interpolation order at all here. See §3.
```

## 2. BACKLOG.md status line to set

Track S5 item 3 → 🟢 done (2026-07-23). Suggested text:

> **3. Order-5 dense output — 🟢 done (2026-07-23).** Dense output between
> accepted steps now uses the Calvo–Montijano–Rández order-5 continuous
> extension (2 extra lazily-evaluated RK stages) instead of the "free"
> quartic, on both the native and the pure-Julia steppers. Landed together
> with a fix for a pre-existing FSAL bug that had been collapsing the
> *existing* quartic to first order on every stepper — see
> `native-port/PORT_LOG.md`'s 2026-07-23 entry.

## 3. NATIVE_SUPPORT_MATRIX.md

No cell changes. Dense output is not a per-geometry capability — the
interpolant is geometry-agnostic (`eval_extra_stage` dispatches through the
same `rhs_*` branches `step` uses, so every geometry already supported by
the native stepper gets the order-5 extension) and the CUDA backend's
fallback is to the order-4 branch, which is not a `NativeIneligible`
fallback and so has no row in that table.

---

## 4. Provenance and verification of the tableau

### Source

Calvo, Montijano & Rández, "A fifth-order interpolant for the Dormand and
Prince Runge–Kutta method", *J. Comput. Appl. Math.* **29** (1990) 91–100.
The coefficients were transcribed from the PyNumAl / Python-Numerical-
Analysis public tableau collection ("Dormand-Prince RK5(4) Pair.txt",
<https://github.com/PyNumAl/Python-Numerical-Analysis>), which cites that
paper and the PDF at <https://core.ac.uk/download/pdf/81164751.pdf>. The
original PDF was **not** independently fetchable at the time of writing, so
the transcription was never taken on faith — it was verified two ways
before use, and both checks are reproducible from this file alone.

### Check A — exact rational consistency (no floating point)

Run in exact `fractions.Fraction` arithmetic against the standard DP5
Butcher tableau:

| Condition | Meaning | Result |
|---|---|---|
| `Σⱼ a₈ⱼ = 2/5` | extra stage 8 sits at its stated node | ✅ exact |
| `Σⱼ a₉ⱼ = 2/5` | extra stage 9 sits at the same node | ✅ exact |
| `bᵢ(1) = b5ᵢ` for i=1..7 | interpolant reproduces the accepted step | ✅ exact |
| `b₈(1) = b₉(1) = 0` | extra stages contribute nothing at θ=1 | ✅ exact |
| `bᵢ′(0) = δᵢ₁` | y′(t₀) = k1 | ✅ exact |
| `bᵢ′(1) = δᵢ₇` | y′(t₀+h) = k7, i.e. FSAL-consistent | ✅ exact |

That last pair matters: they are what makes the interpolant C¹-continuous
with the underlying solution at both ends of every step, so dense output
never shows a kink at a step boundary.

### Check B — numerical order on a scalar ODE

On `y′ = y·cos(t) − y²`, `y(0.3) = 0.7`, comparing the interpolant against a
20000-substep reference over one step, max over θ ∈ {0.15, 0.35, 0.55, 0.75,
0.9}:

| h | local defect | ratio |
|---|---|---|
| 0.2 | 1.94e-09 | — |
| 0.1 | 3.69e-11 | 52.6 |
| 0.05 | 6.17e-13 | 59.8 |
| 0.025 | 9.22e-15 | 67.0 |
| 0.0125 | 8.77e-15 | 1.05 ← FP floor |

Ratios converge to 2⁶ = 64, i.e. local defect O(h⁶) ⇒ a genuine order-5
continuous extension. **The last row is the important one for anyone
extending this work**: the instant the defect reaches ~1e-14 the measured
ratio collapses to ~1 and the experiment stops saying anything. That is a
floor artifact, not a failure of the interpolant.

### Check C — the same measurement on the real propagator

`test/test_native_dense_order5.jl`, mode-averaged Kerr `RealGrid`
(`prop_capillary`, 125 µm He capillary at 1 bar, 800 nm, 1 µJ, 30 fs),
single step from `t0`, reference from a 32×-finer fixed-step stepper:

| h (m) | order-5 defect | ratio | order-4 defect | ratio |
|---|---|---|---|---|
| 4e-2 | 3.30e-07 | — | 9.57e-07 | — |
| 2e-2 | 5.48e-09 | 60.2 | 3.22e-08 | 29.8 |
| 1e-2 | 8.69e-11 | 63.0 | 1.02e-09 | 31.4 |
| 5e-3 | 1.36e-12 | 63.7 | 3.21e-11 | 31.9 |

2⁶ = 64 and 2⁵ = 32 respectively, on the real nonlinear RHS. The Julia
`PreconStepper` reproduces the order-5 column to ~1e-13 relative, and the
two steppers' dense output agrees with each other to **~1e-17** relative
(essentially bitwise), which is what pins Rust's `eval_extra_stage`/
`DP5_EXTRA_*` to Julia's `_dp5_extra_stages!`/`dp5_extra_*` independently of
how accurate either is in absolute terms.

---

## 5. Why the WIP's convergence test appeared to fail — and the two traps in it

The WIP commit's own blocker note read: *"step-doubling confirms the stepper
is O(h⁵) accurate (~1e-14), but the fine-reference convergence test shows
BOTH the order-4 and the new order-5 interpolant degrading as O(h²) — and so
does the accepted-step endpoint, which involves no interpolation at all."*

Two independent problems were stacked on top of each other, which is why
neither was obvious:

**Trap 1 — a real bug, correctly detected.** The O(h²) was genuine: the FSAL
k1 clobber described in §1. The agent's instinct that "the endpoint involves
no interpolation, so suspect the harness" was the one wrong inference — an
accepted-step *endpoint* comparison is not affected by k1 at all, so that
observation (recorded only in the excluded `_tmp_debug_*` scratch files, and
not reproducible from the committed test) was almost certainly itself a
floor artifact of the too-benign regime below. Chasing it is what consumed
the remaining budget. The committed test, run as-is, prints everything
needed to reach the right answer in one shot.

**Trap 2 — a regime that proves nothing.** The WIP test used `h0 = 2e-3` m,
which is the physically sensible step size for a 0.15 m capillary. With the
FSAL bug fixed, the order-5 defect there is **5.7e-15 relative** — the
double-precision floor — and all four measured errors sit within a factor of
3 of each other, giving ratios of 3.2 / 0.92 / 1.10. A test in that regime
cannot distinguish order 5 from order 4 from order 1. The fix is `h0 = 4e-2`
(a quarter of the fibre), which is usable only because `min_dt == max_dt`
makes `stepcontrol_pi` force-accept regardless of the error estimate.

The reason such a coarse step is needed at all is structural, not
incidental: this is an *interaction-picture* integrator, so the entire
linear/dispersive part of the propagation is handled exactly by the
integrating factor and contributes nothing to the interpolation defect. Only
the comparatively weak Kerr nonlinearity does. Any future dense-output order
test on this codebase will need the same treatment — either a very coarse
step or a much more strongly nonlinear configuration.

## 6. Impact beyond this item

The FSAL fix changes *results*, not just accuracy bookkeeping: every saved
output point of every simulation that does not land exactly on an accepted
step boundary — i.e. very nearly all of them — was previously computed with
a first-order-accurate interpolant on **every** stepper Amalthea has
(`Stepper`, `PreconStepper`, `RustPreconStepper`, `RustNativeStepper`, and
the CUDA-resident backend), and is now order-5. This is inherited from
upstream Luna.jl (`RK45.jl`'s `step!`), where the same bug is present; it is
worth reporting upstream.

It also retroactively explains a Phase 8 observation recorded in
`PORT_LOG.md`: switching native dense output from linear to "quartic"
resolved a batch of general-suite failures. The quartic was in fact never
better than O(h²) — the real win there came from the accompanying
`native_apply_prop` call, i.e. from applying the interaction-picture
propagator at all, not from the polynomial order.

## 7. Cost

2 extra nonlinear RHS evaluations per `interpolate` call, i.e. per saved
output point — roughly 2/7 of one accepted step. Nothing is added to the
hot `step!` path. For a typical run (thousands of steps, ~10²–10³ saved
points) this is well under 0.1% of total runtime, and it is entirely absent
from runs that never request dense output.
