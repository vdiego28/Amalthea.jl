# Inbox note — Phase I.5a: multi-mode ZeisbergerMode/VincettiMode on the native modal path

Agent branch: `worktree-agent-a7c5abd2916ae82a3`. Exclusive file zone used:
`src/RK45.jl` (modal guard block, ~line 1639-1728) and a new test file
`test/test_native_modal_zv.jl`. **No `amalthea/src/native.rs` change was
needed** — see "Dependency" section below for why, and why that's not a gap.

---

## 1. PORT_LOG.md entry (paste at the bottom, dated appropriately)

```
## 2026-07-22 — Phase I.5a — Multi-mode ZeisbergerMode/VincettiMode on the native modal path — Claude (sonnet-5)
**Status:** complete
**Did:** Relaxed `RK45.jl`'s native modal guard (~line 1639-1728) to accept
`Antiresonant.ZeisbergerMode`/`Antiresonant.VincettiMode` in addition to bare
`Capillary.MarcatiliMode`, for the *multi-mode* `TransModal` path (the
single-mode/mode-averaged case was already native — see Phase I item 5's
2026-07-16 entry). `StepIndexMode` remains out of scope (no closed-form
`neff`, per the original item's own scoping).
**How:** Confirmed by code trace (not assumed) that `ZeisbergerMode`/
`VincettiMode` delegate `field`/`N`/`dimlimits` to their wrapped inner
`MarcatiliMode` verbatim (`Antiresonant.jl:126-128`, `:381-383`), and that
the native modal RHS (`native.rs`'s `rhs_modal_pointcalc`/`mode_angle_xy`,
~line 1937-1962) synthesizes the transverse field *only* from
`modal_a`/`modal_unm`/`modal_order`/`modal_kind`/`modal_phi`/
`modal_inv_sqrt_n` — i.e. exactly the quantities `field`/`N` determine, and
nothing else. Dispersion (`neff`, where Zeisberger/Vincetti genuinely
differ from bare Marcatili) never enters the native modal RHS at all: it is
baked once into `linop` by Julia's own `LinearOps.make_const_linop(grid,
modes, λ0)`, which already dispatches per-mode through each mode's own
`Modes.neff` method (`LinearOps.jl:539`) — unaffected by this guard.
`Modes.Aeff` — the one field `VincettiMode` does NOT delegate
(`Antiresonant.jl:523`, its own `neff_real`/`Rco_eff`/`Δneff`-based
formula) — never enters `TransModal` at all (`Aeff` is a `TransModeAvg`-only
normalisation quantity); confirmed by grep, no `Aeff` reference anywhere in
`NonlinearRHS.jl`'s `TransModal`/`Erω_to_Prω!`/`pointcalc!`.

Given that, the fix is a pure accessor change: added a small
`_modal_inner_marcatili(m)` helper (RK45.jl, inside the `if is_modal` block)
that returns `m` for a bare `MarcatiliMode`, `m.m` for `ZeisbergerMode`/
`VincettiMode` (both use `.m` as the wrapped-mode field name), and `nothing`
otherwise (drives the type-acceptance `NativeIneligible` throw). Every
subsequent raw struct-field read in the guard/setup block
(`m.kind`/`m.a`/`m.unm`/`m.ϕ`/`m.n` — these are literal struct fields, not
generic functions, so they don't delegate through the wrapper types) now
reads from `inner = _modal_inner_marcatili.(modes)` instead of `modes`
directly. `_modal_inner_marcatili` is the identity on a bare
`MarcatiliMode`, so **every pre-existing modal code path (plain-Marcatili
multi-mode) is bit-for-bit unchanged** — only the type-acceptance check
widened. `invsqrtn` deliberately still calls `Amalthea.Modes.N(m, z=0.0)`
on the *wrapper* `modes` (not `inner`) to go through the generic/delegating
API rather than reaching past it — functionally identical (it delegates to
the same inner call) but keeps the guard's intent legible.

`is_zdep_modal_taper` (tapered-radius Phase E.2 path) needed no new
handling: `Capillary`'s `ZDepLinopModalTaper` constructor reads
`.model`/`.coren` directly on each mode, fields `ZeisbergerMode`/
`VincettiMode` don't delegate, so a wrapped-mode config can never
legitimately produce that linop type in the first place — tapered
Zeisberger/Vincetti stays correctly out of scope (falls back via the
existing `m.a isa Number` check, unchanged behavior).
**Decisions:**
- Scoped to Zeisberger/Vincetti only, per the task; `StepIndexMode` excluded
  (no closed-form `neff`, needs `Roots.find_zeros` — a materially different,
  harder follow-up per the original Phase I item 5 entry's own scoping).
- No `native.rs` change: the premise research (see task brief) was verified
  correct — the native modal RHS never reads `neff`/dispersion at all, only
  the pre-baked `linop` array and the Marcatili field-synthesis parameters,
  both already mode-type-agnostic once unwrapped.
- Kept the `_modal_inner_marcatili` helper local to the `if is_modal` block
  (not a module-level function) — it's small, single-purpose, and this keeps
  the whole change inside the assigned exclusive file zone (RK45.jl's modal
  guard block only).
**Gotchas:**
- Struct-field access (`m.kind`, `m.a`, `m.unm`, `m.ϕ`, `m.n`) does **not**
  go through Julia's method-dispatch delegation the way `field(m,...)`/
  `N(m,...)` do — `Antiresonant.jl`'s `@eval` delegation loop only covers
  generic functions, so any future guard code that reads a struct field
  directly off `modes[i]` (instead of `inner[i]`) will silently break for
  wrapped types (likely a `FieldError`/`MethodError` at construction, not a
  silent wrong-answer, but still worth remembering next time this block is
  touched).
- Vincetti's default `VincettiMode(a; wallthickness=a, tube_radius=a, ...)`-
  style parameterization (a bare `tube_radius=a` with the same value as the
  core radius) triggers a "negative gap between resonators" `@warn` —
  physically degenerate but numerically fine (equivalence still holds to
  ~1e-15). Test file derives a proper `Rco` via `Antiresonant.getRco(r_ext,
  Ntubes, δ)` (same pattern as `test_antiresonant.jl`'s Vincetti testset) to
  avoid it.
**Tests:** New `test/test_native_modal_zv.jl`, two `@testitem`s (Zeisberger,
Vincetti), each with three testsets:
  1. **Non-vacuousness proof — native path genuinely taken.** Drives the
     actual `RK45.solve_precon` integration point (not a direct
     `RustNativeStepper(...)` construction) for one tiny step and asserts
     `RK45._LAST_STEPPER_TYPE[] <: RK45.RustNativeStepper`. Before this
     change this config threw `NativeIneligible` inside `solve_precon`'s
     try/catch and silently fell back to `PreconStepper` — this assertion
     would have failed pre-fix.
  2. **Single-step equivalence:** Zeisberger 6.08e-18, Vincetti 0.0 (exact).
  3. **Full-solve equivalence (fixed `max_dt=min_dt=dt`, per TESTING.md §3):**
     Zeisberger 3.51e-16, Vincetti 2.56e-15 — both far inside the ~1e-6
     tolerance tier the plain-Marcatili Phase 5 gate uses.
     **Non-vacuousness of the two-mode physics itself:** HE11→HE12 energy
     transfer fraction 2.43e-5 (Zeisberger) / 4.15e-5 (Vincetti), both
     comfortably clearing the >1e-6 bar and in the same range as the
     original Phase 5 gate's own 2.0e-5 sanity figure.
  Also ran the pre-existing `test/test_native_modal.jl` (bare-Marcatili,
  same guard block) to confirm no regression: all assertions pass, full-solve
  rel error unchanged (~3.97e-16, matches pre-change baseline) — expected,
  since `_modal_inner_marcatili` is the identity on `MarcatiliMode`.
  `cargo test --release` in `amalthea/`: 69/69 passing (no Rust code
  touched, ran as a sanity check per AGENTS.md §2).
  Did **not** run the other 7 `test_native_modal_*.jl` files (mixture/env/
  full/npol2/raman/taper/threading) — the identity-on-`MarcatiliMode`
  argument above covers them without spending the shared 12-core budget;
  the lead's full gate is the backstop.
**Next:** `StepIndexMode` multi-mode remains open (needs its own
`neff`-independent-of-native-RHS argument reverified once someone adds a
closed-form or LUT-based `neff` for it — currently numerical root-finding
only, no natural way to bake it into `linop` cheaply the way Marcatili's
Chebyshev fit or Zeisberger/Vincetti's analytic forms do). Also open:
Phase J.2/J.3-adjacent, unrelated to this item.
```

## 2. BACKLOG.md status line(s)

Replace the current item 1 under "Open remainders lifted out of the archived
phases" with two lines (Zeisberger/Vincetti now closed, StepIndexMode still
open):

```
1. 🟢 **Phase I.5a — `ZeisbergerMode`/`VincettiMode` multi-mode: native Rust
   port done (2026-07-22).** `RK45.jl`'s native modal guard now accepts these
   two wrapper types (unwraps to the inner `Capillary.MarcatiliMode` for the
   accessor fields the guard/setup read as struct fields — `field`/`N`
   already delegated through generic dispatch). No `native.rs` change needed:
   the native modal RHS never reads `neff`/dispersion, only the pre-baked
   `linop` and Marcatili field-synthesis parameters. See ARCHIVE.md Phase I
   item 5 and `test/test_native_modal_zv.jl`.
1b. 🟡 **Phase I.5b — `StepIndexMode` multi-mode: still native-ineligible.**
    No closed-form `neff` (needs `Roots.find_zeros`), so it can't cheaply
    join Zeisberger/Vincetti's "bake dispersion into `linop`, unwrap for the
    field-synthesis accessors" pattern without further design work. See
    ARCHIVE.md Phase I item 5.
```

(The lead should fold this into ARCHIVE.md's Phase I item 5 prose too, but
per the inbox-note rule I'm not editing ARCHIVE.md/BACKLOG.md myself.)

## 3. NATIVE_SUPPORT_MATRIX.md cell that changed

Current text (Modal table, `NATIVE_SUPPORT_MATRIX.md` line 70):

```
| Multi-mode `StepIndexMode`/`ZeisbergerMode`/`VincettiMode` (i.e. several such modes propagating together via `TransModal`) | ❌ (`RK45.jl`'s native modal guard requires `Capillary.MarcatiliMode` — `native.rs`'s mode-field synthesis is Marcatili-parameterized. High-level-reachable as of 2026-07-16 via `prop_capillary(...; modes=[m1,m2])`/`prop_stepindex`, but this specific multi-mode case still falls back to `PreconStepper`. Native port tracked as a follow-up, see Phase I item 5) | ❌ |
```

Proposed replacement (splits the row now that Zeisberger/Vincetti are ✅ and
StepIndexMode stays ❌):

```
| Multi-mode `ZeisbergerMode`/`VincettiMode` (several such modes propagating together via `TransModal`) | ✅ (2026-07-22: `RK45.jl`'s native modal guard unwraps to the inner `Capillary.MarcatiliMode` for the field-synthesis accessors — `native.rs`'s mode-field synthesis remains Marcatili-parameterized, but both wrapper types delegate `field`/`N` to it verbatim, and dispersion is baked into `linop` by Julia before the native RHS ever runs. See `test/test_native_modal_zv.jl`, Phase I.5a) | ✅ (same) |
| Multi-mode `StepIndexMode` | ❌ (no closed-form `neff` — numerical root-finding only, doesn't fit the "bake dispersion into `linop`, unwrap for accessors" pattern above without further design work. High-level-reachable via `prop_stepindex(...; modes=[m1,m2])`. Phase I.5b, still open) | ❌ |
```

Also update the cross-reference prose just above the Modal table (lines
39-48, in the Mode-averaged section) — "Only *multi-mode* configurations of
these three types (`TransModal`) are still native-ineligible" should become
"...of `StepIndexMode` are still native-ineligible; `ZeisbergerMode`/
`VincettiMode` multi-mode is native as of Phase I.5a."

## 4. Dependencies needed outside my file zone

**None.** The premise held exactly as scoped: no `amalthea/src/native.rs`
change was needed (verified, not assumed — see PORT_LOG entry's "How"
section for the grep/trace). All changes stayed inside `src/RK45.jl`'s
modal guard block and the new test file, both in my exclusive zone.
